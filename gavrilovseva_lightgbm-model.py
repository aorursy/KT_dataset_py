import pandas as pd
import numpy as np
import lightgbm as lgb

from datetime import datetime, date
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm_notebook  # pip install tqdm, para la barra de progreso

pd.set_option('display.max_columns', 999)

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
events = pd.read_csv('../input/events_up_to_01062018.csv', low_memory=False)
labels = pd.read_csv('../input/labels_training_set.csv')
test = pd.read_csv('../input/trocafone_kaggle_test.csv')
def evaluate_model(y_true, model=None, X_test=None, prediction=None, probabilites=None):
    if model is not None:
        if prediction is None:
            prediction = model.predict(X_test)
        if probabilites is None:
            probabilites = model.predict_proba(X_test)[:, 1]
    if prediction is not None:
        print('Accuracy:        ', accuracy_score(y_true, prediction))
        print('ROC AUC Predict: ', roc_auc_score(y_true, prediction))
    if probabilites is not None:
        print('Avg Log loss:    ', log_loss(y_true, probabilites))
        print('Sum Log loss:    ', log_loss(y_true, probabilites, normalize=False))
        print('ROC AUC Proba:   ', roc_auc_score(y_true, probabilites))
def get_train_set(df):
    bar = tqdm_notebook(total=12, desc='Progress')
    last_date_ts = pd.Timestamp(date(2018, 6, 1))
    
    df = df.copy()
    bar.update(1)
    
    df['brand'] = df.model.str.split(' ').str[0]
    df['operating_system'] = df.operating_system_version.str.split(' ').str[:-1].str.join(' ')
    # A los sistemas operativos fuera del top 10 les ponemos "Other"
    top10os = df.operating_system.value_counts().head(10)
    df.loc[(~df.operating_system.isin(top10os.index))&(df.operating_system.notnull()), 'operating_system'] = 'Other'
    
    df['browser'] = df.browser_version.str.split(' ').str[:-1].str.join(' ')
    # Misma lógica que con sistemas operativos
    top20browsers = df['browser'].value_counts().head(20)
    df.loc[(~df['browser'].isin(top20browsers.index))&(df['browser'].notnull()), 'browser'] = 'Other'
    
    df['ts'] = pd.to_datetime(df.timestamp)
    #df['month'] = df.ts.dt.month
    #df['hour'] = df.ts.dt.hour
    df.sort_values('ts', inplace=True)
    
    bar.update(1)
    
    # A partir del screen resolution se puede saber qué versión del sitio se vio
    #resolutions = df['screen_resolution'].str.split('x', expand=True).fillna(0).astype(int)
    #df['screen_width'], df['screen_height'] = resolutions[0], resolutions[1]
    #df['screen_size'] = pd.cut(df['screen_width'], [1, 768, 992, 1200, 100000], right=False,
    #                           labels=['Extra small', 'Small', 'Medium', 'Large'])
    
    bar.update(1)
    
    gb = df.groupby('person')
    
    # El primer feature es simplemente la cantidad de eventos de cada persona
    train = pd.DataFrame(gb.size(), columns=['num_events'])
    #train = train.join(gb.month.value_counts().unstack().add_prefix('month_'))
    #train = train.join(gb.hour.value_counts().unstack().add_prefix('hour_'))
    
    bar.update(1)
    
    # A continuación una serie de features que cuenta las apariciones de valores de 
    # los diferentes campos: cantidad de ocurrencias de cada evento, cantidad de veces 
    # que vio cada marca, cantidad de visitas desde cada browser.
    train = train.join(gb.event.value_counts().unstack().add_prefix('event_'))   
    train = train.join(gb.brand.value_counts().unstack().add_prefix('brand_')) 
    train = train.join(gb.storage.value_counts().unstack().add_prefix('storage_'))
    
    train = train.join(gb.browser.value_counts().unstack().add_prefix('browser_'))
    train = train.join(gb.operating_system.value_counts().unstack().add_prefix('os_'))
    
    bar.update(1)
    
    # Los diferentes colores son demasiados, solo usamos el top 20
    top_20_colors = df.color.value_counts().nlargest(20).index
    train = train.join(gb.color.value_counts().unstack()[top_20_colors].add_prefix('color_'))
    
    train = train.join(gb.condition.value_counts().unstack().add_prefix('condition_'))
    train = train.join(gb.channel.value_counts().unstack().add_prefix('channel_'))
    train = train.join(gb.device_type.value_counts().unstack().add_prefix('device_'))
    
    #train = train.join(gb.screen_size.value_counts().unstack().add_prefix('screen_size'))
    
    bar.update(1)
    
    regions = df.region.value_counts()
    # Las regiones con menos de 140 ocurrencias no son de Brasil
    top_regions = gb.region.value_counts().unstack()[regions[regions>140].index]
    train = train.join(top_regions.add_prefix('region_'))
    
    train = train.join(gb.staticpage.value_counts().unstack().add_prefix('staticpage_'))
    
    bar.update(1)
    
    country_count = gb.country.value_counts().unstack()

    # Hay una cierta cantidad de gente sin visitas, asumo que son de Brasil
    train['is_brasil'] = (country_count.Brazil > 0).astype(int).fillna(1)
    
    last_event = gb.ts.nth(-1)
    first_event = gb.ts.nth(0)
    # Cantidad de dias entre el primer y el último evento
    time_to_last_event = last_event - first_event
    train['days_to_last_event'] = time_to_last_event.dt.days
    
    bar.update(1)
    
    ps = {}
    for t, g in gb:
        # Por cada persona calculo el promedio de diferencias de tiempo entre eventos consecutivos
        ps[t] = g.ts.sub(g.ts.shift()).mean()
    train['avg_seconds_between_events'] = pd.Series(ps).dt.total_seconds().fillna(0)
    
    bar.update(1)
    
    # Segundos hasta la fecha límite (1 de Junio)
    limit_date_series = pd.Series(data=last_date_ts, index=train.index)
    train['seconds_till_limit_date_since_last_event'] = (limit_date_series - last_event).dt.total_seconds()
    train['seconds_till_limit_date_since_first_event'] = (limit_date_series - first_event).dt.total_seconds()
    
    bar.update(1)
    
    # Se hace el mismo cálculo de cantidad de segundos hasta el 1o de Junio por cada tipo de evento
    last_event_by_person = df.groupby(['person', 'event']).ts.last().unstack()
    for col in last_event_by_person:
        time_since_last_event = limit_date_series - last_event_by_person[col]
        train['seconds_till_limit_date_since_last_' + col] = time_since_last_event.dt.total_seconds().fillna(0)
    
    bar.update(1)
    
    # Features sobre la última visita
    
    #visits_gb = df[df.event=='visited site'].groupby('person')
    #last_visits = visits_gb.last()
    #time_since_last_visit = (limit_date_series - last_visits.ts).dt.total_seconds()
    #train['seconds_till_limit_date_since_last_visit'] = time_since_last_visit
    #train = train.join(pd.get_dummies(last_visits.device_type, prefix='last_device'))
    #train['last_screen_width'] = last_visits.screen_width
    #train['last_screen_height'] = last_visits.screen_height
    
    #ps = {}
    #for t, g in visits_gb:
        # Por cada persona calculo el promedio de diferencias de tiempo entre eventos consecutivos
    #    ps[t] = g.ts.sub(g.ts.shift()).mean()
    #train['avg_seconds_between_visits'] = pd.Series(ps).dt.total_seconds().fillna(0)

    train = train.fillna(0)
    
    bar.update(1)
    
    return train
all_false = np.zeros(len(labels))
evaluate_model(labels.label, prediction=all_false, probabilites=all_false)
# Esta llamada puede tardar un poco
full_train = get_train_set(events)
train = full_train.loc[labels.person]
assert all(train.index == labels.person)
scaler = StandardScaler()
features = scaler.fit_transform(train)
features.shape
X_train, y_train = features, labels.label
y_test = test.set_index('person')
X_test = scaler.transform(full_train.loc[y_test.index])
#from sklearn.linear_model import LogisticRegression
#lr_model = LogisticRegression(C=0.1, max_iter=1000)
#lr_model.fit(X_train, y_train)
#evaluate_model(y_train, lr_model, X_train)
import lightgbm as lgb
lgb_model = lgb.LGBMClassifier(boosting_type='dart', num_leaves=5, n_estimators=1000, metric='AUC',
                               learning_rate=0.05, colsample_bytree=0.9)
lgb_model.fit(X_train, y_train)
evaluate_model(y_train, lgb_model, X_train)
import shap  # pip install shap
shap.initjs()
explainer = shap.TreeExplainer(lgb_model)
shap_vals = explainer.shap_values(X_train)
feature_names = full_train.columns.tolist()
shap.summary_plot(shap_vals, X_train, max_display=30, feature_names=feature_names)
conversion_probs = lgb_model.predict_proba(X_test)[:, 1]
subm = pd.DataFrame({'person': y_test.index, 'label': conversion_probs})
subm.to_csv('submission.csv', index=False)