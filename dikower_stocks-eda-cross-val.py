!jupyter nbextension enable --py widgetsnbextension

!pip install colour

# !npm update npm -g

# !jupyter labextension install @jupyter-widgets/jupyterlab-manager

import ipywidgets as widgets

from ipywidgets import interact, interact_manual

from IPython.display import HTML



import warnings

warnings.filterwarnings("ignore")



import os

import json

import pickle

import tqdm

import random



import pandas as pd

import numpy as np



import catboost as cb

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.linear_model import Ridge

from sklearn.metrics import median_absolute_error, r2_score, mean_squared_error, explained_variance_score, accuracy_score, classification_report, roc_curve, auc

from sklearn.preprocessing import StandardScaler

from sklearn.manifold import TSNE

from concurrent.futures import ProcessPoolExecutor

from multiprocessing import cpu_count



import colorlover as cl

from plotly import figure_factory as FF

from plotly.offline import iplot, plot, init_notebook_mode, download_plotlyjs

import plotly.plotly as py

import plotly.graph_objs as go

from colour import Color

import cufflinks as cf



import catboost as cb

from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR

from sklearn.linear_model import BayesianRidge



from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier



from sklearn.neighbors import KNeighborsRegressor

# statsmodels



init_notebook_mode(connected=True)

cf.go_offline()

cf.set_config_file(world_readable=True, theme='white', offline=True)

HTML(cl.to_html(cl.scales['9']))
def compress(data: dict) -> tuple:

    if data == {}:

        return data

    data = pd.DataFrame.from_dict(data, orient='index').reset_index(drop=True)

    data['begin'] = pd.to_datetime(data['begin'])

    data = data.sort_values(by=['begin']).drop(['end', 'begin'], axis=1)

    past = data.iloc[:data.shape[0] // 2]

    central = data.iloc[data.shape[0] // 2]

    future = data.iloc[data.shape[0] // 2:]

    

#   Обработка прошлого

    compressed_data = dict()

    for column in past.columns:

        methdos = {'skew': past[column].skew, 'kurt': past[column].kurt, 'min': past[column].min, 'max': past[column].max, 'std': past[column].std, 'mean': past[column].mean, 'var': past[column].var, 'qua': past[column].quantile, 'std': past[column].std, 'med': past[column].median}

        for method_name, method in methdos.items():

            compressed_data[f'{column}_{method_name}_past'] = method()

            

#   Обработка будущего

    regressor = Ridge(alpha=.1)

    for column in ['mean_profit', 'mean_cost', 'close']:

#       Обучаем регрессию для того чтобы взять производную как таргет

        X_train, Y_train = future[column].index.values.reshape(-1, 1), future[column].values

        regressor.fit(X_train, Y_train)

#       =================================================================================================================================================

        compressed_data[f'{column}_regression_future'] = regressor.coef_[0]

        compressed_data[f'{column}_mean_future'] = future[column].mean()

        compressed_data[f'{column}_mean_percentage_deviation_future'] = ((future[column] - central[column]) / central[column]).mean()

    compressed_data['mean_cost_trend_class_future'] = compressed_data[f'mean_cost_mean_future'] > central['mean_cost']

    return compressed_data
with open('../input/all_stock_data.json', 'r', encoding='utf8') as file:

    data = json.load(file)



datasets = dict()

with ProcessPoolExecutor(max_workers=cpu_count()) as executor:

    for company in data.keys():

#         results = dict()

#         for date, _dict in tqdm.tqdm_notebook(data[company].items()):

#             if _dict != {}:

#                 results[date] = compress(_dict)

        results = dict(zip(data[company].keys(), tqdm.tqdm_notebook(executor.map(compress, list(data[company].values())), total=len(data[company]))))

        datasets[company] = pd.DataFrame(results).T.dropna()
@interact

def draw_candles(company=list(data.keys())):

    time_series = pd.DataFrame(data[company][random.choice(list(data[company].keys()))]).T  # random.choice(list(data[company].keys()))

    time_series.index = pd.to_datetime(time_series['begin'])

    time_series.iplot(title=f'Динамика на {time_series.index[0].date().strftime("%y.%m.%d")}',kind='candle', colors=['magenta','grey'], theme='white',asDates=True)

@interact

def difference_distribution(company=list(data.keys())): 

    _data = datasets['Яндекс'][['close_mean_future', 'close_mean_past']].reset_index(drop=True)

    _data['diff'] = (_data['close_mean_future'] - _data['close_mean_past']).abs()

    _data['diff'].iplot(title=f'Распределение разницы между средними прошлого и будущего в процентах компании {company}', layout_update=dict(height=800, width=1000), xTitle='Распределение', yTitle='Разница в процентах', bargap=0.1, opacity=0.8, kind='histogram', colors=['#2A3D51','grey'], histnorm='percent', linecolor='white', theme='white', width=0.3)

@interact

def amount_distribution(company=list(data.keys())):

    lst = []

    for date, _data in data['Яндекс'].items():

        lst.append(len(_data))

    pd.Series(lst).iplot(title=f'Распределение кол-ва записей {company}', layout_update=dict(height=800, width=1000), xTitle='Распределение', yTitle='Кол-во', bargap=0.1, opacity=0.8, kind='histogram', colors=['#FF3D3C','grey'], linecolor='grey', theme='white', width=0.5)
datasets['Яндекс'] = datasets['Яндекс'].astype(float)

target_columns = [column for column in datasets['Яндекс'].columns if 'future' in column]

reg_targets = [target for target in target_columns if 'class' not in target]

class_targets = [column for column in target_columns if 'class' in column]



X, Y_labels = datasets['Яндекс'].drop(target_columns, axis=1).reset_index(drop=True), datasets['Яндекс'][target_columns].reset_index(drop=True)

Y_labels['mean_cost_trend_class_future'] = Y_labels['mean_cost_trend_class_future'].astype(int)
@interact

def target_dist(target=reg_targets):

    Y_labels[target].iplot(kind='hist', layout_update=dict(height=800, width=1000),  barmode='stack', title='Распределение переменной', xTitle='Ошибка', yTitle='Кол-во вхождений', 

               colors=['#C471EC', '#065280', '#34B2E3', '#64D1DA', '#8C103D', '#E24956', '#FF912B', '#FFC74F', '#dff9fb'], opacity=0.7, bargap=0.1, linecolor='grey')
@interact

def class_balance(target=class_targets):

    Y_labels[target].value_counts().reset_index().iplot(kind='pie', labels='index', values=target, title='Баланс классов',

        pull=.01, hole=.09, colors=['#19D3F3', '#636EFA', '#E763FA', '#BA8BF8'], textposition='outside', textinfo='value+percent', width=0, linecolor='grey')

def convert_colorscale_format(colorscale):

    plotly_colorscale = []

    for index, sec_value in enumerate(np.linspace(0, 1, len(colorscale))):

        plotly_colorscale.append([sec_value, str(colorscale[index])])

    return plotly_colorscale



def error_typer(y, prediction):

    if y == prediction:

        if bool(y): return 'true pos'

        else: return 'true neg'

    if bool(y): return 'false pos'

    else: return 'false neg'



def mean(x):

    return round(sum(x)/len(x), 2)



def gradient_generator(color1, color2, n):

    first = Color(color1)

    second = Color(color2)

    return [str(color) for color in list(first.range_to(second, n))]

@interact

def correlation_matrix(company=list(datasets.keys())):

    colors = convert_colorscale_format(gradient_generator('#43C6AC', '#191654', 500))

    data = [go.Heatmap(z=datasets[company].corr().values, x=datasets[company].columns.tolist(), y=datasets[company].columns.tolist(), colorscale=colors)]

    fig = go.Figure(data=data, layout=go.Layout(height=800, width=800))

    iplot(fig)
def remove_collinear_features(x, threshold):

    # Считаем матрицу кореляций

    corr_matrix = x.corr()

    iters = range(len(corr_matrix.columns) - 1)

    drop_cols = []



    # Сравниваем фичу каждую с каждой

    for i in iters:

        for j in range(i):

            item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]

            col = item.columns

            row = item.index

            val = abs(item.values)

            # Если кореляция больше, чем трешхолд

            if val >= threshold:

                drop_cols.append(col.values[0])



    # Удаляем по одному из каждой пары корелируемых

    drops = list(set(drop_cols))

    compression = []

    random.shuffle(drops)

    for i in tqdm.tqdm_notebook(range(3, len(drops), 4)):

        _compressed_data = TSNE().fit_transform(x[[drops[i], drops[i-1], drops[i-2], drops[i-3]]])

        compression.append(pd.DataFrame({f'TSNE_x{i}': _compressed_data[:, 0], f'TSNE_y{i}': _compressed_data[:, 1]}))

    x = x.drop(drops, axis=1).reset_index(drop=True)

    compression.append(x)

    x = pd.concat(compression, axis=1)

    # Возвращаем таргет в таблицу

    return x



X = remove_collinear_features(X, 0.6)
def correlation_matrix():

    colors = convert_colorscale_format(gradient_generator('#43C6AC', '#191654', 500))

    data = [go.Heatmap(z=X.corr().values, x=X.columns.tolist(), y=X.columns.tolist(), colorscale=colors)]

    fig = go.Figure(data=data, layout=go.Layout(height=800, width=800))

    iplot(fig)

correlation_matrix()
def cross_validate_model(model, classifier=False, folds_number=10):

    kf = KFold(n_splits=folds_number, shuffle=True, random_state=42)

    if not classifier:

        columns = [column for column in target_columns if 'class' not in column]

    else:

        columns = [column for column in target_columns if 'class' in column]

    validation_data = {column: [] for column in columns}

    for train_index, test_index in tqdm.tqdm_notebook(kf.split(X), total=folds_number):

        for target in columns:

            X_train, X_val, Y_train, Y_val = X.iloc[train_index].reset_index(drop=True), X.iloc[test_index].reset_index(drop=True), Y_labels[target].iloc[train_index].reset_index(drop=True), Y_labels[target].iloc[test_index].reset_index(drop=True)

            if type(model) == cb.CatBoostRegressor or type(model) == cb.CatBoostClassifier:

                model.fit(

                    X_train, Y_train,

                    use_best_model=True,

                    eval_set=cb.Pool(X_val, Y_val),

                    logging_level="Silent",  # 'Silent', 'Verbose', 'Info', 'Debug'

                    plot=False)

            else:

                model.fit(X_train, Y_train)

            

            prediction = pd.Series(model.predict(X_val)) # median_absolute_error, r2_score, mean_squared_error

            if not classifier:

                validation_data[target].append({

                    'Y_val': Y_val,'prediction': prediction, 

                    'abs_diffense': (Y_val - prediction).abs(), 

                    'mae': round(median_absolute_error(Y_val, prediction), 2), 

                    'rmse': round(np.sqrt(mean_squared_error(Y_val, prediction)), 2),

                    'r2': round(r2_score(Y_val, prediction), 2)})

            else:

                fpr, tpr, thresholds = roc_curve(Y_val, model.predict_proba(X_val)[:, 1])

                

                validation_data[target].append(

                    {'Y_val': Y_val,'prediction': prediction, 'X_t-SNE': TSNE(n_components=3).fit_transform(X_val), 'fpr': fpr, 'tpr': tpr, 'auc': auc(fpr, tpr), 

                     'report': pd.DataFrame(classification_report(Y_val, prediction, output_dict=True)),

                     'accuracy': accuracy_score(Y_val, prediction)})

    return validation_data



models = [

    ('SVR', SVR(), False),

    ('SVC', SVC(probability=True), True),

    ('KNeighbors', KNeighborsClassifier(), True),

    ('GaussianNB', GaussianNB(), True),

    ('LogisticRegression', LogisticRegression(), True),

#     ('Catboost', cb.CatBoostClassifier(task_type="GPU", early_stopping_rounds=30), True),

#     ('Catboost', cb.CatBoostRegressor(task_type="GPU", early_stopping_rounds=30), False),

    ('RandomForest', RandomForestClassifier(), True),

    ('RandomForest', RandomForestRegressor(), False),

    ('BayesianRidge', BayesianRidge(), False)

]

models_validation = {True: dict(), False: dict()}

for name, model, is_classifier in models:

    print(name)

    models_validation[is_classifier][name] = cross_validate_model(model, is_classifier, 10)
@interact

def regression_distribution(model=models_validation[False].keys(), target=reg_targets):

    dist = []

    mae, rmse, r2 = 0, 0, 0

    counter = {'mae': [], 'rmse': [], 'r2': []}

    for fold in models_validation[False][model][target]:

        dist.append(fold['abs_diffense'])

        counter['mae'].append(str(fold['mae']))

        counter['rmse'].append(str(fold['rmse']))

        counter['r2'].append(str(fold['r2']))

        mae, rmse, r2 = mae + fold['mae'], rmse + fold['rmse'], r2 + fold['r2']

        

    max_len = {'mae': len(max(counter['mae'], key=len)), 'rmse': len(max(counter['rmse'], key=len)), 'r2': len(max(counter['r2'], key=len))}

    names = []

    for _mae, _rmse, _r2 in zip(counter['mae'], counter['rmse'], counter['r2']):

        names.append(f"mae: {_mae.rjust(max_len['mae'])} rmse: {_rmse.rjust(max_len['rmse'])} r2: {_r2.rjust(max_len['r2'])}")

        

    n = len(models_validation[False][model][target])

    mae, rmse, r2 = round(mae/n, 2), round(rmse/n, 2), round(r2/n, 2)

    

    colors = gradient_generator('#C471EC', '#1CBDE9', n)

    dist = pd.DataFrame(dict(zip(names, dist)))

    dist.iplot(kind='hist', title=f'{model}; Распределние ошибок при кросс-валидации. Средние значения mae: {mae}, rmse: {rmse}, r2: {r2}', 

               bargap=0., xTitle='Ошибка', yTitle='Кол-во вхождений', width=0., colors=colors, barmode='stack', opacity=0.8, 

               layout_update=dict(height=800, width=1600)#.to_plotly_json()

              )
@interact

def regr_predictions(model=models_validation[False].keys(), target=reg_targets):

    dots = list()

    n = 0

    for fold in models_validation[False][model][target]:

        local = pd.concat([fold['Y_val'].reset_index(drop=True), fold['prediction'].reset_index(drop=True)], axis=1)

        local.columns = ['Y_val', 'prediction']

        local['fold'] = str(n)

        dots.append(local)

        n += 1

    dots = pd.concat(dots, axis=0)

    

    colors = gradient_generator('#C471EC', '#1CBDE9', n)

    dots.iplot(

        x='Y_val', y='prediction', layout_update=dict(height=600, width=1000),

        kind='scatter', opacity=0.9, mode='markers', size=3, categories='fold', width=0.0, symbol='circle', 

        world_readable=True, title=f'{model} Линейность предсказаний к валидирующей выборке' ,yTitle='Prediction', xTitle='Y validation',

        colors=colors)
@interact

def draw_dots(model=models_validation[True].keys(), target=class_targets, color_info=['all',' guessed']):

    dots = {'X': [], 'Y': [], 'Z': [], 'type': [], 'guessed': []}

    stats = {0: dict(), 1: dict(), 'accuracy': []}

    for fold in models_validation[True][model][target]:

        dots['X'] += fold['X_t-SNE'][:, 0].tolist()

        dots['Y'] += fold['X_t-SNE'][:, 1].tolist()

        dots['Z'] += fold['X_t-SNE'][:, 2].tolist()

        local = pd.concat([fold['Y_val'], fold['prediction'].astype(int)], axis=1)

        local.columns = ['Y_val', 'prediction']

#         local.apply(lambda x: print(x['Y_val'], x['prediction']), axis=1)

        dots['type'] += local.apply((lambda row: error_typer(row['Y_val'], row['prediction'])), axis=1).values.tolist()

        dots['guessed'] += (fold['Y_val'] == fold['prediction']).replace({0: 'Not Guessed', 1: 'Guessed'}).tolist()

        stats['accuracy'].append(fold['accuracy'])

        for i in range(2):

            for key in ['f1-score', 'precision', 'recall']:

                stats[i][key] = stats[i].get(key, []) + [fold['report'][str(i)][key]]

    dots = pd.DataFrame(dots)

    dots.iplot(

        x='X', y='Y', z='Z', layout_update=dict(height=800, width=1000),

        kind='scatter3d', opacity=0.9, mode='markers', size=3, categories='type' if color_info=='all' else 'guessed', width=0.0, symbol='circle', world_readable=True, 

        title=f"{model} a: {mean(stats['accuracy'])}, p-0: {mean(stats[0]['precision'])}, r-0: {mean(stats[0]['recall'])}, f1-0: {mean(stats[0]['f1-score'])}"

        f" f1-1: {mean(stats[1]['f1-score'])}; p-1: {mean(stats[1]['precision'])}, r-1: {mean(stats[1]['recall'])}, f1-1: {mean(stats[1]['f1-score'])}",

        xTitle='X t-SNE', yTitle='Y t-SNE', zTitle='Z t-SNE',

        colors=['#19D3F3', '#E763FA', '#636EFA', '#BA8BF8'])
@interact

def roc_curve(model=models_validation[True].keys(), target=class_targets):

    curve_data = list()

    aucs = list()

    n = 0

    for fold in models_validation[True][model][target]:

        curve_data.append(pd.DataFrame({'x': fold['fpr'], 'y': fold['tpr'], 'name': f"AUC: {round(fold['auc'], 2)}"}))

        aucs.append(fold['auc'])

        n += 1

    pd.concat(curve_data, axis=0).iplot(mode='lines+markers',  x='x', y='y', categories='name', title=f'{model}; ROC-кривая; avg AUC: {mean(aucs)}', xTitle='FPR', yTitle='TPR', 

        layout_update=dict(height=1000, width=1000), colors=gradient_generator('#C471EC', '#1CBDE9', n), width=0.3, size=0.1, fontsize=25)