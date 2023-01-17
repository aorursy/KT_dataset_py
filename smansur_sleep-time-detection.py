import pandas as pd
from datetime import datetime, timedelta
import numpy as np
data = pd.read_csv('../input/daily-activity-test-data/daily_activity.csv', index_col = 'id' , parse_dates=['datetime'])
data.head()
data.info()
data.wakeup.value_counts()
def feature_generation(df_orig):
    '''
    функция принимает на вход датафрейм входного формата конкурса и возвращает расширенный признаками набор данных
    '''
    df = df_orig.copy()
    df['weekday'] = df.apply(lambda x: x.datetime.weekday(), axis = 1)
    df['weekend'] =df.apply(lambda x: 1 if x.weekday in [5,6] else 0, axis=1)
    df['dayid'] = df.apply(lambda x:x.datetime.date(), axis=1)
    df['hour'] = df.apply(lambda x:int(x.datetime.time().hour), axis=1)
    df['month'] = df.apply(lambda x:int(x.datetime.month), axis=1)
    df['day'] = df.apply(lambda x:int(x.datetime.day), axis=1)
    # промежуток времени до предыдущего срабатывания датчика
    df['delta_back'] = df['datetime'] - df['datetime'].shift(1)
    # промежуток времени до СЛЕДУЮЩЕГО срабатывания датчика
    df['delta_forward'] = df['datetime'].shift(-1) - df['datetime'] 
    df['minute'] = df.apply(lambda x:int(x.datetime.time().minute), axis=1)
    df['delta_back'] = df['delta_back'].fillna(0)
    df['delta_forward'] = df['delta_forward'].fillna(0)
    # промежуток времени до предыдущего срабатывания датчика в минутах
    df['delta_back_min'] = df.apply(lambda x: int(x.delta_back.total_seconds()/60), axis = 1)
    # промежуток времени до СЛЕДУЮЩЕГО срабатывания датчика в минутах
    df['delta_forward_min'] = df.apply(lambda x: int(x.delta_forward.total_seconds()/60), axis = 1)
    return(df)
# создаем новый набор данных с признаками для алгоритма
df = feature_generation(data)
df.head()
# определяем значиме признаки и формируем матрицу
features = ['weekday', 'weekend', 'hour', 'minute', 'delta_back_min','delta_forward_min', 'day', 'month']
X = df[features].values
df[features].head()
# отклик
Y = df['wakeup'].values
# сделаем простую проверку на отложенной выборке 
TRAIN_TEST_RATIO = 0.6
data_shape = df.shape[0]
data_shape
train_size = int(data_shape*TRAIN_TEST_RATIO)
train_size
# определям наборы данных для обучения и для теста
X_train, Y_train, X_test, Y_test = X[:train_size,], Y[:train_size], X[train_size:,], Y[train_size:]
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
#model = XGBClassifier(objective='rank:pairwise')
# устраняем дисбаланс классов
model = XGBClassifier(scale_pos_weight=1)
model.fit(X_train, Y_train)
Y_prediction_proba = model.predict_proba(X_test)
Y_prediction = [0 if item < 0.5 else 1 for item in Y_prediction_proba[:,1]]
print(classification_report(Y_test, Y_prediction))
roc_auc_score(Y_test, Y_prediction_proba[:,1], average=None)
