import pandas as pd #pomaga czytać csv. pod spodem używa numpy

import numpy as np #numeryczne operacje, średnia, mediana itp

np.random.seed(2018) #unikamy niepowtarzalności, czyli chcemy żeby wynik mniej więcej był powtarzalny



from collections import defaultdict



from sklearn.model_selection import GroupKFold #element do walidacji

#modele, które wykorzystujemy

from sklearn.dummy import DummyRegressor #najprostszy, model bazowy. Będziemy do niego porównywać wyniki pozostałych modeli.

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor



#boosting models => advanced options

import xgboost as xgb

#import catboost as ctb



import re

from tqdm import tqdm



#!pip install ml_metrics

from ml_metrics import rmsle #



import matplotlib.pyplot as plt #wykresy

# wykres bezpośrednio w notebooku

%matplotlib inline 
train = pd.read_csv('../input/train.csv', parse_dates=['datetime'])

test = pd.read_csv('../input/test.csv', parse_dates=['datetime'])
all = pd.concat([train,test], sort=False)

all.info()
train.info()
test.info()
train.head(10) #wylosowane 10 wierszy, żeby popatrzeć sobie na dane ;) 
def plot_by_hour(data, year=None, agg='sum'):

    dd = data.copy()

    if year: dd = dd[ dd.datetime.dt.year == year ]

    dd.loc[:, ('hour')] = dd.datetime.dt.hour

    

    by_hour = dd.groupby(['hour', 'workingday'])['count'].agg(agg).unstack()

    return by_hour.plot(kind='bar', ylim=(0, 80000), figsize=(15,5), width=0.9, title="Year = {0}".format(year))





plot_by_hour(train, year=2011);

plot_by_hour(train, year=2012);
def plot_by_year(agg_attr, title):

    dd = train.copy()

    dd['year'] = train.datetime.dt.year

    dd['month'] = train.datetime.dt.month

    dd['hour'] = train.datetime.dt.hour

    

    by_year = dd.groupby([agg_attr, 'year'])['count'].agg('sum').unstack()

    return by_year.plot(kind='bar', figsize=(15,5), width=0.9, title=title)



plot_by_year('month', "Rent bikes per month in 2011 and 2012");

plot_by_year('hour', "Rent bikes per hour in 2011 and 2012");
def plot_hours(data, message = ''):

    dd = data.copy()

    dd['hour'] = data.datetime.dt.hour

    

    hours = {}

    for hour in range(24):

        hours[hour] = dd[ dd.hour == hour ]['count'].values



    plt.figure(figsize=(20,10))

    plt.ylabel("Count rent")

    plt.xlabel("Hours")

    plt.title("count vs hours\n" + message)

    plt.boxplot( [hours[hour] for hour in range(24)] )

    

    axis = plt.gca()

    axis.set_ylim([1, 1100])

    



plot_hours( train[train.workingday == 1], 'working day')

plot_hours( train[train.workingday == 0], 'non working day')



plot_hours( train[train.datetime.dt.year == 2011], 'year 2011');

plot_hours( train[train.datetime.dt.year == 2012], 'year 2012');
df_train = pd.read_csv('../input/train.csv', parse_dates=['datetime'])

df_test = pd.read_csv('../input/test.csv', parse_dates=['datetime'])



# **pandas.concat** pobiera listę lub dyktuje obiekty jednorodnie wpisane i łączy je z konfigurowalną obsługą 

# "co zrobić z innymi osiami", osiami - nie osłami :D



df_all = pd.concat([df_train, df_test], sort=False)

df_all.head(10)
fig = plt.figure(figsize=(12, 10))



y_true = [60] # przykladowa wartość którą chcemy przewidzieć

y_preds = np.linspace(0, 100, 100)



err    = [ rmsle(y_true, [y_pred]) for y_pred in y_preds ]

plt.plot(y_preds, err);
df_train.sample(10)

#oglądamy nasz zbiór danych i widzimy, że nie potrzebujemy w zasadzie "casual, registered,count" dlatego, że to są odpowiedzi na nasze pytania. 

# Dlatego jeszcze niżej dodajemy te wartości do czarnej listy :) 
# funkcja: Pobierz Cechy

def get_feats(df, black_list = ['count', 'casual', 'registered']):

    feats = df.select_dtypes(include=[np.int64, np.float64]).columns.values

    

    def allow_feat(feat):

        for block_feat in black_list:

            if block_feat in feat: return False

        return True

    return [feat for feat in feats if allow_feat(feat)]



get_feats(df_train) #cechy które będziemy brać pod uwagę trenując nasz model
# 1. wczytujemy cechy, które później przekażemy do modelu jako data do trenowania

# 2. macierz - df_train[feats] mówimy które cechy chcemy pobrać i następnie mapujemy to do numpy (.values)

# 3. wektor - wyciąga count/ilość



feats = get_feats(df_train)

X = df_train[feats].values

y = df_train['count'].values #wektor



model = DummyRegressor(strategy='median') #albo wartość średnia, albo mediana [można sprawdzić co jak działa: lepiej, gorzej?] **tu im mniej tym lepiej

model2 = DummyRegressor(strategy='mean') #albo wartość średnia, albo mediana [można sprawdzić co jak działa: lepiej, gorzej?] **tu im mniej tym lepiej

model.fit(X, y)

model2.fit(X, y)

y_pred = model.predict(X)

y_pred2 = model2.predict(X)



#wykorzystujemy metryke sukcesu, czyli y to jest błąd - im mniejszy tym lepszy

print("median: " + str(rmsle(y, y_pred))) 

print("mean(ignored): " + str(rmsle(y, y_pred2)))
X = df_test[feats].values



df_test['count'] = model.predict(X)

df_test[['datetime', 'count']].to_csv('dummy_median.csv', index=False)

df_test['count'] = model2.predict(X)

df_test[['datetime', 'count']].to_csv('dummy_mean.csv', index=False)
feats = get_feats(df_train)

X = df_train[feats].values

y = df_train['count'].values



model = DecisionTreeRegressor(max_depth=5, random_state=2018) #im większa głębokość tym bardziej overfittujemy :O , random_state - zapewniamy powtarzalność wyniku

model.fit(X,y)

y_pred = model.predict(X)



rmsle(y, y_pred) 
model.fit(X,y).feature_importances_
def feats_engineering(df):

    df['year'] = df['datetime'].dt.year

    df['month'] = df['datetime'].dt.month

    df['day'] = df['datetime'].dt.day

    df['hour'] = df['datetime'].dt.hour

    return df
feats = get_feats(feats_engineering(df_train))

X = df_train[feats].values

y = df_train['count'].values



model = DecisionTreeRegressor(max_depth=5, random_state=2018)

model.fit(X,y) # ta sama macierz do trenowania

y_pred = model.predict(X) # ta sama macierz do testowania



rmsle(y, y_pred)



def feature_importance_plot(m, n, data_with_columns):

    importances = m.feature_importances_

    indices = np.argsort(importances)[::-1]

    

    indicess = indices[:n]

    feature_names = data_with_columns.columns[indices]

    

    print("Feature ranking:")

    for f in range(n):

        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]),  feature_names[f])

    



    plt.figure()

    plt.title("Feature importances")

    plt.bar(range(n), importances[indicess], color="r", align="center")

    plt.xticks(range(n), feature_names, rotation='vertical')

    plt.xlim([-1, n])

    plt.show()



feature_importance_plot(model,10,df_train[feats])
X = feats_engineering(df_test)[feats].values



df_test['count'] = model.predict(X)

df_test[ ['datetime', 'count'] ].to_csv('..decision_tree_5depth.csv', index=False)
# zbiór do testowania

df = feats_engineering(df_test)

df[ ['year', 'month', 'count'] ].groupby(['year', 'month']).agg(len)
# zbiór do trenowania

df = feats_engineering(df_train)

df[ ['year', 'month', 'count'] ].groupby(['year', 'month']).agg(len)
# testujemy dane w taki sposób żeby doba nie była rozrywana. nie ma tak że przy treningu bierzemy pod uwagę poranne godziny 

# a przy testowaniu dane z wieczora.



def custom_validation(df, feats, target_variable='count', n_folds=3):

    X = df[feats].values

    y = df[target_variable].values



    groups = df['datetime'].dt.month.values

    group_kfold = GroupKFold(n_splits=n_folds) # GroupKFold sprawia, że waliduje per miesiąc, czyli wg miesiąca

    

    for train_idx, test_idx in group_kfold.split(X, y, groups):

        yield X[train_idx], X[test_idx], y[train_idx], y[test_idx]

feats = get_feats(feats_engineering(df_train))



models, scores = [], []



for idx, (X_train, X_test, y_train, y_test) in enumerate(custom_validation(df_train, feats, target_variable='count')):

    model = DecisionTreeRegressor(max_depth=5, random_state=2018)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    score = rmsle(y_test, y_pred)

    

    models.append(model)

    scores.append(score)





np.mean(scores), np.std(scores)
df_train[ df_train['count'] != df_train['registered'] + df_train['casual'] ].shape
df = feats_engineering(df_train)

feats = get_feats(df)



groups = df_train['datetime'].dt.month.values

group_kfold = GroupKFold(n_splits=3)

    

X = df[feats].values

registered = df_train['registered'].values

casual = df_train['casual'].values

count = df_train['count'].values



scores = []

for train_idx, test_idx in group_kfold.split(X, count, groups):



    #predykcja dla przypadków zarejestrowanych

    model = DecisionTreeRegressor(max_depth=5, random_state=2018)

    model.fit(X[train_idx], registered[train_idx])

    registered_pred = model.predict(X[test_idx])

    

    #predykcja dla niezarejestrowanych

    model = DecisionTreeRegressor(max_depth=5, random_state=2018)

    model.fit(X[train_idx], casual[train_idx])

    casual_pred = model.predict(X[test_idx])

    

    count_pred = registered_pred + casual_pred

    

    score = rmsle(count[test_idx], count_pred)

    scores.append(score)



np.mean(scores), np.std(scores)
#rozkład danych COUNT y - ile razy wystąpił przypadek, x - ilość potrzebnych rowerów

# widzimy, że wykres jest skośny z długim ogonem



df_train['count'].hist(bins=40);
#musimy dodać co najmniej 1 do naszego (df_train['casual']) bo w danych znajdują się 0, a logarytm z 0 nie może być

# my dajemy 3, żeby nie występowały takie dziury między danymi --- model stanie się mądrzejszy



#ponoć log2 sprawdza się lepiej

np.log2( df_train['casual'] + 3).hist(bins=50);
df = feats_engineering(df_train)

feats = get_feats(df)



groups = df_train['datetime'].dt.month.values

group_kfold = GroupKFold(n_splits=3)



X = df[feats].values



count = df_train['count'].values

offset_log = 6

count_log = np.log2(count + offset_log)



scores = []

for train_idx, test_idx in group_kfold.split(X, count,groups):

    model = DecisionTreeRegressor(max_depth=5, random_state=2018)

    model.fit(X[train_idx], count_log[train_idx])

    count_log_pred = model.predict(X[test_idx])

    

    count_pred = np.exp2(count_log_pred) - offset_log

    

    score = rmsle(count[test_idx], count_pred)

    scores.append(score)

    

np.mean(scores), np.std(scores)
df = feats_engineering(df_train)

feats = get_feats(df)



groups = df_train['datetime'].dt.month.values

group_kfold = GroupKFold(n_splits=3)

    

X = df[feats].values

offset_log = 3



registered = df['registered'].values

registered_log = np.log2(registered + offset_log)



casual = df['casual'].values

casual_log = np.log2(casual + offset_log)



count = df['count'].values

count_log = np.log2(count + offset_log)



scores = []

for train_idx, test_idx in group_kfold.split(X, count, groups):

    model = DecisionTreeRegressor(max_depth=5, random_state=2018)

    model.fit(X[train_idx], registered_log[train_idx])

    registered_log_pred = model.predict(X[test_idx])

    registered_pred = np.exp2(registered_log_pred) - offset_log

    

    model = DecisionTreeRegressor(max_depth=5, random_state=2018)

    model.fit(X[train_idx], casual_log[train_idx])

    casual_log_pred = model.predict(X[test_idx])

    casual_pred = np.exp2(casual_log_pred) - offset_log

    

#     model = DecisionTreeRegressor(max_depth=5, random_state=2018)

#     model.fit(X[train_idx], count_log[train_idx])

#     count_log_pred = model.predict(X[test_idx])

#     count_pred_1 = np.exp2(count_log_pred) - offset_log

    

    count_pred = registered_pred + casual_pred

    #count_pred = 0.7*(registered_pred + casual_pred) + 0.3*count_pred_1

    

    score = rmsle(count[test_idx], count_pred)

    scores.append(score)



np.mean(scores), np.std(scores)
train = feats_engineering(df_train)

test = feats_engineering(df_test)



feats = get_feats(train)

X_train, X_test = train[feats].values, test[feats].values



offset_log = 3



registered = train['registered'].values

registered_log = np.log2(registered + offset_log)



casual = train['casual'].values

casual_log = np.log2(casual + offset_log)



count = train['count'].values

count_log = np.log2(count + offset_log)
model_registered = DecisionTreeRegressor(max_depth=5, random_state=2018)

model_registered.fit(X_train, registered_log)



model_casual = DecisionTreeRegressor(max_depth=5, random_state=2018)

model_casual.fit(X_train, casual_log)



registered_log_pred = model_registered.predict(X_test)

registered_pred = np.exp2(registered_log_pred) - offset_log



casual_log_pred = model_casual.predict(X_test)

casual_pred = np.exp2(casual_log_pred) - offset_log



test['count'] = registered_pred + casual_pred
test[ ['datetime', 'count'] ].to_csv('dt_log_registered_casual.csv', index=False)
def get_models():

    return [

#         ('dt', DecisionTreeRegressor(max_depth=5, random_state=2018)),

#         ('rf', RandomForestRegressor(max_depth=8, n_estimators=50, random_state=20168)),

        ('xgb', xgb.XGBRegressor(max_depth=10, n_estimators=500, random_state=2018)),

        

    ]

        

#informacja o ważności cech    

def draw_importance_features(model, df, importance_type='gain'):



    fscore = model.get_booster().get_score(importance_type=importance_type) #cover, gain, weight

    maps_name = dict([ ("f{0}".format(i), col) for i, col in enumerate(df.columns)])



    impdf = pd.DataFrame([ {'feature': maps_name[k], 'importance': fscore[k]} for k in fscore ])

    impdf = impdf.sort_values(by='importance', ascending=False).reset_index(drop=True)

    impdf['importance'] /= impdf['importance'].sum()

    impdf.index = impdf['feature']



    impdf.plot(kind='bar', title='{0} - Importance Features'.format(importance_type.title()), figsize=(12, 4))

    

    

def run_model(model, X_train, X_test, y_train, y_test):

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    

    #jeśli predykcja jest ujemna to wtedy zerujemy

    y_pred[y_pred<0] = 0

    return rmsle(y_test, y_pred)
def feats_engineering(df, offset_log=4):

    df['year']       = df['datetime'].dt.year

    df['month']      = df['datetime'].dt.month

    df['day']        = df['datetime'].dt.day

    df['hour']       = df['datetime'].dt.hour

    df['dayofweek']  = df['datetime'].dt.dayofweek

    df['weekofyear'] = df['datetime'].dt.weekofyear

    df['weekend']    = df.dayofweek.map(lambda x: int(x in [5,6]) )

    df['dayofyear']  = df['datetime'].dt.dayofyear

    

    df['rush_hour'] = df['datetime'].apply(lambda i: min([np.fabs(9-i.hour), np.fabs(20-i.hour)]))

    df.loc[:,('rush_hour')] = df['datetime'].apply(lambda i: np.fabs(14-i.hour))

    

    df['peak'] = df[['hour', 'workingday']].apply(lambda x: (0, 1)[(x['workingday'] == 1 and  ( x['hour'] == 8 or 17 <= x['hour'] <= 18 or 12 <= x['hour'] <= 12)) or (x['workingday'] == 0 and  10 <= x['hour'] <= 19)], axis = 1)



    if 'count' in df:

        df['count_log'] = np.log2( df['count'] + offset_log )

        df['registered_log'] = np.log2( df['registered'] + offset_log )

        df['casual_log'] = np.log2( df['casual'] + offset_log )



    return df
df = feats_engineering(df_train)

feats = get_feats(df)



for model_name, model in get_models():

    scores = []

    for X_train, X_test, y_train, y_test in custom_validation(df_train, feats):

        score = run_model(model, X_train, X_test, y_train, y_test)

        scores.append(score)

        

    print("Model: {0}, scores-mean: {1}, scores-std: {2}".format(model_name, np.mean(scores), np.std(scores)))
draw_importance_features(model, df[feats], importance_type='gain')

draw_importance_features(model, df[feats], importance_type='cover')

draw_importance_features(model, df[feats], importance_type='weight')
median_year = df_train[ ['year', 'count'] ].groupby(['year']).median().to_dict()['count']

df_train['year'].map(lambda x: median_year[x]).head()
agg_feats = ['year', 'hour']

agg_func = np.mean



median_hour_year = df_train[ agg_feats + ['count'] ].groupby(agg_feats).agg(agg_func).to_dict()['count']

new_feat = '{0}_{1}'.format( agg_func.__name__, "_".join(agg_feats) )

df_train[agg_feats].apply(lambda x: median_hour_year[ tuple(dict(x).values()) ], axis=1).head()
df_train = feats_engineering(df_train)

df_test  = feats_engineering(df_test)



agg_feats = [ 

    ['hour', 'year'],

    ['hour', 'season'],

    ['hour', 'month'],

]



for agg_feat in tqdm(agg_feats):

    for agg_func in [np.mean, np.median, np.sum, np.std]:

        dict_agg = df_train[ agg_feat + ['count'] ].groupby(agg_feat).agg(agg_func).to_dict()['count']

        new_feat = '{0}_{1}'.format( agg_func.__name__, "_".join(agg_feat) )



        default_dict_val = agg_func( list(dict_agg.values()))

        default_dict_agg = defaultdict(lambda: default_dict_val, dict_agg)



        df_train[new_feat] = df_train[agg_feat].apply(lambda x: dict_agg[ tuple(dict(x).values()) ], axis=1)

        df_test[new_feat] = df_test[agg_feat].apply(lambda x: default_dict_agg[ tuple(dict(x).values()) ], axis=1)