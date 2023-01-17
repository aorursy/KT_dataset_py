import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn import ensemble, neighbors
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_log_error, make_scorer
from sklearn.feature_selection import f_regression
import os
print(os.listdir('../input/'))
test_d = pd.read_csv('../input/test.csv',index_col='Id')
train_d = pd.read_csv('../input/train.csv', index_col='Id')
train_d.head()
train_d.info()
test_d.info()
test_d['median_house_value'] = np.nan
data = train_d.append(test_d, sort=False)
plt.hist(train_d['median_house_value'], bins = 20, edgecolor='white')
plt.xlabel('Faixa de valores'); plt.ylabel('Número de Observações');
plt.title('Histograma da variável de interesse (median_house_value)');
plt.scatter(train_d['latitude'], train_d['longitude'], s=1,c=np.log10(train_d['median_house_value']), cmap=plt.cm.cool)
plt.title('Distribuição geográfica do valor mediano das casas');
plt.scatter(test_d['latitude'], test_d['longitude'], s=1)
plt.title('Distribuição geográfica das observações da base teste');
plt.scatter(train_d['median_income'], train_d['median_house_value'], s=1)
plt.title('Relação entre "median_income" e "median_house_value"');
plt.scatter(train_d['median_age'], train_d['median_house_value'], s=1)
plt.title('Relação entre "median_age" e "median_house_value"');
#cria as features "soma médiana por idade do terreno" e "número de observações de terrenos com a mesma idade que esse"

train_d['median_age_sum'] = train_d.groupby('median_age')['median_house_value'].transform('sum')
ages = train_d['median_age'].value_counts()

for i, row in train_d.iterrows():
    train_d.loc[i,'median_age_freq'] = ages[train_d.loc[i,'median_age']]
plt.scatter(train_d['median_age'], train_d['median_age_sum']/train_d['median_age_freq'], s=1)
plt.title('Média das medianas do valor da cada de uma região por idade mediana das casas da região');
plt.scatter(train_d['median_age'], train_d['median_age_freq'], s=1)
plt.title('Número de observações por idade mediana das casas da região');
plt.scatter(train_d['latitude'], train_d['longitude'], s=1, c=train_d['median_age'], cmap=plt.cm.cool)
plt.title('Distribuição geográfica das idades medianas');
ax = plt.scatter(train_d['total_rooms'], train_d['median_house_value'], s=1)
plt.title('Relação entre "total_rooms" e "median_house_value"');
plt.scatter(train_d['latitude'], train_d['longitude'], s=1, c=(train_d['total_rooms']), cmap=plt.cm.cool)
plt.title('Distribuição geográfica da variável "total_rooms"');
ax = plt.scatter(train_d['total_bedrooms'], train_d['median_house_value'], s=1)
plt.title('Relação entre "total_bedrooms" e "median_house_value"');
plt.scatter(train_d['latitude'], train_d['longitude'], s=1, c=(train_d['total_bedrooms']), cmap=plt.cm.cool)
plt.title('Distribuição geográfica da variável "total_bedrooms"');
ax = plt.scatter(train_d['population'], train_d['median_house_value'], s=1)
plt.title('Relação entre "population" e "median_house_value"');
plt.scatter(train_d['latitude'], train_d['longitude'], s=1, c=(train_d['population']), cmap=plt.cm.cool)
plt.title('Distribuição geográfica da variável "population"');
ax = plt.scatter(train_d['households'], train_d['median_house_value'], s=1)
plt.scatter(train_d['latitude'], train_d['longitude'], s=1, c=(train_d['households']), cmap=plt.cm.cool)
data['rooms_not_bedrooms'] = np.subtract(data['total_rooms'],data['total_bedrooms'])

data['bedrooms_per_house'] = np.divide(data['total_bedrooms'],data['households'])
data['rooms_not_bedrooms_per_house'] = np.divide(data['total_bedrooms'],data['households'])
data['pop_per_house'] = np.divide(data['population'],data['households'])

data['pop_per_bedroom'] = np.divide(data['population'],data['total_bedrooms'])
data['pop_per_room'] = np.divide(data['population'],data['total_bedrooms'])

data['income_per_pop'] = np.divide(data['median_income'],data['pop_per_house'])
ages_disc=[]
for i,row in data.iterrows():
    if row['median_age'] < 16:
        ages_disc.append(0)
    elif row['median_age'] < 36:
        ages_disc.append(1)
    elif row['median_age'] < 52:
        ages_disc.append(2)
    else:
        ages_disc.append(3)
data['age_class'] = ages_disc
correlacao = data.dropna().corr()

correlacao
corr_superior = correlacao.where(np.triu(np.ones(correlacao.shape), k=1).astype(np.bool))

drop = [column for column in corr_superior.columns if any(abs(corr_superior[column]) > 0.9)]

drop
correlacao.loc[correlacao['total_bedrooms'].abs() > 0.9, correlacao['total_bedrooms'].abs() > 0.9]
correlacao.loc[correlacao['rooms_not_bedrooms'].abs() > 0.9, correlacao['rooms_not_bedrooms'].abs() > 0.9]
correlacao.loc[correlacao['rooms_not_bedrooms_per_house'].abs() > 0.9, correlacao['rooms_not_bedrooms_per_house'].abs() > 0.9]
correlacao.loc[correlacao['pop_per_bedroom'].abs() > 0.9, correlacao['pop_per_bedroom'].abs() > 0.9]
data.drop(['rooms_not_bedrooms_per_house','pop_per_room','pop_per_house','total_rooms','median_age'], axis = 1, inplace = True)
normcols = list(data.columns)
[normcols.remove(x) for x in ['latitude','longitude','median_house_value']]
fit_norm = data.dropna()
fit_norm = fit_norm[normcols]

means = {}
stds = {}

for c in normcols:
        means[c] = fit_norm[c].mean()
        stds[c] = fit_norm[c].std()

        
fit_norm = data.copy()
for c in normcols:
        fit_norm.loc[:,c] = fit_norm.loc[:,c].subtract(means[c]).divide(stds[c])

fit_norm['latitude'] = data['latitude']
fit_norm['longitude'] = data['longitude']
fit_norm['median_house_value'] = data['median_house_value']

fit_norm.head()
correlacao_norm = fit_norm.dropna().corr()

correlacao_norm
msle = make_scorer(mean_squared_log_error)
from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbors=10)
scores = cross_val_score(knn,
                         fit_norm.dropna().drop('median_house_value', axis = 1),
                         fit_norm.dropna().loc[:,'median_house_value'],
                         cv=10,
                         scoring = msle)

scores = np.sqrt(scores)
display(scores)
print(f'MSLE = {round(scores.mean(), 4)}, std = {round(scores.std(), 4)}')
scores_array = []
scores_var = []

for i in range(21):
    knn = KNeighborsRegressor(n_neighbors=10, metric='wminkowski', p=2, metric_params={'w': np.divide([i/20]*2+[1-i/20]*9,15)})
    scores = cross_val_score(knn,
                         fit_norm.dropna().drop('median_house_value', axis = 1),
                         fit_norm.dropna().loc[:,'median_house_value'],
                         cv=10,
                         scoring = msle)
    scores = np.sqrt(scores)
    scores_array.append(scores.mean())
    scores_var.append(scores.std())
plt.errorbar(np.divide(range(len(scores_array)),20),scores_array,scores_var, linestyle='None', marker = '.')
plt.title('Erro em função da proporção dos pesos de distância destinados às coordenadas na regressão do KNN');
test_data = fit_norm.loc[fit_norm['median_house_value'].isnull()].drop('median_house_value',axis=1)
classifier = KNeighborsRegressor(n_neighbors=8, metric='wminkowski', p=2, metric_params={'w': np.divide([18/20]*2+[1-18/20]*9,15)})

classifier.fit(fit_norm.dropna().drop('median_house_value', axis = 1),fit_norm.dropna().loc[:,'median_house_value'])

testPred = classifier.predict(test_data)

arq = open ("prediction_knn.csv", "w")
arq.write("Id,median_house_value\n")
for i, j in zip(test_data.index, testPred):
    arq.write(str(i)+ "," + str(int(j))+"\n")
arq.close()
from sklearn.ensemble import RandomForestRegressor

regr = RandomForestRegressor(max_depth=20, random_state=0,n_estimators=400)
regr.fit(fit_norm.dropna().drop('median_house_value', axis = 1),fit_norm.dropna().loc[:,'median_house_value'])

testPred = regr.predict(test_data)

arq = open ("prediction_rfr.csv", "w")
arq.write("Id,median_house_value\n")
for i, j in zip(test_data.index, testPred):
    arq.write(str(i)+ "," + str(int(j))+"\n")
arq.close()
from sklearn.ensemble import AdaBoostRegressor

boost = AdaBoostRegressor(RandomForestRegressor(max_depth=5, random_state=0, n_estimators=300),
                          n_estimators=100)
boost.fit(fit_norm.dropna().drop('median_house_value', axis = 1),fit_norm.dropna().loc[:,'median_house_value'])

testPred = boost.predict(test_data)

arq = open ("prediction_ada_rfr.csv", "w")
arq.write("Id,median_house_value\n")
for i, j in zip(test_data.index, testPred):
    arq.write(str(i)+ "," + str(int(j))+"\n")
arq.close()