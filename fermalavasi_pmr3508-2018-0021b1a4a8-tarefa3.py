import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, chi2

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
testCA = pd.read_csv("../input/californiadataset/test.csv",
                     sep=r'\s*,\s*',
                     engine='python',
                     na_values="?")
testCA
trainCA = pd.read_csv("../input/californiadataset/train.csv",
                      sep=r'\s*,\s*',
                      engine='python',
                      na_values="?")
trainCA
features = ['longitude','latitude','median_age','total_rooms',
            'total_bedrooms','population','households','median_income']
for f in features:
    plt.scatter(trainCA.median_house_value, trainCA[f], s=3)
    plt.xlabel('median_house_value')
    plt.ylabel(f)
    plt.title('Distribuição da feature na base de dados')
    plt.show()
dataset = trainCA.iloc[:,1:]
pearson = dataset.corr(method='pearson')
pearson
pearson_values = abs(pearson.median_house_value)
pearson_values.pop('median_house_value')
pearson_values.plot(kind='bar')
f_p = {}
for i in range(len(features)):
    f_p[features[i]] = pearson_values[i]
features_ordenadas = sorted(f_p, key=f_p.__getitem__, reverse=True)
features_ordenadas
longitudes = [-122.205107,-122.257010,-118.660497,-122.360984,-122.139629,
              -122.467144,-118.403503,-118.484295,-122.560761,-122.442169,
              -117.824792,
              -119.268043,-119.733119,-119.447774,-118.555709,-115.042510,-114.832796,-120.873526]
latitudes = [37.455204,37.430998,34.163592,37.556827,37.368736,
             37.872094,34.076851,34.020278,37.963242,37.757915,
             33.666484,
             37.504785,38.140382,37.842019,37.902616,33.236779,33.789509,39.885386]
areas = ['d_0', 'd_1', 'd_2', 'd_3', 'd_4', 'd_5', 'd_6', 'd_7', 'd_8', 'd_9', 'd_I',
         'd_F1', 'd_F2', 'd_F3', 'd_F4', 'd_Q1', 'd_Q2', 'd_A']
for area in range(len(areas)):
    trainCA[areas[area]] = ((longitudes[area]-trainCA.longitude)**2 + (latitudes[area]-trainCA.latitude)**2)**(1/2)
trainCA
dataset = trainCA.iloc[:,1:]
pearson = dataset.corr(method='pearson')
pearson
pearson_values = abs(pearson.median_house_value)
pearson_values.pop('median_house_value')
pearson_values.plot(kind='bar')
features = features + areas
f_p = {}
for i in range(len(features)):
    f_p[features[i]] = pearson_values[i]
features_ordenadas = sorted(f_p, key=f_p.__getitem__, reverse=True)
features_ordenadas
X = trainCA[features]
Y = trainCA.median_house_value
def rmsle(y_pred, y_test):
    y_pred = abs(y_pred)
    y_test = abs(y_test)
    return np.sqrt(np.mean((np.log(1+y_pred) - np.log(1+y_test))**2))
scorer_rmsle = make_scorer(rmsle)
def gerar_grafico(classificadores,X,f_o):
    if X is None:
        acc_total = {}
        for clf in  classificadores:
            acc_clf = []
            for n in range(len(f_o)):
                Xn = trainCA[f_o[:n+1]]
                clf.fit(Xn,Y)
                scores = cross_val_score(clf, Xn, Y, cv=10, scoring=scorer_rmsle)
                acc_clf.append(scores.mean())
            acc_total[clf] = acc_clf
    else:
        acc_total = {}
        for clf in  classificadores:
            acc_clf = []
            for n in range(len(f_o)):
                Xn = SelectKBest(chi2, k=n+1).fit_transform(abs(X),Y)
                clf.fit(Xn,Y)
                scores = cross_val_score(clf, Xn, Y, cv=10, scoring=scorer_rmsle)
                acc_clf.append(scores.mean())
            acc_total[clf] = acc_clf
    for clf in acc_total:
        plt.plot(np.arange(1,len(f_o)+1), acc_total[clf], 'o-', label=classificadores[clf])
    plt.ylabel('RMSLE esperada')
    plt.xlabel('Quantidade de features')
    plt.title('Uso das melhores features vs RMSLE esperada')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()
    return acc_total
LA = Lasso(alpha=10000)
LA.fit(X,Y)
scores = cross_val_score(LA, X, Y, cv=10, scoring=scorer_rmsle)
scores.mean()
RI = Ridge()
RI.fit(X,Y)
scores = cross_val_score(RI, X, Y, cv=10, scoring=scorer_rmsle)
scores.mean()
LR = LinearRegression()
LR.fit(X,Y)
scores = cross_val_score(LR, X, Y, cv=10, scoring=scorer_rmsle)
scores.mean()
classificadores_lineares = {LA:'LA',
                            RI:'RI',
                            LR:'LR'}
acc_lineares = gerar_grafico(classificadores_lineares,None,features_ordenadas)
acc_lineares = gerar_grafico(classificadores_lineares,X,features_ordenadas)
X_new = X
LA = Lasso(alpha=10000)
LA.fit(X_new,Y)
scores = cross_val_score(LA, X_new, Y, cv=10, scoring=scorer_rmsle)
scores.mean()
X_new = SelectKBest(chi2, k=17).fit_transform(abs(X),Y)
RI = Ridge()
RI.fit(X_new,Y)
scores = cross_val_score(RI, X_new, Y, cv=10, scoring=scorer_rmsle)
scores.mean()
X_new = trainCA[features_ordenadas[:25]]
LR = LinearRegression()
LR.fit(X_new,Y)
scores = cross_val_score(LR, X_new, Y, cv=10, scoring=scorer_rmsle)
scores.mean()
knn = KNeighborsRegressor()
knn.fit(X,Y)
scores = cross_val_score(knn, X, Y, cv=10, scoring=scorer_rmsle)
scores.mean()
classificador_knn = {knn:'knn'}
acc_knn = gerar_grafico(classificador_knn,None,features_ordenadas)
acc_knn = gerar_grafico(classificador_knn,X,features_ordenadas)
X_new = SelectKBest(chi2, k=6).fit_transform(abs(X),Y)
knn = KNeighborsRegressor()
knn.fit(X_new,Y)
scores = cross_val_score(knn, X_new, Y, cv=10, scoring=scorer_rmsle)
scores.mean()
RFR = RandomForestRegressor(random_state=0)
RFR.fit(X,Y)
scores = cross_val_score(RFR, X, Y, cv=10, scoring=scorer_rmsle)
scores.mean()
ETR = ExtraTreesRegressor(random_state=0)
ETR.fit(X,Y)
scores = cross_val_score(ETR, X, Y, cv=10, scoring=scorer_rmsle)
scores.mean()
classificadores_ensemble = {RFR:'RFR',
                            ETR:'ETR'}
acc_ensemble = gerar_grafico(classificadores_ensemble,None,features_ordenadas)
acc_ensemble = gerar_grafico(classificadores_ensemble,X,features_ordenadas)
ETR = ExtraTreesRegressor(n_estimators=200, max_features='sqrt', min_samples_split=3,
                          min_impurity_decrease=0.2, max_depth=31)
ETR.fit(X,Y)
scores = cross_val_score(ETR, X, Y, cv=10, scoring=scorer_rmsle)
scores.mean()
acc_total={}
acc_total.update(acc_lineares)
acc_total.update(acc_knn)
acc_total.update(acc_ensemble)

classificadores={}
classificadores.update(classificadores_lineares)
classificadores.update(classificador_knn)
classificadores.update(classificadores_ensemble)

for clf in acc_total:
    plt.plot(np.arange(1,len(features_ordenadas)+1), acc_total[clf], 'o-', label=classificadores[clf])

plt.plot([1,len(features_ordenadas)], [scores.mean(),scores.mean()], 'k', label='ETR_top')

plt.ylabel('RMSLE esperada')
plt.xlabel('Quantidade de features')
plt.title('Uso das melhores features vs RMSLE esperada')
plt.legend(loc='best')
plt.grid(True)
plt.show()
for area in range(len(areas)):
    testCA[areas[area]] = ((longitudes[area]-testCA.longitude)**2 + (latitudes[area]-testCA.latitude)**2)**(1/2)
testCA
Xtest = testCA[features]
YtestETR = ETR.predict(Xtest)
resultETR = np.vstack((testCA['Id'], abs(YtestETR))).T.astype(int)
x = ["Id","median_house_value"]
ResultadoETR = pd.DataFrame(columns = x, data = resultETR)
ResultadoETR.to_csv("resultadosETR.csv", index = False)
resultETR
