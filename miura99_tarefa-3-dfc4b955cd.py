import pandas as pd
import numpy as np

raw_ds = pd.read_csv("../input/train.csv")
raw_ds.head()
raw_ds.info()
raw_ds.describe()
from matplotlib import pyplot as plt
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.hist(raw_ds['median_house_value'] - 179950, bins = 50)
plt.title("median_house_value")
plt.subplot(1, 2, 2)
plt.hist(np.log(raw_ds['median_house_value'] ), bins = 50)
plt.title("log of median_house_value")
plt.show()
from matplotlib import pyplot as plt
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.hist(raw_ds['total_rooms'], bins = 50)
plt.title("total rooms")
plt.subplot(1, 2, 2)
plt.hist(np.log(raw_ds['total_rooms'] ), bins = 50)
plt.title("log of total rooms")
plt.show()
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.hist(raw_ds['population'], bins = 50)
plt.title("population")
plt.subplot(1, 2, 2)
plt.hist(np.log(raw_ds['population'] ), bins = 50)
plt.title("log of population")
plt.show()
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.hist(raw_ds['households'], bins = 50)
plt.title("households")
plt.subplot(1, 2, 2)
plt.hist(np.log(raw_ds['households'] ), bins = 50)
plt.title("log of households")
plt.show()
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.hist(raw_ds['total_bedrooms'], bins = 50)
plt.title("total bedrooms")
plt.subplot(1, 2, 2)
plt.hist(np.log(raw_ds['total_bedrooms'] ), bins = 50)
plt.title("log of total bedrooms")
plt.show()
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.hist(raw_ds['median_income'], bins = 50)
plt.title("median income")
plt.subplot(1, 2, 2)
plt.hist(np.log(raw_ds['median_income'] ), bins = 50)
plt.title("log of median income")
plt.show()
raw_ds.plot(kind="scatter", x="longitude", y="latitude",
    s=raw_ds['population']/100, label="population",
    c="median_house_value", cmap=plt.get_cmap("inferno"),
    colorbar=True, alpha=0.4, figsize=(10,7))


raw_ds[raw_ds['median_house_value'] >= 500000].dropna().plot(kind="scatter", x="longitude", y="latitude",
    s=raw_ds['population']/100, label="population",
    c="median_house_value", cmap=plt.get_cmap("viridis"),
    colorbar=True, alpha=0.4, figsize=(10,7))
plt.legend()
plt.show()
raw_ds.var()[1:].sort_values()
raw_ds.cov()['median_house_value'][1:].sort_values()
raw_ds.corr()['median_house_value'][1:].sort_values()
from sklearn import ensemble
iso3 = ensemble.IsolationForest(contamination = 0.005)
iso3.fit(raw_ds.iloc[:,1:])
scores = iso3.decision_function(raw_ds.iloc[:,1:])
plt.hist(scores, bins = 90)
plt.show()
outliers = iso3.predict(raw_ds.iloc[:,1:])
y = np.bincount(outliers+1)
ii = np.nonzero(y)[0]
np.vstack((ii,y[ii])).T
clean_ds = raw_ds[outliers == 1]
reg_ds = clean_ds.iloc[:,1:-1]
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, ElasticNet, BayesianRidge
scores = []
#Verificando os seguintes modelos de regressão para o dataset:
Models = {"Lasso": linear_model.Lasso(alpha=1.0, fit_intercept=True, normalize=True, precompute=False, 
                                      copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, 
                                      positive=False, random_state=None, selection= 'cyclic'),
          
          "Ridge": linear_model.Ridge(alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, 
                                      max_iter=None, tol=0.001, solver='auto', random_state=None),
          
          "kNN": KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30,
                                     p=1, metric='minkowski', metric_params=None),
          
          "Lin_Reg": LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=None),
          
          "ElasticNet": ElasticNet(alpha=1.0, l1_ratio=0.5, fit_intercept=True, normalize=False, precompute=False,
                                   max_iter=1000, copy_X=True, tol=0.001, warm_start=False, positive=False, 
                                   random_state=None, selection='cyclic'),
          
          "BayesianRidge": BayesianRidge(n_iter=300, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, 
                                         lambda_2=1e-06, compute_score=False, fit_intercept=True, normalize=False, 
                                         copy_X=True, verbose=False)
         }
#Selecionando atributos de maior importância 
from sklearn.feature_selection import SelectFromModel

Models['Ridge'].fit(reg_ds, clean_ds.iloc[:,-1])
fs = SelectFromModel(Models['Ridge'], prefit=True)
s_reg_ds = fs.transform(reg_ds)
from sklearn import preprocessing
import warnings
warnings.filterwarnings('ignore')
#definindo métrica de scoring
def rmsle_K(y, y0):
    return np.sqrt(np.mean(np.square(np.log1p(y) - np.log1p(y0))))


from sklearn.metrics import make_scorer
rmsle_score = make_scorer(rmsle_K, greater_is_better=False)
#Verificando o desempenho de cada um dos modelos:

for name, model in Models.items():
    scores = cross_val_score(model, s_reg_ds, clean_ds['median_house_value'], cv = 50, scoring= rmsle_score)
    print("Model:", name)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print()
Models['kNN'].fit(s_reg_ds, clean_ds['median_house_value'])
test_raw = pd.read_csv("../input/test.csv")
test_input = test_raw.iloc[:,1:] 
X_test = fs.transform(test_input)
y_test = Models['kNN'].predict(X_test)
Y_test = pd.Series(y_test)
output = pd.DataFrame([test_raw['Id'],Y_test])
output = output.transpose()
output = output.rename(columns={'Id': 'Id', 'Unnamed 0': 'median_house_value'})
output['Id'] = output['Id'].astype(int) 
output.to_csv("output.csv", index = False)
