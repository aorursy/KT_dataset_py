import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from time import time
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, RidgeCV, LinearRegression, SGDRegressor, Lasso, ElasticNet, Lars, LassoLars, HuberRegressor, BayesianRidge, PassiveAggressiveRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import max_error, mean_absolute_error, explained_variance_score, mean_squared_error, r2_score
import xgboost as xgb

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
sub = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
train.head(2)
test.head(2)
sub.head(2)
train.shape
test.shape
# Verificar nulos base de treino
train.isna().sum().sort_values(ascending=False)/train.shape[0]
# Verificar nulos base de treino
test.isna().sum().sort_values(ascending=False)/test.shape[0]
# Juntar os dataset de treino e teste
df = train.append(test)
df.shape
df.isna().sum()
df['Fence'].fillna('NA', inplace=True)
colunas_nulas_perc = (df.isna().sum()/df.shape[0]*100) > 50.0 
colunas_nulas_perc = colunas_nulas_perc[colunas_nulas_perc.isin([True])]
len(list(colunas_nulas_perc.index))
lista_colunas_nulas_perc = list(colunas_nulas_perc.index)
qtd_lista_colunas_nulas_perc = len(list(colunas_nulas_perc.index))
print('Existem', qtd_lista_colunas_nulas_perc, 'São elas', lista_colunas_nulas_perc)
df['Fence'].value_counts()
df['Fence'].fillna('NA', inplace=True)
df.drop(columns=['Alley', 'PoolQC', 'MiscFeature'], inplace=True)
df.shape
colunas_nulas_perc = (df.isna().sum()/df.shape[0]*100) > 20.0 
colunas_nulas_perc = colunas_nulas_perc[colunas_nulas_perc.isin([True])]
len(list(colunas_nulas_perc.index))
lista_colunas_nulas_perc = list(colunas_nulas_perc.index)
qtd_lista_colunas_nulas_perc = len(list(colunas_nulas_perc.index))
print('Existem', qtd_lista_colunas_nulas_perc, 'São elas', lista_colunas_nulas_perc)
df['FireplaceQu'].value_counts()
df['FireplaceQu'].fillna('NA', inplace=True)
colunas_nulas_perc = (df.isna().sum()/df.shape[0]*100) > 5.0 
colunas_nulas_perc = colunas_nulas_perc[colunas_nulas_perc.isin([True])]
len(list(colunas_nulas_perc.index))
lista_colunas_nulas_perc = list(colunas_nulas_perc.index)
qtd_lista_colunas_nulas_perc = len(list(colunas_nulas_perc.index))
print('Existem', qtd_lista_colunas_nulas_perc, 'São elas', lista_colunas_nulas_perc)
columns_fill = {'GarageType': 'NA', 
                'GarageYrBlt': 'NA', 
                'GarageFinish': 'NA', 
                'GarageQual': 'NA', 
                'GarageCond': 'NA', 
                'LotFrontage': 0}
df.fillna(columns_fill, inplace=True)
colunas_nulas_perc = (df.isna().sum()/df.shape[0]*100) > 2.0 
colunas_nulas_perc = colunas_nulas_perc[colunas_nulas_perc.isin([True])]
len(list(colunas_nulas_perc.index))
lista_colunas_nulas_perc = list(colunas_nulas_perc.index)
qtd_lista_colunas_nulas_perc = len(list(colunas_nulas_perc.index))
print('Existem', qtd_lista_colunas_nulas_perc, 'São elas', lista_colunas_nulas_perc)
columns_fill = {'BsmtQual': 'NA', 
                'BsmtCond': 'NA', 
                'BsmtExposure': 'NA', 
                'BsmtFinType1': 'NA', 
                'BsmtFinType2': 'NA'}
df.fillna(columns_fill, inplace=True)
colunas_nulas_perc = (df.isna().sum()/df.shape[0]*100) > 0.0 
colunas_nulas_perc = colunas_nulas_perc[colunas_nulas_perc.isin([True])]
len(list(colunas_nulas_perc.index))
lista_colunas_nulas_perc = list(colunas_nulas_perc.index)
qtd_lista_colunas_nulas_perc = len(list(colunas_nulas_perc.index))
print('Existem', qtd_lista_colunas_nulas_perc, 'São elas', lista_colunas_nulas_perc)
columns_fill = {'MSZoning': 'NA', 
                'Utilities': 'NA', 
                'Exterior1st': 'NA', 
                'Exterior2nd': 'NA', 
                'MasVnrType': 'None', 
                'MasVnrArea': 0, 
                'BsmtFinSF1': 0, 
                'BsmtFinSF2': 0, 
                'BsmtUnfSF': 0, 
                'TotalBsmtSF': 0, 
                'Electrical': 'SBrkr', 
                'BsmtFullBath': 0, 
                'BsmtHalfBath': 0, 
                'KitchenQual': 'NA', 
                'Functional': 'NA', 
                'GarageCars': 0, 
                'GarageArea': 0, 
                'SaleType': 'WD'}
df.fillna(columns_fill, inplace=True)
df.info()
#Ajustando algumas variáveis categoricas
df['MSSubClass'] = df['MSSubClass'].apply(str)
df['OverallCond'] = df['OverallCond'].astype(str)
df['YrSold'] = df['YrSold'].astype(str)
df['MoSold'] = df['MoSold'].astype(str)
df['GarageYrBlt'] = df['GarageYrBlt'].astype(str)
df = pd.get_dummies(df)
# Transformando as colunas com o LabelEncoder
colunas_transform = list(df.select_dtypes(include=['object']).columns)

encoder = LabelEncoder()
for label in colunas_transform:
    df[label] = encoder.fit_transform(df[label])
# Adicionando um feature
df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
df.shape
df_train = df.dropna()
df_train.shape
X = df_train.drop(columns=['Id','SalePrice'])
y = df_train.SalePrice
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
# Função criada para aplicação do modelo e medição da performance
def benchmark(reg):
    print('_' * 80)
    print("Training: ")
    print(reg)
    t0 = time()
    reg.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = reg.predict(X_test)
    y_true = list(y_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = reg.score(X_test, y_test)
    if score < 0 or score > 1:
        print("score: FAIL")
        score = 0
    else:
        print("score:   %0.3f" % score)
        
    if hasattr(reg, 'coef_'):
        print("intercept_: ", reg.intercept_)
        evs = explained_variance_score(y_test, pred)
        print('EVS (Explained Variance Score): {0:.2f}'.format(evs), ' [Best value is 1.0]')
        mre = max_error(y_test, pred)
        print('MRE (Max Residual Error): {0:.2f}'.format(mre), ' [Best value is 0.0]')
        mae = mean_absolute_error(y_test, pred)
        print('MAE (Mean Absolute Error): {0:.2f}'.format(mae), ' [Best value is 0.0]')
        mse = mean_squared_error(y_test, pred)
        print('MSE (Mean Squared Error): {0:.2f}'.format(mse), ' [Best value is 0.0]')
        r_2 = r2_score(y_test, pred)
        print('R^2 (Coefficient of Determination): {0:.2f}'.format(r_2), ' [Best value is 1.0]')
        print()

    print()
    reg_descr = str(reg).split('(')[0]
    return reg_descr, score, train_time, test_time, reg
# Lista que vai armazennar as métricas de cada modelo
results = []

print('=' * 80)
print('LinearSVR')
results.append(benchmark((LinearSVR())))
print('=' * 80)
print('DecisionTreeRegressor')
results.append(benchmark((DecisionTreeRegressor())))
print('=' * 80)
print('KNeighborsRegressor')
results.append(benchmark((KNeighborsRegressor())))
print('=' * 80)
print('Ridge')
results.append(benchmark((Ridge(alpha=1.0))))
print('=' * 80)
print('RidgeCV')
results.append(benchmark((RidgeCV())))
print('=' * 80)
print('LinearRegression')
results.append(benchmark((LinearRegression())))
print('=' * 80)
print('Lasso')
results.append(benchmark((Lasso(max_iter=4000))))
print('=' * 80)
print('ElasticNet')
results.append(benchmark((ElasticNet())))
print('=' * 80)
print('Lars')
results.append(benchmark((Lars())))
print('=' * 80)
print('LassoLars')
results.append(benchmark((LassoLars(max_iter=100))))
print('=' * 80)
print('HuberRegressor')
results.append(benchmark((HuberRegressor(max_iter=100))))
print('=' * 80)
print('BayesianRidge')
results.append(benchmark((BayesianRidge())))
print('=' * 80)
print('PassiveAggressiveRegressor')
results.append(benchmark((PassiveAggressiveRegressor())))
print('=' * 80)
print('RandomForestRegressor')
results.append(benchmark((RandomForestRegressor(1000))))
print('=' * 80)
print('MLPRegressor')
results.append(benchmark((MLPRegressor())))
# Gráfico de avalização dos modelos
indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(4)]

reg_names, score, training_time, test_time = results
training_time = np.array(training_time) / np.max(training_time)
test_time = np.array(test_time) / np.max(test_time)

plt.figure(figsize=(12, 8))
plt.title("Score")
plt.barh(indices, score, .2, label="score", color='navy')
plt.barh(indices + .3, training_time, .2, label="training time", color='c')
plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')
plt.yticks(())
plt.legend(loc='best')
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)
plt.subplots_adjust(bottom=.05)

for i, c in zip(indices, reg_names):
    plt.text(-.3, i, c)

plt.show()
model = benchmark(LassoLars(max_iter=100))[4]
# Preparando base de teste
df_test = df.loc[(df.SalePrice.isnull())].copy()

X_test = df_test.drop(columns=['Id','SalePrice'])

# Realizando a predição
df_test['SalePrice'] = model.predict(X_test)

# Preparando df para exportação
df_test = df_test[['Id', 'SalePrice']]

# Exportação da predição para csv
df_test.to_csv('submission.csv', index=False)

df_test.head()
X = X.values
y = y.values
print("Boston Housing: regression")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_index, test_index in kf.split(X):
    xgb_model = xgb.XGBRegressor().fit(X[train_index], y[train_index])
    predictions = xgb_model.predict(X[test_index])
    actuals = y[test_index]
    print('MSE:', mean_squared_error(actuals, predictions))

print("Parameter optimization")
xgb_model = xgb.XGBRegressor()
reg = GridSearchCV(xgb_model,
                   {'max_depth': [2, 4, 6],
                    'n_estimators': [200, 400, 800]}, verbose=1)
reg.fit(X, y)
print('Best Score', reg.best_score_)
print('Best Params', reg.best_params_)
# Preparando base de teste
df_test = df.loc[(df.SalePrice.isnull())].copy()

X_test = df_test.drop(columns=['Id','SalePrice']).values

# Realizando a predição
df_test['SalePrice'] = reg.predict(X_test)

# Preparando df para exportação
df_test = df_test[['Id', 'SalePrice']]

# Exportação da predição para csv
df_test.to_csv('submission_xgb.csv', index=False)

df_test.head()