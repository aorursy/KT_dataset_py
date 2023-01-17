import pandas as pd
import numpy as np

import matplotlib.pyplot as plt # geração de gráficos
import seaborn as sns # geração de gráficos

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

# Ignorar warnings
import warnings
warnings.filterwarnings('ignore')
df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
# verificando o formato do arquivo para submissão
ss = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')
ss.tail()
print(df_train.shape)
print(df_test.shape)
df_train.head()
df_train['SalePrice'].head()
df_pivot = pd.DataFrame({'types': df_train.dtypes,
                         'nulls': df_train.isna().sum(),
                          '% nulls': df_train.isna().sum() / df_train.shape[0],
                          'size': df_train.shape[0],
                          'uniques': df_train.nunique()})
df_pivot
df_train.corr()['SalePrice'].sort_values(ascending=False).head(12)
# features numericas mais correlacionadas
features_correlacao = ['SalePrice','OverallQual','GrLivArea',
                       'GarageCars','GarageArea','TotalBsmtSF',
                       '1stFlrSF','FullBath','TotRmsAbvGrd',
                       'YearBuilt','YearRemodAdd']
# plotando a matrix de correlação somente com estas features
correlacao = df_train[features_correlacao].corr()
axis = plt.subplots(figsize = (12,8))
sns.heatmap(correlacao, annot=True, annot_kws = {"size":12})
# exemplo correlação fortemente positiva
plt.xlabel('SalePrice')
plt.ylabel('OverallQual')
plt.scatter(df_train['SalePrice'], df_train['OverallQual'])
plt.show()
# variáveis categóricas e numéricas relevantes
df_train_final = df_train[['GarageType','KitchenQual','HouseStyle', 
                          'Condition1', 'Functional','SalePrice',
                           'OverallQual','GrLivArea','GarageCars',
                           'GarageArea','TotalBsmtSF','1stFlrSF',
                           'FullBath','TotRmsAbvGrd','YearBuilt','YearRemodAdd']]


df_test_final = df_test[['GarageType','KitchenQual','HouseStyle', 
                          'Condition1', 'Functional','OverallQual',
                           'GrLivArea','GarageCars','GarageArea',
                           'TotalBsmtSF','1stFlrSF','FullBath',
                           'TotRmsAbvGrd','YearBuilt','YearRemodAdd']]

df_train_final.shape
# tratando valores nulos
df_train_final.isnull().sum()
df_test_final.isnull().sum()
# base de teste não tem o valor "2.5Fin" no house style. 
print(df_train_final.HouseStyle.value_counts()) 
print(df_test_final.HouseStyle.value_counts())
#df_train_final.GarageType.value_counts() # Attchd
#df_test_final.GarageType.value_counts() # Attchd
#df_test_final.KitchenQual.value_counts() # TA
#df_test_final.Functional.value_counts() # Typ
df_test_final['TotalBsmtSF'].fillna(df_test_final['TotalBsmtSF'].mean(), inplace = True)
df_test_final['GarageArea'].fillna(df_test_final['GarageArea'].mean(), inplace = True)
df_test_final['GarageCars'].fillna(df_test_final['GarageCars'].mean(), inplace = True)

df_train_final['GarageType'].fillna('Attchd', inplace = True)
df_test_final['GarageType'].fillna('Attchd', inplace = True)
df_test_final['KitchenQual'].fillna('TA', inplace = True)
df_test_final['Functional'].fillna('Typ', inplace = True)
# convertendo dados categóricos em numéricos
df_new_train = pd.get_dummies(df_train_final)
df_new_test = pd.get_dummies(df_test_final)
df_new_train.head()
# Eliminando a coluna "HouseStyle_2.5Fin", pois ela não existe em teste
df_new_train = df_new_train.drop("HouseStyle_2.5Fin", axis=1)
id_number = df_test['Id']

y_train = df_new_train['SalePrice']

x_train = df_new_train.drop("SalePrice", axis=1)
x_test = df_new_test
# Multiple linear regression
model_linear = LinearRegression()
model_linear.fit(x_train, y_train)

predict_lr = model_linear.predict(x_test)

acc_linear_regression = round(model_linear.score(x_train, y_train) * 100, 2)
acc_linear_regression
# Random forest
model_random_forest = RandomForestRegressor()
model_random_forest.fit(x_train, y_train)

predict_rf = model_random_forest.predict(x_test)

acc_random_forest = round(model_random_forest.score(x_train, y_train) * 100, 2)
acc_random_forest
# Decision Tree
model_decision_tree = DecisionTreeRegressor()
model_decision_tree.fit(x_train, y_train)

predict_dt = model_decision_tree.predict(x_test)

acc_decision_tree = round(model_decision_tree.score(x_train, y_train) * 100, 2)
acc_decision_tree
# gerar arquivo csv para submissão ao kaggle
submission = pd.DataFrame({
    "Id": id_number,
    "SalePrice": predict_rf # regressão linear
})

submission.to_csv('submission.csv', index=False)
print("Arquivo gerado com sucesso")
submission
