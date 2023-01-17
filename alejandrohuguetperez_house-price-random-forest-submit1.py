# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#import libraries
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, classification_report
from sklearn.model_selection import train_test_split, cross_val_score

from warnings import filterwarnings
filterwarnings('ignore')
# Import data. Read files
train_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train_data.shape
# Drop Id from train data
train_data.drop("Id", axis = 1, inplace = True)
#delete excess null values
def delete_null (df, p100):
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    list_null = percent[percent > p100].index
    df.drop(list_null, axis=1, inplace=True)
    return df
#handle null values
def handle_null(df, func):
    na_cols=df.columns[df.isna().sum()>0]
    for col in na_cols:
        if func=='mean':
            df[col]=df[col].fillna(df[col].mean())
        if func=='mode':
            df[col]=df[col].fillna(df[col].mode()[0])
    return df
#check null values
total = train_data.isnull().sum().sort_values(ascending=False)
percent = (train_data.isnull().sum()/train_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data = missing_data.loc[missing_data.Percent>0]
missing_data
#delete variable with more than 17% of null values
train_data = delete_null(train_data, 0.17)
train_data.shape
test_data = delete_null(test_data, 0.17)
test_data.shape
train_data.describe()
# check correlation between independent variables
corrmat = train_data.corr()
c1= corrmat.abs().unstack()
c1[c1==1] = 0
c2= c1.sort_values(ascending = False)
c2= c2[c2 > 0.70]
c2
# delete correlational variables with less impact on SalePrice
train_data = train_data.drop(['TotRmsAbvGrd','GarageArea','1stFlrSF','GarageYrBlt'], axis=1)
#same for test data
test_data = test_data.drop(['TotRmsAbvGrd','GarageArea','1stFlrSF','GarageYrBlt'], axis=1)
# Move target variable SalePrice to another data frame
Y_data = train_data.SalePrice
train_data.drop(['SalePrice'], axis=1, inplace=True)
# Split in numeric and categorical data
#Get numerical data
num_data = train_data.select_dtypes(include=np.number)
num_test = test_data.select_dtypes(include=np.number)
num_data
#Get categorical data
cat_data = train_data.select_dtypes(include=np.object)
cat_test = test_data.select_dtypes(include=np.object)
cat_data
num_data.describe()
#Numeric variables with null values
null_data = num_data.isnull().sum().sort_values(ascending=False)
null_data = null_data[null_data > 0]
null_data
#Replace null values with mean
num_data=handle_null(num_data, 'mean')
num_test=handle_null(num_test, 'mean')
#Describe categorical variables
cat_data.describe()
#List values in categorical variables
for variable in cat_data.columns:
    print(
        f"{variable} :{len(cat_data[variable].unique())}: {cat_data[variable].unique()}"
    )
#Replace null values with mean
cat_data=handle_null(cat_data, 'mode')
cat_test=handle_null(cat_test, 'mode')
# Encoding of categorical variables
label_encoder = LabelEncoder()
for col in cat_data.columns:
    cat_data[col] = label_encoder.fit_transform(cat_data[col])
label_encoder = LabelEncoder()
for col in cat_test.columns:
    cat_test[col] = label_encoder.fit_transform(cat_test[col])
cat_data.head()
#concat numerical and categorical variable
data_final = pd.concat([num_data,cat_data], axis=1)
data_final_test = pd.concat([num_test,cat_test], axis=1)
data_final.shape
Y_data.shape
#split in train and test data
X_train, X_test, Y_train, Y_test = train_test_split(data_final, Y_data, train_size=0.75, random_state=23)
print('X_train :'+str(X_train.shape))
print('Y_train :'+str(Y_train.shape))
print('---')
print('Y_test :'+str(Y_test.shape))
print('X_test :'+str(X_test.shape))
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k=30)
X_new = selector.fit_transform(X_train, Y_train)
# Get back the kept features as a DataFrame with dropped columns as all 0s
selected_features = pd.DataFrame(selector.inverse_transform(X_new), index=X_train.index, columns=X_train.columns)
# Find the columns that were dropped
dropped_columns = selected_features.columns[selected_features.var() == 0]
dropped_columns
X_features = selected_features.columns[selected_features.var() != 0]
X_features
X_train[X_features].shape
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train[X_features],Y_train)
#Model accuracy
print('Precisión del entrenamiento: '+str(lr.score(X_train[X_features], Y_train)))
print('Precisión en la validación: '+str(lr.score(X_test[X_features], Y_test)))
#Cross-validation 7 folds
lr_cross = cross_val_score(lr, data_final[X_features], Y_data, cv=7)
print(lr_cross)
print('Mean Absolute Error %2f' %(lr_cross.mean()))
#model prediction
lr_pred= lr.predict(X_test[X_features])
lr_mse = mean_squared_error(Y_test, lr_pred)
lr_rmse = sqrt(lr_mse)
lr_mae = mean_absolute_error(Y_test, lr_pred)
print("Error Cuadratico Medio (MSE) en la prediccion: {:.4f}".format(lr_mse))
print("Raíz Cuadrada del Error Cuadratico Medio (RMSE) en la prediccion: {:.4f}".format(lr_rmse))
print("Error Absoluto Medio (MAE): {:.4f}".format(lr_mae))
from sklearn.tree import DecisionTreeRegressor
# with pruning = 8
tree_model = DecisionTreeRegressor(max_depth=8)
tree_model.fit(X_train[X_features], Y_train)
#Model accuracy
print('Precisión del entrenamiento: '+str(tree_model.score(X_train[X_features], Y_train)))
print('Precisión en la validación: '+str(tree_model.score(X_test[X_features], Y_test)))
#Cross-validation 7 folds
tree_cross = cross_val_score(tree_model, data_final[X_features], Y_data, cv=7)
print(tree_cross)
print('Mean Absolute Error %2f' %(tree_cross.mean()))
#model prediction
tree_pred= tree_model.predict(X_test[X_features])
tree_mse = mean_squared_error(Y_test, tree_pred)
tree_rmse = sqrt(tree_mse)
tree_mae = mean_absolute_error(Y_test, tree_pred)
print("Error Cuadratico Medio (MSE) en la prediccion: {:.4f}".format(tree_mse))
print("Raíz Cuadrada del Error Cuadratico Medio (RMSE) en la prediccion: {:.4f}".format(tree_rmse))
print("Error Absoluto Medio (MAE): {:.4f}".format(tree_mae))
forest_model=RandomForestRegressor(n_estimators=300, n_jobs=None, max_features='auto', oob_score=True)
forest_model.fit(X_train[X_features], Y_train)
#Model accuracy
print('Precisión del entrenamiento: '+str(forest_model.score(X_train[X_features], Y_train)))
print('Precisión en la validación: '+str(forest_model.score(X_test[X_features], Y_test)))
print('Error oob: '+str(forest_model.oob_score_))
#Cross-validation 7 folds
forest_cross = cross_val_score(forest_model, data_final[X_features], Y_data, cv=7)
print(forest_cross)
print('Mean Absolute Error %2f' %(forest_cross.mean()))
#model prediction
forest_pred= forest_model.predict(X_test[X_features])
forest_mse = mean_squared_error(Y_test, forest_pred)
forest_rmse = sqrt(forest_mse)
forest_mae = mean_absolute_error(Y_test, forest_pred)
print("Error Cuadratico Medio (MSE) en la prediccion: {:.4f}".format(forest_mse))
print("Raíz Cuadrada del Error Cuadratico Medio (RMSE) en la prediccion: {:.4f}".format(forest_rmse))
print("Error Absoluto Medio (MAE): {:.4f}".format(forest_mae))
plt.figure(figsize=(15,12))

plt.subplot(221)
# Mostramos en una gráfica para Arboles de Decisión
plt.scatter(tree_pred, Y_test)
plt.ylabel('Real price')
plt.xlabel('Predicted price')
plt.title(r'Decision Tree')
# Añadir la linea ideal de predicción
diagonal = np.linspace(0, np.max(tree_pred), 100)
plt.plot(diagonal, diagonal, '-r')

plt.subplot(222)
# Mostramos en una gráfica precisión de Random Forest
plt.scatter(forest_pred, Y_test)
plt.ylabel('Real price')
plt.xlabel('Predicted price')
plt.title(r'Random Forest')
# Añadir la linea ideal de predicción
diagonal = np.linspace(0, np.max(forest_pred), 100)
plt.plot(diagonal, diagonal, '-r')

plt.subplot(223)
# Mostramos en una gráfica para Arboles de Decisión
plt.scatter(lr_pred, Y_test)
plt.ylabel('Real price')
plt.xlabel('Predicted price')
plt.title(r'Linear Regression')
# Añadir la linea ideal de predicción
diagonal = np.linspace(0, np.max(lr_pred), 100)
plt.plot(diagonal, diagonal, '-r')

plt.show()
test_pred= forest_model.predict(data_final_test[X_features])
data_final_test['SalePrice']=test_pred
df_solution = data_final_test[['Id','SalePrice']]
df_solution.head(10)
df_solution.to_csv ('sample_submission.csv', index = False, header=True)