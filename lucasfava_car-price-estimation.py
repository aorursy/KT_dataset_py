#Importação de bibliotecas
import pandas as pd
import seaborn as sns
from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
import pickle
import os
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv("/kaggle/input/car-price-estimation/datasets_33080_43333_car data.csv")
df.shape
df.describe()
df.head()
df.sample(6)
final_dataset = df[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]
final_dataset.head()
final_dataset['Current_Year'] = 2020
final_dataset['Num_of_Years'] = final_dataset['Current_Year'] - final_dataset['Year']
final_dataset.head()
final_dataset.drop(['Num_of_Years'],axis=1,inplace=True)
final_dataset.head()
final_dataset = pd.get_dummies(final_dataset, drop_first = True)
final_dataset.head()
final_dataset = final_dataset.drop(['Current_Year'], axis=1)
sns.pairplot(final_dataset)
#Correlação entre as variáveis
final_dataset.corr()
#Correlações entre as features do dataset
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#Plotagem do  heat map
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
X = final_dataset.iloc[:,1:]
y = final_dataset.iloc[:,0]
X['Owner'].unique()
X.head()
y.head()
#Extração de feature importance
model = ExtraTreesRegressor()
model.fit(X,y)
print(model.feature_importances_)
#Plotagem das features mais relevantes
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(6).plot(kind='barh')
plt.show()
#Split de treino e teste
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
regressor = RandomForestRegressor()

import numpy as np
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
print(n_estimators)
#Randomized Search CV

n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)] #total de árvores na random forest
max_features = ['auto', 'sqrt'] #número de  features a se considerar em cada split
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)] #máximo de níveis por árvore
# max_depth.append(None)
min_samples_split = [2, 5, 10, 15, 100] 
min_samples_leaf = [1, 2, 5, 10] 

# Criação do random grid, para seleção de hiperparâmetros
random_grid = {'n_estimators': n_estimators,
              'max_features': max_features,
              'max_depth': max_depth,
              'min_samples_split': min_samples_split,
              'min_samples_leaf': min_samples_leaf}

print(random_grid)
rf = RandomForestRegressor()
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
print(n_estimators)
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)
rf_random.fit(X_train,y_train)
rf_random.best_params_
rf_random.best_score_
predictions = rf_random.predict(X_test)
sns.distplot(y_test-predictions)
plt.scatter(y_test,predictions)
#Métricas:
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
