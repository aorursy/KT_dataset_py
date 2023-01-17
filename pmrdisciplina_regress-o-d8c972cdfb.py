# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
train = pd.read_csv('../input/train.csv')
train = train.set_index('Id')
y_train = train['median_house_value']
train.head() #mostra os 5 primeiros itens do meu datastet
train.tail() #mostra os 5 ultimos itens do meu dataset
train.describe()
train.shape
train.columns[:-1]
plt.scatter(train['median_income'], y_train, marker = '+')
plt.ylabel('Valor da casa')
plt.xlabel('Renda media')
plt.title('Relação entre o valor da casa e a renda média')
plt.show()
plt.scatter(train['median_age'], y_train, marker = '+', color = 'r')
plt.ylabel('Valor da casa')
plt.xlabel('Idade media')
plt.title('Relação entre o valor da casa e a idade média')
plt.show()
plt.scatter(x=train['longitude'], y=train['latitude'], alpha = 0.4)
plt.show()
train.plot(kind ='scatter', x = 'longitude', y = 'latitude', s = train['population']/100, label = 'population',
           c = 'median_house_value', cmap=plt.get_cmap("jet"), colorbar=True, alpha = 0.4, figsize = (10, 7))
plt.legend()
plt.show()
import scipy.stats as st
plt.title('Distribuição normal')
sns.distplot(y_train, kde=False, fit = st.norm)
plt.show()
plt.title('Johnson SUl')
sns.distplot(y_train, kde = False, fit = st.johnsonsu)
plt.show()
corr = train.corr()
corr.head()
corr['median_house_value'].sort_values()
mask = np.zeros_like(corr, dtype=np.bool) 
mask[np.triu_indices_from(mask)] = True
fig, axis = plt.subplots(figsize=(16, 12))
plt.title('Matriz de correlação',fontsize=25)

sns.heatmap(corr,linewidths=0.25,vmax=1.0,square=True, 
            linecolor='w',annot=True,mask=mask,cbar_kws={"shrink": .75})
plt.show()

sns.pairplot(train, size = 2, kind = 'scatter')
plt.show()
fig, ((ax1, ax2), (ax3, ax4),(ax5,ax6)) = plt.subplots(nrows=3, ncols=2, figsize=(14,10))
sns.regplot(x = train['total_bedrooms'], y=train['households'], scatter = True, fit_reg = True, ax = ax1)
sns.regplot(x = train['total_bedrooms'], y=train['total_rooms'], scatter = True, fit_reg = True, ax = ax2)
sns.regplot(x = train['population'], y=train['total_rooms'], scatter = True, fit_reg = True, ax = ax3)
sns.regplot(x = train['population'], y=train['households'], scatter = True, fit_reg = True, ax = ax4)
sns.regplot(x = train['total_rooms'], y=train['households'], scatter = True, fit_reg = True, ax = ax5)
sns.regplot(x = train['latitude'], y=train['longitude'], scatter = True, fit_reg = True, ax = ax6)
plt.show()
train.columns
train['income_x_pop'] = train['median_income']/train['population']
train['rooms_x_pop'] = train['total_rooms']/train['population']
train['bedrooms_x_rooms'] = train['total_bedrooms']/train['total_rooms']
train['location'] = (train['latitude'] + train['longitude']) / 2
train.corr()['median_house_value'].sort_values()
x_train =  train[['location', 'bedrooms_x_rooms', 'income_x_pop', 'rooms_x_pop', 'median_income']]
y_train = train['median_house_value']
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
reg = linear_model.LinearRegression()
reg.fit(x_train, y_train)
cross_val_score(reg, x_train, y_train, cv = 5).mean()
ridge = linear_model.Ridge(alpha = .5)
ridge.fit(x_train, y_train)
cross_val_score(ridge, x_train, y_train, cv = 5).mean()
lasso = linear_model.Lasso(alpha = 0.1)
lasso.fit(x_train, y_train)
cross_val_score(lasso, x_train, y_train, cv = 5).mean()
from sklearn.neighbors import KNeighborsRegressor
for i in range(1, 31):
    knn = KNeighborsRegressor(n_neighbors = i)
    knn.fit(x_train, y_train)
    print(i, cross_val_score(knn, x_train, y_train, cv = 5).mean())
from xgboost import XGBRegressor
model = XGBRegressor()
model.fit(x_train, y_train)
cross_val_score(model, x_train, y_train, cv = 5)
test = pd.read_csv('../input/test.csv')
submit = pd.read_csv('../input/sample_sub_1.csv')
test['income_x_pop'] = test['median_income']/test['population']
test['rooms_x_pop'] = test['total_rooms']/test['population']
test['bedrooms_x_rooms'] = test['total_bedrooms']/test['total_rooms']
test['location'] = (test['latitude'] + test['longitude']) / 2
x_test = test[['location', 'bedrooms_x_rooms', 'income_x_pop', 'rooms_x_pop', 'median_income']]
predict = model.predict(x_test)
submit['median_house_value'] = np.abs(predict)
submit = submit.set_index('Id')
submit.head()
submit.to_csv('submission_2.csv')