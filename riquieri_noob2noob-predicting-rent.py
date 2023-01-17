import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
csv_path = '/kaggle/input/brasilian-houses-to-rent/houses_to_rent_v2.csv'
df = pd.read_csv(csv_path)
df.head(3)
df.shape
df.info()
miss = df.isnull().sum()
miss_= (df.isnull().sum()/df.isnull().count())*100

missing_data = pd.concat([miss, miss_], axis=1, keys=['Total', 'Percent'])
missing_data
df['city'].value_counts()
df['floor'].value_counts()
df['floor'] = df['floor'].replace(['-','301'], 0)
df['floor'].value_counts()
df['floor'] = df['floor'].astype(int)
df.info()
df['floor'] = df['floor'].replace(0, df['floor'].median())
df['floor'].value_counts()
ax = df['city'].value_counts().plot(kind='bar', figsize=(15,6))
ax.set_xlabel('Cities')
ax.set_title('Cities visualization', fontsize = 25)
ax = df['rooms'].value_counts().plot(kind='bar', figsize=(20,6))
ax.set_xlabel('Number of rooms')
ax.set_title('Rooms visualization')
ax = df['bathroom'].value_counts().plot(kind='bar', figsize=(15,6))
ax.set_xlabel('Number of bathroom')
ax.set_title('Bathroom visualization', fontsize = 25)
df['animal'].value_counts()
df['animal'] = df['animal'].replace(['acept'], 1)
df['animal'] = df['animal'].replace(['not acept'], 0)
df['furniture'].value_counts()
df['furniture'] = df['furniture'].replace(['furnished'], 1)
df['furniture'] = df['furniture'].replace(['not furnished'], 0)
df.head()
df.corr()
import matplotlib.pyplot as plt

x= df['city']
y= df['rent amount (R$)']

plt.scatter(x,y)
import matplotlib.pyplot as plt

x= df['area']
y= df['rent amount (R$)']

plt.scatter(x,y)
df.sort_values(by = 'area', ascending = False)[:2]
df = df.drop(df[df['area'] == 46335].index)
df = df.drop(df[df['area'] == 12732].index)
import matplotlib.pyplot as plt

x= df['area']
y= df['total (R$)']

plt.scatter(x,y)
x= df['rooms']
y= df['rent amount (R$)']

plt.scatter(x,y)
x= df['bathroom']
y= df['rent amount (R$)']

plt.scatter(x,y)
x= df['parking spaces']
y= df['rent amount (R$)']

plt.scatter(x,y)
import matplotlib.pyplot as plt

x= df['hoa (R$)']
y= df['rent amount (R$)']

plt.scatter(x,y)
y= df['rent amount (R$)']
x= df['total (R$)']

plt.scatter(x,y)
import matplotlib.pyplot as plt

x= df['property tax (R$)']
y= df['rent amount (R$)']

plt.scatter(x,y)
df.sort_values(by = 'property tax (R$)', ascending = False)[:2]
df = df.drop(df[df['property tax (R$)'] == 10830].index)

x= df['property tax (R$)']
y= df['rent amount (R$)']

plt.scatter(x,y)
x= df['fire insurance (R$)']
y= df['rent amount (R$)']

plt.scatter(x,y)
x = df.drop(['total (R$)','city','rent amount (R$)'], axis=1)
y = df['rent amount (R$)']
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
algorithm = SelectKBest(score_func=chi2, k=5)

best_features = algorithm.fit_transform(x,y)
print('scores:',algorithm.scores_)
#print('Resultado da transformacao:\n',dados_das_melhores_preditoras)
chi_scores = algorithm.scores_
chi_ = pd.DataFrame(data=chi_scores,columns = ['Chi^2'], index=['area','rooms','bathroom','parking spaces','floor','animal','furniture','hoa (R$)','property tax (R$)','fire insurance (R$)'])
chi_['Chi^2'].sort_values(ascending=False)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)
from sklearn.linear_model import LinearRegression
#lm = linear model
lm = LinearRegression()
lm.fit(x_train, y_train)
a_lm =lm.score(x_test, y_test)
print('R^2=', a_lm)         
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
kfold = KFold(n_splits=5,shuffle=True)
b_lm = cross_val_score(lm,x,y,cv=kfold)
print(b_lm.mean())
from sklearn.linear_model import Ridge
#rm = Ridge model
rm = Ridge()
rm.fit(x_train, y_train)
a_rm =rm.score(x_test, y_test)
print('R^2=',a_rm)
kfold = KFold(n_splits=5,shuffle=True)
b_rm = cross_val_score(rm,x,y,cv=kfold)
print(b_rm.mean())
from sklearn.linear_model import Lasso
#lassom = lasso model
lassom = Lasso(alpha=1000, max_iter=1000, tol=0.1)
lassom.fit(x_train, y_train)

a_lassom = lassom.score(x_test, y_test)
print('R^2=',a_lassom)
kfold = KFold(n_splits=5,shuffle=True)
b_lassom = cross_val_score(lassom,x,y,cv=kfold)
print(b_lassom.mean())
from sklearn.linear_model import ElasticNet
#em = elastic model
em = ElasticNet(alpha=1,max_iter=5000, l1_ratio=0.5, tol=0.2)
em.fit(x_train, y_train)

a_em = em.score(x_test, y_test)
print('R^2=',a_em)
kfold = KFold(n_splits=5,shuffle=True)
b_em = cross_val_score(em,x,y,cv=kfold)
print(b_em.mean())
import numpy as np
from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeRegressor
min_splits = np.array([2,3,4,5,6,7])
max_lever = np.array([3,4,5,6,7,9,11])
algorithm = ['mse', 'friedman_mse', 'mae']

valores_grid = {'min_samples_split':min_splits,
                'max_depth':max_lever,
                'criterion':algorithm
               }
tm = DecisionTreeRegressor()
grid_tm = GridSearchCV(estimator=tm, param_grid=valores_grid, cv=5, n_jobs=-1)
grid_tm.fit(x,y)
print('R2:',grid_tm.best_score_)
print('Min to split:',grid_tm.best_estimator_.min_samples_split)
print('Max depth:',grid_tm.best_estimator_.max_depth)
print('Algorithm:',grid_tm.best_estimator_.criterion)
score_df_data = {'Linear regression':[0.9870,0.9871],
        'Ridge regression':[0.9870,0.9872],
        'Lasso regression':[0.9828,0.9823],
        'ElasticNet':[0.9866,0.9866],
        'Decision tree regression':['Nan',0.9929]}



score_df = pd.DataFrame(data=score_df_data ,index=['Train_test','Cross_val'])

score_df
a = input('AREA:\n')
b = input('ROOMS:\n')
c = input('BATHROOM:\n')
d = input('PARKING SPACES:\n')
e = input('FLOOR:\n')
f = input('ANIMAL 1-YES/0-NO:\n')
g = input('FORNITURE 1-YES/0-NO:\n')
h = input('HOA:\n')
i = input('PROPERTY TAX:\n')
j = input('FIRE INSURANCE:\n')
l = rm.predict([[int(a),int(b),int(c),int(d),int(e),int(f),int(g),int(h),int(i),int(j)]])
print('The total value suggest is R$',l[0])
