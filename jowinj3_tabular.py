# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
!pip install statsmodels
!pip install missingno 
import statsmodels.imputation.mice as st
import pandas as pd

weather= pd.read_csv("../input/hourly-weather-surface-brazil-southeast-region/sudeste.csv")
path="../input/hourly-weather-surface-brazil-southeast-region/sudeste.csv"
df1 = weather.iloc[:int(len(weather)/2)]
df2 = weather.iloc[int(len(weather)/2):]

import missingno as msno 
msno.matrix(df1) 
#msno.matrix(df2) 
df1.info()
#df1['hmax'].plot.kde()
print(df1.isna().mean())
reduce_df1=df1.loc[:, df1.isna().mean() < .07]


reduce_df1.info()
print(reduce_df1.isna().mean())
#imp=st.MICEData(reduce_df1)
#reduce_df1.columns[weather.isnull().mean() < 0.8]
#weather[weather.columns[weather.isnull().mean() < 0.8]]
#weather.info()

"""thresh = len(weather) * .02
print(weather)
weather_new = weather.dropna(thresh = thresh, axis = 1, inplace = True)
print(weather_new)"""
#print (weather.isin([' ','NULL',0]))
#reduce_df2=np.empty()
#corr_matrix
import numpy as np
import matplotlib.pyplot as plt
plt.matshow(reduce_df1.corr())
plt.show()
import matplotlib.pyplot as plt
f = plt.figure(figsize=(31, 15))
plt.matshow(reduce_df1.corr(), fignum=f.number)
plt.xticks(range(reduce_df1.shape[1]),reduce_df1.columns, fontsize=14, rotation=45)
plt.yticks(range(reduce_df1.shape[1]), reduce_df1.columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=31)
plt.title('Correlation Matrix', fontsize=16);
corr_matrix = reduce_df1.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
reduce_df1=reduce_df1.drop(reduce_df1[to_drop], axis=1)
reduce_df1.corr()
cols = reduce_df1.columns
num_cols = reduce_df1._get_numeric_data().columns
cat_data=list(set(cols) - set(num_cols))
con_data=list(set(num_cols))
cat_data

len(con_data)

from fastai.tabular import * 
#procs = [FillMissing, Categorify, Normalize]
#valid_idx = range(len(reduce_df1)-2000, len(reduce_df1))
#dep_var = 'wdct'


#valid_idx
#data = TabularDataBunch.from_df(path,reduce_df1, dep_var, valid_idx=valid_idx, procs=procs, bs=64, cat_names=cat_data)
#print(type(data))
#print(data.train_ds.cont_names)
#print(data.train_ds.cat_names)

#learn = tabular_learner(data, layers=[1000,500], metrics=accuracy)
#learn
#learn.fit_one_cycle(1, 2.5)

reduce_df1.info()
tfm = FillMissing(cat_data, con_data)
#tfm(reduce_df1)
#tfm(valid_df, test=True)
#train_df[cont_names].head()
tfm(reduce_df1)


reduce_df1[0:2000000]=reduce_df1[0:2000000].interpolate(method='quadratic',limit_direction='forward')

reduce_df1[2000000:len(reduce_df1)]=reduce_df1[2000000:len(reduce_df1)].interpolate(method='quadratic',limit_direction='forward')
reduce_df1.isna().mean()
reduce_df1.info()
con_data.remove('hr')
con_data.remove('yr')
con_data.remove('temp')
con_data
from sklearn.preprocessing import StandardScaler
# Separating out the features
x = reduce_df1.loc[:, con_data].values
# Separating out the target
y = reduce_df1.loc[:, 'temp'].values
# Standardizing the features
#x = StandardScaler().fit_transform(x)
y.shape
import sklearn
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
pca = PCA(n_components = "mle")
#pca = PCA(n_components == 'mle',svd_solver == 'full'
from sklearn.linear_model import LinearRegression
X_pca=pca.fit_transform(x)
x_train,x_test,y_train,y_test=train_test_split(X_pca,y,test_size=0.5)
lr=LinearRegression().fit(x_train,y_train)

y_pred=lr.predict(x_test)
from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))
con_data

one_hot_encoded_training_predictors = pd.get_dummies(reduce_df1[cat_dataw])
pca.fit(reduce_df1[con_data])

print(pca.explained_variance_ratio_)
print(pca.explained_variance_)
print(pca.singular_values_)
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = con_data
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
%matplotlib inline
reduce_df1.plot(x='temp', y='dewp', style='o')  
plt.title('MinTemp vs MaxTemp')  
plt.xlabel('MinTemp')  
plt.ylabel('MaxTemp')  
plt.show()
X = reduce_df1['temp'].values.reshape(-1,1)
y = reduce_df1['dewp'].values.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = LinearRegression()  
regressor.fit(X_train, y_train) #training the algorithm
#To retrieve the intercept:
print(regressor.intercept_)
#For retrieving the slope:
print(regressor.coef_)
y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df


























