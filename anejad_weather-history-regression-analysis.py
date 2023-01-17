# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split

from sklearn import linear_model
data  = pd.read_csv('../input/weatherHistory.csv')
data.head()
data.shape
data.describe()
data.isna().any()
data=data.dropna()
modeling_data=data.copy()

modeling_data=modeling_data.drop(['Daily Summary','Loud Cover'], axis=1)

le = LabelEncoder()

modeling_data['Summary']=le.fit(modeling_data['Summary']).transform(modeling_data['Summary'])

le2 = LabelEncoder()

modeling_data['Precip Type']=le2.fit(modeling_data['Precip Type']).transform(modeling_data['Precip Type'])
corr = modeling_data.corr()

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(11, 9))

cmap = sns.diverging_palette(0, 150, as_cmap=True)

sns.heatmap(corr, mask=mask, cmap=cmap,  center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
modeling_data=modeling_data.drop(['Apparent Temperature (C)','Formatted Date','Summary',],axis=1)

modeling_data=modeling_data[modeling_data['Humidity']>0]
X_train, X_test, y_train, y_test = train_test_split( modeling_data['Humidity'], 

                                                      modeling_data['Temperature (C)'], 

                                                      test_size=0.33, random_state=42)
reg = linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)

reg.fit(X_train.values.reshape(-1, 1),y_train.values.reshape(-1, 1))

reg.coef_
print ('In sample regression score: ' + str(reg.score(X_train.values.reshape(-1, 1), y_train.values.reshape(-1, 1))))
print ('Out of sample regression score: ' + str(reg.score(X_test.values.reshape(-1, 1), y_test.values.reshape(-1, 1)))) 
font = {'size'   : 20}

plt.rc('font', **font)

plt.figure(figsize=(13,10))

plt.plot(modeling_data['Humidity'],modeling_data['Temperature (C)'],'o',label='Data')

I=np.linspace(np.floor(min(modeling_data['Humidity'])*0.95),np.ceil(max(modeling_data['Humidity'])*0.11),50)

plt.plot(I,reg.predict(I.reshape(-1, 1)),color='r', linewidth=3,label='Regression Line')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.xlabel('Humidity');plt.ylabel('Temperature (C)')

Preds=reg.predict( modeling_data['Humidity'].values.reshape(-1, 1))

R2=r2_score(modeling_data['Temperature (C)'],Preds )

plt.title('R2: '+ str(np.round(R2,decimals=3)))

plt.show()
Residuals=modeling_data['Temperature (C)'].values.reshape(-1, 1)-Preds

font = {'size'   : 20}

plt.rc('font', **font)

fig, ax = plt.subplots(1,2,figsize=(20,10))

num_bins = 50

n, bins, patches = ax[0].hist(Residuals, num_bins, density=1)

ax[0].title.set_text('Regression Residuals');





ax[1].plot(modeling_data['Humidity'],Residuals,'o')

ax[1].set(xlabel='Fitted Value', ylabel='Residual Value')

ax[1].hlines(0, np.min(modeling_data['Humidity'])*0.95, np.max(modeling_data['Humidity'])*1.1, colors='r', linestyles='solid',zorder=10, linewidth=4 )

plt.show()
!pip install plotly_express 
import plotly_express as px
iris = px.data.iris()
px.scatter(iris, x="sepal_width", y="sepal_length")