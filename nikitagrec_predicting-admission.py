import numpy as np

import pandas as pd

import seaborn as sns

import random

import scipy.stats as stt

import warnings

import sklearn

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_curve,roc_auc_score

from sklearn.metrics import classification_report

from sklearn.model_selection import cross_val_score

from sklearn.metrics import r2_score

from sklearn import linear_model

from sklearn.model_selection import GridSearchCV

warnings.filterwarnings('ignore')

%pylab inline
data = pd.read_csv('../input/Admission_Predict.csv')

#data2 = pd.read_csv('../input/Admission_Predict_Ver1.1.csv')
data.drop(['Serial No.'],axis=1,inplace=True)

data.head(5)

print(data.shape)
sns.pairplot(data[['GRE Score', 'TOEFL Score','CGPA','Chance of Admit ']]);
fig = plt.figure(figsize=(20,20))

sns.set(style="white",font_scale=2)

sns.heatmap(data.corr(), fmt='.2f',annot=True,linewidth=2);
fig = plt.figure(figsize=(20,20));

sns.set(font_scale=1.5);

pd.crosstab(data['University Rating'],data.Research).plot(kind='barh');
fig = plt.figure(figsize=(20,10))

ax1 = fig.add_subplot(2,2,1)

#ax1.set_title('SOP')

data["Chance of Admit "].groupby(data['SOP']).mean().plot();

ax1.legend()

ax2 = fig.add_subplot(2,2,2)

#ax2.set_title("LOR")

data["Chance of Admit "].groupby(data['LOR ']).mean().plot(color='red');

ax2.legend();
fig = plt.figure(figsize=(7,7))

sns.set(font_scale=1.5)

sns.regplot(x=data['GRE Score'],y=data['TOEFL Score'],marker='+');

lin1 = sklearn.linear_model.LinearRegression()

x = np.transpose(np.atleast_2d(data['GRE Score']))

lin1.fit(x,data['TOEFL Score'])

print('R_coeff = ',r2_score(data['TOEFL Score'],lin1.predict(x)))
fig = plt.figure(figsize=(7,7))

sns.set(font_scale=1.5)

sns.regplot(x=data['CGPA'],y=data['Chance of Admit '],color='green');

lin1 = sklearn.linear_model.LinearRegression()

x = np.transpose(np.atleast_2d(data['CGPA']))

lin1.fit(x,data['Chance of Admit '])

print('R_coeff = ',r2_score(data['Chance of Admit '],lin1.predict(x)))
lin_full = linear_model.LinearRegression()

x = np.transpose(np.atleast_2d(data[['GRE Score', 'TOEFL Score', 'University Rating',\

                            'SOP', 'LOR ', 'CGPA','Research']]))

xx = data[['GRE Score', 'TOEFL Score', 'University Rating',\

                            'SOP', 'LOR ', 'CGPA','Research']]

lin_full.fit(xx,data['Chance of Admit '])

print('R_coeff = ',r2_score(data['Chance of Admit '],lin_full.predict(xx)))
ss = pd.DataFrame(np.hstack((np.array(list(data.drop(['Chance of Admit '],\

                                                     axis=1).columns)).reshape(7,1),lin_full.coef_.reshape(7,1))))

features = ss[0]

importances = lin_full.coef_

indices = np.argsort(importances)



plt.figure(figsize=(10,10))

plt.title('Regression coefficients')

plt.barh(range(len(indices)), importances[indices], color='b', align='center');

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Сoefficient value');
lin_full = linear_model.LinearRegression()

#x = np.transpose(np.atleast_2d(data[['LOR ', 'CGPA','Research']]))

xx = data[['LOR ', 'CGPA','Research']]

lin_full.fit(xx,data['Chance of Admit '])

print('R_coeff = ',r2_score(data['Chance of Admit '],lin_full.predict(xx)))
lin_reg = linear_model.ElasticNet(alpha=0.05, l1_ratio=0.05,)

x = np.transpose(np.atleast_2d(data[['GRE Score', 'TOEFL Score', 'University Rating',\

                            'SOP', 'LOR ', 'CGPA','Research']]))

xx = data[['GRE Score', 'TOEFL Score', 'University Rating',\

                            'SOP', 'LOR ', 'CGPA','Research']]

lin_reg.fit(xx,data['Chance of Admit '])

print('R_coeff = ',r2_score(data['Chance of Admit '],lin_reg.predict(xx)))
parameters = {'alpha':[0.001, 3],"l1_ratio":[0.001,3]}

clf = GridSearchCV(lin_reg, parameters, cv=5)

clf.fit(xx,data['Chance of Admit '])

clf.best_params_