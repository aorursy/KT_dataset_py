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
# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier


COVID19_line_list_data = pd.read_csv("../input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv")

COVID19_open_line_list = pd.read_csv("../input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv")

covid_19_data = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")

time_series_covid_19_confirmed = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")

time_series_covid_19_deaths = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")

time_series_covid_19_recovered = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")
COVID19_line_list_data.head()
#Removing unwanted columns

COVID19_line_list_data.drop(columns=['id','case_in_country','summary','symptom','link'],inplace=True)
#Found an unnamed column which is no use to us

cols=[-1,-2,-3,-4,-5,-6,1]

COVID19_line_list_data=COVID19_line_list_data.drop(COVID19_line_list_data.columns[cols],axis=1)





COVID19_line_list_data.head(2)
#Converted Gender to numeric

COVID19_line_list_data=COVID19_line_list_data.drop(COVID19_line_list_data.columns[-1],axis=1)

COVID19_line_list_data.head(2)
COVID19_line_list_data['gender_num']=COVID19_line_list_data['gender'].map({'male': 1, 'female':0})

COVID19_line_list_data.head(2)
#To find which gender was infected more

gend=pd.DataFrame(COVID19_line_list_data['gender'].value_counts())

gend['Sex']=gend.index

gend.columns=['count','sex']

gend.head()
plt.figure(figsize=(12,5))

sns.barplot(x='sex',y='count',data=gend.iloc[:])

#plt.show()
import random

import matplotlib.colors as mcolors

plt.figure(figsize=(15,5))

gend.plot(kind='pie', y = 'count', autopct='%1.1f%%', 

 startangle=90, shadow=False, labels=gend['sex'], legend = True, fontsize=14)

plt.show()
#To find which age group was hit badly

age=pd.DataFrame(COVID19_line_list_data['age'].value_counts())

age['Age']=age.index

age.columns=['count','Age']

plt.figure(figsize=(15,10))

sns.barplot(x='Age',y='count',data=age.iloc[:])

plt.xticks(rotation=90)

plt.show()
age['age_bins'] = pd.cut(x=age['Age'], bins=[0, 25, 50, 75, 100],labels=["0-25","25-50","50-75","75-100"])

a=age.groupby('age_bins').size()

a.plot.pie(figsize=(10,5),autopct='%1.1f%%',legend=True)
#Finding which country and location has the highest count

locate=COVID19_line_list_data.groupby(['country','location']).count()[['reporting date']].sort_values(by='reporting date',ascending=False).reset_index()

locate
#FInd which country highly affected using seaborn count plot

plt.figure(figsize=(15,5))

sns.countplot(data=COVID19_line_list_data,x='country')

plt.xticks(rotation=90)

plt.show()
#FInd which country highly affected using matplotlib

plt.figure(figsize=(15,5))

COVID19_line_list_data['country'].value_counts().plot(kind='bar')

plt.show
#confirmed cases with Time series file



from collections import defaultdict

confirmed_date_data=time_series_covid_19_confirmed.iloc[:,4:]

d=defaultdict(int)

#for i in range(len(confirmed_date_data.columns)):

#    d[confirmed_date_data.columns[i]]=confirmed_date_data.iloc[:,i].sum()



for i in confirmed_date_data.columns:

      d[i]=  confirmed_date_data[i].sum()

    

print(d)
#Find which country highly affected using matplotlib

dates=d.keys()

vals=d.values()

plt.figure(figsize=(15,5))

plt.bar(dates,vals)

plt.xticks(rotation=90)

plt.show
#death cases with Time series file



death_date_data=time_series_covid_19_deaths.iloc[:,4:]

d1=defaultdict(int)

#for i in range(len(confirmed_date_data.columns)):

#    d[confirmed_date_data.columns[i]]=confirmed_date_data.iloc[:,i].sum()



for i in death_date_data.columns:

      d1[i]=  death_date_data[i].sum()

    

print(d1)

#Find which country highly affected using matplotlib

dates_death=d1.keys()

vals_death=d1.values()

plt.figure(figsize=(15,5))

plt.bar(dates_death,vals_death)

plt.xticks(rotation=90)

plt.show
#recovered cases with Time series file



recovered_date_data=time_series_covid_19_recovered.iloc[:,4:]

d2=defaultdict(int)

#for i in range(len(confirmed_date_data.columns)):

#    d[confirmed_date_data.columns[i]]=confirmed_date_data.iloc[:,i].sum()



for i in recovered_date_data.columns:

      d2[i]=  recovered_date_data[i].sum()

#Find which country highly affected using matplotlib

dates_recovered=d2.keys()

vals_recovered=d2.values()

plt.figure(figsize=(15,5))

plt.bar(dates_recovered,vals_recovered)

plt.xticks(rotation=90)

plt.show
country_confirm=time_series_covid_19_confirmed.iloc[:,np.r_[1,4:57]] #Using numpy object to get slices of columns

country_deaths=time_series_covid_19_deaths.iloc[:,np.r_[1,4:57]]

country_recovered=time_series_covid_19_recovered.iloc[:,np.r_[1,4:57]]

country_recovered.head(2)
country_confirm.head(3)
#Top 20 confirmed coutries

dict_cnf=defaultdict(int)

l_cnf=country_confirm['Country/Region'].unique()

cn=country_confirm.groupby(['Country/Region']).sum().reset_index()

for i in l_cnf:

    l=list(cn[cn['Country/Region']==i].sum(axis=1))

    dict_cnf[i]=l[0]

dict_cnf={k: v for k, v in sorted(dict_cnf.items(), key=lambda item: item[1],reverse=True)[:20]}

country_cnf=dict_cnf.keys()

vals_cnf=dict_cnf.values()

plt.figure(figsize=(15,5))

plt.bar(country_cnf,vals_cnf)

plt.xticks(rotation=90)

plt.show
#Top 20 confirmed coutries

dict_dea=defaultdict(int)

l_dea=country_deaths['Country/Region'].unique()

de=country_deaths.groupby(['Country/Region']).sum().reset_index()

for i in l_dea:

    l=list(de[de['Country/Region']==i].sum(axis=1))

    dict_dea[i]=l[0]

dict_dea={k: v for k, v in sorted(dict_dea.items(), key=lambda item: item[1],reverse=True)[:20]}

country_dea=dict_dea.keys()

vals_dea=dict_dea.values()

plt.figure(figsize=(15,5))

plt.bar(country_dea,vals_dea)

plt.xticks(rotation=90)

plt.show
#Top 20 confirmed coutries

dict_rec=defaultdict(int)

l_rec=country_recovered['Country/Region'].unique()

re=country_recovered.groupby(['Country/Region']).sum().reset_index()

for i in l_rec:

    l=list(re[re['Country/Region']==i].sum(axis=1))

    dict_rec[i]=l[0]

dict_rec={k: v for k, v in sorted(dict_rec.items(), key=lambda item: item[1],reverse=True)[:20]}

country_rec=dict_rec.keys()

vals_rec=dict_rec.values()

plt.figure(figsize=(15,5))

plt.bar(country_rec,vals_rec)

plt.xticks(rotation=90)

plt.show
dates
#Thanks to therealcyberlord to give an insight on ML techinques

days_in_future = 5

future_forcast = np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1,1)

adjusted_dates = future_forcast[:-5]

adjusted_dates
import datetime

start = '1/22/2020'

start_date = datetime.datetime.strptime(start, '%m/%d/%Y')

future_forcast_dates = []

for i in range(len(future_forcast)):

    future_forcast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))



future_forcast_dates
vals

days_since_1_22 = np.array([i for i in range(len(dates))]).reshape(-1, 1)

confirmed_cases = np.array(list(vals)).reshape(-1,1)
days_since_1_22
confirmed_cases
from sklearn.linear_model import LinearRegression, BayesianRidge

from sklearn.model_selection import RandomizedSearchCV, train_test_split

from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error, mean_absolute_error

import datetime

import operator

plt.style.use('seaborn')

%matplotlib inline 



X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(days_since_1_22, confirmed_cases, test_size=0.20, shuffle=False) 
#Implementing support vector machine

kernel = ['poly', 'sigmoid', 'rbf']

c = [0.01, 0.1, 1]

gamma = [0.01, 0.1, 1]

epsilon = [0.01, 0.1, 1]

shrinking = [True, False]

svm_grid = {'kernel': kernel, 'C': c, 'gamma' : gamma, 'epsilon': epsilon, 'shrinking' : shrinking}



svm = SVR()

svm_search = RandomizedSearchCV(svm, svm_grid, scoring='neg_mean_squared_error', cv=5, return_train_score=True, n_jobs=-1, n_iter=100, verbose=1)

svm_search.fit(X_train_confirmed, y_train_confirmed)
svm_search.best_params_
svm_confirmed = SVR(shrinking=True, kernel='poly', gamma=0.1, epsilon=1, C=0.01)

svm_confirmed.fit(X_train_confirmed, y_train_confirmed)

svm_pred = svm_confirmed.predict(future_forcast)
# check against testing data

svm_test_pred = svm_confirmed.predict(X_test_confirmed)

plt.plot(svm_test_pred,color='red')

plt.plot(y_test_confirmed,color='green')
print('MAE:', mean_absolute_error(svm_test_pred, y_test_confirmed))

print('MSE:',mean_squared_error(svm_test_pred, y_test_confirmed))
#Predicting using linear regression



linear_model = LinearRegression(normalize=True, fit_intercept=True)

linear_model.fit(X_train_confirmed, y_train_confirmed)

test_linear_pred = linear_model.predict(X_test_confirmed)

linear_pred = linear_model.predict(future_forcast)

print('MAE:', mean_absolute_error(test_linear_pred, y_test_confirmed))

print('MSE:',mean_squared_error(test_linear_pred, y_test_confirmed))
print(linear_model.coef_)

print(linear_model.intercept_)
plt.plot(y_test_confirmed,color='red')

plt.plot(test_linear_pred,color='green')
#Using Bayesian regression



tol = [1e-4, 1e-3, 1e-2]

alpha_1 = [1e-7, 1e-6, 1e-5, 1e-4]

alpha_2 = [1e-7, 1e-6, 1e-5, 1e-4]

lambda_1 = [1e-7, 1e-6, 1e-5, 1e-4]

lambda_2 = [1e-7, 1e-6, 1e-5, 1e-4]



bayesian_grid = {'tol': tol, 'alpha_1': alpha_1, 'alpha_2' : alpha_2, 'lambda_1': lambda_1, 'lambda_2' : lambda_2}



bayesian = BayesianRidge()

bayesian_search = RandomizedSearchCV(bayesian, bayesian_grid, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=50, verbose=1)

bayesian_search.fit(X_train_confirmed, y_train_confirmed)
bayesian_search.best_params_
bayesian_confirmed = bayesian_search.best_estimator_

test_bayesian_pred = bayesian_confirmed.predict(X_test_confirmed)

bayesian_pred = bayesian_confirmed.predict(future_forcast)

print('MAE:', mean_absolute_error(test_bayesian_pred, y_test_confirmed))

print('MSE:',mean_squared_error(test_bayesian_pred, y_test_confirmed))
plt.plot(y_test_confirmed,color='red')

plt.plot(test_bayesian_pred,color='green')