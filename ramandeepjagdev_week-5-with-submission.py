# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sb
train = pd.read_csv('../input/covid19-global-forecasting-week-5/train.csv')
test = pd.read_csv('../input/covid19-global-forecasting-week-5/test.csv')
submission = pd.read_csv('../input/covid19-global-forecasting-week-5/submission.csv')
#exploring the data
train = pd.read_csv('../input/covid19-global-forecasting-week-5/train.csv')
test = pd.read_csv('../input/covid19-global-forecasting-week-5/test.csv')
submission_example = pd.read_csv('../input/covid19-global-forecasting-week-5/submission.csv')
test.head()
train.describe()
#print("Number of Country_Region: ", train['Country_Region'].nunique())
#print("Dates go from day", max(train['Date']), "to day", min(train['Date']), ", a total of", train['Date'].nunique(), "days")
#print("Countries with Province/State informed: ", train.loc[train['Province_State']!='None']['Country_Region'].unique())
test.head()

train.describe()
test.describe()
train.head()

train.info

apple = train["TargetValue"]
np.asarray(apple)
confirmed_cases = []


#Afghanistan
for a in range(105):
    y = apple[2*a]
    confirmed_cases.append(y)
    
    
L = confirmed_cases

def add_one_by_one(L):
    new_L = []
    for elt in L:
        if len(new_L)>0:
            new_L.append(new_L[-1]+elt)
        else:
            new_L.append(elt)
    return new_L

new_L = add_one_by_one(L)
confirmed_cases_Afghanistan = new_L
confirmed_cases_Afghanistan
total_deaths = []
for a in range(105):
    y = apple[2*a+1]
    total_deaths.append(y)
    
L = total_deaths


def add_one_by_one(L):
    new_L = []
    for elt in L:
        if len(new_L)>0:
            new_L.append(new_L[-1]+elt)
        else:
            new_L.append(elt)
    return new_L
new_L = add_one_by_one(L)
total_deaths_Afghanistan = new_L
total_deaths_Afghanistan
total_date_afghanistan = []
for a in range(105):
    total_date_afghanistan.append(a+105)
total_date_afghanistan 

import matplotlib.pyplot as plt
plt.legend(loc='upper left')

#plt.plot(total_date, confirmed_cases_Afghanistan, 'r')
#plt.plot(total_date, total_deaths_Afghanistan)
plt.title('Total deaths: Afghanistan')

ax = plt.subplot(111)
ax.plot(total_date_afghanistan, confirmed_cases_Afghanistan , label='confirmed cases')
ax.plot(total_date_afghanistan, total_deaths_Afghanistan , label='total deaths')
plt.yscale("log")


ax.legend()
plt.show()
apple = train["TargetValue"]
np.asarray(apple)
confirmed_cases = []


#Afghanistan
for a in range(30243,30243+209):
    y = apple[2*a]
    confirmed_cases.append(y)
    
    
L = confirmed_cases

def add_one_by_one(L):
    new_L = []
    for elt in L:
        if len(new_L)>0:
            new_L.append(new_L[-1]+elt)
        else:
            new_L.append(elt)
    return new_L

new_L = add_one_by_one(L)
confirmed_cases_India = new_L
confirmed_cases_India
total_date_india = []
for a in range(209):
    total_date_india.append(a)
total_date_inida = np.asarray(total_date_india)
total_deaths = []
for a in range(30243,30243+209):
    y = apple[2*a+1]
    total_deaths.append(y)
    
L = total_deaths


def add_one_by_one(L):
    new_L = []
    for elt in L:
        if len(new_L)>0:
            new_L.append(new_L[-1]+elt)
        else:
            new_L.append(elt)
    return new_L
new_L = add_one_by_one(L)
total_deaths_India = new_L
total_deaths_India
import matplotlib.pyplot as plt
plt.legend(loc='upper left')

#plt.plot(total_date, confirmed_cases_Afghanistan, 'r')
#plt.plot(total_date, total_deaths_Afghanistan)
plt.title('Total deaths: Inida')

ax = plt.subplot(111)
ax.plot(total_date_india, confirmed_cases_India , label='confirmed cases:inida')
ax.plot(total_date_india, total_deaths_India , label='total deaths:india')
ax.plot(total_date_afghanistan, confirmed_cases_Afghanistan , label='confirmed cases: afghanistan')
ax.plot(total_date_afghanistan, total_deaths_Afghanistan , label='total deaths:afghanistan')
#plt.yscale("log")


ax.legend()
plt.show()
US = pd.read_csv('../input/us-counties-covid-19-dataset/us-counties.csv')
US['cases']
US.info()
confirmed_total_date_Italy = train[train['Country_Region']=='Italy'].groupby(['Date']).agg({'TargetValue':['sum']})

dt = pd.read_excel('../input/us-total/Book 2.xlsx')
dt.info()

dt.head()
dt['Deaths']
total_deathsUS = []
apple = dt['Deaths']
for a in range(64):
    y = apple[64-a]
    total_deathsUS.append(y)
    
total_deathsUS
dt['Positive']
apple = dt['Positive']
np.asarray(apple)
confirmed_casesUS = []


#Afghanistan
for a in range(64):
    y = apple[64-a]
    confirmed_casesUS.append(y)
confirmed_casesUS 

total_dateUS = []
for a in range(64):
    total_dateUS.append(a+209-64)
total_dateUS = np.asarray(total_dateUS)
import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
plt.legend(loc='upper left')

#plt.plot(total_date, confirmed_cases_Afghanistan, 'r')
#plt.plot(total_date, total_deaths_Afghanistan)
plt.title('Total deaths: US')

ax = plt.subplot(111)
ax.plot(total_date_india, confirmed_cases_India , label='confirmed cases:inida')
ax.plot(total_date_india, total_deaths_India , label='total deaths:india')
ax.plot(total_date_afghanistan, confirmed_cases_Afghanistan , label='confirmed cases: afghanistan')
ax.plot(total_date_afghanistan, total_deaths_Afghanistan , label='total deaths:afghanistan')
ax.plot(total_dateUS, confirmed_casesUS , label='confirmed cases: US')
ax.plot(total_dateUS, total_deathsUS , label='total deaths: US')
plt.yscale("log")

#afghanistan
x = np.linspace(0,200,100)
y = 19.20357661*x + 0.001
ax.plot(x,y, label= "lin reg afghan")
#Inida
x = np.linspace(0,200,100)
y = 1.53645039*x + 0.001
ax.plot(x,y, label= "lin reg india")
#US
x = np.linspace(0,200,100)
y = 21597.92248168*x + 0.001
ax.plot(x,y, label= "lin reg US")


ax.legend()
plt.show()
from sklearn import linear_model
a= total_date_afghanistan
n=len(total_date_afghanistan)
X = np.c_[np.ones((n,1)),a]
y = confirmed_cases_Afghanistan
lm = linear_model.LinearRegression()
model = lm.fit(X,y)
lm.coef_
a= total_date_india
n=len(total_date_india)
X = np.c_[np.ones((n,1)),a]
y = confirmed_cases_India
lm = linear_model.LinearRegression()
model = lm.fit(X,y)
lm.coef_
a= total_dateUS
n=len(total_dateUS)
X = np.c_[np.ones((n,1)),a]
y = confirmed_casesUS
lm = linear_model.LinearRegression()
model = lm.fit(X,y)
lm.coef_
import sklearn.linear_model as skl_lm
log_clf = skl_lm.LogisticRegression(solver='newton-cg')
a= total_date_afghanistan
n=len(total_date_afghanistan)
X = np.c_[np.ones((n,1)),a]
y = confirmed_cases_Afghanistan
log_clf.fit(X,y)
print('coefficient: ',log_clf.coef_)
print('intercept: ',log_clf.intercept_)
print('classes: ',log_clf.classes_)
ols = linear_model.LinearRegression()
ols.fit(X, confirmed_cases_Afghanistan)
plt.plot(X, ols.coef_ * X + ols.intercept_, linewidth=1)
plt.axhline(.5, color='.5')
plt.legend(('Logistic Regression Model', 'Linear Regression Model'),
           loc="lower right", fontsize='small')
clf = linear_model.LogisticRegression(C=1e5)
clf.fit(X, y)
# and plot the result
plt.figure(1, figsize=(4, 3))
plt.clf()
plt.plot(X, ols.coef_ * X + ols.intercept_, linewidth=1)
plt.axhline(.5, color='.5')
plt.legend(('Logistic Regression Model', 'Linear Regression Model'),
           loc="lower right", fontsize='small')
from scipy.special import expit
X_test = np.linspace(0, 210, 54)

loss = expit(X_test * clf.coef_ + clf.intercept_).flatten()
plt.plot(X_test, loss, color='red', linewidth=3)
clf.coef_ = clf.coef_.flatten()
clf.coef_
len(clf.coef_)
clf.intercept_ = clf.intercept_.flatten()
clf.intercept_
len(clf.intercept_)
from scipy.special import expit
X_test = np.linspace(0, 210, 54)

#plt.plot(X_test, loss, color='red', linewidth=3)

coef = []

for a in range(len(X_test)):
    coef.append(clf.coef_[a])
coef 
len(coef)
clf.coef_
len(clf.coef_)
from scipy.special import expit
X_test = np.linspace(0, 210, 54)

loss = expit(X_test * coef + clf.intercept_).flatten()
plt.plot(X_test, loss, color='red', linewidth=3)
plt.plot(X, ols.coef_ * X + ols.intercept_, linewidth=1)
plt.axhline(.5, color='.5')
plt.legend(('Logistic Regression Model', 'Linear Regression Model'),
           loc="lower right", fontsize='small')
from scipy.special import expit
X_test = np.linspace(0, 210, 54)

loss = expit(X_test * coef + clf.intercept_).flatten()
plt.plot(X_test, loss, color='red', linewidth=3)
submission = pd.read_csv('../input/covid19-global-forecasting-week-5/submission.csv')
sub = submission = pd.read_csv('../input/covid19-global-forecasting-week-5/submission.csv')
sub.to_csv("submission.csv", index = False)

