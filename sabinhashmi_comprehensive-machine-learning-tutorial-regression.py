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
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')
data=pd.read_excel('/kaggle/input/Admission.xlsx')
data.head()
data.info()
print('Categorical Columns are ','\n',list(data.select_dtypes(include='object')),'\n')

print('Numerical Columns are ','\n',list(data.select_dtypes(exclude='object')))
data.isnull().mean()*100
data.nunique()
data.describe(include='object').T
data.describe().T
data.isnull().mean()*100
data.plot(kind='box',layout=(3,4),subplots=1,figsize=(10,8))

plt.show()
sns.distplot(data['Salary'])

print('Skewness of Y- Variable is ',data['Salary'].skew())
data['Entrance_Test'].value_counts()
data['Entrance_Test'].isnull().sum()
data['Entrance_Test']=data['Entrance_Test'].fillna('No-Exam')
plt.figure(figsize=(10,6))

sns.heatmap(data.corr(),cmap='magma',annot=True)

plt.show()
data.corr()['Salary'] # To get exact values of correlation by X-Variables on Y-Variable.
for i in data.columns:

    sns.scatterplot(data[i],data['Salary'],color='black')

    plt.show()
from sklearn.model_selection import train_test_split #Importing the library
x=data.drop('Salary',axis=1)

y=data['Salary']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=43)
print(x_train.shape)

print(x_test.shape)

print(y_train.shape)

print(y_test.shape)
data[data['Placement']!='Placed']['Salary'].value_counts() # Verifying is there any mismatch between Placement and Salary.
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()
data.select_dtypes(include='object').head()
from scipy.stats import zscore #or we can use Standard Scalar, since the degrees of freedom is zero in this case we can use either of them.
x=pd.concat([x.select_dtypes(include='object').apply(le.fit_transform),(x.select_dtypes(exclude='object').apply(zscore))],axis=1)

x.head()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=43) #Splitting the data
from sklearn.linear_model import LinearRegression # Importing the package for Linear Regression

lr=LinearRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
print('R2-Score ',r2_score(y_test,y_pred))

print('RMSE  ',np.sqrt(mean_squared_error(y_test,y_pred)))
x_test['Predicted Salary']=y_pred # Adding the predicted value to the dataframe (Test Data Set)
x_test.head()
from sklearn.feature_selection import RFE
#no of features

nof_list=np.arange(1,19)            

high_score=0

#Variable to store the optimum features

nof=0          

score_list =[]

for n in range(len(nof_list)):

    X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state = 43)

    model = LinearRegression()

    rfe = RFE(model,nof_list[n])

    X_train_rfe = rfe.fit_transform(X_train,y_train)

    X_test_rfe = rfe.transform(X_test)

    model.fit(X_train_rfe,y_train)

    score = model.score(X_test_rfe,y_test)

    score_list.append(score)

    if(score>high_score):

        high_score = score

        nof = nof_list[n]

print("Optimum number of features: %d" %nof)

print("Score with %d features: %f" % (nof, high_score))

rfe = RFE(model,10)

rfe.fit_transform(X_train,y_train)

y_pred = rfe.predict(X_test)



r2_score(y_test,y_pred)

pd.DataFrame(rfe.support_,index=x.columns)
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()

vif["VIF Values"]=[variance_inflation_factor(x.values, col) 

                   for col in range(0, x.shape[1])]

vif.index=x.columns

vif.T
from statsmodels.stats.api import het_goldfeldquandt

het_goldfeldquandt(y_train, x_train)
import statsmodels.api as sm
xc=sm.add_constant(x_train)
ols=sm.OLS(y_train,xc).fit()
ols.summary()
from statsmodels.stats.stattools import durbin_watson

durbin_watson(ols.resid)
import statsmodels.tsa.api as smt



acf = smt.graphics.plot_acf(ols.resid, lags=40 , alpha=0.05)

acf.show()
from statsmodels.stats.diagnostic import linear_rainbow
linear_rainbow(ols,frac=0.5)
x=data.drop('Salary',axis=1)

y=data['Salary']

x=pd.concat([x.select_dtypes(include='object').apply(le.fit_transform),(x.select_dtypes(exclude='object').apply(zscore))],axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=43) #Re-splitting the data since 

                                        #there was an addition of prediction column to the table for the previous question.
from sklearn.tree import DecisionTreeRegressor

dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
r2_score(y_test,y_pred)
from sklearn.ensemble import RandomForestRegressor

rf=RandomForestRegressor()
rf.fit(x_train,y_train)
y_pred=rf.predict(x_test)
r2_score(y_test,y_pred)
from sklearn.model_selection import RandomizedSearchCV
rf=RandomForestRegressor()
parameters={'n_estimators':range(1,50),'min_samples_split':range(2,10),'min_samples_leaf':range(2,10)}
rand=RandomizedSearchCV(estimator=rf,param_distributions=parameters,scoring='r2',cv=5,return_train_score=True)
rand.fit(x_train,y_train)
y_pred=rand.predict(x_test)
r2_score(y_test,y_pred)
pd.DataFrame(rand.cv_results_).sort_values(by='param_n_estimators').set_index('param_n_estimators')['mean_test_score'].plot.line()

pd.DataFrame(rand.cv_results_).sort_values(by='param_n_estimators').set_index('param_n_estimators')['mean_train_score'].plot.line()
rf.fit(x_train,y_train)

pd.DataFrame(rf.feature_importances_,index=x.columns).sort_values(by=0,ascending=False)
pd.DataFrame(rf.feature_importances_,index=x.columns).sort_values(by=0,ascending=False).head()
pd.DataFrame(rf.feature_importances_,index=x.columns).sort_values(by=0,ascending=False).head().plot.bar()