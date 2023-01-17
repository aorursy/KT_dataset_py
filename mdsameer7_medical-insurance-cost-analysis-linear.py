# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
%matplotlib inline
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
import statsmodels as sm

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

df=pd.read_csv("/kaggle/input/insurance/insurance.csv")
df_org=df
df.head(10)
df.describe(include='all')
sns.distplot(df['charges'])
#Taking the 99th percentile value and filtering the dataset to show entries above the 99th percentile
q=df['charges'].quantile(0.99)
#print(q)
df_out=df[(df['charges']>q)]
df_out.describe(include='all')
#Removing outlier data(99 percentile)
df_nonout=df[df['charges']<q]
sns.distplot(df_nonout['charges'])
df_nonout.reset_index(drop='True')
df_nonout.describe(include='all')
#df_nonout.info()
a=df_nonout._get_numeric_data()
print(a.columns.values)
l=df_nonout.select_dtypes(include=['object'])
print(l.columns.values)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
#sex
le = LabelEncoder()
le.fit(df_nonout.sex.drop_duplicates()) 
df_nonout.sex = le.transform(df_nonout.sex)
# smoker or not
le.fit(df_nonout.smoker.drop_duplicates()) 
df_nonout.smoker = le.transform(df_nonout.smoker)
#region
le.fit(df_nonout.region.drop_duplicates()) 
df_nonout.region = le.transform(df_nonout.region)
print(df_nonout.head(10))
from sklearn.preprocessing import LabelEncoder

#sex
le = LabelEncoder()
le.fit(df.sex.drop_duplicates()) 
df.sex = le.transform(df.sex)
# smoker or not
le.fit(df.smoker.drop_duplicates()) 
df.smoker = le.transform(df.smoker)
#region
le.fit(df.region.drop_duplicates()) 
df.region = le.transform(df.region)
print(df.head(10))
f,ax=plt.subplots(2,3, sharey=False,figsize=(15,8))
ax[0][0].scatter(df_nonout['age'],df_nonout['charges'])
ax[0][0].set_title('age')
ax[0][1].scatter(df_nonout['bmi'],df_nonout['charges'])
ax[0][1].set_title('bmi')
ax[0][2].scatter(df_nonout['smoker'],df_nonout['charges'])
ax[0][2].set_title('smoker')
ax[1][0].scatter(df_nonout['region'],df_nonout['charges'])
ax[1][0].set_title('region')
ax[1][1].scatter(df_nonout['sex'],df_nonout['charges'])
ax[1][1].set_title('sex')
ax[1][2].scatter(df_nonout['children'],df_nonout['charges'])
ax[1][2].set_title('children')
plt.show()
df_nonout.corr()['charges'].sort_values()

df_nonout.loc[(df_nonout['bmi'] >=18.5) & (df_nonout['bmi'] <=25), 'health'] = 'healthy'  
df_nonout.loc[(df_nonout['bmi'] <18.5) | (df_nonout['bmi'] >25), 'health'] = 'unhealthy'
df_nonout.describe(include='all')
f,ax=plt.subplots(1,2, sharey=False,figsize=(15,8))
sns.boxplot(y="health", x="charges", data =  df_nonout , orient="h", palette = 'magma',ax=ax[0])
ax[0].set_title('health')
sns.boxplot(y="children", x="charges", data =  df_nonout , orient="h", palette = 'magma',ax=ax[1])
ax[1].set_title('children')
# Test for difference in variability for sex

from scipy import stats
df_anova = df_nonout[['charges','sex']]
grps = pd.unique(df_anova.sex.values)
d_data = {grp:df_anova['charges'][df_anova.sex == grp] for grp in grps}
#print(d_data)
F, p = stats.f_oneway(d_data[0], d_data[1])
print("p-value for significance is: ", p)
if p<0.05:
    print("reject null hypothesis")
else:
    print("accept null hypothesis")
# Test for difference in variability for bmi

from scipy import stats
df_anova = df_nonout[['charges','health']]

grps = pd.unique(df_anova.health.values)
d_data = {grp:df_anova['charges'][df_anova.health == grp] for grp in grps}

F, p = stats.f_oneway(d_data['healthy'], d_data['unhealthy'])#, d_data[2] )#, d_data[3], d_data[4], d_data[5])
print("p-value for significance is: ", p)
if p<0.05:
    print("reject null hypothesis")
else:
    print("accept null hypothesis")
# Test for difference in variability for children

from scipy import stats
df_anova = df_nonout[['charges','children']]
grps = pd.unique(df_anova.children.values)
d_data = {grp:df_anova['charges'][df_anova.children == grp] for grp in grps}
#print(d_data)
F, p = stats.f_oneway(d_data[0], d_data[1], d_data[2] , d_data[3], d_data[4], d_data[5])
print("p-value for significance is: ", p)
if p<0.05:
    print("reject null hypothesis")
else:
    print("accept null hypothesis")
# Test for difference in variability for children

from scipy import stats
df_anova = df_nonout[['charges','region']]
grps = pd.unique(df_anova.region.values)
d_data = {grp:df_anova['charges'][df_anova.region == grp] for grp in grps}
#print(d_data)
F, p = stats.f_oneway(d_data[0], d_data[1], d_data[2] , d_data[3])
print("p-value for significance is: ", p)
if p<0.05:
    print("reject null hypothesis")
else:
    print("accept null hypothesis")
#lets encode health column too for future reference.
le.fit(df_nonout.health.drop_duplicates()) 
df_nonout.health = le.transform(df_nonout.health)

df_xstand=df_nonout.drop(['charges'],axis=1)
x_cols=df_xstand.columns.values

scaler=StandardScaler()
scaler.fit(df_xstand)
x_scaled=scaler.transform(df_xstand)
df_xstand = pd.DataFrame(data=x_scaled, columns=x_cols)
print(df_xstand.head(10))
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

#we declare a variable where we put all features where we want to check for multicollinearity
#since health column is a derived column from bmi lets drop bmi
variables =df_nonout.drop(['charges'],axis=1)
# we create a new data frame which will include all the VIFs
# note that each variable has its own variance inflation factor as this measure is variable specific (not model specific)
vif = pd.DataFrame()

# here we make use of the variance_inflation_factor, which will basically output the respective VIFs 
vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
# Finally to include names so it is easier to explore the result
vif["Features"] = variables.columns
print(vif)
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

#we declare a variable where we put all features where we want to check for multicollinearity
#since health column is a derived column from bmi lets drop bmi
variables =df_nonout.drop(['charges','sex','region','bmi'],axis=1)
# we create a new data frame which will include all the VIFs
# note that each variable has its own variance inflation factor as this measure is variable specific (not model specific)
vif = pd.DataFrame()

# here we make use of the variance_inflation_factor, which will basically output the respective VIFs 
vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
# Finally to include names so it is easier to explore the result
vif["Features"] = variables.columns
print(vif)
#fitting regression on original data without outlier treatment.

df.loc[(df['bmi'] >=18.5) & (df['bmi'] <=25), 'health'] = 'healthy'  
df.loc[(df['bmi'] <18.5) | (df['bmi'] >25), 'health'] = 'unhealthy'
df.describe(include='all')
le.fit(df.health.drop_duplicates()) 
df.health = le.transform(df.health)

from statsmodels.regression.linear_model import OLS

y_org=df['charges']
x_org=df.drop(['charges'],axis=1)
x_org=add_constant(x_org)
result=OLS(y_org,x_org).fit()
result.summary()
y_org1=df['charges']
x_org1=df.drop(['charges','sex','region','bmi'],axis=1)
x_org1=add_constant(x_org1)
result=OLS(y_org1,x_org1).fit()
result.summary()
y_org1=df['charges']
x_org1=df[['age','smoker','bmi','children']]
x_org1=add_constant(x_org1)
result=OLS(y_org1,x_org1).fit()
result.summary()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# Split the variables with an 80-20 split and some random state
# To have the same split always random_state = 365
x_train, x_test, y_train, y_test = train_test_split(x_org, y_org, test_size=0.2, random_state=365)
x_train.reset_index(drop=True)
y_train.reset_index(drop=True)
x_test.reset_index(drop=True)
y_test.reset_index(drop=True)
reg = LinearRegression()
# Fit the regression with the scaled TRAIN inputs and targets
reg.fit(x_train,y_train)
y_hat_train = reg.predict(x_train)
print("------Train R2 value-----")
print(reg.score(x_train,y_train))
print("------Test R2 value-----")
print(reg.score(x_test,y_test))
print("-----Mean error of Train---")
q=abs(y_train-y_hat_train)
sns.distplot(y_train-y_hat_train)
print(q.mean())
print("-----Mean % error of train---")
q=((abs(y_train-y_hat_train))/(y_train))*100
print(q.mean())
y_hat_test = reg.predict(x_test)
print("-----Mean error of Test---")
q=abs(y_test-y_hat_test)
print(q.mean())
print("-----Mean % error of Test---")
q=((abs(y_test-y_hat_test))/(y_test))*100
print(q.mean())

# Sometimes the plot will have different scales of the x-axis and the y-axis
# This is an issue as we won't be able to interpret the '45-degree line'
# We want the x-axis and the y-axis to be the same
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# Split the variables with an 80-20 split and some random state
# To have the same split always random_state = 365
x_train, x_test, y_train, y_test = train_test_split(x_org1, y_org1, test_size=0.2, random_state=365)
x_train.reset_index(drop=True)
y_train.reset_index(drop=True)
x_test.reset_index(drop=True)
y_test.reset_index(drop=True)

reg = LinearRegression()
# Fit the regression with the scaled TRAIN inputs and targets
reg.fit(x_train,y_train)
y_hat_train = reg.predict(x_train)
print("------Train R2 value-----")
print(reg.score(x_train,y_train))
print("------Test R2 value-----")
print(reg.score(x_test,y_test))
print("-----Mean error of Train---")
q=abs(y_train-y_hat_train)
sns.distplot(y_train-y_hat_train)
print(q.mean())
print("-----Mean % error of train---")
q=((abs(y_train-y_hat_train))/(y_train))*100
print(q.mean())
y_hat_test = reg.predict(x_test)
print("-----Mean error of Test---")
q=abs(y_test-y_hat_test)
print(q.mean())
print("-----Mean % error of Test---")
q=((abs(y_test-y_hat_test))/(y_test))*100
print(q.mean())
# Sometimes the plot will have different scales of the x-axis and the y-axis
# This is an issue as we won't be able to interpret the '45-degree line'
# We want the x-axis and the y-axis to be the same
from statsmodels.regression.linear_model import OLS

y=df_nonout['charges']
y=y.values.reshape(-1,1)

x=df_xstand

x=add_constant(x)

result=OLS(y,x).fit()
result.summary()
from statsmodels.regression.linear_model import OLS

y1=df_nonout['charges']
y1=y1.values.reshape(-1,1)

x1=df_xstand.drop(['sex','region'],axis=1)

x1=add_constant(x1)

result=OLS(y1,x1).fit()
result.summary()
from statsmodels.regression.linear_model import OLS

y1=df_nonout['charges']
y1=y1.values.reshape(-1,1)

x1=df_xstand[['age','smoker','health','children']]

x1=add_constant(x1)

result=OLS(y1,x1).fit()
result.summary()
from statsmodels.regression.linear_model import OLS

y1=df_nonout['charges']
y1=y1.values.reshape(-1,1)

x1=df_xstand[['age','smoker','bmi','children']]

x1=add_constant(x1)

result=OLS(y1,x1).fit()
result.summary()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# Split the variables with an 80-20 split and some random state
# To have the same split always random_state = 365
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=365)
x_train.reset_index(drop=True)
#y_train.reset_index(drop=True)
x_test.reset_index(drop=True)
#y_test.reset_index(drop=True)

reg = LinearRegression()
# Fit the regression with the scaled TRAIN inputs and targets
reg.fit(x_train,y_train)
y_hat_train = reg.predict(x_train)
print("------Train R2 value-----")
print(reg.score(x_train,y_train))
print("------Test R2 value-----")
print(reg.score(x_test,y_test))
print("-----Mean error of Train---")
q=abs(y_train-y_hat_train)
sns.distplot(y_train-y_hat_train)
print(q.mean())
print("-----Mean % error of train---")
q=((abs(y_train-y_hat_train))/(y_train))*100
print(q.mean())
y_hat_test = reg.predict(x_test)
print("-----Mean error of Test---")
q=abs(y_test-y_hat_test)
print(q.mean())
print("-----Mean % error of Test---")
q=((abs(y_test-y_hat_test))/(y_test))*100
print(q.mean())
# Sometimes the plot will have different scales of the x-axis and the y-axis
# This is an issue as we won't be able to interpret the '45-degree line'
# We want the x-axis and the y-axis to be the same
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# Split the variables with an 80-20 split and some random state
# To have the same split always random_state = 365
x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size=0.2, random_state=365)
x_train.reset_index(drop=True)
#y_train.reset_index(drop=True)
x_test.reset_index(drop=True)
#y_test.reset_index(drop=True)

reg = LinearRegression()
# Fit the regression with the scaled TRAIN inputs and targets
reg.fit(x_train,y_train)
y_hat_train = reg.predict(x_train)
print("------Train R2 value-----")
print(reg.score(x_train,y_train))
print("------Test R2 value-----")
print(reg.score(x_test,y_test))
print("-----Mean error of Train---")
q=abs(y_train-y_hat_train)
sns.distplot(y_train-y_hat_train)
print(q.mean())
print("-----Mean % error of train---")
q=((abs(y_train-y_hat_train))/(y_train))*100
print(q.mean())
y_hat_test = reg.predict(x_test)
print("-----Mean error of Test---")
q=abs(y_test-y_hat_test)
print(q.mean())
print("-----Mean % error of Test---")
q=((abs(y_test-y_hat_test))/(y_test))*100
print(q.mean())
# Sometimes the plot will have different scales of the x-axis and the y-axis
# This is an issue as we won't be able to interpret the '45-degree line'
# We want the x-axis and the y-axis to be the same