import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("/kaggle/input/advertising/advertising.csv")
df.head()
df.shape
df.describe()
df.info()
df.duplicated().sum()

df.isnull().sum()
pd.crosstab(df['Male'],df['Clicked on Ad']).sort_values(1,ascending=False)
##data is balanced and equally distributed between gender
no_click = df[df['Clicked on Ad'] == 0]
click = df[df['Clicked on Ad'] == 1]

click['Age'].hist(bins=10,label = 'click', alpha=0.5)
no_click['Age'].hist(bins=10,label = 'no click', alpha=0.5)
plt.legend(loc = 'age_click')
plt.show()

click['Area Income'].hist(bins=10,label = 'click', alpha=0.5)
no_click['Area Income'].hist(bins=10,label = 'no click', alpha=0.5)
plt.legend(loc = 'income_click')
plt.show()

click['Daily Time Spent on Site'].hist(bins=10,label = 'click', alpha=0.5)
no_click['Daily Time Spent on Site'].hist(bins=10,label = 'Clicked on Ad', alpha=0.5)
plt.legend(loc = 'time_click')
plt.show()

click['Daily Internet Usage'].hist(bins=10,label = 'click', alpha=0.5)
no_click['Daily Internet Usage'].hist(bins=10,label = 'no click', alpha=0.5)
plt.legend(loc = 'fulltime_click')
plt.show()
import datetime

df['Date'] = pd.to_datetime(df['Timestamp'], errors='coerce')

df['Hour']=df['Date'].dt.hour
df['Month']=df['Date'].dt.month
df['Weekdays']= df['Date'].dt.weekday
pd.crosstab(df['Month'],df['Clicked on Ad'])
#no season
sns.countplot('Month',hue='Clicked on Ad',data= df)
sns.countplot('Weekdays',hue='Clicked on Ad',data= df)

sns.countplot('Hour',hue='Clicked on Ad',data= df)
df.corr()
df['Age_bins'] = pd.cut(df['Age'], bins=[0, 29, 35, 42, 70], labels=['Young','Adult','Mid', 'Elder'])
df['Salary_bins'] = pd.cut(df['Area Income'], bins=[0, 30000.00, 55000.00, 65000.00, 85000.00], labels=['Low Income','Below Average','Above Average', 'Wealth'])
df['Daily_bins'] = pd.cut(df['Daily Internet Usage'], bins=[0, 139, 183, 218, 300], labels=['Short Time','Below Average','Above Average', 'Addict'])
df['Website_bins'] = pd.cut(df['Daily Time Spent on Site'], bins=[0, 51, 68, 78, 100], labels=['Short time','Below Average','Above Average', 'Addict'])
a = df.groupby(['Age_bins', 'Salary_bins', 'Male'])['Clicked on Ad'].sum().unstack('Salary_bins')
a.fillna(0)
df.groupby(['Age_bins', 'Website_bins', 'Male'])['Clicked on Ad'].sum().unstack('Website_bins')
print('The number of towns is equal to {}.'.format(df['City'].nunique()))
print('The number of coutnries is equal to {}.'.format(df['Country'].nunique()))
X = df.drop(['Date','Timestamp','Clicked on Ad', 'Ad Topic Line', 'Age_bins','City', 'Country', 'Salary_bins', 'Daily_bins', 'Website_bins'], axis=1)
y = df['Clicked on Ad']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
from  sklearn.preprocessing  import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)
import  statsmodels.api  as sm
from scipy import stats

X2   = sm.add_constant(X_train)
model  = sm.Logit(y_train, X2)
model2 = model.fit()
print(model2.summary(xname=['Const','Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male', 'Hour', 'Month', 'Weekdays']))
X.drop(['Male','Hour', 'Month', 'Weekdays'], axis= 1, inplace = True)
from sklearn.linear_model import LogisticRegression                                                                  
lr = LogisticRegression()                
lr.fit(X_train, y_train)                                                                        
y_pred = lr.predict(X_test)   
from sklearn import metrics
print (metrics.accuracy_score(y_test, y_pred))
print (metrics.confusion_matrix(y_test, y_pred))
print (metrics.classification_report(y_test, y_pred))