# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/car-milage.csv')
print(data.info())
print(data.columns)
data = data.rename(columns=str.lower)
data.dropna(inplace=True)
data['automatic'].value_counts()
sns.countplot(x=data['automatic'],data=data,palette='hls')
plt.show()
count_no = len(data[data['automatic']==0])
count_yes = len(data[data['automatic']==1])

pct_of_no = count_no/(count_no +count_yes)
print("orneklerin 0 olma  olasılığı yuzde", pct_of_no*100)
pct_of_yes= count_yes/(count_no+count_yes)
print("orneklerin 1 olma olasılığı yuzde", pct_of_yes*100)

data.groupby('automatic').mean()
f,ax = plt.subplots()
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
data.groupby('mpg').mean()
pd.crosstab(data.mpg,data.automatic).plot(kind='bar')
plt.title('mpg-fig')
plt.xlabel('')
plt.ylabel('değer oranı')
plt.savefig('mpg-fig')

data.groupby('hp').mean()
pd.crosstab(data.hp,data.automatic).plot(kind='bar')
plt.title('hp-fig')
plt.xlabel('hp')
plt.ylabel('değer oranı')
plt.savefig('hp-fig')
data.groupby('displacement').mean()
pd.crosstab(data.displacement,data.automatic).plot(kind='bar')
plt.title('displacement-fig')
plt.xlabel('displacement')
plt.ylabel('değer oranı')
plt.savefig('displacement-fig')
data.groupby('torque').mean()
pd.crosstab(data.torque,data.automatic).plot(kind='bar')
plt.title('torque_fig')
plt.xlabel('torque')
plt.ylabel('değer oranı')
data.groupby('cratio').mean()
pd.crosstab(data.cratio,data.automatic).plot(kind='bar')
plt.title('cratio_fig')
plt.xlabel('cratio')
plt.ylabel('değer oranı')
data.groupby('raratio').mean()
pd.crosstab(data.raratio,data.automatic).plot(kind='bar')
plt.title('raratio_fig')
plt.xlabel('raratio')
plt.ylabel('değer oranı')
data.groupby('carbbarrells').mean()
pd.crosstab(data.carbbarrells,data.automatic).plot(kind='bar')
plt.title('carbbarrells_fig')
plt.xlabel('carbbarrells')
plt.ylabel('değer oranı')

data.groupby('noofspeed').mean()
pd.crosstab(data.noofspeed,data.automatic).plot(kind='bar')
plt.title('noofspeed_fig')
plt.xlabel('noofspeed')
plt.ylabel('değer oranı')
data.groupby('length').mean()
pd.crosstab(data.length,data.automatic).plot(kind='bar')
plt.title('length_fig')
plt.xlabel('length')
plt.ylabel('değer oranı')
plt.show()
data.groupby('width').mean()
pd.crosstab(data.width,data.automatic).plot(kind='bar')
plt.title('width_fig')
plt.xlabel('width')
plt.ylabel('değer oranı')
data.groupby('weight').mean()
pd.crosstab(data.weight,data.automatic).plot(kind='bar')
plt.title('weight_fig')
plt.xlabel('weight')
plt.ylabel('değer oranı')
data2_df= data.loc[:, data.columns != 'automatic']
automatic_df = data.loc[:, data.columns == 'automatic']



x_train, x_test,y_train,y_test = train_test_split(data2_df,automatic_df,test_size=0.33, random_state=0)


X = x_train.as_matrix().astype(np.float)
data2x_df = pd.DataFrame(data =X,index = range(20),columns =['mpg', 'displacement', 'hp', 'torque', 'cratio', 'raratio',
       'carbbarrells', 'noofspeed', 'length', 'width', 'weight'])

X_test =x_test.as_matrix().astype(np.float)
data2xtest_df = pd.DataFrame(data =X_test,index = range(10),columns =['mpg', 'displacement', 'hp', 'torque', 'cratio', 'raratio',
       'carbbarrells', 'noofspeed', 'length', 'width', 'weight'])

from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(x_train,y_train)

print(regressor.score(x_test,y_test))

import statsmodels.formula.api as sm
X = np.append(arr = np.ones((30,1)).astype(int),values =data2_df,axis =1)
X_l = data2_df.iloc[:,[0,1,2,3,4,5,6,7,8,9,10]].values
r = sm.OLS(endog =automatic_df, exog =X_l).fit()
print(r.summary())
X_l = data2_df.iloc[:,[1,2,3,8,10]].values
r = sm.OLS(endog =automatic_df, exog =X_l).fit()
print(r.summary())
data5 = data.iloc[:,[1,2,3,8,10]]
x_train, x_test,y_train,y_test = train_test_split(data5,automatic_df,test_size=0.33, random_state=0)
regressor.fit(x_train,y_train)
print(regressor.score(x_test,y_test))
X_l = data2_df.iloc[:,[1,6,7,9,10]].values
r = sm.OLS(endog =automatic_df, exog =X_l).fit()
print(r.summary())
data3 = data.iloc[:,[1,6,7,9,10]]
#data3_df = pd.DataFrame(data =data3,index = range(30),columns =['displacement', 'carbbarelles', 'noofspeed', 'width', 'weight'])

x_train, x_test,y_train,y_test = train_test_split(data3,automatic_df,test_size=0.33, random_state=0)
regressor.fit(x_train,y_train)

print(regressor.score(x_test,y_test))  #71
y_prediction = regressor.predict(x_test)


X_l = data2_df.iloc[:,[1,7,9,10]].values
r = sm.OLS(endog =automatic_df, exog =X_l).fit()
print(r.summary())

data4 = data.iloc[:,[1,7,9,10]].values
data4_df = pd.DataFrame(data =data4,index = range(30),columns =['displacement', 'noofspeed', 'width', 'weight'])

print(regressor.score(x_test,y_test))  #71
print('linear regression modeli doğruluk oranı: {:.2f}'.format(regressor.score(x_test, y_test)))



