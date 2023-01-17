import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
from scipy import stats
from scipy.stats import skew,norm
import warnings
def ignore_warn(*args,**kwargs):
    pass
warnings.warn = ignore_warn
import os
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
data = pd.read_csv('/kaggle/input/brasilian-houses-to-rent/houses_to_rent_v2.csv')
data.head(10)
corrmat = data.corr()
#print(corrmat)
f,ax = plt.subplots(figsize=(6,6))
sns.heatmap(corrmat,square=True)
#Checking for missing data
data_missing = (data.isnull().sum()/len(data)) * 100
data_missing = data_missing.drop(data_missing[data_missing==0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Percentage':data_missing})
missing_data
data[['floor']] = data[['floor']].replace(['-'], ['0'])
#replacing - values in floor with 0
cat_col = ['city','animal','furniture']
for col in cat_col:
    sns.set()
    cols = ['city','area','rooms','bathroom','parking spaces','floor','animal','furniture','hoa (R$)','rent amount (R$)','property tax (R$)','fire insurance (R$)','total (R$)']
    plt.figure()
    sns.pairplot(data[cols],size=5.0,hue=col)
    plt.show()
data.head(10)
data.dtypes
data['floor'] = data['floor'].astype(int)
data3 = data[data['city'] == 'Porto Alegre']
data1 = data[data['city'] == 'Rio de Janeiro']
data2 = data[data['city'] == 'Campinas']
data4 = data[data['city'] == 'São Paulo']
data5 = data[data['city'] == 'Belo Horizonte']
data2.head()
y = data.city
ax = sns.countplot(y,label="Count") 

average1 = data1.mean(axis=0)
average2 = data2.mean(axis=0)
average3 = data3.mean(axis=0)
average4 = data4.mean(axis=0)
average5 = data5.mean(axis=0)
average1.plot(figsize=(10,10))
average2.plot(figsize=(10,10))
average3.plot(figsize=(10,10))
average4.plot(figsize=(10,10))
average5.plot(figsize=(10,10))
plt.show()
#Plot of averages city wise
mean1 = average1.to_frame() 
mean1 =mean1.rename(columns={0: 'Rio de Janeiro'})
mean2 = average2.to_frame() 
mean2 = mean2.rename(columns={0: 'Campinas'})
mean3 = average3.to_frame()
mean3 = mean3.rename(columns={0: 'Porto Alegre'}) 
mean4 = average4.to_frame() 
mean4 = mean4.rename(columns={0: 'Sao Paulo'}) 
mean5 = average5.to_frame()
mean5 = mean5.rename(columns={0: 'Belo Horizonte'})
result = pd.concat([mean1, mean2,mean3,mean4,mean5], axis=1, sort=False)
result
#averages table
ax = result.plot.bar(rot=0,logy=True,figsize=(15,7))
#scaled y axis for better comparison
#comparison of averages through a bar graph
mode1 = data1.mode(axis=0).T
mode1 = mode1.rename(columns={0: 'Rio de Janeiro'})
mode2 = data2.mode(axis=0).T
mode2 = mode2.rename(columns={0: 'Campinas'})
mode3 = data3.mode(axis=0).T
mode3 = mode3.rename(columns={0: 'Porto Alegre'})
mode4 = data4.mode(axis=0).T
mode4 = mode4.rename(columns={0: 'Sao Paulo'})
mode5 = data5.mode(axis=0).T
mode5 = mode5.rename(columns={0: 'Belo Horizonte'})

ax = result.plot.bar(rot=0,logy=True,figsize=(15,7))
#scaled y axis for better comparison
#comparison of modes
ax = result.plot(rot=0,figsize=(15,15))
#comparison of mode values
# label encoding the data 
from sklearn.preprocessing import LabelEncoder 
  
le = LabelEncoder() 
  
data['city']= le.fit_transform(data['city']) 
data['furniture']= le.fit_transform(data['furniture']) 
data['animal']= le.fit_transform(data['animal']) 
data.head()
y = data["rent amount (R$)"]
X = data[['city','area','rooms','bathroom','parking spaces','floor','animal','fire insurance (R$)','furniture','hoa (R$)','property tax (R$)','total (R$)']]
data_dia = y.copy()
data = X[['city']]
data_n_2 = (data - data.mean()) / (data.std())              # standardization
data = pd.concat([y,data_n_2.iloc[:,0:1000]],axis=1)
data = pd.melt(data,id_vars="city",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
sns.violinplot(x="features", y="value", hue="city", data=data, inner="quart")
plt.xticks(rotation=90)
X_train = X[0:10000]
y_train = y[0:10000]
X_test = X[10000:]
y_test = y[10000:]
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
regressor = LinearRegression()  
regressor.fit(X_train, y_train) #training the algorithm
y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df
df1 = df.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
y_test.mean()
print('Coefficients: \n', regressor.coef_)
import statsmodels.api as sm # import statsmodels 

X = X_train 
y = y_train 
X = sm.add_constant(X) 
model = sm.OLS(y, X).fit() 
predictions = model.predict(X)

# Print out the statistics
model.summary()