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

import matplotlib.pyplot as plt
import statsmodels.api as sm
data  = pd.read_csv('/kaggle/input/satandgpa-lr/SATandGPA_LinearRegression.csv')

column_names = pd.Series(data.columns)
print(column_names)
# x_cols = [col for col in data.columns if col not in ['ARR_DEL15']]

# for col in x_cols:
#   corr_coeffs = np.corrcoef(data[col].values, data.ARR_DEL15.values)
  
# # Get the number of missing values in each column / feature variable
# data.isnull().sum()

# Drop a feature variable 
#data = data.drop('feature_name', 1)

# df.index.values
# df.isnull().values.any().any() # ==> was false



# Follow this syntax
# np.where(if_this_condition_is_true, do_this, else_this)
# Example
# df['new_column'] = np.where(df[i] > 10, 'foo', 'bar) 

#nan_rows
# this drop the entire column if any value conatin NAN, i only want to drop rows that contains na
# clean_dataDF = df.dropna(how='any',axis=0) # number of rows --> 565963

#drop only row that contains nan
clean_dataDF = data.dropna() # number of rows --> 565963

# clean_dataDF.info()
# clean_dataDF.describe()
#equatins y^ = b0 + b1.X1
y = clean_dataDF['GPA']
x1 = clean_dataDF['SAT'] 
#explore the data first
plt.scatter(x1,y)
plt.xlabel('SAT Score', fontsize = 20)
plt.ylabel('Student GPA', fontsize = 20)
plt.show()
#regression itself

x = sm.add_constant(x1)
results = sm.OLS(y,x).fit()
results.summary()
#plot it in scatter

plt.scatter(x1,y)
yhat = 0.0017*x1+ 0.275
fig = plt.plot(x1, yhat, lw = 4 , c= 'orange' , label = 'reg line')
plt.xlabel('SAT Score', fontsize = 20)
plt.ylabel('Student GPA', fontsize = 20)
# plt.xlim(0)
# plt.ylim(0)
plt.show()
data = pd.read_csv('/kaggle/input/multiplelineregressionsampledata/1.02. Multiple linear regression.csv')
data.head(10)
y = data['GPA']
x1 = data[['SAT' , 'Rand 1,2,3']]
x = sm.add_constant(x1)
results = sm.OLS(y,x).fit()
results.summary()
raw_data = pd.read_csv('/kaggle/input/103-dummiescsv/1.03. Dummies.csv')
raw_data
data = raw_data.copy()
data['Attendance'] = data['Attendance'].map({'Yes':1 , 'No':0})
data.describe()
y = data['GPA']
x1 = data[['SAT', 'Attendance']]
x = sm.add_constant(x1)
results = sm.OLS(y, x).fit()
results.summary()
#plot it in scatter

plt.scatter(data['SAT'],y)
yhat_NO = 0.6439+0.0014*data['SAT']
yhat_yes = 0.8685 + 0.0014*data['SAT']
fig = plt.plot(data['SAT'], yhat_NO, lw = 2 , c= 'orange' , label = 'reg line')
fig = plt.plot(data['SAT'], yhat_yes, lw = 2 , c= 'red' , label = 'reg line')
plt.xlabel('SAT Score', fontsize = 20)
plt.ylabel('Student GPA', fontsize = 20)
plt.show()
x
new_data = pd.DataFrame({'const' : 1, 'SAT':[1700, 1670], 'Attendance' : [0,1]})
new_data = new_data[['const', 'SAT' , 'Attendance']]
new_data
new_data.rename(index={0 : 'Bob' , 1 : 'Alice'})
predictions = results.predict(new_data)
predictions
predictionsdf = pd.DataFrame({'predictions' : predictions})
joined = new_data.join(predictionsdf)
joined.rename(index = {0 : 'bob' , 1 : 'Alice'})