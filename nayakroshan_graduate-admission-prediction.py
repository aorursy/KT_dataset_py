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
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict.csv')
data.head()
#Drop the serial column
data = data.drop('Serial No.', axis = 1)
data.head()
#Describe the dataframe.
data.info()
#Dataset has no null values.
bool_data = data.isnull()
for col in list(data.columns):
    print(col + ': ' + str(bool_data[col].sum()))
sns.regplot(x = data['Chance of Admit '], y = data['GRE Score'])
sns.regplot(x = data['Chance of Admit '], y = data['TOEFL Score'])
#SOP vs Chance of Admit plot.
sns.boxplot('SOP', 'Chance of Admit ', data = data)
#LOR vs Chance of Admit.
sns.boxplot('LOR ', 'Chance of Admit ', data = data)
sns.regplot(x = data['Chance of Admit '], y = data['CGPA'])
sns.swarmplot(x = data['Research'], y = data['Chance of Admit '])
corr_data = data.corr()
figure = plt.figure()
plt.figure(figsize = (8, 8))
sns.heatmap(corr_data, cmap = 'YlGnBu', annot = True)
dict_corr = dict(corr_data['Chance of Admit '])
sorted_list = sorted(dict_corr.items(), key = lambda kv:(kv[1], kv[0]))

#Print the features from the most influencing one to the least influencing one.
for feature, value in sorted_list[-2::-1]:
    print(feature + ' --> ' + str(value))
#linear regression, rmse, train_test_split, MinMax Scalar.
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor 
#split the data.
values = data['Chance of Admit ']
train_data = data.drop('Chance of Admit ', axis = 1)

X_train, X_test, Y_train, Y_test = train_test_split(train_data, values, test_size = 0.25)
print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

#Scale the data.
scaler = MinMaxScaler(feature_range = (0, 1))
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

model = LinearRegression()
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

import numpy as np
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(Y_test, Y_pred)))
results_df = pd.DataFrame({'Predicted' : Y_pred.tolist(), 'Actual' : Y_test.tolist()}) #Dataframe haviing predicted and actual values.

sns.scatterplot(x = results_df['Actual'], y = results_df['Predicted'])
corr_df = results_df.corr() #slope of that reg line is about 0.91. It should have been 1. More less than 1 means the error is high.
print(corr_df)
sns.regplot(x = results_df['Actual'], y = results_df['Predicted'])
vlues = data['Chance of Admit ']
train_data = data.drop('Chance of Admit ', axis = 1)

X_train, X_test, Y_train, Y_test = train_test_split(train_data, values, test_size = 0.25)
print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

forest = RandomForestRegressor(n_estimators = 250, random_state = 20) 

forest.fit(X_train, Y_train)

Y_pred=forest.predict(X_test)

print('Root Mean Squared Error:', np.sqrt(mean_squared_error(Y_test, Y_pred)))
results_df = pd.DataFrame({'Predicted' : Y_pred.tolist(), 'Actual' : Y_test.tolist()})
sns.scatterplot(x = results_df['Actual'], y = results_df['Predicted'])
corr_df = results_df.corr()
print(corr_df)
sns.regplot(x = results_df['Actual'], y = results_df['Predicted'])
