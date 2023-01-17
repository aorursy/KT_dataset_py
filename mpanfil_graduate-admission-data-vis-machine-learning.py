# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, ru"nning this (by clicking run or pressing Shift+Enter) will list all files under the input directory



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

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn import metrics



sns.set_style('darkgrid')
data = pd.read_csv('../input/graduate-admissions/Admission_Predict.csv')
data.head()



# Serial No is the same what is index

data.drop('Serial No.', axis=1, inplace=True)
data.describe()
data.info()
# Seeing correlations



plt.figure(figsize=(16,8))

plt.title('Correlations between features', fontsize=15)

data_corr = data.corr()

mask = np.zeros_like(data_corr)

mask[np.triu_indices_from(mask)] = True



with sns.axes_style('white'):

    sns.heatmap(data=data_corr, mask=mask, vmax=1, square=True, annot=True, cmap='Reds', linewidths=0.2)
data.isnull().sum()
data.head()
# Visualisation 1 - Research



plt.figure(figsize=(12,5))

plt.title('Research among students', fontsize=15)

sns.countplot(x=data['Research'], palette='Reds')
with_research = data[data['Research']==1].sum()['Research']

without_research = len(data['Research']) - with_research

percent_with_research = with_research/(with_research+without_research)*100



print('Number of students who did research: %s' %with_research)

print('Number of students who did not do research: %s' %without_research)

print('Percentage of student who did research: %s' %percent_with_research), 

print('%')
# Visualisation 2 - Exams scores, uni ratings



plt.figure(figsize=(15,12))

plt.subplot(2,2,1)

plt.title('GRE Score', fontsize=15)

sns.distplot(data['GRE Score'], color='red')



plt.subplot(2,2,2)

plt.title('TOEFL Score', fontsize=15)

sns.distplot(data['TOEFL Score'], color='red')



plt.subplot(2,2,3)

plt.title('CGPA', fontsize=15)

sns.distplot(data['CGPA'], color='red')



plt.subplot(2,2,4)

plt.title('University rating', fontsize=15)

sns.countplot(x=data['University Rating'], color='red')

plt.ylabel('Count of students')
print('Number of students for each university ranking: \n')

print('Rating 1: %s' %(data['University Rating'] == 1).sum())

print('Rating 2: %s' %(data['University Rating'] == 2).sum())

print('Rating 3: %s' %(data['University Rating'] == 3).sum())

print('Rating 4: %s' %(data['University Rating'] == 4).sum())

print('Rating 5: %s' %(data['University Rating'] == 5).sum())
# Visualisation 3 - Chnce of admit vs CGPA, TOEFL and GRE



sns.pairplot(data=data, x_vars=['GRE Score','TOEFL Score','CGPA'], y_vars='Chance of Admit ', hue='University Rating',size=5)
# Visualisation 4 - LOR, SOP vs Chance of admit



sns.pairplot(data=data, x_vars=['LOR ', 'SOP'], y_vars='Chance of Admit ', size=5)
# Visualisation 5 - LOR, SOR



plt.figure(figsize=(8,7))

plt.title('GRE Score vs TOEFL Score', fontsize=15)

sns.regplot(x=data['TOEFL Score'], y=data['GRE Score'], color='red')
# Linear Regression model - predicting 
X = data.drop('Chance of Admit ', axis=1)

y = data['Chance of Admit ']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
linmodel = LinearRegression()
linmodel.fit(X_train, y_train)
coef = linmodel.coef_

df_coef = pd.DataFrame(data.columns[:-1])

df_coef.columns = ['Feature']

df_coef['Coefficients'] = pd.Series(linmodel.coef_)

df_coef.sort_values(by='Coefficients', ascending=False)
print('Linear regression intercept: %s' %linmodel.intercept_)
predictions = linmodel.predict(X_test)
# Visualisation 1 - seeing evaluated predictions with y_test values



plt.figure(figsize=(8,7))

plt.title('Evaluated predictions', fontsize=15)

plt.xlabel('Predictions')

sns.regplot(y=y_test, x=predictions, color='red')
# Visualisation 2 - seeing distribution of errors: y_test-predictions



plt.figure(figsize=(8,7))

plt.title('Error rate', fontsize=15)

sns.distplot(y_test-predictions, color='red')
# Evaluated metrics



mae = metrics.mean_absolute_error(y_test, predictions)

mse = metrics.mean_squared_error(y_test, predictions)

rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions))



print('MAE (Mean Absolute Error): %s' %mae)

print('MSE (Mean Squared Error): %s' %mse)

print('RMSE (Root mean squared error): %s' %rmse)