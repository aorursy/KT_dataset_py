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
import matplotlib.pyplot as plt

import seaborn as sns #Visualization

plt.rcParams['figure.figsize'] = [8,5]

plt.rcParams['font.size'] =14

plt.rcParams['font.weight']= 'bold'

plt.style.use('seaborn-whitegrid')


df = pd.read_csv('/kaggle/input/advertising-data/Advertising.csv')

print('\nNumber of rows and columns in the data set: ',df.shape)

print('')



#Lets look into top few rows and columns in the dataset

print(df.head())
df =df.round(2)

df.head()



#Drop unnecessary columns

drop_elements = ['Unnamed: 0']

df = df.drop(drop_elements, axis=1)



df.tail()

df.describe().round(2)



df.describe()
#null value check



df.isnull().sum()*100/df.shape[0]



ax = sns.boxplot(data=df,orient='v',palette="Set1")
sns.pairplot(df, x_vars=['TV','Radio','Newspaper'],y_vars='Sales',height=4,aspect=1 )

plt.show()
# correlation plot

corr = df.corr()

sns.heatmap(corr, cmap = 'Wistia', annot= True);
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn import metrics



import statsmodels.api as sm



x=df['TV']

y=df['Sales']



x_train,x_test,y_train, y_test=train_test_split(x,y,train_size=0.7, test_size=0.3, random_state=100)



x_train.head()

y_train.head()

x_train_sm =sm.add_constant(x_train)



lr=sm.OLS(y_train, x_train_sm).fit()



lr.params
print(lr.summary())
plt.scatter(x_train, y_train)

plt.plot(x_train, 6.9897+0.0465*x_train,'r')

plt.show()
y_train_pred=lr.predict(x_train_sm)



res =(y_train-y_train_pred)



fig = plt.figure()



sns.distplot(res,bins=15)



fig.suptitle('Error Terms')



plt.xlabel('y_train - y_train_pred')



plt.show()

plt.scatter(x_train,res)

plt.show()
x_test_sm=sm.add_constant(x_test)

y_pred = lr.predict(x_test_sm)



y_pred.head
from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score



np.sqrt(mean_squared_error(y_test,y_pred))
r_squared = r2_score(y_test,y_pred)

r_squared
plt.scatter(x_test,y_test)

plt.plot(x_test, 6.9897+0.0465*x_test,'r')

plt.show()