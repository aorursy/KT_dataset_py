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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sb
dataset = pd.read_csv('../input/insurance/insurance.csv')

dataset.head()
dataset.shape
dataset.describe()
plt.figure(figsize=(20,8))



plt.subplot(1,2,1)

plt.title('Insurance Charges Distribution Plot')

sb.distplot(dataset.charges,color ='red')



plt.subplot(1,2,2)

plt.title('Insurance Charges Spread')

ax = sb.boxplot(y = dataset.charges ,color = 'red' )

#ax = sb.swarmplot(y = dataset.charges,color = 'black')



plt.show()
X = dataset.iloc[: ,:-1].values



y = dataset.iloc[:, 6].values
dataset["age_range"] = dataset['age'].apply(lambda x : "young" if x < 30

                                                     else ("middle adulthood" if 30 <= x < 50

                                                           else ("late adulthood")))

dataset.head()
df = pd.DataFrame(dataset.groupby(['age_range'])['charges'].mean().sort_values(ascending = False))

df.plot.bar(color = 'cyan')

plt.title('age_range vs Average charges')

plt.show()
df = pd.DataFrame(dataset.groupby(['sex'])['charges'].mean().sort_values(ascending = False))

df.plot.bar(color = 'red')

plt.title('sex vs Average charges')



plt.show()
dataset["bmi_range"] = dataset['bmi'].apply(lambda x : "thin" if x < 19

                                                     else ("fit" if  19 <= x < 25

                                                           else ("overweight" if  25 <= x < 28

                                                                else ("Obese"))))

                                            

dataset.head()
df = pd.DataFrame(dataset.groupby(['bmi_range'])['charges'].mean().sort_values(ascending = False))

df.plot.bar(color = 'cyan')

plt.title('bmi_range vs Average charges')

plt.show()
df = pd.DataFrame(dataset.groupby(['children'])['charges'].mean().sort_values(ascending = False))

df.plot.bar(color = 'red')

plt.title('children vs Average charges')

plt.show()
df = pd.DataFrame(dataset.groupby(['smoker'])['charges'].mean().sort_values(ascending = False))

df.plot.bar(color = 'cyan')

plt.title('smoker vs Average charges')

plt.show()
df = pd.DataFrame(dataset.groupby(['region'])['charges'].mean().sort_values(ascending = False))

df.plot.bar(color = 'red')

plt.title('region vs Average charges')

plt.show()
def attributes(x,y):

    sb.pairplot(dataset, x_vars=[x,y], y_vars='charges',size=4, aspect=1, kind='scatter')

    plt.show()

    

    

attributes('age','bmi')

plt1 = sb.scatterplot(x = 'age', y = 'charges', hue = 'age_range', data = dataset)

plt1.set_xlabel('age')

plt1.set_ylabel('Charges')

plt.show()
plt1 = sb.scatterplot(x = 'bmi', y = 'charges', hue = 'bmi_range', data = dataset)

plt1.set_xlabel('bmi')

plt1.set_ylabel('Charges')

plt.show()
attributes = dataset[['age_range','sex','bmi_range','children','smoker','region','charges']]

attributes.head()
# Defining the map function

def dummies(x,df):

    temp = pd.get_dummies(df[x], drop_first = True)

    df = pd.concat([df, temp], axis = 1)

    df.drop([x], axis = 1, inplace = True)

    return df



# Applying the function to the  attributes



attributes = dummies('age_range',attributes)

attributes = dummies('sex',attributes)

attributes = dummies('bmi_range',attributes)

attributes = dummies('smoker',attributes)

attributes = dummies('region',attributes)
attributes.head()
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(attributes, y, test_size = 0.20, random_state = 0)
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

num_vars = ['charges']

X_train[num_vars] = scaler.fit_transform(X_train[num_vars])
X_train.head()
y_train = X_train.pop('charges')

X_train_new = X_train
import statsmodels.api as sm

model = sm.OLS(y_train, X_train_new.astype(float)).fit()

model.summary()
def build_model(X,y):

    X = sm.add_constant(X) #Adding the constant

    lm = sm.OLS(y,X).fit() # fitting the model

    print(lm.summary()) # model summary

    return X
X_train_new = build_model(X_train.astype(float),y_train)
X_train_new = X_train.drop(['male'], axis = 1)

X_train_new = build_model(X_train_new.astype(float),y_train)
X_train_new = X_train_new.drop(['southeast'], axis = 1)

X_train_new = X_train_new.drop(['northwest'], axis = 1)

X_train_new = X_train_new.drop(['southwest'], axis = 1)



X_train_new = build_model(X_train_new.astype(float),y_train)
lm = sm.OLS(y_train,X_train_new).fit()

y_train_price = lm.predict(X_train_new)
# Plot the histogram of the error terms

fig = plt.figure()

sb.distplot((y_train - y_train_price), bins = 20)

fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 

plt.xlabel('Errors', fontsize = 18)  
num_vars = ['charges']

X_test[num_vars] = scaler.fit_transform(X_test[num_vars])
#Dividing into X and y

y_test = X_test.pop('charges')

XX_test = X_test
# Now let's use our model to make predictions.

X_train_new = X_train_new.drop('const',axis=1)

# Creating X_test_new dataframe by dropping variables from X_test

X_test_new = XX_test[X_train_new.columns]



# Adding a constant variable 

X_test_new = sm.add_constant(X_test_new)
y_pred = lm.predict(X_test_new.astype(float))
from sklearn.metrics import r2_score 

r2_score(y_test, y_pred)
#EVALUATION OF THE MODEL

# Plotting y_test and y_pred to understand the spread.

fig = plt.figure()

plt.scatter(y_test,y_pred)

fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 

plt.xlabel('y_test', fontsize=18)                          # X-label

plt.ylabel('y_pred', fontsize=16)  