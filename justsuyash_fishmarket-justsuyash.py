#Let me know if you need anymore details or if you have suggestions

#Linkedin : @justsuyash
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data  = pd.read_csv('../input/fish-market/Fish.csv')
df = data.copy() #Copying it just in case

df.head(10)
df['Species'].unique()
df.info()
#Lets Check if there are any Null Values in the data set:

df.isnull().values.any()
df.describe()
first_quarlitle_weight = df['Weight'].quantile(0.25)

third_quarlitle_weight = df['Weight'].quantile(0.75)



inter_quartile_weight = third_quarlitle_weight - first_quarlitle_weight



lower_range_weight = first_quarlitle_weight - 1.5*inter_quartile_weight

upper_range_weight = third_quarlitle_weight + 1.5*inter_quartile_weight
#This way we get a resonable estimate of an outlier

df[ (df['Weight'] < lower_range_weight) | (df['Weight']>upper_range_weight)]
first_quarlitle_length1 = df['Length1'].quantile(0.25)

third_quarlitle_length1 = df['Length1'].quantile(0.75)



inter_quartile_length1 = third_quarlitle_length1 - first_quarlitle_length1



lower_range_length1 = first_quarlitle_length1 - 1.5*inter_quartile_length1

upper_range_length1 = third_quarlitle_length1 + 1.5*inter_quartile_length1

df[ (df['Length1'] < lower_range_length1) | (df['Length1']>upper_range_length1)]
excess_weight  = df[ (df['Weight'] < lower_range_weight) | (df['Weight']>upper_range_weight)]

df.drop(excess_weight.index,inplace=True)
df.describe()
zero_weights  = df [ data['Weight'] == 0]

df.drop(zero_weights.index,inplace=True)
df.describe()
sns.heatmap(df.corr(),annot=True, cmap='YlGnBu')
#Lets do a pairplot to see if we can find something.

sns.pairplot(df, hue='Species')
df['Species'] = df['Species'].astype('category')

df['species_cat'] = df['Species'].cat.codes
c = df['Species'].astype('category')



d = dict(enumerate(c.cat.categories))

print (d)
sns.pairplot(df,hue='species_cat')
data_with_dummies = df.drop(['species_cat'],axis=1)

data_with_dummies = pd.get_dummies(df, drop_first=True)
data_with_dummies
data_with_dummies.columns
x = data_with_dummies.drop(['Weight'],axis=1)

y = data_with_dummies['Weight']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 42)
lin = LinearRegression()
lin.fit(x_train,y_train)
y_train_predicted = lin.predict(x_train)

r2_score(y_train, y_train_predicted)
from sklearn.model_selection import cross_val_score

cross_val_score_train = cross_val_score(lin, x_train, y_train, cv=10, scoring='r2')

print(cross_val_score_train)
cross_val_score_train.mean()
predict = lin.predict(x_test)
# calculate these metrics by hand!

from sklearn import metrics



print('MAE:', metrics.mean_absolute_error(y_test, predict))

print('MSE:', metrics.mean_squared_error(y_test, predict))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predict)))
print(r2_score(y_test, predict))
plt.scatter(y_test,predict)

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')
sns.pairplot(df,hue='species_cat')
c = df['Species'].astype('category')



d = dict(enumerate(c.cat.categories))

print (d)