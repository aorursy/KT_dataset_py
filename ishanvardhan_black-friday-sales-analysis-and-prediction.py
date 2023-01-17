# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

dataset=pd.read_csv('../input/BlackFriday.csv')

dataset.head()
dataset.info()
dataset.describe()
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

imputer = imputer.fit(dataset.iloc[:, 9:11].values)

dataset.iloc[:,9:11] = imputer.transform(dataset.iloc[:, 9:11].values)

dataset.info() 
dataset.drop(['User_ID','Product_ID'], axis=1, inplace=True)

dataset.info()
dataset.head()
dataset['Age']=(dataset['Age'].str.strip('+'))

dataset['Stay_In_Current_City_Years']=(dataset['Stay_In_Current_City_Years'].str.strip('+').astype('float'))
dataset.info()

dataset.head()
sns.heatmap(

    dataset.corr(),

    annot=True

)

g = sns.FacetGrid(dataset,col="Stay_In_Current_City_Years")

g.map(sns.barplot, "Marital_Status", "Purchase");
sns.jointplot(x='Occupation',y='Purchase',

              data=dataset, kind='hex'

             )
g = sns.FacetGrid(dataset,col="City_Category")

g.map(sns.barplot, "Gender", "Purchase");
g = sns.FacetGrid(dataset,col="Age",row="City_Category")

g.map(sns.barplot, "Gender", "Purchase");
sns.violinplot(x='City_Category',y='Purchase',hue='Marital_Status',

               data=dataset)
fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(18,14))

ax = sns.pointplot(y='Product_Category_1', x='City_Category',hue='Age',

                 data=dataset,

                 ax=axes[0,0]

                )

ax = sns.pointplot(y='Product_Category_2', x='City_Category',hue='Age',

                 data=dataset,

                 ax=axes[0,1]

                )

ax = sns.pointplot(y='Product_Category_3', x='City_Category', hue='Age',

                 data=dataset,

                 ax=axes[1,0]

                )

ax = sns.pointplot(y='Purchase', x='City_Category', hue='Age',

                 data=dataset,

                 ax=axes[1,1]

                )

#Dividing the data into test and train datasets

X = dataset.iloc[:, 0:9].values

y = dataset.iloc[:, 9].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

X_train

y_train
X_test

y_test
from sklearn.preprocessing import LabelEncoder

labelencoder_X_train = LabelEncoder()

X_train



X_train[:, 0] = labelencoder_X_train.fit_transform(X_train[:, 0])

X_train
X_train[:, 1] = labelencoder_X_train.fit_transform(X_train[:, 1])

X_train
X_train[:, 3] = labelencoder_X_train.fit_transform(X_train[:, 3])

X_train
labelencoder_X_test = LabelEncoder()

X_test
X_test[:, 0] = labelencoder_X_test.fit_transform(X_test[:, 0])

X_test
X_test[:, 1] = labelencoder_X_test.fit_transform(X_test[:, 1])

X_test
X_test[:, 3] = labelencoder_X_test.fit_transform(X_test[:, 3])

X_test
# Feature Scaling of training and test set

from sklearn.preprocessing import StandardScaler

sc_X_train = StandardScaler()

X_train = sc_X_train.fit_transform(X_train)



sc_X_test = StandardScaler()

X_test = sc_X_test.fit_transform(X_test)

#Fitting the model

from sklearn.ensemble import RandomForestRegressor



from sklearn.metrics import mean_absolute_error

# compare MAE with differing values of max_leaf_nodes

def get_mae(max_leaf_nodes, X_train, X_test, y_train, y_test):

    model = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test,y_pred)

    return(mae)
for max_leaf_nodes in [5, 50, 100, 300, 500, 700, 800, 850]:

    my_mae = get_mae(max_leaf_nodes, X_train, X_test, y_train, y_test)

    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))

    
y_test
regressor = RandomForestRegressor(n_estimators=700, random_state=0)

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
y_pred