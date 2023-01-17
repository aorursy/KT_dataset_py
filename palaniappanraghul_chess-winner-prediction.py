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
df = pd.read_csv('/kaggle/input/chess/games.csv')

df.head(5)
df.isnull().sum()
df.describe()
x = df.drop(columns=['opening_eco','id'])

print(x)
y = df['opening_eco']

print(y)
import matplotlib.pyplot as plt

import seaborn as sns
#plotting histogram 

plt.hist(df['winner'],rwidth=0.9,alpha=0.3,color='blue',bins=15,edgecolor='red') 



#x and y-axis labels 

plt.xlabel('winner') 

plt.ylabel('opening_eco') 



#plot title 

plt.title('Chess Prediction') 

plt.show();
z = x['opening_ply']

# Change the style of the figure to the "dark" theme

sns.set_style("dark")

# Line chart 

plt.figure(figsize=(12,6))

sns.lineplot(data=z)
plt.scatter(df.index,df['opening_ply'])

plt.show()
sns.scatterplot(x=df.index,y=df['opening_eco'],hue=df['winner'])
df['winner'].value_counts().head(10).plot.pie()

# Unsquish the pie.

import matplotlib.pyplot as plt

plt.gca().set_aspect('equal')
plt.boxplot(df['opening_ply'])
sns.boxplot(df['opening_ply'])
from sklearn import preprocessing 

label_encoder = preprocessing.LabelEncoder()  

x= x.apply(label_encoder.fit_transform)

print(x)

y= label_encoder.fit_transform(y)

print(y)
#Normalisation

from sklearn import preprocessing

# normalize the data attributes

normalized_X = preprocessing.normalize(x)

normalized_X
#Standardisation

# standardize the data attributes

standardized_X = preprocessing.scale(x)

standardized_X
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)
# data normalization with sklearn

from sklearn.preprocessing import MinMaxScaler



# fit scaler on training data

norm = MinMaxScaler().fit(x_train)



# transform training data

X_train_norm = norm.transform(x_train)



# transform testing dataabs

X_test_norm = norm.transform(x_test)


# fit scaler on training data

norm = MinMaxScaler().fit(x_train)



# transform training data

X_train_norm = norm.transform(x_train)

print("Scaled Train Data: \n\n")

print(X_train_norm)



# transform testing dataabs

X_test_norm = norm.transform(x_test)

print("\n\nScaled Test Data: \n\n")

print(X_test_norm)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

# random forest model creation

rfc = RandomForestClassifier()

rfc.fit(x_train,y_train)

# predictions

rfc_predict = rfc.predict(x_test)



print("Accuracy:",accuracy_score(y_test, rfc_predict))
from sklearn.tree import DecisionTreeClassifier

# Create Decision Tree classifer object

clf = DecisionTreeClassifier()

# Train Decision Tree Classifer

clf = clf.fit(x_train,y_train)

#Predict the response for test dataset

y_pred = clf.predict(x_test)





print("Accuracy:",accuracy_score(y_test, y_pred))