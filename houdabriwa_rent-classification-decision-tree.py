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

# Load the dataset into a pandas dataframe.

data = pd.read_csv('/kaggle/input/brasilian-houses-to-rent/houses_to_rent_v2.csv')


# Google file system

#from google.colab import drive

#drive.mount('/content/drive')
#extract the text file

#File='/drive/My Drive/Colab Notebooks/houses_to_rent_v2.csv'
import pandas as pd

# Load the dataset into a pandas dataframe.

#data = pd.read_csv('/content/drive/My Drive/Colab Notebooks/houses_to_rent_v2.csv')
data
data.isna().sum()
data[['area', 'rooms', 'bathroom', 'parking spaces', 'floor' ,'hoa (R$)', 'rent amount (R$)', 'property tax (R$)', 'fire insurance (R$)', 'total (R$)']].describe()
data['rent-class'] = pd.cut(x=data['rent amount (R$)'], bins=[450, 1530, 2661, 5000, 45000], labels=['cheap', 'average', 'expensive', 'very expensive'], include_lowest=True)
data.animal.unique()
data.city.unique()
data.furniture.unique()
data.info()
import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

ax = data['rent-class'].value_counts().sort_values().plot(kind="barh")

totals= []

for i in ax.patches:

    totals.append(i.get_width())

total = sum(totals)

for i in ax.patches:

     ax.text(i.get_width()+.3, i.get_y()+.20, 

     str(round((i.get_width()/total)*100, 2))+'%', 

     fontsize=10, color='black')

ax.grid(axis="x")

plt.suptitle('rent-class', fontsize=20)

plt.show()


plt.figure(figsize=(15, 6))

plt.title('Rent amount (R$) distribution')

sns.distplot(data['rent amount (R$)'])

plt.xticks(np.arange(data['rent amount (R$)'].min(), data['rent amount (R$)'].max(), step=3000));
plt.figure(figsize=(15,6))

plt.title('Rent amount (R$) distribution by rent class and city')

sns.violinplot(x=data['city'], y=data['rent amount (R$)'], hue=data['rent-class'])
k=max(data['rent amount (R$)'])

name=data[data['rent amount (R$)']==k].index

data.drop(name, inplace=True)
sns.catplot(x ='bathroom', y ='rent amount (R$)', data = data, height=5, aspect=3)

plt.title("Relationship between rent and number of bathrooms", size=17)
sns.catplot(x ='rooms', y ='rent amount (R$)', data = data, height=5, aspect=3)

plt.title("Relationship between rent and number of rooms", size=17)
plt.figure(figsize=(15,6))

sns.countplot(x ='rooms' , hue = data['city'], data = data)
sns.catplot(x ='area', y ='rent amount (R$)', data = data, height=5, aspect=3)

plt.title("Relationship between rent and house area", size=17)
import plotly.express as px

fig = px.scatter(data, x="rent amount (R$)",

                y="fire insurance (R$)", 

                color="city"          

                 )

fig.update_traces(marker=dict(size=12,

                              line=dict(width=1, color='LightSkyBlue')),

                  selector=dict(mode='markers'))

fig.show(renderer='colab')
plt.figure(figsize=(15,6))

sns.countplot(x ='animal', hue = data['city'], data = data)
plt.figure(figsize=(15,6))

sns.countplot(x ='furniture',  data = data)
plt.figure(figsize=(15,6))

plt.title('Rent amount (R$) distribution by rent class and city')

sns.violinplot(x=data['furniture'], y=data['rent amount (R$)'], hue=data['city'])
#we will locate our independent variables (predictors)

X = data.iloc[:,[0,1,2,3,6,7]].values

#Here we try to locate our target variable

y = data.iloc[:,13].values
# encoding categorical data e.g. gender as a dummy variable

from sklearn.preprocessing import LabelEncoder

labelencoder_X = LabelEncoder()

X[:,0] = labelencoder_X.fit_transform(X[:,0])

X[:,4] = labelencoder_X.fit_transform(X[:,4])

X[:,5] = labelencoder_X.fit_transform(X[:,5])

#X[:,0] = labelencoder_X.fit_transform(X[:,0])

# encoding categorical data e.g. disease outcome as a dummy variable

y,class_names = pd.factorize(y)
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
# Fitting Classifier to the Training Set

from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(criterion='entropy',max_depth=3)

classifier.fit(X_train, y_train)
# Model performance on training set

y_pred_train =classifier.predict(X_train)

from sklearn import metrics

from sklearn.metrics import confusion_matrix, classification_report



accuracy = metrics.accuracy_score(y_train, y_pred_train)

print("Accuracy: {:.2f}".format(accuracy))
from sklearn.metrics import classification_report, confusion_matrix

#print(confusion_matrix(y_train, y_pred_train))

print(classification_report(y_train, y_pred_train))