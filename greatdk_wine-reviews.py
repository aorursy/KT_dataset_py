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
# import required libraries

import numpy as np  # for numerical computations

import pandas as pd # for dataframes



#import visualization libraries

import matplotlib.pyplot as plt

%matplotlib inline 

import seaborn as sns

from IPython.display import display, Image



from matplotlib import animation

from IPython.display import HTML
Image(url='https://media.giphy.com/media/3o6ZtcLKytmWgdDOFO/giphy.gif')
Image(url='https://media.giphy.com/media/MUlmRFnTQxwJ2/giphy.gif')
# import required libraries

import numpy as np  # for numerical computations

import pandas as pd # for dataframes



#import visualization libraries

import matplotlib.pyplot as plt

%matplotlib inline 

import seaborn as sns

from IPython.display import display, Image
from matplotlib import animation

from IPython.display import HTML
sns.set_style('whitegrid')
# read the data from the csv files

white=pd.read_csv('/kaggle/input/wine-quality-dataset/winequality-white.csv')

red=pd.read_csv('/kaggle/input/wine-quality-dataset/winequality-red.csv')
# check head of the two data files

red.head()
len(red)
white.head()
len(white)
red=pd.read_csv('/kaggle/input/wine-quality-dataset/winequality-white.csv',sep=';')

white=pd.read_csv('/kaggle/input/wine-quality-dataset/winequality-red.csv',sep=';')
len(red)
color_red = np.repeat('red', 4898)

color_white = np.repeat('white', 1599)
len(color_red)
len(color_white)
len(color_red)
red['wine_type']=color_red
red['wine_type'] = color_red

white['wine_type'] = color_white
wine = red.append(white, ignore_index=True) 

wine.head()
wine.shape
sum(wine.duplicated())
wine.drop_duplicates(inplace=True)
wine.isnull().sum()
#checking the datatypes of the variables

wine.info()
wine.describe()
wine.shape
wine['quality'].describe()
sns.countplot(x='quality',data=wine,palette='Reds')

plt.title('Distribution of Quality')

plt.xlabel('Quality Scaler')

plt.tight_layout()
sns.countplot(x='quality',hue='wine_type',data=wine,palette='Reds_r')

plt.title('Distribution of Quality-Red and White wine')

plt.xlabel('Quality Scaler')

plt.tight_layout()
plt.figure(figsize=(12,7))

sns.boxplot(x='quality',y='fixed acidity',hue='wine_type',data=wine,palette='Reds_r')

plt.title('Distribution between Quality vs Fixed acidity')

plt.tight_layout()

Image(url='https://media.giphy.com/media/dXICCcws9oxxK/giphy.gif')
plt.figure(figsize=(12,7))

sns.boxplot(x='quality',y='volatile acidity',hue='wine_type',data=wine,palette='Reds_r')

plt.title('Distribution between Quality vs Volatile acidity')

plt.tight_layout()
plt.figure(figsize=(12,7))

sns.boxplot(x='quality',y='citric acid',hue='wine_type',data=wine,palette='Reds_r')

plt.title('Distribution between Quality vs Citric Acid')

plt.tight_layout()
plt.figure(figsize=(12,7))

sns.boxplot(x='quality',y='residual sugar',hue='wine_type',data=wine,palette='Reds_r')

plt.title('Distribution between Quality vs Residual Sugar')

plt.tight_layout()
plt.figure(figsize=(12,7))

sns.boxplot(x='quality',y='chlorides',hue='wine_type',data=wine,palette='Reds_r')

plt.title('Distribution between Quality vs Chlorides')

plt.tight_layout()
plt.figure(figsize=(12,7))

sns.boxplot(x='quality',y='free sulfur dioxide',hue='wine_type',data=wine,palette='Reds_r')

plt.title('Distribution between Quality vs Free Sulfur Dioxide')

plt.tight_layout()
plt.figure(figsize=(12,7))

sns.boxplot(x='quality',y='total sulfur dioxide',hue='wine_type',data=wine,palette='Reds_r')

plt.title('Distribution between Quality vs Total Sulfur Dioxide')

plt.tight_layout()
plt.figure(figsize=(12,7))

sns.boxplot(x='quality',y='density',hue='wine_type',data=wine,palette='Reds_r')

plt.title('Distribution between Quality vs Density')

plt.tight_layout()
plt.figure(figsize=(12,7))

sns.boxplot(x='quality',y='pH',hue='wine_type',data=wine,palette='Reds_r')

plt.title('Distribution between Quality vs Total pH')

plt.tight_layout()
plt.figure(figsize=(12,7))

sns.boxplot(x='quality',y='sulphates',hue='wine_type',data=wine,palette='Reds_r')

plt.title('Distribution between Quality vs Total Sulphates')

plt.tight_layout()
plt.figure(figsize=(12,7))

sns.boxplot(x='quality',y='alcohol',hue='wine_type',data=wine,palette='Reds_r')

plt.title('Distribution between Quality vs Alcohol')

plt.tight_layout()
plt.figure(figsize=(12,7))

sns.heatmap(data=wine.corr(),annot=True)

plt.title('Correlation Between features')

plt.tight_layout()
Image(url='https://media.giphy.com/media/g6VWn4bujbz2w/giphy.gif')
Image(url='https://media.giphy.com/media/11e56tPCqD9kjK/giphy.gif')
from sklearn.ensemble import RandomForestClassifier
X=wine.drop(['quality','wine_type'],axis=1)

y=wine.quality
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=101)
print(X_train.shape)

print(X_test.shape)
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
# fitting the transformer API

scaler.fit(X_train)
#confirm applying transformer to training data

X_train_scaled=scaler.transform(X_train)
X_test_scaled=scaler.transform(X_test)
rfc=RandomForestClassifier()
rfc.fit(X_train,y_train)
from sklearn.pipeline import make_pipeline
pipeline = make_pipeline(MinMaxScaler(),RandomForestClassifier(n_estimators=100))
# Use tunable parameters

print(pipeline.get_params())
hyperparameters = { 'randomforestclassifier__max_features' : ['auto', 'sqrt', 'log2'],'randomforestclassifier__max_depth': [None, 5, 50, 70]}
from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(pipeline,param_grid=hyperparameters, cv=10)
# Fit and tune model

clf.fit(X_train,y_train)
print(clf.best_params_)

print(clf.refit)
from sklearn.metrics import mean_squared_error,mean_absolute_error,classification_report,accuracy_score
predictions=clf.predict(X_test)
print(mean_squared_error(y_test, predictions))
print(classification_report(y_test, predictions))
print(accuracy_score(y_test,predictions))