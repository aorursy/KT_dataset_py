import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#preprocessing

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from pandas.plotting import scatter_matrix

#visualizations

import seaborn as sns

from matplotlib import pyplot as plt

%matplotlib inline



#algorithms

from xgboost import XGBClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression



#score metrics

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

import os

print(os.listdir("../input"))
df = pd.read_csv('../input/ALF_Data.csv')

copy_df=df

df.head(10)
df.shape
df.info()
#dropping samples that dont have value fore ALF

df = df.dropna(axis = 0, subset=['ALF'])
total_missingvalues = df.isnull().sum()

total_missingvalues
#selecting a sample of the features for easier understanding

df = df[['Age','Gender','Region','Weight','Height','Body Mass Index','Obesity','Waist',

         'Maximum Blood Pressure','Minimum Blood Pressure','Good Cholesterol','Bad Cholesterol',

         'Total Cholesterol','Dyslipidemia','PVD','ALF']]



df.head()
#refer to slide for heat map 

# calculate the correlation matrix

corr = df.corr()



# plot the heatmap

sns.heatmap(corr, 

        xticklabels=corr.columns,

        yticklabels=corr.columns)
df.corr()
y = df['ALF']

df = df.drop('ALF',axis=1)

df.head()
total_missingvalues = df.isnull().sum()

total_missingvalues
#Taking care of missing values

from sklearn.preprocessing import Imputer



imputer = Imputer(missing_values = 'NaN',strategy = 'median',axis=0)

imputer = imputer.fit(df.iloc[:,3:13]) #SELECTING THE COLUMN WITH MISSING VALUES

df.iloc[:,3:13] = imputer.transform(df.iloc[:,3:13])
df.info()
df.head()
#checking number of classes in the categorical feature

df['Region'].unique()
df['Gender'].unique()
#Encode categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder



labelencoder_X = LabelEncoder()

df.iloc[:,1] = labelencoder_X.fit_transform(df.iloc[:,1]) #SELECTING THE COLUMN WITH OBJECT TYPE



df=pd.get_dummies(df, columns=["Region"], prefix=["Region"])





df.head()
#dropping Region_west because the model can infer the values for this from the other 3 columns

df = df.drop('Region_west',axis = 1)

df.head()
X=df
#standardizing the input feature

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X = sc.fit_transform(X)
#splitting our dataset into training sets and teset sets



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,stratify = y)
from keras import Sequential

from keras.layers import Dense
classifier = Sequential()

#First Hidden Layer

classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal', input_dim=X_train.shape[1]))

#Second  Hidden Layer

classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal'))

#Output Layer

classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))
#Compiling the neural network

classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])
#Fitting the data to the training dataset

classifier.fit(X_train,y_train, batch_size=10, epochs=10)
y_pred=classifier.predict(X_test)

y_pred =(y_pred>0.5)
count = 0

for i in range( len(y_test) ):

    if y_pred[i] != y_test.iloc[i]: 

        count = count + 1

error = count/len(y_pred)

print( "Error for Neural Network= %f " % (error*100) + '%' )

accuracy = (1-error)

print( "Accuracy for Neural Network = %f " % (accuracy*100) + '%' )