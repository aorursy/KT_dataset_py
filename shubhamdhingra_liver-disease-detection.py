import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
# Reading the file as saving to dataframe(df)
df=pd.read_csv('../input/liver.csv')
df.head()
# Checking dataframe to see if there are any missing values. There are two missing values
sns.heatmap(df.isnull(), cmap='coolwarm',xticklabels=True,yticklabels=False,cbar=False)
# Impute missing values by importing the Imputer class from sklearn.preprocessing
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer = imputer.fit(df.iloc[:,9:10])
df.iloc[:,9:10]= imputer.transform(df.iloc[:,9:10])
# Checking for missing data again
sns.heatmap(df.isnull(), cmap='coolwarm',xticklabels=True,yticklabels=False,cbar=False)
# Get some info on the dataset
df.info()
df.drop('Selector',axis=1).describe()
# Checking if data is skewed
df.skew()
# As the heatmap shows there are no missing values. Now let's do some visalization.
#plt.figure(figsize= (6
#df.hist()
sns.pairplot(df)
# Looks like there may be some linear correlations between some of the features. More data visualizations
sns.barplot(x='Selector',y='Age',data =df)
# Mean Age is roughly the same for both selctors
sns.jointplot(x='Selector',y='Age',data =df)
sns.distplot(df['Age'])
# Age looks almost normally distributed
sns.countplot(x='Gender',data=df)
# More Males than Females in the dataset
sns.countplot(x='Gender',data=df,hue='Selector')
# The percentage of females falling under category 2 is higher than that of of males when compared to the total of
#their gender.
sns.violinplot(x='Gender',y='Age',hue='Selector',data=df)
sns.heatmap(df.corr())
# Some of the features are highly correlated
df.head()
# Encoding gender 
Gender = pd.get_dummies(df.iloc[: ,1], drop_first=True)
df = pd.concat([df,Gender],axis=1)
df.head()
df.drop('Gender',axis=1,inplace=True)
df.head()
#Encoding Selector and Renaming it as Prognosis
Result = pd.get_dummies(df['Selector'],drop_first=True)
df=pd.concat([df,Result],axis=1)
df.head(10)
df.drop('Selector',axis=1,inplace=True)
# This turned the categories in the Selector column: Category 2 is now category 1 and Category 1 is now category 0
df.head()
#renaming column 2 to Prognosis
df['Prognosis'] = df[2]
df.drop(2,axis=1,inplace=True)
df.head(10)
#checking if target variable is imbalanced
df['Prognosis'].value_counts()
from sklearn.utils import resample
# Creating 2 different dataframes df_majority and df_minority
df_majority = df[df['Prognosis']==0]
df_minority = df[df['Prognosis']==1]
# Upsample minority class
df_minority_upsampled = resample(df_minority,replace=True,n_samples=416, random_state=123)
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority,df_minority_upsampled])
df_upsampled['Prognosis'].value_counts()
X =df_upsampled.drop('Prognosis', axis=1)
y = df_upsampled['Prognosis']
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=101)
import keras
from keras.models import Sequential
from keras.layers import Dense
# Initializing the Network
nn_classifier = Sequential()
# Adding the first input layer and the first hidden layer
nn_classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=10))
# Adding second Layer
nn_classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
# Adding output layer
nn_classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
# Compiling Neural Network
nn_classifier.compile(optimizer='adam', loss = 'binary_crossentropy',metrics=['accuracy'])
nn_classifier.fit(X_train,y_train,batch_size=10,epochs=1000)
# Neural Network gives us an 80-81% accuracy