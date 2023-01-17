#Libraries required

# Import Dependencies

%matplotlib inline



# Start Python Imports

import math, time, random, datetime



# Data Manipulation

import numpy as np

import pandas as pd



# Visualization 

import matplotlib.pyplot as plt

import missingno

import seaborn as sns

plt.style.use('seaborn-whitegrid')



# Machine learning

from sklearn.model_selection import train_test_split

from sklearn import model_selection, tree, preprocessing, metrics, linear_model

from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier

from sklearn.tree import DecisionTreeClassifier



#ignore warnings for now

import warnings

warnings.filterwarnings('ignore')



import os

print(os.listdir("../input"))
data = pd.read_csv('../input/weatherAUS.csv')
# Let's see how our data looks like

print('Weather dataframe dimension: ',data.shape)

data.describe()
# We can see that there are lot of NaN values in the dataframe.  

# Let's check which column has maximum Nan values

print(data.count().sort_values())



#Graph to find missing values in the dataframe

import missingno

missingno.matrix(data, figsize = (30,10))
data = data.drop(columns = ['Sunshine','Evaporation','Cloud3pm','Cloud9am','Location','Date','RISK_MM'],axis=1)
print(data.shape)

data.head()
def find_missing_values(df,columns):

    missing_vals = {}

    df_length = len(df)

    for column in columns:

        total_column_values = df[column].value_counts().sum()

        missing_vals[column] = df_length - total_column_values

    return missing_vals



missing_values = find_missing_values(data,data.columns)

missing_values
data = data.dropna(axis = 'index',how='any')

print(data.shape)



missing_values = find_missing_values(data,data.columns)

missing_values
final = pd.DataFrame()
#Data transformation

#For the categorical columns, we will change the value 'Yes' and 'No' to '1' and '0' respectively

data['RainTomorrow'].replace({'No': 0, 'Yes': 1},inplace = True)

data['RainToday'].replace({'No': 0, 'Yes': 1},inplace = True)



#See unique values and convert them to int using pd.getDummies()

categorical_columns = ['WindGustDir', 'WindDir3pm', 'WindDir9am']

for col in categorical_columns:

    print(np.unique(data[col]))

# transform the categorical columns

final = pd.get_dummies(data, columns=categorical_columns)

final.head()
from sklearn import preprocessing

scaler = preprocessing.MinMaxScaler()

scaler.fit(final)

final = pd.DataFrame(scaler.transform(final), index=final.index, columns=final.columns)

final.head()
#Now we will just see how many times it rained the next day?

fig = plt.figure(figsize = (20,3))

sns.countplot(y='RainTomorrow', data=final);

print(final.RainTomorrow.value_counts())
missing_values['MinTemp']
data.MinTemp.value_counts()
final_bin = pd.DataFrame()

final_bin['RainTomorrow'] = final['RainTomorrow']
final_bin['MinTemp'] = pd.cut(data['MinTemp'],bins = 5) #discretising the float numbers into categorical
final_bin.MinTemp.value_counts()
final.head()
def plot_count_dist(df,label_column,target_column,figsize=(20,5)):

        fig = plt.figure(figsize=figsize)

        plt.subplot(1,2,1)

        sns.countplot(y=target_column, data = df);

        plt.subplot(1,2,2)

        sns.distplot(data.loc[data[label_column] == 1][target_column],

                    kde_kws={"label" : "Yes"});

        sns.distplot(data.loc[data[label_column] == 0][target_column],

                    kde_kws={"label" : "No"});

    
#Calling the function above we will visualise the MinTemp bin counts as well as the MinTemp distribution versus RainTomorrow

plot_count_dist(df= final_bin, label_column = 'RainTomorrow', target_column = 'MinTemp', figsize = (20,10))
# Let's cross check the missing values

missing_values['MaxTemp']
data['MaxTemp'].value_counts()
final_bin['MaxTemp'] = pd.cut(data['MaxTemp'],bins = 5) #discretising the float numbers into categorical
final_bin['MaxTemp'].value_counts()
final.head()
#Calling the function above we will visualise the MaxTemp bin counts as well as the MaxTemp distribution versus RainTomorrow

plot_count_dist(df= final_bin, label_column = 'RainTomorrow', target_column = 'MaxTemp', figsize = (20,10))
# Let's cross check the missing values

missing_values['Rainfall']
data['Rainfall'].value_counts()
print("There are {} unique minimum temperature values.".format(len(data.Rainfall.unique())))
final_bin['Rainfall'] = pd.cut(data['Rainfall'],bins = 5) #discretising the float numbers into categorical
final_bin['Rainfall'].value_counts()
final.head()
#Calling the function above we will visualise the MaxTemp bin counts as well as the MaxTemp distribution versus RainTomorrow

plot_count_dist(df= final_bin, label_column = 'RainTomorrow', target_column = 'Rainfall', figsize = (20,10))
missing_values['WindGustDir']
WindGustDir_table = pd.crosstab(index=data["WindGustDir"], columns=data["RainTomorrow"])

WindGustDir_table
WindGustDir_table.plot(kind="bar", figsize=(15,8),stacked=False)
missing_values['WindGustSpeed']
plot_count_dist(df= final, label_column = 'RainTomorrow', target_column = 'WindGustSpeed', figsize = (20,10))
missing_values['WindDir9am']
WindDir9am_table = pd.crosstab(index=data["WindDir9am"], columns=data["RainTomorrow"])

WindDir9am_table
WindDir9am_table.plot(kind="bar", figsize=(15,8),stacked=False)
missing_values['WindDir3pm']
WindDir3pm_table = pd.crosstab(index=data["WindDir3pm"], columns=data["RainTomorrow"])

WindDir3pm_table
WindDir3pm_table.plot(kind="bar", figsize=(15,8),stacked=False)
missing_values['WindSpeed9am']
plot_count_dist(df= final, label_column = 'RainTomorrow', target_column = 'WindSpeed9am', figsize = (20,10))
missing_values['WindSpeed3pm']
plot_count_dist(df= final, label_column = 'RainTomorrow', target_column = 'WindSpeed3pm', figsize = (20,10))
missing_values['Humidity9am']
plot_count_dist(df= final, label_column = 'RainTomorrow', target_column = 'Humidity9am', figsize = (20,10))
missing_values['Humidity3pm']
plot_count_dist(df= final, label_column = 'RainTomorrow', target_column = 'Humidity3pm', figsize = (20,10))
missing_values['Pressure9am']
final['Pressure9am'].value_counts()
final_bin['Pressure9am'] = pd.cut(data['Pressure9am'],bins = 5) #discretising the float numbers into categorical
final_bin['Pressure9am'].value_counts()
final.head()
plot_count_dist(df= final_bin, label_column = 'RainTomorrow', target_column = 'Pressure9am', figsize = (20,10))
missing_values['Pressure3pm']
final['Pressure3pm'].value_counts()
final_bin['Pressure3pm'] = pd.cut(data['Pressure3pm'],bins = 5) #discretising the float numbers into categorical

final_bin['Pressure3pm'].value_counts()
final.head()
plot_count_dist(df= final_bin, label_column = 'RainTomorrow', target_column = 'Pressure3pm', figsize = (20,10))
missing_values['Temp9am']
final['Temp9am'].value_counts()
final_bin['Temp9am'] = pd.cut(data['Temp9am'],bins = 5) #discretising the float numbers into categorical
final_bin['Temp9am'].value_counts()
final.head()
plot_count_dist(df= final_bin, label_column = 'RainTomorrow', target_column = 'Temp9am', figsize = (20,10))
missing_values['Temp3pm']
final['Temp3pm'].value_counts()
final_bin['Temp3pm'] = pd.cut(data['Temp3pm'],bins = 5) #discretising the float numbers into categorical

final_bin['Temp3pm'].value_counts()
final.head()
plot_count_dist(df= final_bin, label_column = 'RainTomorrow', target_column = 'Temp3pm', figsize = (20,10))
#Now we will just see how many times it rained the current day?

fig = plt.figure(figsize = (20,3))

sns.countplot(y='RainToday', data=final);

print(final.RainToday.value_counts())
final.head()
f, ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.xticks(rotation=90)
final_bin.head()
final.shape
#Let's get hold of the independent variables and assign them as X



X = final.loc[:, final.columns != 'RainTomorrow']

y = final['RainTomorrow']

X.shape
# PCA to find the best number of features based on explained variance for each attribute

#Fitting the PCA algorithm with our Data

from sklearn.decomposition import PCA

pca = PCA().fit(X)

#Plotting the Cumulative Summation of the Explained Variance

plt.figure()

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('Number of Components')

plt.ylabel('Variance (%)') #for each component

plt.title('WeatherAUS Dataset Explained Variance')

plt.show()
#Using SelectKBest to get the top features!

from sklearn.feature_selection import SelectKBest, chi2

selector = SelectKBest(chi2, k=40)

selector.fit(X, y)

X_new = selector.transform(X)

print(X.columns[selector.get_support(indices=True)]) #top 40 columns
X = final[['MaxTemp', 'Rainfall', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm',

       'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Temp3pm',

       'RainToday', 'WindGustDir_E', 'WindGustDir_ENE', 'WindGustDir_ESE',

       'WindGustDir_N', 'WindGustDir_NNW', 'WindGustDir_NW', 'WindGustDir_W',

       'WindGustDir_WNW', 'WindDir3pm_E', 'WindDir3pm_ENE', 'WindDir3pm_ESE',

       'WindDir3pm_N', 'WindDir3pm_NNW', 'WindDir3pm_NW', 'WindDir3pm_SE',

       'WindDir3pm_SW', 'WindDir3pm_W', 'WindDir3pm_WNW', 'WindDir9am_E',

       'WindDir9am_ENE', 'WindDir9am_ESE', 'WindDir9am_N', 'WindDir9am_NNE',

       'WindDir9am_NNW', 'WindDir9am_NW', 'WindDir9am_SE', 'WindDir9am_SSE',

       'WindDir9am_W', 'WindDir9am_WNW']] # let's use all 40 features

y = final[['RainTomorrow']]
#Split the data into train and test data

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)
from sklearn.metrics import accuracy_score

import time

t0=time.time()

logreg = LogisticRegression(random_state=0, class_weight={0:0.3,1:0.7})

logreg = logreg.fit(X_train,y_train)

y_predLR = logreg.predict(X_test)

score = accuracy_score(y_test,y_predLR)

print('Accuracy :',score)

print('Time taken :' , time.time()-t0)







t0=time.time()

#X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)

dt = DecisionTreeClassifier(random_state=0,class_weight={0:0.3,1:0.7})

dt.fit(X_train,y_train)

y_predDT = dt.predict(X_test)

score = accuracy_score(y_test,y_predDT)

print('Accuracy :',score)

print('Time taken :' , time.time()-t0)
from sklearn.ensemble import RandomForestClassifier

t0=time.time()

rf = RandomForestClassifier(n_estimators=100, max_depth=4,random_state=0,class_weight={0:0.3,1:0.7})

rf.fit(X_train,y_train)

y_predRF = rf.predict(X_test)

score = accuracy_score(y_test,y_predRF)

print('Accuracy :',score)

print('Time taken :' , time.time()-t0)
from imblearn.ensemble import BalancedBaggingClassifier

from sklearn.tree import DecisionTreeClassifier



#Creating an object of the classifier.

bbc = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(),

                                sampling_strategy='auto',

                                replacement=False,

                                random_state=0)



#Training the classifier.

bbc.fit(X_train, y_train)

y_predBBC = bbc.predict(X_test)

score = accuracy_score(y_test,y_predBBC)

print('Accuracy :',score)

print('Time taken :' , time.time()-t0)
from sklearn.metrics import confusion_matrix

pred_models = []

pred_models.append(('LogisticRegression', y_predLR))

pred_models.append(('DecisionTree', y_predDT))

pred_models.append(('RandomForest', y_predRF))

pred_models.append(('BalancedBaggingClassifier', y_predBBC))





for name, pred_model in pred_models:

    cm = confusion_matrix(y_test, pred_model)

    #print(cm)

    plt.figure(figsize = (3,3))

    sns.heatmap(cm,fmt="d",annot=True,xticklabels=["No","Yes"],yticklabels=["No","Yes"],cbar=False)

    plt.title(name+" "+"Confusion Matrix")

    plt.xlabel("Predicted")

    plt.ylabel("Actuals")

    plt.show()