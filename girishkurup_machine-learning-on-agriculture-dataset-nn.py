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
# This librarys is to work with vectors
import numpy as np
# This library is to create some graphics algorithmn
import seaborn as sns
# to render the graphs
import matplotlib.pyplot as plt
# import module to set some ploting parameters
from matplotlib import rcParams
# Library to work with Regular Expressions
import re

# This function makes the plot directly on browser
%matplotlib inline

# Seting a universal figure size 
rcParams['figure.figsize'] = 10,8
train=pd.read_csv('/kaggle/input/train.csv')
train.info()
train['Crop_Damage'].value_counts()

#coln =['Estimated_Insects_Count','Number_Doses_Week','Number_Weeks_Used','Number_Weeks_Quit']
#for col in coln:
 #   train[col] = train[col].replace('None', np.nan)

#coln =['Estimated_Insects_Count','Number_Doses_Week','Number_Weeks_Used','Number_Weeks_Quit']
#for col in coln:
   # train[col] = train[col].fillna(train[col].mode()[0], inplace=True)
train.isnull().sum()
train['Number_Weeks_Used'].fillna(train['Number_Weeks_Used'].mode()[0], inplace=True)
train.isnull().sum()
train['Number_Doses_Week'].value_counts()
#printing the chance to cropdamage by each Croptype
print("Chances to crop damange based on crop type: ") 
print(train.groupby("Crop_Type")["Crop_Damage"].mean())

# figure size
plt.figure(figsize=(12,5))

#Plotting the count of title by Crop damage or not category
sns.countplot(x='Crop_Type', data=train, palette="hls",
              hue="Crop_Damage")
plt.xlabel("Crop_Type", fontsize=16)
plt.ylabel("Count", fontsize=16)
plt.title("Crop_Type Grouped Count", fontsize=20)
plt.xticks(rotation=45)
plt.show()
inscect_high_Damage_low = train[(train["Estimated_Insects_Count"] > 0) & 
                              (train["Crop_Damage"] == 0)]

inscect_high_Damage_medium = train[(train["Estimated_Insects_Count"] > 0) & 
                              (train["Crop_Damage"] == 1)]                                   
inscect_high_Damage_high= train[(train["Estimated_Insects_Count"] > 0) & 
                              (train["Crop_Damage"] == 2)]

#figure size
plt.figure(figsize=(10,5))

# Ploting the 2 variables that we create and compare the three
sns.distplot(inscect_high_Damage_low["Estimated_Insects_Count"], bins=24, color='g')
sns.distplot(inscect_high_Damage_medium["Estimated_Insects_Count"], bins=24, color='b')
sns.distplot(inscect_high_Damage_high["Estimated_Insects_Count"], bins=24, color='r')
                                   
                                   
plt.title("Distribuition and density by Estimated_Insects_Count",fontsize=20)
plt.xlabel("Estimated_Insects_Count",fontsize=15)
plt.ylabel("Distribuition Crop Damage",fontsize=15)
plt.show()
# figure size
plt.figure(figsize=(12,5))

# using facetgrid that is a great way to get information of our dataset
g = sns.FacetGrid(train, col='Crop_Damage',size=5)
g = g.map(sns.distplot, "Estimated_Insects_Count")
plt.show()
train['Estimated_Insects_Count'].value_counts()
print(pd.crosstab(train.Soil_Type, train.Crop_Damage))
#Plotting the result
plt.subplot(2,1,1)
sns.countplot("Soil_Type",data=train,hue="Crop_Damage", palette="hls")
plt.ylabel("Count", fontsize=18)
plt.xlabel("Soil_Type", fontsize=18)
plt.title("Soil_Type Distribution ", fontsize=20)

#plt.subplot(2,1,2)
#sns.swarmplot(x='Soil_Type',y="Estimated_Insects_Count",data=train,
             # hue="Crop_Damage", palette="hls", )
#plt.ylabel("Estimated_Insects_Count", fontsize=18)
#plt.xlabel("Soil_Type", fontsize=18)
#plt.title("Estimated_Insects Distribution by Soil Categorys ", fontsize=20)

#plt.subplots_adjust(hspace = 0.5, top = 0.9)

plt.show()

print(pd.crosstab(train.Pesticide_Use_Category, train.Crop_Damage))
#Plotting the result
plt.subplot(2,1,1)
sns.countplot("Pesticide_Use_Category",data=train,hue="Crop_Damage", palette="hls")
plt.ylabel("Count", fontsize=18)
plt.xlabel("Pesticide_Use_Category", fontsize=18)
plt.title("Pesticide_Use_Category Distribution ", fontsize=20)
plt.show()

print(pd.crosstab(train.Number_Doses_Week, train.Crop_Damage))
#Plotting the result
plt.subplot(2,1,1)
sns.countplot("Number_Doses_Week",data=train,hue="Crop_Damage", palette="hls")
plt.ylabel("Count", fontsize=18)
plt.xlabel("Number_Doses_Week", fontsize=18)
plt.title("Number_Doses_Week_Category Distribution ", fontsize=20)
plt.show()

#Number_Weeks_Used

print(pd.crosstab(train.Number_Weeks_Used, train.Crop_Damage))
#Plotting the result
plt.subplot(2,1,1)
sns.countplot("Number_Weeks_Used",data=train,hue="Crop_Damage", palette="hls")
plt.ylabel("Count", fontsize=18)
plt.xlabel("Number_Weeks_Used", fontsize=18)
plt.title("Number_Weeks_Used_Category Distribution ", fontsize=20)
plt.show()

print(pd.crosstab(train.Season, train.Crop_Damage))
#Plotting the result
plt.subplot(2,1,1)
sns.countplot("Season",data=train,hue="Crop_Damage", palette="hls")
plt.ylabel("Count", fontsize=18)
plt.xlabel("Season", fontsize=18)
plt.title("Season_Category Distribution ", fontsize=20)
plt.show()


print(pd.crosstab(train.Number_Weeks_Quit, train.Crop_Damage))
#Plotting the result
plt.subplot(2,1,1)
sns.countplot("Number_Weeks_Quit",data=train,hue="Crop_Damage", palette="hls")
plt.ylabel("Count", fontsize=18)
plt.xlabel("Number_Weeks_Quit", fontsize=18)
plt.title("Number_Weeks_Quit Distribution ", fontsize=20)
plt.show()
# Explore Parch feature vs Survived
g  = sns.factorplot(x="Pesticide_Use_Category",y="Crop_Damage",data=train, kind="bar", size = 6,palette = "hls")
g = g.set_ylabels("survival probability Pesticide")

g  = sns.factorplot(x="Season",y="Crop_Damage",data=train, kind="bar", size = 6,palette = "hls")
g = g.set_ylabels("survival probability Season")
#lets understand the impact of number of weeks of pestiside use on the Crop

inscect_high_Damage_low = train[(train["Number_Weeks_Used"] > 0) & 
                              (train["Crop_Damage"] == 0)]

inscect_high_Damage_medium = train[(train["Number_Weeks_Used"] > 0) & 
                              (train["Crop_Damage"] == 1)]                                   
inscect_high_Damage_high= train[(train["Number_Weeks_Used"] > 0) & 
                              (train["Crop_Damage"] == 2)]

#figure size
plt.figure(figsize=(10,5))

# Ploting the 2 variables that we create and compare the three
sns.distplot(inscect_high_Damage_low["Number_Weeks_Used"], bins=24, color='g')
sns.distplot(inscect_high_Damage_medium["Number_Weeks_Used"], bins=24, color='b')
sns.distplot(inscect_high_Damage_high["Number_Weeks_Used"], bins=24, color='r')
                                   
                                   
plt.title("Distribution by Number_Weeks_Used",fontsize=20)
plt.xlabel("Number_Weeks_Used",fontsize=15)
plt.ylabel("Distribuition Crop Damage",fontsize=15)
plt.show()


# using facetgrid that is a great way to get information of our dataset
g = sns.FacetGrid(train, col='Crop_Damage',size=5)
g = g.map(sns.distplot, "Number_Weeks_Used")
plt.show()
#lets understand the impact of number of doses of pestiside use on the Crop

inscect_high_Damage_low = train[(train["Number_Doses_Week"] > 0) & 
                              (train["Crop_Damage"] == 0)]

inscect_high_Damage_medium = train[(train["Number_Doses_Week"] > 0) & 
                              (train["Crop_Damage"] == 1)]                                   
inscect_high_Damage_high= train[(train["Number_Doses_Week"] > 0) & 
                              (train["Crop_Damage"] == 2)]

#figure size
plt.figure(figsize=(10,5))

# Ploting the 2 variables that we create and compare the three
sns.distplot(inscect_high_Damage_low["Number_Doses_Week"], bins=24, color='g')
sns.distplot(inscect_high_Damage_medium["Number_Doses_Week"], bins=24, color='b')
sns.distplot(inscect_high_Damage_high["Number_Doses_Week"], bins=24, color='r')
                                   
                                   
plt.title("Distribution by Number_Doses",fontsize=20)
plt.xlabel("Number_Doses_Week",fontsize=15)
plt.ylabel("Distribuition Crop Damage",fontsize=15)
plt.show()

# using facetgrid that is a great way to get information of our dataset
g = sns.FacetGrid(train, col='Crop_Damage',size=5)
g = g.map(sns.distplot, "Number_Doses_Week")
plt.show()
df_train1=train.drop('ID', axis=1)

plt.show()
plt.figure(figsize=(15,12))
sns.heatmap(df_train1.astype(float).corr(),vmax=1.0,  annot=True)
plt.show()
df_train = pd.get_dummies(df_train1, columns=["Crop_Type","Soil_Type","Pesticide_Use_Category","Season","Crop_Damage"],\
                         prefix=["Crop","Soil","Pesticide","Season","Damage"], drop_first=False)
plt.figure(figsize=(15,12))
plt.title('Correlation of Features for Train Set')
sns.heatmap(df_train.astype(float).corr(),vmax=1.0,  annot=True)
plt.show()
df_train
df_train['Estimated_Insects_Count'] = df_train['Estimated_Insects_Count'].astype("int16")
df_train['Number_Doses_Week'] = df_train['Number_Doses_Week'].astype("int16")
df_train['Number_Weeks_Used'] = df_train['Number_Weeks_Used'].astype("int16")
df_train['Number_Weeks_Quit'] = df_train['Number_Weeks_Quit'].astype("int16")

#df_train['Estimated_Insects_Count']=np.log(df_train['Estimated_Insects_Count'])
num_cols = ['Estimated_Insects_Count','Number_Doses_Week','Number_Weeks_Used','Number_Weeks_Quit']
    
fig,ax = plt.subplots(4,1,figsize=(8,8),squeeze=False)
r=0
c=0
for i in num_cols:
    sns.distplot(df_train[i],ax=ax[r][c])
    r+=1
df_train.info()
df_train.head()
df_train['Estimated_Insects_Count']=np.log(df_train['Estimated_Insects_Count'])
#df_train['Number_Doses_Week1']=np.log(df_train['Number_Doses_Week'])
#df_train['Number_Weeks_Used1']=np.log(df_train['Number_Weeks_Used'])
#df_train['Number_Weeks_Quit1']=np.log(df_train['Number_Weeks_Quit'])

num_cols = ['Estimated_Insects_Count','Number_Doses_Week','Number_Weeks_Used','Number_Weeks_Quit']
    
fig,ax = plt.subplots(4,1,figsize=(8,8),squeeze=False)
r=0
c=0
for i in num_cols:
    sns.distplot(df_train[i],ax=ax[r][c])
    r+=1
df_train
train=pd.read_csv('/kaggle/input/train.csv')
train['Estimated_Insects_Count']=np.log(train['Estimated_Insects_Count'])
train.describe()
#Implementing  Encoding for some of the numeric columns
train['Number_Doses_Week_bin'] = np.where(train['Number_Doses_Week']>20,1,0)
train['Number_Weeks_Used_bin'] = np.where(train['Number_Weeks_Used']>36,1,0)
train['Number_Weeks_Quit_bin'] = np.where(train['Number_Weeks_Quit']>7,1,0)

#Implementing  Dummy Encoding for cate gorical columns
train = pd.get_dummies(train, columns=["Crop_Type","Soil_Type","Pesticide_Use_Category","Season"],\
                         prefix=["Crop","Soil","Pesticide","Season"], drop_first=False)

#train=train.drop(['Number_Doses_Week','Number_Weeks_Used','Number_Weeks_Quit'], axis=1)
train.info()
col=['Crop_0','Crop_1','Soil_0','Soil_1','Pesticide_1','Pesticide_2','Pesticide_3','Season_1','Season_2','Season_3']
for i in col:
    train[i] = train[i].astype('category')
    train[i] = train[i].cat.codes.astype("int16")
train=train.drop(['Number_Doses_Week','Number_Weeks_Used','Number_Weeks_Quit'],axis=1)
train=train.drop(['ID'],axis=1)
#Importing the auxiliar and preprocessing librarys 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import accuracy_score
#Models
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier, SGDClassifier, LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, BaggingClassifier, VotingClassifier, RandomTreesEmbedding

X= train.drop(['Crop_Damage'],axis=1)
y= train['Crop_Damage']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 22)
clfs = []
seed = 3

clfs.append(("LogReg", 
             Pipeline([("Scaler", StandardScaler()),
                       ("LogReg", LogisticRegression())])))

clfs.append(("XGBClassifier",
             Pipeline([("Scaler", StandardScaler()),
                       ("XGB", XGBClassifier())]))) 
clfs.append(("KNN", 
             Pipeline([("Scaler", StandardScaler()),
                       ("KNN", KNeighborsClassifier())]))) 

clfs.append(("DecisionTreeClassifier", 
             Pipeline([("Scaler", StandardScaler()),
                       ("DecisionTrees", DecisionTreeClassifier())]))) 

clfs.append(("RandomForestClassifier", 
             Pipeline([("Scaler", StandardScaler()),
                       ("RandomForest", RandomForestClassifier())]))) 

clfs.append(("GradientBoostingClassifier", 
             Pipeline([("Scaler", StandardScaler()),
                       ("GradientBoosting", GradientBoostingClassifier(max_features=15, n_estimators=150))]))) 

clfs.append(("RidgeClassifier", 
             Pipeline([("Scaler", StandardScaler()),
                       ("RidgeClassifier", RidgeClassifier())])))

clfs.append(("BaggingRidgeClassifier",
             Pipeline([("Scaler", StandardScaler()),
                       ("BaggingClassifier", BaggingClassifier())])))

clfs.append(("ExtraTreesClassifier",
             Pipeline([("Scaler", StandardScaler()),
                       ("ExtraTrees", ExtraTreesClassifier())])))

#'neg_mean_absolute_error', 'neg_mean_squared_error','r2'
scoring = 'accuracy'
n_folds = 10

results, names  = [], [] 

for name, model  in clfs:
    kfold = KFold(n_splits=n_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, y_train, cv= 5, scoring=scoring, n_jobs=-1)*100    
    names.append(name)
    results.append(cv_results)    
    #msg = "%s: %f (+/- %f)" % (name, cv_results.mean(),  cv_results.std())
    #print(msg)
    
# boxplot algorithm comparison
fig = plt.figure(figsize=(15,6))
fig.suptitle('Classifier Algorithm Comparison', fontsize=22)
ax = fig.add_subplot(111)
sns.boxplot(x=names, y=results)
ax.set_xticklabels(names)
ax.set_xlabel("Algorithmn", fontsize=20)
ax.set_ylabel("Accuracy of Models", fontsize=18)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
plt.show()
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.preprocessing import *
from sklearn.model_selection import *
from sklearn.metrics import *
import gc
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, mean_absolute_error,accuracy_score, classification_report
kfold = KFold(n_splits=10, random_state=7)
from lightgbm import LGBMClassifier
from sklearn import metrics
from sklearn.metrics import mean_squared_log_error
lgbc = LGBMClassifier(n_estimators=550,
                     learning_rate=0.03,
                     min_child_samples=40,
                     random_state=1,
                     colsample_bytree=0.5,
                     reg_alpha=2,
                     reg_lambda=2)

resultsLGB = cross_val_score(lgbc,X_train, y_train,cv=kfold)
print("LightGBM",resultsLGB.mean()*100)
LGB=lgbc.fit(X_train,y_train)
y_predict_LGBM = LGB.predict(X_test)
print(100*(np.sqrt(mean_squared_log_error(np.exp(y_test), np.exp(y_predict_LGBM)))))
resultsLGB_test = cross_val_score(lgbc,X_test, y_test,cv=kfold)
print("LightGBM",resultsLGB_test.mean()*100)


sorted(zip(LGB.feature_importances_, X_train), reverse = True)
from catboost import CatBoostRegressor 
from catboost import  CatBoostClassifier
from sklearn import metrics
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import roc_auc_score

#cb = CatBoostRegressor(
    #n_estimators = 1000,
    #learning_rate = 0.11,
    #iterations=1000,
    #loss_function = 'RMSE',
    #eval_metric = 'RMSE',
    #verbose=0)
    
cb= CatBoostClassifier(
    iterations=100, 
    learning_rate=0.1, 
    #loss_function='CrossEntropy'
)

#rmsle = 0
#for i in ratio:
 # x_train,y_train,x_val,y_val = train_test_split(i)

#CAT=cb.fit(X_train,y_train)
#resultsCAT = cross_val_score(cb,X_train, y_train,cv=kfold)
#print("CAT",resultsCAT.mean()*100)
                        
cb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50,early_stopping_rounds = 100)
kfold = KFold(n_splits=10, random_state=7)
resultsCAT = cross_val_score(cb,X_train, y_train,cv=kfold)
print("CAT",resultsCAT.mean()*100)

y_predict_CAT = cb.predict(X_train)
print(100*(np.sqrt(mean_squared_log_error(np.exp(y_train), np.exp(y_predict_CAT)))))
resultsCAT_train = cross_val_score(cb,X_train, y_train,cv=kfold)
print("CAT",resultsCAT_train.mean()*100)
test=pd.read_csv('/kaggle/input/test.csv')
test.info()

test.describe()
test.isnull().sum()
test['Number_Weeks_Used'].fillna(test['Number_Weeks_Used'].mode()[0], inplace=True)
test.isnull().sum()
submissiontest=test
test['Estimated_Insects_Count']=np.log(test['Estimated_Insects_Count'])
#Implementing  Encoding for some of the numeric columns
test['Number_Doses_Week_bin'] = np.where(test['Number_Doses_Week']>20,1,0)
test['Number_Weeks_Used_bin'] = np.where(test['Number_Weeks_Used']>36,1,0)
test['Number_Weeks_Quit_bin'] = np.where(test['Number_Weeks_Quit']>7,1,0)
test = pd.get_dummies(test, columns=["Crop_Type","Soil_Type","Pesticide_Use_Category","Season"],\
                         prefix=["Crop","Soil","Pesticide","Season"], drop_first=False)

test.info()
col=['Crop_0','Crop_1','Soil_0','Soil_1','Pesticide_1','Pesticide_2','Pesticide_3','Season_1','Season_2','Season_3']
for i in col:
    test[i] = test[i].astype('category')
    test[i] = test[i].cat.codes.astype("int16")
test.info()
test=test.drop(['ID'],axis=1)

test=test.drop(['Number_Doses_Week','Number_Weeks_Used','Number_Weeks_Quit'],axis=1)
y_predict_CAT_TEST = cb.predict(test)
y_predict_CAT_TEST
df_solution = pd.DataFrame()
df_solution['ID'] = submissiontest.ID
df_solution['Crop_Damage'] = y_predict_CAT_TEST
df_solution
df_solution.to_csv("CATBOOST_Implementation_Agriculture_Analytics_Submission.csv", index=False)
X=train.drop('Crop_Damage',axis=1)
y=train.Crop_Damage
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X)
from keras.utils import np_utils
dummy_y = np_utils.to_categorical(y)
dummy_y
X.shape
m=X.shape[1]
m
X_train
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot
# prepare sequence
X = X_train
y = dummy_y
# create model
model = Sequential()
model.add(Dense(20, input_dim=14, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# train model
NN = model.fit(X, y, epochs=30, batch_size=m, verbose=2)
# plot metrics
#pyplot.plot(NN.history['accuracy'])
#pyplot.show()
print(NN.history.keys())
#history = model.fit(X, y, epochs=30, batch_size=m, verbose=2)
# plot metrics
pyplot.plot(NN.history['accuracy'])
#pyplot.plot(history.history['loss'])
pyplot.show()
test
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
final_test = sc.fit_transform(test)
final_test
#predicting the Crop Damange by using Neural Network
y_predict_NN=model.predict(final_test)
df=pd.DataFrame(y_predict_NN)
# converting the NP array to Pandas data frame. Identifying the the colum which has maximum probability predicted
df.idxmax(axis=1)
df_solution_NN = pd.DataFrame()
df_solution_NN['ID'] = submissiontest.ID
df_solution_NN['Crop_Damage'] = df.idxmax(axis=1)
df_solution_NN

df_solution_NN.to_csv("Neural Network_Implementation_Agriculture_Analytics_Submission.csv", index=False)