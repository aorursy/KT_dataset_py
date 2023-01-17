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
# This dataset is about heart disease, more specifically it's about coronary artery disease and angina. I will go through this dataset. First I want to test my Python skills. Second, I want to test some ideas with machine learning. 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
HD=pd.read_csv ('../input/heart.csv')
HD.shape
HD.sample(3)
# the dataset is very clean, which makes it easy to go next step.
HD.isnull().sum()
HD.info()
hd=HD.copy()
# from the data, we can see that there seems no clear 'outcome' information. So test several basic features and also learn the number meanings from data description.
print('cp values are', hd['cp'].unique())
print('thal values are', hd['thal'].unique())
print('restecg values are', hd['restecg'].unique())
print('ca values are', hd['ca'].unique())
# to make the column name more readable, such as 'target' means how narrow the coronary artery is.
hd=hd.rename(columns={'target': 'narrow', 'thal': 'duration', 'thalach':'max_HR', 'fbs':'high_fasting_sugar', 'exang':'exercise_angina','ca': 'coronary_num'})
hd.sample(3)
# Correlation test
sns.heatmap(hd.corr(),annot=True,cmap='RdYlGn') 
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show
# two factors correlation
fig, ax=plt.subplots( figsize=(6,6))
sns.regplot(x="cp", y="age", data=hd,)
sns.distplot(HD["ca"] )
# distribution. 
fig, ax=plt.subplots(1,3, figsize=(15,5))
sns.kdeplot(hd['age'], ax=ax[0])
sns.kdeplot(hd['trestbps'], ax=ax[1])
sns.kdeplot(hd['max_HR'], ax=ax[2])
from sklearn.ensemble import RandomForestClassifier
# I set chest pain as the disease outcome
x=hd.drop(['cp'] , axis=1)
y=hd['cp']
# look at which features are important for the 'Outcome'
model= RandomForestClassifier(n_estimators=100,random_state=0)
model.fit(x,y)
pd.Series(model.feature_importances_,index=x.columns).sort_values(ascending=False)

#remember that 'max_HR' has the highest score
# Since we already set the 'outcome', some supervised models are considered
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
Xtrain,Xtest, Ytrain, Ytest=train_test_split(x, y, test_size=0.2,random_state=42,stratify=y) # split data to train and test parts
# evaluate the accuracies of the models
abc=[]
classifiers=['Linear Svm','Radial Svm','Logistic Regression','KNN','Decision Tree']
models=[svm.SVC(kernel='linear'),svm.SVC(kernel='rbf'),LogisticRegression(),KNeighborsClassifier(n_neighbors=3),DecisionTreeClassifier()]
for i in models:
    model = i
    model.fit(Xtrain,Ytrain)
    prediction=model.predict(Xtest)
    abc.append(metrics.accuracy_score(prediction,Ytest))

models_dataframe=pd.DataFrame(abc,index=classifiers)   
models_dataframe
# Notice that the highest score is only 0.6. Based on the feature score list, to 8 features have scores above 0.5. To see whether these important features will produce more accurate prediction
hd_8=hd.loc[:, ['max_HR', 'chol', 'age', 'trestbps', 'oldpeak', 'narrow', 'exercise_angina', 'coronary_num']]
Xtrain,Xtest, Ytrain, Ytest=train_test_split(hd_8, y, test_size=0.2,random_state=42,stratify=y)
abc8=[]
classifiers=['Linear Svm','Radial Svm','Logistic Regression','KNN','Decision Tree']
models=[svm.SVC(kernel='linear'),svm.SVC(kernel='rbf'),LogisticRegression(),KNeighborsClassifier(n_neighbors=3),DecisionTreeClassifier()]
for i in models:
    model = i
    model.fit(Xtrain,Ytrain)
    prediction=model.predict(Xtest)
    abc8.append(metrics.accuracy_score(prediction,Ytest))

models_df8=pd.DataFrame(abc8,index=classifiers)   
models_df8
# surprisingly, top 8 important features are less accurate than the total features. How about the top 4?
hd_4=hd.loc[:, ['max_HR', 'chol', 'age', 'trestbps']]
Xtrain,Xtest, Ytrain, Ytest=train_test_split(hd_4, y, test_size=0.2,random_state=42,stratify=y)
abc4=[]
classifiers=['Linear Svm','Radial Svm','Logistic Regression','KNN','Decision Tree']
models=[svm.SVC(kernel='linear'),svm.SVC(kernel='rbf'),LogisticRegression(),KNeighborsClassifier(n_neighbors=3),DecisionTreeClassifier()]
for i in models:
    model = i
    model.fit(Xtrain,Ytrain)
    prediction=model.predict(Xtest)
    abc4.append(metrics.accuracy_score(prediction,Ytest))

models_df4=pd.DataFrame(abc4,index=classifiers)   
models_df4
# Even worse!
# Maybe standarlization could help?
from sklearn.preprocessing import StandardScaler
features=x[x.columns[:13]]
features_standard=StandardScaler().fit_transform(features)
hd_std=pd.DataFrame(features_standard,columns=[[  'age', 'sex', 'trestbps', 'chol', 'high_fasting_sugar', 'restecg', 'max_HR','exercise_angina', 'oldpeak', 'slope', 'coronary_num','duration','narrow' ]])

hd_std.head(1)
# try again with standarlizaed data
Xtrain,Xtest, Ytrain, Ytest=train_test_split(hd_std, y, test_size=0.2,random_state=42,stratify=y)
abc_std=[]
classifiers=['Linear Svm','Radial Svm','Logistic Regression','KNN','Decision Tree']
models=[svm.SVC(kernel='linear'),svm.SVC(kernel='rbf'),LogisticRegression(),KNeighborsClassifier(n_neighbors=3),DecisionTreeClassifier()]
for i in models:
    model = i
    model.fit(Xtrain,Ytrain)
    prediction=model.predict(Xtest)
    abc_std.append(metrics.accuracy_score(prediction,Ytest))

models_df_std=pd.DataFrame(abc_std,index=classifiers)   
models_df_std
hd_8std=hd_std.loc[:, ['max_HR', 'chol', 'age', 'trestbps', 'oldpeak', 'narrow', 'exercise_angina', 'coronary_num']]
Xtrain,Xtest, Ytrain, Ytest=train_test_split(hd_8std, y, test_size=0.2,random_state=42,stratify=y)
abc_8std=[]
classifiers=['Linear Svm','Radial Svm','Logistic Regression','KNN','Decision Tree']
models=[svm.SVC(kernel='linear'),svm.SVC(kernel='rbf'),LogisticRegression(),KNeighborsClassifier(n_neighbors=3),DecisionTreeClassifier()]
for i in models:
    model = i
    model.fit(Xtrain,Ytrain)
    prediction=model.predict(Xtest)
    abc_8std.append(metrics.accuracy_score(prediction,Ytest))

models_df_8std=pd.DataFrame(abc_8std,index=classifiers)   
models_df_8std
# For other models, it seems improve a little. For the LInear Svm, not better.
# So look back at the data, whether we miss something?
# For 'cp' column, it records chest pain type. Number 3 mean no chest pain, number 0-2 means different tyoe of angina. So is it possible that this kind of recording makes the problem too complicated?
# To simplify it, I will group the number0-2 together as disease positive, number 3 as disease negative
hd['cp'].isin([0,1,2]).value_counts()
# The dataset contain 23 ' disease negative' patient
hd_mod=HD.copy()
number=[0,1,2]
for col in hd.itertuples():

    if col.cp in number:
        hd_mod['cp'].replace(to_replace=col.cp, value=1, inplace=True)

hd_mod['cp'].value_counts()
y_mod=hd_mod['cp']
x.sample(3)
model= RandomForestClassifier(n_estimators=100,random_state=0)
model.fit(x,y_mod)
pd.Series(model.feature_importances_,index=x.columns).sort_values(ascending=False)
# Notice that after this change, the feature with highest score became 'the rest blood pressure', instead of 'max_HR'
hd_std.sample(3)
Xtrain,Xtest, Ytrain, Ytest=train_test_split(hd_std, y_mod, test_size=0.2,random_state=42,stratify=y)
abc_std=[]
classifiers=['Linear Svm','Radial Svm','Logistic Regression','KNN','Decision Tree']
models=[svm.SVC(kernel='linear'),svm.SVC(kernel='rbf'),LogisticRegression(),KNeighborsClassifier(n_neighbors=3),DecisionTreeClassifier()]
for i in models:
    model = i
    model.fit(Xtrain,Ytrain)
    prediction=model.predict(Xtest)
    abc_std.append(metrics.accuracy_score(prediction,Ytest))

models_df_std=pd.DataFrame(abc_std,index=classifiers)   
models_df_std
# Wow! The accuracy is more than 90%?!
# IS that too good to be true?
# we look at the percentage of so called 'disease positve' patient, after we grouped number 0-2 together
280/303
# the patient percentage is 92%. That means the model just keep saying the prediction is '1', it still has more than 90% to be correct...
# Let's do a test
hd.head(1)
model=svm.SVC(kernel='linear')
model.fit(Xtrain,Ytrain)
print(model.predict([[0.952197,
0.681005,
0.763956,
-0.256334,
2.394438,
-1.005832,
0.015443,
-0.696631,
1.087338,
-2.274579,
-0.714429,
-2.148873,
0.914529]]))
# Use the row 1 data to do the test and the answer should be 3, but the prediction is 1...as I thought.
# So go back to look at the data more carefully..
sns.violinplot(x=HD['cp'], y=HD['target'])
# Typical angina usually have less than 50% narrow artery.
sns.violinplot(x=HD['cp'], y=HD['thalach'])
sns.violinplot(x=HD['cp'], y=HD['age'])
# look at data with 'cp' grouped
sns.violinplot(x=hd_mod['cp'], y=hd_mod['chol'])
sns.violinplot(x=hd_mod['cp'], y=hd_mod['thalach'])
sns.violinplot(x=hd_mod['cp'], y=hd_mod['ca'])
# first check the value of 'narrow' column
hd['narrow'].value_counts()
hd_narrow=HD.copy()
hd_narrow=hd_narrow.rename(columns={'target': 'narrow','pred_attribute': 'narrow', 'thal': 'duration', 'thalach':'max_HR', 'fbs':'high_fasting_sugar', 'exang':'exercise_angina','ca': 'coronary_num'})
x_narrow=hd_narrow.drop(['narrow'] , axis=1)
y_narrow=hd_narrow['narrow']
model= RandomForestClassifier(n_estimators=100,random_state=0)
model.fit(x_narrow,y_narrow)
pd.Series(model.feature_importances_,index=x_narrow.columns).sort_values(ascending=False)
Xtrain_n,Xtest_n, Ytrain_n, Ytest_n=train_test_split(x_narrow, y_narrow, test_size=0.2,random_state=42,stratify=y_narrow)
abc_narrow=[]
classifiers=['Linear Svm','Radial Svm','Logistic Regression','KNN','Decision Tree']
models=[svm.SVC(kernel='linear'),svm.SVC(kernel='rbf'),LogisticRegression(),KNeighborsClassifier(n_neighbors=3),DecisionTreeClassifier()]
for i in models:
    model = i
    model.fit(Xtrain_n,Ytrain_n)
    prediction=model.predict(Xtest_n)
    abc_narrow.append(metrics.accuracy_score(prediction,Ytest_n))

models_df_narrow=pd.DataFrame(abc_narrow,index=classifiers, columns=['Score'])   
models_df_narrow
model=svm.SVC(kernel='linear')
model.fit(Xtrain_n,Ytrain_n)
print(model.predict([[61,
1,
0,
140,
207,
0,
0,
138,
1,
1.9,
2,
1,
3]]))
model=svm.SVC(kernel='linear')
model.fit(Xtrain_n,Ytrain_n)
print(model.predict([[43,
1,
0,
110,
211,
0,
1,
161,
0,
0.0,
2,
0,
3]]))
hd_8nar=hd_narrow.loc[:, ['max_HR', 'chol', 'age', 'trestbps', 'oldpeak', 'cp', 'duration', 'coronary_num']]
Xtrain,Xtest, Ytrain, Ytest=train_test_split(hd_8nar, y_narrow, test_size=0.2,random_state=42,stratify=y_narrow)
abc_8nar=[]
classifiers=['Linear Svm','Radial Svm','Logistic Regression','KNN','Decision Tree']
models=[svm.SVC(kernel='linear'),svm.SVC(kernel='rbf'),LogisticRegression(),KNeighborsClassifier(n_neighbors=3),DecisionTreeClassifier()]
for i in models:
    model = i
    model.fit(Xtrain,Ytrain)
    prediction=model.predict(Xtest)
    abc_8nar.append(metrics.accuracy_score(prediction,Ytest))

models_df_8nar=pd.DataFrame(abc_8nar,index=classifiers, columns=['Scores'])   
models_df_8nar
fig, ax=plt.subplots(1, 4, figsize=(25,5))
sns.violinplot(x="narrow", y="cp", data=hd_narrow, ax=ax[0])
ax[0].set_xlabel('narrow', fontsize=20)
ax[0].set_ylabel('cp', fontsize=20)
sns.violinplot(x='narrow', y='oldpeak', data=hd_narrow, ax=ax[1])
ax[1].set_xlabel('narrow', fontsize=20)
ax[1].set_ylabel('ST depression', fontsize=20)
sns.violinplot(x='narrow', y='duration', data=hd_narrow, ax=ax[2])
ax[2].set_xlabel('narrow', fontsize=20)
ax[2].set_ylabel('duration of excercise', fontsize=20)
sns.violinplot(x='narrow', y='max_HR', data=hd_narrow, ax=ax[3])
ax[3].set_xlabel('narrow', fontsize=20)
ax[3].set_ylabel('max_HR', fontsize=20)
# I want to thank Kaggle for providing the wonderful place to learn and to communicate. Many code in here are learned from other kaggle members, such as I,coder.
