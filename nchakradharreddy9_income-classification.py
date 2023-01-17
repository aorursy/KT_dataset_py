import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
#from pandas.plotiing import scatter_matrix
from sklearn.linear_model import LogisticRegression # logistic regression
from sklearn import svm # support vector machine
from sklearn.ensemble import RandomForestClassifier #Random_forest
from sklearn.tree import DecisionTreeClassifier #Decision tree
from sklearn.naive_bayes import GaussianNB #Naive_bayes
from sklearn.neighbors import KNeighborsClassifier #K nearest neighbors
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
income_df = pd.read_csv("/kaggle/input/adult-income-dataset/adult.csv")

income_df.info()
                        
income_df.head()
income_df.isin(['?']).sum(axis=0)
#Replacing the special character to nan and then drop the columns
income_df['native-country'] = income_df['native-country'].replace('?',np.nan)
income_df['workclass'] = income_df['workclass'].replace('?',np.nan)
income_df['occupation'] = income_df['occupation'].replace('?',np.nan)
#Dropping the NaN rows now 
income_df.dropna(how='any',inplace=True)
#once again checking for any symbols
income_df.isin(['?']).sum(axis=0)
#standard quantities in dataset
income_df.describe()
#converting the given dataset(age) into float type
income_df.age = income_df.age.astype(float)
income_df['hours-per-week'] = income_df['hours-per-week'].astype(float)
#storing the new dataset in my_df(data without dropped columns)
my_df = income_df.dropna()
my_df['predclass'] = my_df['income']
del my_df['income']
my_df['education-num'] = my_df['educational-num']
del my_df['educational-num']
my_df.info()
#once again checking for the symbols in newdata
my_df.isin(['?']).sum(axis=0)
#reading each column entry
print('workclass',my_df.workclass.unique())
print('education',my_df.education.unique())
print('marital-status',my_df['marital-status'].unique())
print('occupation',my_df.occupation.unique())
print('relationship',my_df.relationship.unique())
print('race',my_df.race.unique())
print('gender',my_df.gender.unique())
print('native-country',my_df['native-country'].unique())
print('predclass',my_df.predclass.unique())
#creating two classes '<=50k' and '>50k'
my_df.loc[income_df['income'] == ' >50K', 'predclass'] = 1
my_df.loc[income_df['income'] == ' <=50K', 'predclass'] = 0
#properly classifying the data with education column entries
my_df['education'].replace('Preschool', 'dropout',inplace=True)
my_df['education'].replace('10th', 'dropout',inplace=True)
my_df['education'].replace('11th', 'dropout',inplace=True)
my_df['education'].replace('12th', 'dropout',inplace=True)
my_df['education'].replace('1st-4th', 'dropout',inplace=True)
my_df['education'].replace('5th-6th', 'dropout',inplace=True)
my_df['education'].replace('7th-8th', 'dropout',inplace=True)
my_df['education'].replace('9th', 'dropout',inplace=True)
my_df['education'].replace('HS-Grad', 'HighGrad',inplace=True)
my_df['education'].replace('HS-grad', 'HighGrad',inplace=True)
my_df['education'].replace('Some-college', 'CommunityCollege',inplace=True)
my_df['education'].replace('Assoc-acdm', 'CommunityCollege',inplace=True)
my_df['education'].replace('Assoc-voc', 'CommunityCollege',inplace=True)
my_df['education'].replace('Bachelors', 'Bachelors',inplace=True)
my_df['education'].replace('Masters', 'Masters',inplace=True)
my_df['education'].replace('Prof-school', 'Masters',inplace=True)
my_df['education'].replace('Doctorate', 'Doctorate',inplace=True)
#similarly classifying martial status column entries
my_df['marital-status'].replace('Never-married', 'NotMarried',inplace=True)
my_df['marital-status'].replace(['Married-AF-spouse'], 'Married',inplace=True)
my_df['marital-status'].replace(['Married-civ-spouse'], 'Married',inplace=True)
my_df['marital-status'].replace(['Married-spouse-absent'], 'NotMarried',inplace=True)
my_df['marital-status'].replace(['Separated'], 'Separated',inplace=True)
my_df['marital-status'].replace(['Divorced'], 'Separated',inplace=True)
my_df['marital-status'].replace(['Widowed'], 'Widowed',inplace=True)
#Assigning the numeric values to the string type variables
number = LabelEncoder()
my_df['workclass'] = number.fit_transform(my_df['workclass'])
my_df['education'] = number.fit_transform(my_df['education'])
my_df['marital-status'] = number.fit_transform(my_df['marital-status'])
my_df['occupation'] = number.fit_transform(my_df['occupation'])
my_df['relationship'] = number.fit_transform(my_df['relationship'])
my_df['race'] = number.fit_transform(my_df['race'])
my_df['gender'] = number.fit_transform(my_df['gender'])
my_df['native-country'] = number.fit_transform(my_df['native-country'])
my_df['predclass'] = number.fit_transform(my_df['predclass'])
my_df['age_bin'] = pd.cut(my_df['age'], 20)
my_df['hours-per-week_bin'] = pd.cut(my_df['hours-per-week'], 10)
my_df['hours-per-week'] = my_df['hours-per-week']
my_df[['predclass', 'age']].groupby(['predclass'], as_index=False).mean().sort_values(by='age', ascending=False)
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split #training and testing data split
my_df = my_df.apply(LabelEncoder().fit_transform)
my_df.head()
my_df['age-hours'] = my_df['age']*my_df['hours-per-week']
my_df['age-hours_bin'] = pd.cut(my_df['age-hours'], 10)
drop_elements = ['education', 'native-country', 'predclass', 'age_bin', 'age-hours_bin','hours-per-week_bin']
y = my_df["predclass"]
X = my_df.drop(drop_elements, axis=1)
X.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,train_size=0.6, random_state=0)
#using standard scaler we can normalize the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns = X.columns)
X_train

from sklearn import linear_model
# Decision Tree
DT = DecisionTreeClassifier()
DT.fit(X_train,y_train)
y_pred = DT.predict(X_test)
score_DT = DT.score(X_test,y_test)
print("The accuracy of the Decision tree model is ",score_DT)
targets = ['<=50k' , '>50k']
print(classification_report(y_test, y_pred,target_names=targets))
# Gaussian Naive Bayes
GNB = GaussianNB()
GNB.fit(X_train, y_train)
y_pred = GNB.predict(X_test)
score_GNB = GNB.score(X_test,y_test)
print('The accuracy of Gaussian Naive Bayes model is', score_GNB)
targets = ['<=50k' , '>50k']
print(classification_report(y_test, y_pred,target_names=targets))
# K-Nearest Neighbors
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
score_knn = knn.score(X_test,y_test)
print('The accuracy of the KNN Model is',score_knn)
targets = ['<=50k' , '>50k']
print(classification_report(y_test, y_pred,target_names=targets))
# Support Vector Classifier (SVM/SVC)
from sklearn.svm import SVC
svc = SVC(gamma=0.22)
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
score_svc = svc.score(X_test,y_test)
print('The accuracy of SVC model is', score_svc)
targets = ['<=50k' , '>50k']
print(classification_report(y_test, y_pred,target_names=targets))
# Logistic Regression
LR = LogisticRegression()
LR.fit(X_train, y_train)
y_pred = LR.predict(X_test)
score_LR = LR.score(X_test,y_test)
print('The accuracy of the Logistic Regression model is', score_LR)
targets = ['<=50k' , '>50k']
print(classification_report(y_test, y_pred,target_names=targets))
# Random Forest Classifier
RF = RandomForestClassifier()
RF.fit(X_train, y_train)
y_pred = RF.predict(X_test)
score_RF = RF.score(X_test,y_test)
print('The accuracy of the Random Forest Model is', score_RF)
targets = ['<=50k' , '>50k']
print(classification_report(y_test, y_pred,target_names=targets))
tabular_form = {'CLASSIFICATION':['LogisticRegression','SupportVectorClassifier','RandomForestClassifier','DecisionTree','GaussianNaiveBayes','K-NearestNeighbors'],
                'ACCURACY':[score_LR,score_svc,score_RF,score_DT,score_GNB,score_knn]
                }
Tabular_form = pd.DataFrame(tabular_form,columns= ['CLASSIFICATION','ACCURACY'])
print(Tabular_form)
!pip install texttable
from texttable import Texttable
# texttable takes the first reocrd in the list as the column names
# of the table
l = [["CLASSIFICATION", "ACCURACY"],['LogisticRegression',score_LR],['SupportVectorClassifier',score_svc],['RandomForestClassifier',score_RF],['DecisionTree',score_DT],['GaussianNaiveBayes',score_GNB],['K-NearestNeighbors',score_knn]]
table = Texttable()
table.add_rows(l)
print(table.draw())