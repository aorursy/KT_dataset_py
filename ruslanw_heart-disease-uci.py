# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns



from datetime import datetime



from sklearn.model_selection import GridSearchCV,train_test_split,cross_val_score

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.svm import SVC

from sklearn.metrics import roc_curve, auc

import os

import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/heart.csv')
print('Data First 5 Rows Show\n')

data.head()
print('Data Last 5 Rows Show\n')

data.tail()
print('Data columns\n')

data.columns
print('Data Show Describe\n')

data.describe()
print('Data Show Info\n')

data.info()
data.sample(frac=0.01)
data=data.rename(columns={'age':'Age','sex':'Sex','cp':'Cp','trestbps':'Trestbps','chol':'Chol','fbs':'Fbs','restecg':'Restecg','thalach':'Thalach','exang':'Exang','oldpeak':'Oldpeak','slope':'Slope','ca':'Ca','thal':'Thal','target':'Target'})
#And, how many rows and columns are there for all data?

print('Data Shape Show\n')

data.shape  #first one is rows, other is columns
print('Data Sum of Null Values \n')

data.isnull().sum()
#all rows control for null values

data.isnull().values.any()
plt.figure(figsize=(10,10))

sns.heatmap(data.corr(),annot=True,fmt='.1f')

plt.show()
plt.figure(figsize=(10,10))

sns.heatmap(data.corr(),vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5},annot=True)

plt.tight_layout()

plt.show()
data.drop('Target', axis=1).corrwith(data.Target).plot(kind='bar', grid=True, figsize=(12, 8), title="Correlation with target")
data.Age.value_counts()[:10]

#data age show value counts for age least 10
sns.barplot(x=data.Age.value_counts()[:10].index,y=data.Age.value_counts()[:10].values)

plt.xlabel('Age')

plt.ylabel('Age Counter')

plt.title('Age Analysis System')

plt.show()
#firstly find min and max ages

minAge=min(data.Age)

maxAge=max(data.Age)

meanAge=data.Age.mean()



print('Min Age :',minAge)

print('Max Age :',maxAge)

print('Mean Age :',meanAge)
young_ages=data[(data.Age>=29)&(data.Age<40)]

middle_ages=data[(data.Age>=40)&(data.Age<55)]

elderly_ages=data[(data.Age>55)]



print('Young Ages :',len(young_ages))

print('Middle Ages :',len(middle_ages))

print('Elderly Ages :',len(elderly_ages))
sns.barplot(x=['young ages','middle ages','elderly ages'],y=[len(young_ages),len(middle_ages),len(elderly_ages)])

plt.xlabel('Age Range')

plt.ylabel('Age Counts')

plt.title('Ages State in Dataset')

plt.show()
data['AgeRange']=0

youngAge_index=data[(data.Age>=29)&(data.Age<40)].index

middleAge_index=data[(data.Age>=40)&(data.Age<55)].index

elderlyAge_index=data[(data.Age>55)].index
for index in elderlyAge_index:

    data.loc[index,'AgeRange']=2

    

for index in middleAge_index:

    data.loc[index,'AgeRange']=1



for index in youngAge_index:

    data.loc[index,'AgeRange']=0
# Draw a categorical scatterplot to show each observation

sns.swarmplot(x="AgeRange", y="Age",hue='Sex', palette=["r", "c", "y"], data=data)

plt.show()
# Plot the total crashes

sns.set_color_codes("pastel")

sns.barplot(y="AgeRange", x="Sex", data=data, label="Total", color="b")

plt.show()
sns.countplot(elderly_ages.Sex)

plt.title("Elderly Sex Operations")

plt.show()
elderly_ages.groupby(elderly_ages['Sex'])['Thalach'].agg('sum')
sns.violinplot(data.Age, palette="Set3", bw=.2, cut=1, linewidth=1)

plt.xticks(rotation=45)

plt.title("Age Rates")

plt.show()
plt.figure(figsize=(15,7))

sns.violinplot(x=data.Age,y=data.Target)

plt.xticks(rotation=45)

plt.legend()

plt.title("Age & Target System")

plt.show()
colors = ['blue','green','yellow']

explode = [0,0,0.1]

plt.figure(figsize = (5,5))



plt.pie([len(young_ages),len(middle_ages),len(elderly_ages)], labels=['young ages','middle ages','elderly ages'], explode=explode, colors=colors, autopct='%1.1f%%')

plt.title('Age States', color = 'blue', fontsize = 15)

plt.show()
data.Sex.value_counts()
#Sex (1 = male; 0 = female)

sns.countplot(data.Sex)

plt.show()
sns.countplot(data.Sex,hue=data.Slope)

plt.title('Slope & Sex Rates Show')

plt.show()
total_genders_count=len(data.Sex)

male_count=len(data[data['Sex']==1])

female_count=len(data[data['Sex']==0])



print('Total Genders :',total_genders_count)

print('Male Count    :',male_count)

print('Female Count  :',female_count)
#Percentage ratios

print("Male State: {:.2f}%".format((male_count / (total_genders_count)*100)))

print("Female State: {:.2f}%".format((female_count / (total_genders_count)*100)))
#Male State & target 1 & 0

male_andtarget_on=len(data[(data.Sex==1)&(data['Target']==1)])

male_andtarget_off=len(data[(data.Sex==1)&(data['Target']==0)])





sns.barplot(x=['Male Target On','Male Target Off'],y=[male_andtarget_on,male_andtarget_off])

plt.xlabel('Male and Target State')

plt.ylabel('Count')

plt.title('State of the Gender')

plt.show()
#Female State & target 1 & 0

female_andtarget_on=len(data[(data.Sex==0)&(data['Target']==1)])

female_andtarget_off=len(data[(data.Sex==0)&(data['Target']==0)])





sns.barplot(x=['Female Target On','Female Target Off'],y=[female_andtarget_on,female_andtarget_off])

plt.xlabel('Female and Target State')

plt.ylabel('Count')

plt.title('State of the Gender')

plt.show()
# Plot miles per gallon against horsepower with other semantics

sns.relplot(x="Trestbps", y="Age",

            sizes=(40, 400), alpha=.5, palette="muted",

            height=6, data=data)
#As seen, there are 4 types of chest pain.

data.Cp.value_counts()
sns.countplot(data.Cp)

plt.xlabel('Chest Type')

plt.ylabel('Count')

plt.title('Chest Type vs Count State')

plt.show()



# 0 status at least

# 1 condition slightly distressed

# 2 condition medium problem

# 3 condition too bad
cp_zero_target_zero=len(data[(data.Cp==0)&(data.Target==0)])

cp_zero_target_one=len(data[(data.Cp==0)&(data.Target==1)])
sns.barplot(x=['cp_zero_target_zero','cp_zero_target_one'],y=[cp_zero_target_zero,cp_zero_target_one])

plt.show()
cp_one_target_zero=len(data[(data.Cp==1)&(data.Target==0)])

cp_one_target_one=len(data[(data.Cp==1)&(data.Target==1)])
sns.barplot(x=['cp_one_target_zero','cp_one_target_one'],y=[cp_one_target_zero,cp_one_target_one])

plt.show()
cp_two_target_zero=len(data[(data.Cp==2)&(data.Target==0)])

cp_two_target_one=len(data[(data.Cp==2)&(data.Target==1)])
sns.barplot(x=['cp_two_target_zero','cp_two_target_one'],y=[cp_two_target_zero,cp_two_target_one])

plt.show()
cp_three_target_zero=len(data[(data.Cp==3)&(data.Target==0)])

cp_three_target_one=len(data[(data.Cp==3)&(data.Target==1)])
sns.barplot(x=['cp_three_target_zero','cp_three_target_one'],y=[cp_three_target_zero,cp_three_target_one])

plt.show()
# Show the results of a linear regression within each dataset

sns.lmplot(x="Trestbps", y="Chol",data=data,hue="Cp")

plt.show()
sns.barplot(x=data.Thalach.value_counts()[:20].index,y=data.Thalach.value_counts()[:20].values)

plt.xlabel('Thalach')

plt.ylabel('Count')

plt.title('Thalach Counts')

plt.xticks(rotation=45)

plt.show()
age_unique=sorted(data.Age.unique())

age_thalach_values=data.groupby('Age')['Thalach'].count().values

mean_thalach=[]

for i,age in enumerate(age_unique):

    mean_thalach.append(sum(data[data['Age']==age].Thalach)/age_thalach_values[i])
plt.figure(figsize=(10,5))

sns.pointplot(x=age_unique,y=mean_thalach,color='red',alpha=0.8)

plt.xlabel('Age',fontsize = 15,color='blue')

plt.xticks(rotation=45)

plt.ylabel('Thalach',fontsize = 15,color='blue')

plt.title('Age vs Thalach',fontsize = 15,color='blue')

plt.grid()

plt.show()
for i,col in enumerate(data.columns.values):

    plt.subplot(5,3,i+1)

    plt.scatter([i for i in range(303)],data[col].values.tolist())

    plt.title(col)

    fig,ax=plt.gcf(),plt.gca()

    fig.set_size_inches(10,10)

    plt.tight_layout()

plt.show()
#Let's see how the correlation values between them

data.corr()
X=data.drop('Target',axis=1)

y=data['Target']
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print('X_train', X_train.shape)

print('X_test', X_test.shape)

print('y_train', y_train.shape)

print('y_test', y_test.shape)
def print_score(clf, X_train, y_train, X_test, y_test, train=True):

    if train:

        pred = clf.predict(X_train)

        print("Train Result:\n================================================")

        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")

        print("_______________________________________________")

        print("Classification Report:", end='')

        print(f"\tPrecision Score: {precision_score(y_train, pred) * 100:.2f}%")

        print(f"\t\t\tRecall Score: {recall_score(y_train, pred) * 100:.2f}%")

        print(f"\t\t\tF1 score: {f1_score(y_train, pred) * 100:.2f}%")

        print("_______________________________________________")

        print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")

        

    elif train==False:

        pred = clf.predict(X_test)

        print("Test Result:\n================================================")        

        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")

        print("_______________________________________________")

        print("Classification Report:", end='')

        print(f"\tPrecision Score: {precision_score(y_test, pred) * 100:.2f}%")

        print(f"\t\t\tRecall Score: {recall_score(y_test, pred) * 100:.2f}%")

        print(f"\t\t\tF1 score: {f1_score(y_test, pred) * 100:.2f}%")

        print("_______________________________________________")

        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")
log_reg = LogisticRegression(solver='liblinear')

log_reg.fit(X_train, y_train)
print_score(log_reg, X_train, y_train, X_test, y_test, train=True)

print_score(log_reg, X_train, y_train, X_test, y_test, train=False)
test_score = accuracy_score(y_test, log_reg.predict(X_test)) * 100

train_score = accuracy_score(y_train, log_reg.predict(X_train)) * 100



results_df = pd.DataFrame(data=[["Logistic Regression", train_score, test_score]], columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])

results_df
knn_classifier = KNeighborsClassifier()

knn_classifier.fit(X_train, y_train)



print_score(knn_classifier, X_train, y_train, X_test, y_test, train=True)

print_score(knn_classifier, X_train, y_train, X_test, y_test, train=False)
knn_classifier = KNeighborsClassifier()

knn_classifier.fit(X_train, y_train)
test_score = accuracy_score(y_test, knn_classifier.predict(X_test)) * 100

train_score = accuracy_score(y_train, knn_classifier.predict(X_train)) * 100



results_df_2 = pd.DataFrame(data=[["K-nearest neighbors", train_score, test_score]], columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])

results_df = results_df.append(results_df_2, ignore_index=True)

results_df
svm_model = SVC(kernel='rbf', gamma=0.1, C=1.0)

svm_model.fit(X_train, y_train)
print_score(svm_model, X_train, y_train, X_test, y_test, train=True)

print_score(svm_model, X_train, y_train, X_test, y_test, train=False)
test_score = accuracy_score(y_test, svm_model.predict(X_test)) * 100

train_score = accuracy_score(y_train, svm_model.predict(X_train)) * 100



results_df_2 = pd.DataFrame(data=[["Support Vector Machine", train_score, test_score]], columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])

results_df = results_df.append(results_df_2, ignore_index=True)

results_df
tree = DecisionTreeClassifier(random_state=42)

tree.fit(X_train, y_train)



print_score(tree, X_train, y_train, X_test, y_test, train=True)

print_score(tree, X_train, y_train, X_test, y_test, train=False)
train_score = accuracy_score(y_train, tree.predict(X_train)) * 100



results_df_2 = pd.DataFrame(data=[["Decision Tree Classifier", train_score, test_score]], columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])

results_df = results_df.append(results_df_2, ignore_index=True)

results_df
from xgboost import XGBClassifier



xgb = XGBClassifier()

xgb.fit(X_train, y_train)



print_score(xgb, X_train, y_train, X_test, y_test, train=True)

print_score(xgb, X_train, y_train, X_test, y_test, train=False)
test_score = accuracy_score(y_test, xgb.predict(X_test)) * 100

train_score = accuracy_score(y_train, xgb.predict(X_train)) * 100



results_df_2 = pd.DataFrame(data=[["XGBoost Classifier", train_score, test_score]], columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])

results_df = results_df.append(results_df_2, ignore_index=True)

results_df