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
from sklearn import linear_model,metrics,preprocessing
from sklearn import ensemble,tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
df1= pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
df1.head()
#we map the quality values from 0 to 5
qual_map={
    3:0,
    4:1,
    5:2,
    6:3,
    7:4,
    8:5
}
df1.loc[:,"quality"]=df1.quality.map(qual_map)
#For our convenience we have mapped the quality indexes from 3-8 to 0-5, and shall treat this as a multi-class 
#classification problem.
df1

#Naively splitting data without visualization 
#using dataframe.sample for random sampling of the data.We will reset the indices,as they change after 
#shuffling the data.

df1= df1.sample(frac=1).reset_index(drop=True)

#top 1000 out of 1600 rows are our training data.
df1_train=df1.head(1000)

#bottom 600 out of 1600 rows are our testing data.
df1_test=df1.tail(600)



df1_train
df1_test
#Now as this problem can be treated as a classification task, we train this using Decision Tree model
import sklearn
from sklearn import tree
from sklearn import metrics

classifier= tree.DecisionTreeClassifier(max_depth=3)

cols=['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']
classifier.fit(df1_train[cols],df1_train.quality)
train_predictions= classifier.predict(df1_train[cols])

test_predictions= classifier.predict(df1_test[cols])
train_accuracy= metrics.accuracy_score(df1_train.quality,train_predictions)

test_accuracy= metrics.accuracy_score(df1_test.quality,test_predictions)

train_accuracy
test_accuracy
#Importing libraries for visualization.
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
#Global size of texts on the plots.
matplotlib.rc('xtick', labelsize=15)
matplotlib.rc('ytick', labelsize=15)

#To ensure that plot is displayed inside the notebook itself
%matplotlib inline
#initializing the list to store the training & testing accuracies, beginning from
#50%

train_accuracies=[0.5]
test_accuracies=[0.5]

#looping over some values of max_depths

for d in range(1,25):
    classifier1= tree.DecisionTreeClassifier(max_depth=d)
    classifier1.fit(df1_train[cols],df1_train.quality)
    
    train_predictions=classifier1.predict(df1_train[cols])
    test_predictions=classifier1.predict(df1_test[cols])
    
    train_acc= metrics.accuracy_score(df1_train.quality,train_predictions)
    test_acc = metrics.accuracy_score(df1_test.quality,test_predictions)
    
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)
    
    

#creating plots using matplotlib and seaborn
plt.figure(figsize=(12,5))
sns.set_style("whitegrid")
plt.plot(train_accuracies,label="Training Accuracy")
plt.plot(test_accuracies,label="Testing Accuracy")

plt.legend(loc="upper left", prop={'size':15})
plt.xticks(range(0,26,5))
plt.xlabel("max_depth",size=20)
plt.ylabel("accuracy",size=20)
plt.show()
print("The train accuracies are given as :")
print(train_accuracies)
print("The corresponding testing accuracies are given as :")
print(test_accuracies)

plt.figure(figsize=(15,10))
sns.heatmap(df1.corr(),annot=True)
plt.show()
#Plot a boxplot to check for Outliers
#Target variable is Quality. So will plot a boxplot each column against target variable
sns.boxplot('quality', 'fixed acidity', data = df1)
sns.boxplot('quality', 'volatile acidity', data = df1)
sns.boxplot('quality', 'citric acid', data = df1)
sns.boxplot('quality', 'chlorides', data = df1)
sns.boxplot('quality', 'free sulfur dioxide', data = df1)
sns.boxplot('quality', 'total sulfur dioxide', data = df1)
sns.boxplot('quality', 'density', data = df1)
sns.boxplot('quality', 'pH', data = df1)
sns.boxplot('quality', 'sulphates', data = df1)
sns.boxplot('quality', 'alcohol', data = df1)
#To ostify our arguement lets look at the distribution of quality labels in the given data.
q= sns.countplot(x='quality',data=df1)
q.set_xlabel("quality", fontsize=20)
q.set_ylabel("count", fontsize=20)


df2= pd.read_csv('/kaggle/input/folds-winequality/Train_folds_winequality-red.csv')
def help_Decision_tree(fold):
    df2= pd.read_csv('/kaggle/input/folds-winequality/Train_folds_winequality-red.csv')
    qual_map={
    3:0,
    4:1,
    5:2,
    6:3,
    7:4,
    8:5
    }
    df2.loc[:,"quality"]=df2.quality.map(qual_map)
    features=[
        f for f in df2.columns if f not in("quality","kfold")
        ]
    for col in features:
        df2.loc[:,col]=df2[col].astype(str).fillna("NONE")
    
    df2_train=df2[df2.kfold!=fold].reset_index(drop=True)
    df2_valid=df2[df2.kfold==fold].reset_index(drop=True)
    
    
    x_train= df2_train.drop("quality",axis=1).values
    y_train= df2_train.quality.values
    
    x_valid= df2_valid.drop("quality",axis=1).values
    y_valid= df2_valid.quality.values
    
    model= tree.DecisionTreeClassifier()
    model.fit(x_train,y_train)
    
    valid_preds=model.predict(x_valid)
    accur= metrics.accuracy_score(y_valid,valid_preds)
    print(f"Fold = {fold}, Accuracy = {accur}")
    
for i in range(0,5):
    print(help_Decision_tree(i))
    print()
def help_Random_Forest(fold):
    df2= pd.read_csv('/kaggle/input/folds-winequality/Train_folds_winequality-red.csv')
    qual_map={
    3:0,
    4:1,
    5:2,
    6:3,
    7:4,
    8:5
    }
    df2.loc[:,"quality"]=df2.quality.map(qual_map)
    features=[
        f for f in df2.columns if f not in("quality","kfold")
        ]
    for col in features:
        df2.loc[:,col]=df2[col].astype(str).fillna("NONE")
    
    df2_train=df2[df2.kfold!=fold].reset_index(drop=True)
    df2_valid=df2[df2.kfold==fold].reset_index(drop=True)
    
    
    x_train= df2_train.drop("quality",axis=1).values
    y_train= df2_train.quality.values
    
    x_valid= df2_valid.drop("quality",axis=1).values
    y_valid= df2_valid.quality.values
    
    model2= RandomForestClassifier()
    model2.fit(x_train,y_train)
    
    valid_preds=model2.predict(x_valid)
    accur= metrics.accuracy_score(y_valid,valid_preds)
    print(f"Fold = {fold}, Accuracy = {accur}")
    
for i in range(0,5):
    print(help_Random_Forest(i))
    print()
def help_svc(fold):
    df2= pd.read_csv('/kaggle/input/folds-winequality/Train_folds_winequality-red.csv')
    qual_map={
    3:0,
    4:1,
    5:2,
    6:3,
    7:4,
    8:5
    }
    df2.loc[:,"quality"]=df2.quality.map(qual_map)
    features=[
        f for f in df2.columns if f not in("quality","kfold")
        ]
    for col in features:
        df2.loc[:,col]=df2[col].astype(str).fillna("NONE")
    
    df2_train=df2[df2.kfold!=fold].reset_index(drop=True)
    df2_valid=df2[df2.kfold==fold].reset_index(drop=True)
    
    
    x_train= df2_train.drop("quality",axis=1).values
    y_train= df2_train.quality.values
    
    x_valid= df2_valid.drop("quality",axis=1).values
    y_valid= df2_valid.quality.values
    
    model3= SVC()
    model3.fit(x_train,y_train)
    
    valid_preds=model3.predict(x_valid)
    accur= metrics.accuracy_score(y_valid,valid_preds)
    print(f"Fold = {fold}, Accuracy = {accur}")
    
for i in range(0,5):
    print(help_svc(i))
    print()
def help_mlp(fold):
    df2= pd.read_csv('/kaggle/input/folds-winequality/Train_folds_winequality-red.csv')
    qual_map={
    3:0,
    4:1,
    5:2,
    6:3,
    7:4,
    8:5
    }
    df2.loc[:,"quality"]=df2.quality.map(qual_map)
    features=[
        f for f in df2.columns if f not in("quality","kfold")
        ]
    for col in features:
        df2.loc[:,col]=df2[col].astype(str).fillna("NONE")
    
    df2_train=df2[df2.kfold!=fold].reset_index(drop=True)
    df2_valid=df2[df2.kfold==fold].reset_index(drop=True)
    
    
    x_train= df2_train.drop("quality",axis=1).values
    y_train= df2_train.quality.values
    
    x_valid= df2_valid.drop("quality",axis=1).values
    y_valid= df2_valid.quality.values
    
    model4= MLPClassifier()
    model4.fit(x_train,y_train)
    
    valid_preds=model4.predict(x_valid)
    accur= metrics.accuracy_score(y_valid,valid_preds)
    print(f"Fold = {fold}, Accuracy = {accur}")
    
for i in range(0,5):
    print(help_mlp(i))
    print()
def help_gbc(fold):
    df2= pd.read_csv('/kaggle/input/folds-winequality/Train_folds_winequality-red.csv')
    qual_map={
    3:0,
    4:1,
    5:2,
    6:3,
    7:4,
    8:5
    }
    df2.loc[:,"quality"]=df2.quality.map(qual_map)
    features=[
        f for f in df2.columns if f not in("quality","kfold")
        ]
    for col in features:
        df2.loc[:,col]=df2[col].astype(str).fillna("NONE")
    
    df2_train=df2[df2.kfold!=fold].reset_index(drop=True)
    df2_valid=df2[df2.kfold==fold].reset_index(drop=True)
    
    
    x_train= df2_train.drop("quality",axis=1).values
    y_train= df2_train.quality.values
    
    x_valid= df2_valid.drop("quality",axis=1).values
    y_valid= df2_valid.quality.values
    
    model5= GradientBoostingClassifier()
    model5.fit(x_train,y_train)
    
    valid_preds=model5.predict(x_valid)
    accur= metrics.accuracy_score(y_valid,valid_preds)
    print(f"Fold = {fold}, Accuracy = {accur}")
    
for i in range(0,5):
    print(help_gbc(i))
    print()
def help_Knn(fold):
    df2= pd.read_csv('/kaggle/input/folds-winequality/Train_folds_winequality-red.csv')
    qual_map={
    3:0,
    4:1,
    5:2,
    6:3,
    7:4,
    8:5
    }
    df2.loc[:,"quality"]=df2.quality.map(qual_map)
    features=[
        f for f in df2.columns if f not in("quality","kfold")
        ]
    for col in features:
        df2.loc[:,col]=df2[col].astype(str).fillna("NONE")
    
    df2_train=df2[df2.kfold!=fold].reset_index(drop=True)
    df2_valid=df2[df2.kfold==fold].reset_index(drop=True)
    
    
    x_train= df2_train.drop("quality",axis=1).values
    y_train= df2_train.quality.values
    
    x_valid= df2_valid.drop("quality",axis=1).values
    y_valid= df2_valid.quality.values
    
    model6= KNeighborsClassifier()
    model6.fit(x_train,y_train)
    
    valid_preds=model6.predict(x_valid)
    accur= metrics.accuracy_score(y_valid,valid_preds)
    print(f"Fold = {fold}, Accuracy = {accur}")
    
for i in range(0,5):
    print(help_Knn(i))
    print()
def help_ExtraTreeClassifier(fold):
    df2= pd.read_csv('/kaggle/input/folds-winequality/Train_folds_winequality-red.csv')
    qual_map={
    3:0,
    4:1,
    5:2,
    6:3,
    7:4,
    8:5
    }
    df2.loc[:,"quality"]=df2.quality.map(qual_map)
    features=[
        f for f in df2.columns if f not in("quality","kfold")
        ]
    for col in features:
        df2.loc[:,col]=df2[col].astype(str).fillna("NONE")
    
    df2_train=df2[df2.kfold!=fold].reset_index(drop=True)
    df2_valid=df2[df2.kfold==fold].reset_index(drop=True)
    
    
    x_train= df2_train.drop("quality",axis=1).values
    y_train= df2_train.quality.values
    
    x_valid= df2_valid.drop("quality",axis=1).values
    y_valid= df2_valid.quality.values
    
    model7= ExtraTreeClassifier()
    model7.fit(x_train,y_train)
    
    valid_preds=model7.predict(x_valid)
    accur= metrics.accuracy_score(y_valid,valid_preds)
    print(f"Fold = {fold}, Accuracy = {accur}")
    
for i in range(0,5):
    print(help_ExtraTreeClassifier(i))
    print()





