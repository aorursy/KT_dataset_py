# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

#2.4.1 Decision Tree
#2.4.2 Naïve Bayes
#2.4.3 Neural Network
#    แบ่งชุดข้อมูลเป็น 5-fold cross validation
#2.7 แสดงผลลัพท์การจําแนกในรูปแบบของ
#2.7.1 Recall ของแต่ละ class
#2.7.2 Precision ของแต่ละ class
#2.7.3 F-Measure ของแต่ละ class
#2.7.4 Average F-Measure ของทั้งชุดข้อมูล

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_train = pd.read_csv ("/kaggle/input/titanic/train.csv")
df_test = pd.read_csv("/kaggle/input/titanic/test.csv")
df_gender = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
#Checking data : df_train
sns.heatmap(df_train.isnull(),yticklabels=False,cbar=False,cmap='bwr')
#Checking data : df_test
sns.heatmap(df_test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#Cleanning data for train data
plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=df_train,palette='winter')
#Cleanning data for test data
plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=df_test,palette='winter')
def average_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age
#Fill the average age to all missing age data for train data
df_train['Age'] = df_train[['Age','Pclass']].apply(average_age,axis=1)
#Fill the average age to all missing age data for test data
df_test['Age'] = df_test[['Age','Pclass']].apply(average_age,axis=1)
#Check age data for train data
sns.heatmap(df_train.isnull(),yticklabels=False,cbar=False,cmap='bwr')
#Check age data for test data
sns.heatmap(df_test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
df_train.drop('Cabin',axis=1,inplace=True)
#Check age data for train data
sns.heatmap(df_train.isnull(),yticklabels=False,cbar=False,cmap='bwr')
df_test.drop('Cabin',axis=1,inplace=True)
#Check age data for test data
sns.heatmap(df_test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
df_train
df_test
df_gender
def features_data(data):
  features = data[['Pclass', 'Sex', 'Age','SibSp', 'Parch']].copy()
  features['Age'].fillna(features['Age'].mean(),inplace = True)
  features[['Sex']] = features[['Sex']].astype('category')
  features = pd.get_dummies(features, columns=["Sex"], prefix=["ohe"] ) 
  return features.values
def evaluate_model(model,X_train,X_test,y_train,y_test):
  model.fit(X_train,y_train)
  y_pred = model.predict(X_test)
  tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()
  #Recall
  recall = tp/(tp+fn)
  #Precision
  precision = tp / (tp+fp)
  #F-Measure
  f = 2*precision*recall/(precision+recall)
  print(type(model).__name__)
  print('Recall : {}'.format(recall))
  print('Precision : {}'.format(precision))
  print('F : {}'.format(f)) 
  return recall,precision,f
decisiontree = DecisionTreeClassifier()
bayes = GaussianNB()
neuralnetwork = MLPClassifier(hidden_layer_sizes=(100,100), max_iter=500,activation='logistic')
X_training = features_data(df_train)
y_training = df_train[['Survived']].copy().values.reshape(-1,)
X_test = features_data(df_test)
y_test = df_gender[['Survived']].copy().values.reshape(-1,)
Tree = evaluate_model(decisiontree,X_training,X_test,y_training,y_test)
print('------------------------------')  
Bay = evaluate_model(bayes,X_training,X_test,y_training,y_test)
print('------------------------------')
MLP = evaluate_model(neuralnetwork,X_training,X_test,y_training,y_test)
print('------------------------------')
select  = KFold(n_splits=5)
set = 1
decisiontree_eval = []
bayes_eval = []
neuralnetwork_eval = []
for train_index,test_index in select.split(X_training):
  print('SET : ' + str(set))

  X_train,y_train = X_training[train_index],y_training[train_index]
  X_val,y_val = X_training[test_index],y_training[test_index]
  y_train = y_train.reshape(len(y_train),)
  y_val = y_val.reshape(len(y_val),)

  f1 = decisiontree 
  f2 = bayes
  f3 = neuralnetwork

  recall_1,precision_1,f1_1 = evaluate_model(f1,X_train,X_val,y_train,y_val)
  print('------------------------------')
  recall_2,precision_2,f1_2 = evaluate_model(f2,X_train,X_val,y_train,y_val)
  print('------------------------------')
  recall_3,precision_3,f1_3 = evaluate_model(f3,X_train,X_val,y_train,y_val)  

  decisiontree_eval.append([recall_1,precision_1,f1_1])
  bayes_eval.append([recall_2,precision_2,f1_2])
  neuralnetwork_eval.append([recall_3,precision_3,f1_3])
  print('======================================')  
  set +=1
pd.DataFrame([np.mean(decisiontree_eval,axis=0),np.mean(bayes_eval,axis=0),np.mean(neuralnetwork_eval,axis=0)],columns=['Average Recall','Average Precision','Average F'],
             index = ['Decision Tree', 'Naive Bayes', 'MLP'])
gender_submission = (pd.DataFrame(list(zip(df_test['PassengerId'],f3.predict(X_test))),columns=['PassengerId','Survived']))
gender_submission.to_csv('gender_submission.csv',index = False)