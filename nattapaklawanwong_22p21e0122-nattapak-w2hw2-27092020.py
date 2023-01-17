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
# Machine learning
from sklearn import tree 
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
# Preprocessing
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize
# Model
from sklearn.model_selection import KFold
# Visualization 
import seaborn as sns
import matplotlib.pyplot as plt
dt = pd.read_csv('/kaggle/input/titanic/train.csv')
dt.head(10)
dt.info()
dt.describe()
fig = plt.figure(figsize=(20,1))
sns.countplot(y='Survived', data=dt)
print(dt.Survived.value_counts())
fig = plt.subplots(figsize = (20,5))
sns.countplot(x = 'Survived', hue = 'Pclass', data = dt)
fig = plt.subplots(figsize = (20,5))
sns.countplot(x = 'Survived', hue = 'Sex', data = dt)
fig = plt.subplots(figsize = (20,5))
sns.countplot(x = 'Survived', hue = 'Embarked', data = dt)
dt.Age.quantile(.99)
plt.figure(figsize = (16,12))
temp = dt[dt.Age < 65.87]
sns.violinplot(x = 'Survived', y = 'Age', data = temp)
corrMatrix = dt.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()
dt.isna().sum()
dt.Age.plot.hist()
np.random.seed(0)
nan_rows = dt['Age'].isna()
random_age = np.random.choice(dt['Age'][~nan_rows], replace=True, size=sum(nan_rows))
dt.loc[nan_rows,'Age'] = random_age
dt.Age.plot.hist()
dt.Embarked.value_counts()
dt.Embarked.mode()[0]
dt.Embarked.fillna(dt.Embarked.mode()[0], inplace=True)
dt = dt.drop(['Ticket','Cabin','PassengerId','Name'],axis=1)
mean = dt['Age'].mean()
std = dt['Age'].std()
   
# Any value higher than upper limit or below lower limit is an outlier
upper_limit = mean + 3*std
lower_limit = mean - 3*std
upper_limit, lower_limit
outlier_rows = (dt['Age'] > upper_limit) | (dt['Age'] < lower_limit)  
dt['Age'][outlier_rows]
dt.loc[outlier_rows, 'Age'] = dt['Age'][~outlier_rows].mean()
condition = (dt['Age']>60) & (dt['Sex'] == 'male')
condition_2 = (dt['Age']>60) & (dt['Sex'] == 'female')
dt['ElderMale'] = np.where(condition, 1, 0)
dt['ElderFemale'] = np.where(condition_2, 1, 0)
dt.head()
dummy = pd.get_dummies(dt['Embarked'], prefix='Embarked')
dt = pd.concat([dt, dummy], axis=1)
dt = dt.drop(['Embarked'], axis=1)
dt.head(10)
dummy = pd.get_dummies(dt['Pclass'], prefix='Pclass')
dt = pd.concat([dt, dummy], axis=1)
dt = dt.drop(['Pclass'], axis=1)
dt.head(10)
dummy = pd.get_dummies(dt['Sex'], prefix='Sex')
dt = pd.concat([dt, dummy], axis=1)
dt = dt.drop(['Sex'], axis=1)
dt.head(10)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(dt['Age'].values.reshape(-1,1))
dt['Age'] = scaler.transform(dt['Age'].values.reshape(-1,1))
dt['Age'].describe()
scaler.fit(dt['Fare'].values.reshape(-1,1))
dt['Fare'] = scaler.transform(dt['Fare'].values.reshape(-1,1))
dt['Fare'].describe()
X = dt.loc[:, dt.columns != 'Survived']
Y = dt.loc[:, 'Survived']
X.head(10)
Y.head(10)
K = 5
KF = KFold(n_splits=5,random_state=2020,shuffle=True)
K_Fold_list = []
for train_index,test_index in KF.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
    K_Fold_list.append([[X_train,y_train],[X_test,y_test]])
K_Fold_list[0][0][0]
K_Fold_list[0][0][1]
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

def evaluate(y_true,y_pred,label=1):
    precision = precision_score(y_true, y_pred, pos_label=label)
    recall = recall_score(y_true, y_pred, pos_label=label)
    f1 = 2 * (precision * recall) / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}
roc_curve_dict = {}
def show_result(clf, K_Fold_list):
    c = 1
    f1_list_class_survived = []
    f1_list_class_not_survived = []
    for fold in K_Fold_list:
        print("#"*50)
        print("Fold #{}".format(c))
        print("#"*50)
        clf = clf.fit(fold[0][0], fold[0][1])
        y_pred = clf.predict(fold[1][0]) # Predict
        y_true = fold[1][1]

        print("Class survived")
        metrics = evaluate(y_pred,y_true,label=1) # Lable 1 positive
        print("Precision survived:",metrics['precision'])
        print("Recall survived:",metrics['recall'])
        print("F1 survived:",metrics['f1'])
        f1_list_class_survived.append(metrics['f1'])

        print("")
        print("Class not survived")
        metrics = evaluate(y_pred,y_true,label=0) # Lable 0 positive
        print("Precision not survived:",metrics['precision'])
        print("Recall not survived:",metrics['recall'])
        print("F1 not survived:",metrics['f1'])
        f1_list_class_not_survived.append(metrics['f1'])
        print("#"*50)
        print("")

    avg_f1_class_survived = sum(f1_list_class_survived)/len(f1_list_class_survived)
    avg_f1_class_not_survived = sum(f1_list_class_not_survived)/len(f1_list_class_not_survived)

    print("#"*50)
    print("Summary")
    print("#"*50)
    print("Average F1 survived:",avg_f1_class_survived)
    print("Average F1 not survived:",avg_f1_class_not_survived)
    return {"Average_F1_survived":avg_f1_class_survived,
            "Average_F1_not_survived":avg_f1_class_not_survived
           }
clf_tree = tree.DecisionTreeClassifier()
clf_tree = clf_tree.fit(K_Fold_list[0][0][0], K_Fold_list[0][0][1])
import graphviz 
dot_data = tree.export_graphviz(clf_tree,out_file=None,
                        feature_names = K_Fold_list[0][0][0].columns.to_list(),
                        class_names = ['Survived','No Survived'],
                        filled=True, rounded=True,special_characters=True)  
graph = graphviz.Source(dot_data) 
graph
clf_tree = tree.DecisionTreeClassifier()
f1_DT_avg = show_result(clf_tree,K_Fold_list)
NB = GaussianNB()
f1_NB_avg = show_result(NB,K_Fold_list)
clf_NN = MLPClassifier(random_state=2020, max_iter=5000, hidden_layer_sizes = 7)
f1_NN_avg = show_result(clf_NN,K_Fold_list)
dt_Average_F1 = pd.DataFrame([f1_DT_avg,f1_NB_avg,f1_NN_avg])
dt_Average_F1['Name'] = ['Decision Tree','Naive Bayes','Neural Network']
ax = sns.barplot(x = 'Name', y = 'Average_F1_survived', data = dt_Average_F1)
ax.set(xlabel='Name', ylabel='Average F1')
ax.set_title("Average F1 Predict Survived")
ax = sns.barplot(x = 'Name', y = 'Average_F1_not_survived', data = dt_Average_F1)
ax.set(xlabel='Name', ylabel='Average F1')
ax.set_title("Average F1 Predict Not Survived")