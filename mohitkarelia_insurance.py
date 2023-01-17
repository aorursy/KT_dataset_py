import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sb

sb.set_style('whitegrid')

sb.set_palette('dark')

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_data = pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/train.csv')

test_data = pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/test.csv')
train_data
test_data
id=test_data.id
print('Training Data:\n')

print(train_data.isnull().sum())

print('')

print('test data:\n')

print(test_data.isnull().sum())
train_data['Region_Code'].unique()

train_data.dtypes
from sklearn.preprocessing import LabelBinarizer,LabelEncoder

lb = LabelBinarizer()

le = LabelEncoder()

train_data['Gender'] = lb.fit_transform(train_data['Gender'])

test_data['Gender'] = lb.fit_transform(test_data['Gender'])

train_data['Vehicle_Damage'] = lb.fit_transform(train_data['Vehicle_Damage'])

test_data['Vehicle_Damage'] = lb.fit_transform(test_data['Vehicle_Damage'])



train_data.drop(['id'],1,inplace=True)
test_data.drop(['id','Vehicle_Age'],1,inplace=True)
test_data
plt.figure(figsize = (25,8))

sb.heatmap(train_data.corr(),annot=True)
plt.figure(figsize = (25,8))

sb.countplot(y = train_data['Response'])
plt.figure(figsize = (25,9))

plt.hist(train_data['Gender'],color='steelblue',edgecolor='black')

plt.xticks([0,1],labels = ['Female','Male'])

plt.xlabel('Gender')

plt.ylabel('Number of People')

plt.title('Age Distribution')
plt.figure(figsize = (25,8))

plt.hist(x = [train_data[train_data['Response']==1]['Gender'],train_data[train_data['Response']==0]['Gender']],color=['skyblue','slategrey'],stacked=True,edgecolor='black',label = ['yes','no'])

plt.xticks(ticks = [0,1],labels = ['Female','Male'])

plt.xlabel('Gender')

plt.ylabel('Count')

plt.title('Gender Distribution of People having a positive and a Negetive Response')

plt.legend()
plt.figure(figsize = (25,8))

sb.regplot(train_data['Gender'],train_data['Response'],color='red')
plt.figure(figsize = (25,9))

plt.hist(train_data['Age'],color='darkmagenta',edgecolor='violet')

plt.xticks([20,25,30,35,40,45,50,55,60,65,70,75,80,85,90])

plt.xlabel('Age')

plt.ylabel('Number of Peoplw')

plt.title('Age Distribution')
plt.figure(figsize = (25,10))

plt.hist(x = [train_data[train_data['Response']==1]['Age'],train_data[train_data['Response']==0]['Age']],color=['orange','orangered'],stacked=True,edgecolor='black',label = ['yes','no'])

plt.xticks([20,27,33,40,46,53,55,60,65,73,79,80,85])

plt.yticks([4500,8000,20000,40000,50000,60000,80000,100000,120000,130000])

plt.xlabel('Age')

plt.ylabel('Count')

plt.title('Age Distribution of People having a positive and a Negetive Response')

plt.legend()
plt.figure(figsize = (28,9))

sb.regplot(train_data['Response'],train_data['Age'],color= 'indigo')
plt.figure(figsize = (25,9))

plt.hist(train_data['Driving_License'],color='forestgreen',edgecolor='black')

plt.xticks([0,1],labels = ['No','Yes'])

plt.xlabel('License')

plt.ylabel('Number of People')

plt.title('Driving License Distribution')
plt.figure(figsize = (25,8))

plt.hist(x = [train_data[train_data['Response']==1]['Driving_License'],train_data[train_data['Response']==0]['Driving_License']],color=['crimson','gold'],stacked=True,edgecolor='black',label = ['yes','no'])

plt.xticks(ticks = [0,1],labels = ['No','Yes'])

plt.xlabel('License')

plt.ylabel('Count')

plt.legend()

plt.title('License Distribution of People having a positive and a Negetive Response')
plt.figure(figsize = (25,9))

plt.hist(train_data['Previously_Insured'],color='deeppink',edgecolor='black')

plt.xticks([0,1],labels = ['No','Yes'])

plt.xlabel('Previously insured')

plt.ylabel('Number of People')

plt.title('Previously Insurance Distribution')
plt.figure(figsize = (25,8))

plt.hist(x = [train_data[train_data['Response']==1]['Previously_Insured'],train_data[train_data['Response']==0]['Previously_Insured']],color=['blueviolet','navy'],stacked=True,edgecolor='black',label = ['yes','no'])

plt.xticks(ticks = [0,1],labels = ['No','Yes'])

plt.xlabel('Previously Insured')

plt.ylabel('Count')

plt.yticks([0,25000,46710,75000,100000,125000,175000,20000])

plt.title('Distribution of people who had and had not previously insured')

plt.legend()
plt.figure(figsize = (25,8))

sb.regplot(train_data['Response'],train_data['Previously_Insured'],color = 'lightcoral')
plt.figure(figsize = (25,9))

plt.hist(train_data['Vehicle_Damage'],color='lavenderblush',edgecolor='orchid')

plt.xticks([0,1],labels = ['No','Yes'])

plt.xlabel('Vehicle Damage')

plt.ylabel('Number of People')

plt.title('Vehicle Damage Distribution')

plt.tight_layout()
plt.figure(figsize = (25,8))

plt.hist(x = [train_data[train_data['Response']==1]['Vehicle_Damage'],train_data[train_data['Response']==0]['Vehicle_Damage']],color=['aliceblue','azure'],stacked=True,edgecolor='royalblue',label = ['yes','no'])

plt.xticks(ticks = [0,1],labels = ['No','Yes'])

plt.xlabel('Vehicle Damage')

plt.ylabel('Count')

plt.title('Distribution of people with vehicle damage')

plt.legend()
plt.figure(figsize = (25,8))

sb.regplot(train_data['Response'],train_data['Vehicle_Damage'],color = 'greenyellow')
plt.figure(figsize = (25,8))

sb.regplot(train_data['Vehicle_Damage'],train_data['Previously_Insured'],color='skyblue')
plt.figure(figsize = (25,9))

plt.hist(train_data['Annual_Premium'],color='whitesmoke',edgecolor='gold')

plt.xticks([2630. , 56383.5, 110137. , 163890.5, 217644. , 271397.5,325151. , 378904.5, 432658. , 486411.5, 540165.])

plt.yticks([20300,50000,100000,200000,300000,360000])

plt.xlabel('Premium')

plt.ylabel('count')

plt.title('Premium Distribution')
plt.figure(figsize = (25,10))

plt.hist(x = [train_data[train_data['Response']==1]['Annual_Premium'],train_data[train_data['Response']==0]['Annual_Premium']],color=['lightblue','slategrey'],stacked=True,edgecolor='black',label = ['yes','no'])

plt.xlabel('Annual Premium')

plt.ylabel('Count')

plt.xticks([2630. , 56383.5, 110137. , 163890.5, 217644. , 271397.5,325151. , 378904.5, 432658. , 486411.5, 540165.])

plt.yticks([20300,50000,100000,200000,300000,360000])

plt.title('Distribution of People having a positive and a Negetive Response wrt to Annual Premium')

plt.legend()
plt.figure(figsize = (25,9))

plt.hist(train_data['Vehicle_Age'],color='palegreen',edgecolor='black')

plt.xlabel('Vehicle Age')

plt.ylabel('Count')

plt.title('Vehicle Age Distribution')
plt.figure(figsize = (25,10))

plt.hist(x = [train_data[train_data['Response']==1]['Vehicle_Age'],train_data[train_data['Response']==0]['Vehicle_Age']],color=['violet','indigo'],stacked=True,edgecolor='black',label = ['yes','no'])

plt.xlabel('Vehicle Age')

plt.ylabel('Count')

plt.title('Distribution of People having a positive and a Negetive Response wrt Vehicle Age')

plt.legend()
plt.figure(figsize = (25,9))

plt.hist(train_data['Policy_Sales_Channel'],color='orange',edgecolor='black')

plt.xlabel('Policy Sales Channel')

plt.ylabel('Count')

plt.xticks([ 1. ,  17.2,  33.4,  49.6,  65.8,  82. ,  98.2, 114.4, 130.6,

        146.8, 163. ])

plt.title('Policy Sales Channel Distribution')
plt.figure(figsize = (25,10))

plt.hist(x = [train_data[train_data['Response']==1]['Policy_Sales_Channel'],train_data[train_data['Response']==0]['Policy_Sales_Channel']],color=['lightcyan','skyblue'],stacked=True,edgecolor='black')

plt.xlabel('Vehicle Age')

plt.ylabel('Count')

plt.title('Distribution of People having a positive and a Negetive Response wrt Vehicle Age')

plt.legend()
plt.figure(figsize = (25,8))

sb.regplot(train_data['Policy_Sales_Channel'],train_data['Response'],color = 'darkcyan')
plt.figure(figsize = (25,9))

plt.hist(train_data['Vintage'],color='mediumpurple',edgecolor='black')

plt.xlabel('Vintage')

plt.ylabel('Count')

plt.title('Vintage Distribution')
plt.figure(figsize = (25,10))

plt.hist(x = [train_data[train_data['Response']==1]['Vintage'],train_data[train_data['Response']==0]['Vintage']],color=['lightcyan','mediumpurple'],stacked=True,edgecolor='black')

plt.xlabel('Vintage')

plt.ylabel('Count')

plt.title('Distribution of People having a positive and a Negetive Response wrt Vintage')

plt.legend()
plt.figure(figsize = (25,8))

sb.regplot(train_data['Response'],train_data['Vintage'],color = 'deeppink')
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score,cross_val_predict

from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score,accuracy_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB
X = train_data[['Gender', 'Age', 'Driving_License', 'Region_Code', 'Previously_Insured', 'Vehicle_Damage', 'Annual_Premium','Policy_Sales_Channel', 'Vintage', ]]

y = train_data['Response']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)
clf1 = LogisticRegression(C=0.01,max_iter = 1000)

clf1.fit(X_train,y_train)
y_train_pred = cross_val_predict(clf1,X_train,y_train,cv=4)

print("Confusion Matrix: \n",confusion_matrix(y_train,y_train_pred))

print('Precision Score:',precision_score(y_train,y_train_pred))

print("Recall Score:",recall_score(y_train,y_train_pred))

print("Accuracy Score:",accuracy_score(y_train,y_train_pred))

print("Cross Val Score Insample",cross_val_score(clf1,X_train,y_train,cv=4,scoring='accuracy').mean())

print("Cross Val Score Outsample",cross_val_score(clf1,X_test,y_test,cv=4,scoring='accuracy').mean())
clf2 = GaussianNB()

clf2.fit(X_train,y_train)
y_train_pred = cross_val_predict(clf2,X_train,y_train,cv=4)

print("Confusion Matrix: \n",confusion_matrix(y_train,y_train_pred))

print('Precision Score:',precision_score(y_train,y_train_pred))

print("Recall Score:",recall_score(y_train,y_train_pred))

print("Accuracy Score:",accuracy_score(y_train,y_train_pred))

print("Cross Val Score Insample",cross_val_score(clf2,X_train,y_train,cv=4,scoring='accuracy').mean())

print("Cross Val Score Outsample",cross_val_score(clf2,X_test,y_test,cv=4,scoring='accuracy').mean())
param_grid = {'max_depth':np.arange(1,10),'min_samples_leaf':np.arange(1,8)}

grid = GridSearchCV(DecisionTreeClassifier(),param_grid,cv=5)

grid.fit(X,y)
grid.best_params_
clf3 = grid.best_estimator_
clf3.fit(X_train,y_train)
y_train_pred = cross_val_predict(clf3,X_train,y_train,cv=4)

print("Confusion Matrix: \n",confusion_matrix(y_train,y_train_pred))

print('Precision Score:',precision_score(y_train,y_train_pred))

print("Recall Score:",recall_score(y_train,y_train_pred))

print("Accuracy Score:",accuracy_score(y_train,y_train_pred))

print("Cross Val Score Insample",cross_val_score(clf3,X_train,y_train,cv=4,scoring='accuracy').mean())

print("Cross Val Score Outsample",cross_val_score(clf3,X_test,y_test,cv=4,scoring='accuracy').mean())
clf5 = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,

                       criterion='gini', max_depth=3, max_features='auto',

                       max_leaf_nodes=2, max_samples=None,

                       min_impurity_decrease=0.0, min_impurity_split=None,

                       min_samples_leaf=2, min_samples_split=15,

                       min_weight_fraction_leaf=0.0, n_estimators=10,

                       n_jobs=-1, random_state=1, verbose=0,

                       warm_start=False)

clf5.fit(X_train,y_train)
y_train_pred = cross_val_predict(clf5,X_train,y_train,cv=4)

print("Confusion Matrix: \n",confusion_matrix(y_train,y_train_pred))

print('Precision Score:',precision_score(y_train,y_train_pred))

print("Recall Score:",recall_score(y_train,y_train_pred))

print("Accuracy Score:",accuracy_score(y_train,y_train_pred))

print("Cross Val Score Insample",cross_val_score(clf5,X_train,y_train,cv=4,scoring='accuracy').mean())

print("Cross Val Score Outsample",cross_val_score(clf5,X_test,y_test,cv=4,scoring='accuracy').mean())
clf6 = BaggingClassifier(DecisionTreeClassifier(max_depth=1,min_samples_leaf=1),bootstrap=True)

clf6.fit(X_train,y_train)
y_train_pred = cross_val_predict(clf6,X_train,y_train,cv=4)

print("Confusion Matrix: \n",confusion_matrix(y_train,y_train_pred))

print('Precision Score:',precision_score(y_train,y_train_pred))

print("Recall Score:",recall_score(y_train,y_train_pred))

print("Accuracy Score:",accuracy_score(y_train,y_train_pred))

print("Cross Val Score Insample",cross_val_score(clf6,X_train,y_train,cv=4,scoring='accuracy').mean())

print("Cross Val Score Outsample",cross_val_score(clf6,X_test,y_test,cv=4,scoring='accuracy').mean())
clf7 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1,min_samples_leaf=1))

clf7.fit(X_train,y_train)
y_train_pred = cross_val_predict(clf7,X_train,y_train,cv=4)

print("Confusion Matrix: \n",confusion_matrix(y_train,y_train_pred))

print('Precision Score:',precision_score(y_train,y_train_pred))

print("Recall Score:",recall_score(y_train,y_train_pred))

print("Accuracy Score:",accuracy_score(y_train,y_train_pred))

print("Cross Val Score Insample",cross_val_score(clf7,X_train,y_train,cv=4,scoring='accuracy').mean())

print("Cross Val Score Outsample",cross_val_score(clf7,X_test,y_test,cv=4,scoring='accuracy').mean())
classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1,min_samples_leaf=1))

classifier.fit(X,y)
classifier.predict(test_data)