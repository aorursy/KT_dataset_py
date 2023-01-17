# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
sample=pd.read_csv('/kaggle/input/data-science-nigeria-ai-in-citie/sample_submission.csv', delimiter=',')

train=pd.read_csv('/kaggle/input/data-science-nigeria-ai-in-citie/train.csv', delimiter=',')

test=pd.read_csv('/kaggle/input/data-science-nigeria-ai-in-citie/test.csv', delimiter=',')

#The shape of the train and test dataset

print('This is the shape of the Train dataset:', train.shape)

print('This is the shape of the Test dataset:', test.shape)



#Info about the dataset

print('The Train Info of the Dataset')

print(train.info())

print('')

print('+'*60)

print('')

print('The Test Info of the Dataset')

print(test.info())
#Creating a new column name Title from the name column 

test['Title'] = test.name.str.extract(' ([A-Za-z]+)\.', expand=False)

test['Title'] = test['Title'].replace(['Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Title')

test['Title'] = test['Title'].replace('Mlle', 'Miss')

test['Title'] = test['Title'].replace('Ms', 'Miss')

test['Title'] = test['Title'].replace(['Mme','Lady'], 'Mrs')



train['Title'] = train.name.str.extract(' ([A-Za-z]+)\.', expand=False)

train['Title'] = train['Title'].replace(['Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Title')

train['Title'] = train['Title'].replace('Mlle', 'Miss')

train['Title'] = train['Title'].replace('Ms', 'Miss')

train['Title'] = train['Title'].replace(['Mme','Lady'], 'Mrs')



#A boxplot which shows the outliers present in each of the ticket class before filling the Nan values with either mean or median

import seaborn as sns

sns.boxplot(x='ticket_class', y='age', data=test);
#filling the empty value in the age column

train.age=train.age.fillna(train.age.mean())



#filling the empty value in the age column

test.age=test.age.fillna(test.age.mean())



##filling the empty value in the fare column with the mode since it does not give a Gaussian curve from the kde plot

train.fare=train.fare.fillna((train.fare.mode()[0]+train.fare.mode()[1])/2)

test.fare=test.fare.fillna(test.fare.mode())



#filling the empty value in the embarked  column

train.embarked=train.embarked.fillna(value='S')

test.embarked=test.embarked.fillna(value='S')



train['MedBoat']=train['MedBoat'].isnull()

test['MedBoat']=test['MedBoat'].isnull()
#A countplot showing the effect of MedBoat to people that survived

'''

From thi countplot we could say that people on the MedBoat Column

tends to survive than people not on the MedBoat



'''

sns.countplot(train['MedBoat'], hue=train['Survived'])
train_object=train.select_dtypes(include='object')

test_object=test.select_dtypes(include='object')



train_num = train.drop(labels=['name', 'sex', 'TickNum', 'embarked',], axis=1)

test_num = test.drop(labels=['name', 'sex', 'TickNum', 'embarked'], axis=1)



#Checking the correlation between the Numeric features to the survived column

train_num_corr=train.corr()

train_num_corr
#A Heatmap showing the correlation of the numerical features

import matplotlib.pyplot as plt

colormap = plt.cm.viridis

plt.figure(figsize=(12,12));

plt.title(' Correlation of Features', y=1.05, size=15);

sns.heatmap(train_num_corr,linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True);
#A countplot of the Gender to the Survived column

'''

From this graph we can imply that Female tends to survive that male 

'''

sns.countplot(train.Survived, hue=train.sex,palette='rainbow' ,saturation=1)
train_num_drop =train_num.drop(['traveller_ID', 'Survived'], axis=1)

test_num

train_num_drop



from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()



#Creating new column from the categorical column using pandas built in function get_dummies 

train_num[[ 'Male']]=pd.get_dummies(train.sex, drop_first=True).astype('int64')

test_num[[ 'Male']]=pd.get_dummies(test.sex,drop_first=True).astype('int64')



train_num[['S','Q']]=pd.get_dummies(train.embarked,drop_first=True).astype('int64')

test_num[['S','Q']]=pd.get_dummies(test.embarked,drop_first=True).astype('int64')



train_num[[2,1]]=pd.get_dummies(train.ticket_class,drop_first=True).astype('int64')

test_num[[2,1]]=pd.get_dummies(test.ticket_class,drop_first=True).astype('int64')



#Converting the Title Column to an Integer using the LabelEncoder

train.Title=le.fit_transform(train.Title).astype('int64')

test.Title=le.transform(test.Title).astype('int64')

test_num['Title']=test.Title

train_num['Title']=train.Title



#Converting the MedBoat Column to an Integer using the LabelEncoder

train.MedBoat=le.fit_transform(train.MedBoat).astype('int64')

test.MedBoat=le.transform(test.MedBoat).astype('int64')

test_num['MedBoat']=test.MedBoat

train_num['MedBoat']=train.MedBoat



train.Title.value_counts()
test_num
#assigniong pur Numerical Feature to our Dependent and Independent Variables for prediction

X = train_num[['age','Siblings_spouses','Parchil','fare','MedBoat','Title','Male','S','Q',2,1]]

y=train['Survived']

X.head(1)





from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.30, random_state=101,shuffle=True)



print(X_train.shape)

print(y_train.shape)
test_num=test_num.drop(labels=['ticket_class'], axis=1)

test_num=test_num.drop(labels=['traveller_ID'], axis=1)

test_num=test_num.drop(labels=['cabin'], axis=1)
from sklearn.preprocessing import StandardScaler

sc_X=StandardScaler()

col=X_train.columns



#Scaling our dataset beforoe fitting into our Algorithm

X =sc_X.fit_transform(X_train)

y=sc_X.transform(test_num)

y=pd.DataFrame(test_num, columns=col)

X=pd.DataFrame(X, columns=col)



X_train.info()
'''

Using Random Forest Classifier for the prediction and passing

our Dependent and Independet variable into our Machine Learning Algorithm



'''

from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics

from sklearn.metrics import roc_curve

from sklearn.metrics import auc

classifier = RandomForestClassifier(min_samples_leaf=5,max_features='auto',n_estimators = 50, criterion = 'entropy', random_state =42)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)



print("Accuracy: ", metrics.accuracy_score(y_test, y_pred)*100)

print("Precision: ", metrics.precision_score(y_test, y_pred)*100)

print("Recall: ", metrics.recall_score(y_test, y_pred)*100)

fpr, tpr, thresholds = roc_curve (y_test, y_pred)

roc_auc= auc (fpr, tpr)

print  ("ROC AUC", roc_auc*100)



#Random Forest Algorithm

submission_file=sample

from sklearn.metrics import f1_score

pred2=classifier.predict(X_train)

pred3=classifier.predict(X_test)

print('F1 Score for Training dataset:', f1_score(y_train, pred2)*100)

print('F1 Score for Testing dataset:', f1_score(y_test, pred3)*100)



#Confusion Matrix

plt.title('Confusion matrix')

sns.heatmap(metrics.confusion_matrix(y_test, y_pred), annot=True, cmap='coolwarm');

plt.ylabel('GroundTruth');

plt.xlabel('Predicted');

plt.figure(figsize=(3,2))
#Passing our Prediction into our a csv file for submission

newrf=sample.copy()

newrf.Survived =classifier.predict(y)

newrf.to_csv('newrf.csv', index=False)

newrf=pd.read_csv('newrf.csv', delimiter=',')

print('This is the shape of the dataset:',newrf.shape)

print('This is the amount of people that did not Survived:',newrf.Survived.value_counts()[0])

print('This is the amount of people that did Survived:',newrf.Survived.value_counts()[1])

sns.countplot(newrf.Survived, saturation=1);