import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')



train = pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/train.csv')

test = pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/test.csv')
for df in [train,test]:

    df['type_binary']=df['type'].map({'old':1,'new':0})

    df.drop('type', axis=1, inplace=True)

train.head()

test.head()

avg_f3 = train["feature3"].mean()

avg_f4 = train["feature4"].mean()

avg_f5 = train["feature5"].mean()

avg_f8 = train["feature8"].mean()

avg_f9 = train["feature9"].mean()

avg_f10 = train["feature10"].mean()

avg_f11 = train["feature11"].mean()

train["feature3"].fillna(value=avg_f3, inplace=True)

train["feature4"].fillna(value=avg_f4, inplace=True)

train["feature5"].fillna(value=avg_f5, inplace=True)

train["feature8"].fillna(value=avg_f8, inplace=True)

train["feature9"].fillna(value=avg_f9, inplace=True)

train["feature10"].fillna(value=avg_f10, inplace=True)

train["feature11"].fillna(value=avg_f11, inplace=True)





av_f3 = test["feature3"].mean()

av_f4 = test["feature4"].mean()

av_f5 = test["feature5"].mean()

av_f8 = test["feature8"].mean()

av_f9 = test["feature9"].mean()

av_f10 = test["feature10"].mean()

av_f11 = test["feature11"].mean()

test["feature3"].fillna(value=av_f3, inplace=True)

test["feature4"].fillna(value=av_f4, inplace=True)

test["feature5"].fillna(value=av_f5, inplace=True)

test["feature8"].fillna(value=av_f8, inplace=True)

test["feature9"].fillna(value=av_f9, inplace=True)

test["feature10"].fillna(value=av_f10, inplace=True)

test["feature11"].fillna(value=av_f11, inplace=True)
sns.distplot(train['feature1'],kde = False)
train['feature1'] = np.log(train['feature1'])

sns.distplot(train['feature1'],kde = False)
sns.distplot(train['feature2'],kde = False)
train['feature2'] = np.log(train['feature2'])

sns.distplot(train['feature2'],kde = False)
sns.distplot(train['feature3'],kde = False)
train['feature3'] = np.log(train['feature3'])

sns.distplot(train['feature3'],kde = False)
sns.distplot(train['feature4'],kde = False)
sns.distplot(train['feature5'],kde = False)
train['feature5'] = np.log(train['feature5'])

sns.distplot(train['feature5'],kde = False)
sns.distplot(train['feature6'],kde = False)
train['feature6'] = np.log(train['feature6'])

sns.distplot(train['feature6'],kde = False)
sns.distplot(train['feature7'],kde = False)
sns.distplot(train['feature9'],kde = False)
train['feature9'] = np.log(train['feature9'])

sns.distplot(train['feature9'],kde = False)
sns.distplot(train['feature10'],kde = False)
train['feature10'] = np.log(train['feature10'])

sns.distplot(train['feature10'],kde = False)
sns.distplot(train['feature11'],kde = False)
train['feature11'] = np.log(train['feature11'])

sns.distplot(train['feature11'],kde = False)
numerical_features = ['feature3','feature5','feature6','feature7']

#categorical_features = ['type_binary']

X = train[numerical_features]

y = train["rating"]
from sklearn.model_selection import train_test_split

X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.33,random_state=42)



from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])

X_val[numerical_features] = scaler.transform(X_val[numerical_features])  

X_train[numerical_features].head()
#from sklearn.preprocessing import RobustScaler



#scaler = RobustScaler()

#X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])

#X_val[numerical_features] = scaler.transform(X_val[numerical_features])  



#X_train[numerical_features].head()
train.head(1950)
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier



# Initialize and train

clf1 = DecisionTreeClassifier().fit(X_train,y_train)

clf2 = RandomForestClassifier().fit(X_train,y_train)
from sklearn.metrics import accuracy_score  #Find out what is accuracy_score



y_pred_1 = clf1.predict(X_val)

y_pred_2 = clf2.predict(X_val)



acc1 = accuracy_score(y_pred_1,y_val)*100

acc2 = accuracy_score(y_pred_2,y_val)*100



print(y_pred_1)

print("Accuracy score of clf1: {}".format(acc1))



print(y_pred_2)

print("Accuracy score of clf2: {}".format(acc2))
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer



#TODO

clf = RandomForestClassifier()        #Initialize the classifier object



parameters = {'n_estimators':[10,50,100]}    #Dictionary of parameters



scorer = make_scorer(accuracy_score)         #Initialize the scorer using make_scorer



grid_obj = GridSearchCV(clf,parameters,scoring=scorer)         #Initialize a GridSearchCV object with above parameters,scorer and classifier



grid_fit = grid_obj.fit(X_train,y_train)        #Fit the gridsearch object with X_train,y_train



best_clf = grid_fit.best_estimator_         #Get the best estimator. For this, check documentation of GridSearchCV object



unoptimized_predictions = (clf.fit(X_train, y_train)).predict(X_val)      #Using the unoptimized classifiers, generate predictions

optimized_predictions = best_clf.predict(X_val)        #Same, but use the best estimator



acc_unop = accuracy_score(y_val, unoptimized_predictions)*100       #Calculate accuracy for unoptimized model

acc_op = accuracy_score(y_val, optimized_predictions)*100         #Calculate accuracy for optimized model



print("Accuracy score on unoptimized model:{}".format(acc_unop))

print("Accuracy score on optimized model:{}".format(acc_op))
from sklearn.naive_bayes import GaussianNB

#reate a Gaussian Classifier

model = GaussianNB()



# Train the model using the training sets

clf = model.fit(X_train,y_train)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train)



X_train = scaler.transform(X_train)

X_val = scaler.transform(X_val)
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=5)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_val)
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_val, y_pred))

print(classification_report(y_val, y_pred))
acc1 = accuracy_score(y_pred,y_val)*100

print(y_pred)

print("Accuracy score of clf1: {}".format(acc1))
submission = pd.DataFrame({'id':test['id'],'rating':y_pred})



#Visualize the first 5 rows

submission.head()
y_pred = svclassifier.predict(X_val)

acc3 = accuracy_score(y_pred,y_val)*100

print(y_pred)

print("Accuracy score of clf1: {}".format(acc3))

from sklearn.linear_model import LogisticRegression#create an instance and fit the model 

logmodel = LogisticRegression()

logmodel.fit(X_train, y_train)
#predictions

predictions = logmodel.predict(X_val)
from sklearn.metrics import classification_report

print(classification_report(y_val,predictions))
from sklearn.metrics import confusion_matrix 

print(confusion_matrix(y_val, predictions))
acc4 = accuracy_score(predictions,y_val)*100

print(predictions)

print("Accuracy score of clf1: {}".format(acc3))
submission = pd.DataFrame({'id':test['id'],'rating':predictions})



#Visualize the first 5 rows

submission.head()













submission.to_csv(r'/home/bharaths/BITS/ML_LAB/sub14.csv', index=False)