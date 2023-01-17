import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor





import seaborn as sns

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

# Load the training set

train_data = pd.read_csv('/kaggle/input/titanic/train.csv')



# Load the test set

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')



# df List to impute or perform any common operation on both datasets. 

df_all_data = [train_data, test_data]



print (train_data.shape)

print (test_data.shape)
print('Train columns with null values:\n', train_data.isnull().sum())

print("-"*10)



print('Test/Validation columns with null values:\n', test_data.isnull().sum())

print("-"*10)
# fetch empty 'Embarked' values from train set



train_data[train_data['Embarked'].isnull()]
# get average Fare values for Pclass=1 across the Embarkation ports



train_data[train_data['Pclass'] ==1].groupby('Embarked')['Fare'].median()
# apply 'Embarked' with 'C'



train_data['Embarked'].fillna('C', inplace=True)
# fetch empty 'Fare' values from test set



test_data[test_data['Fare'].isnull()]
# get avergae fare where Pclass=3, Embarked='S' and Sex='male'



avg_fare = test_data[(test_data.Pclass == 3) & 

                     (test_data.Embarked == "S") & 

                     (test_data.Sex == "male")].Fare.mean()



# apply the average fare to the empty record



test_data.Fare.fillna(avg_fare, inplace=True)
train_data[train_data['Age'].isnull()]

test_data[test_data['Age'].isnull()]
# function to predict age based on other independent features using machine learning model



def predictAge(df):

    

    df_withAge = df[pd.isnull(df['Age']) == False]

    df_WithoutAge = df[pd.isnull(df['Age'])]



    colums = ['Pclass','SibSp','Parch','Fare','Age']



    df_withAge = df_withAge[colums]

    df_WithoutAge = df_WithoutAge[colums]



    features = ['Pclass','SibSp','Parch','Fare']



    model = RandomForestRegressor()

    model.fit(df_withAge[features], df_withAge['Age'])



    predicted_age = model.predict(X = df_WithoutAge[features])

    

    df.loc[df.Age.isnull(), "Age"] = predicted_age.astype(int)

    



def getAgeGroup(age):

    a=''

    if age<=15:

        a='Child'

    elif age<=30:

        a='Young'

    elif age<=50:

        a='Adult'

    else:

        a='Old'

    return a
#call the function to apply missing age on train set



predictAge(train_data)



#call the function to apply missing age on test set



predictAge(test_data)   
train_data

test_data
# loop through both train and test datasets



for dataset in df_all_data:   



    #If SibSp and/or Parch is > 0, add it to determine the FamilySize

    dataset['TotalFamilyMembers'] = dataset.SibSp + dataset.Parch + 1

    

    family_map = {1: 'Alone', 2: 'Small', 3: 'Small', 4: 'Small', 5: 'Medium', 6: 'Medium', 7: 'Large', 8: 'Large', 11: 'Large'}

    dataset['FamilySize'] = dataset['TotalFamilyMembers'].map(family_map)



   

    #this is now not required as we have mapped the FamilySize and IsAlone feature using the count

    dataset.drop(['TotalFamilyMembers'], axis=1, inplace=True)

    



    #get the title of the passenger

    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

  

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',\

                                                    'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

  

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')



    

    dataset['AgeGroup']=dataset['Age'].map(getAgeGroup)

    

  
train_data

test_data
cat_features = ['Embarked', 'Parch', 'Pclass', 'Sex', 'SibSp', 'FamilySize', 'AgeGroup','Title']



fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(20, 10))

plt.subplots_adjust(right=1.5, top=1.25)



for i, feature in enumerate(cat_features, 1):    

    plt.subplot(2, 4, i)

    sns.countplot(x=feature, hue='Survived', data=train_data)

    

    plt.xlabel('{}'.format(feature), size=15, labelpad=10)

    plt.ylabel('Passenger Count', size=15, labelpad=15)    

    plt.tick_params(axis='x', labelsize=15)

    plt.tick_params(axis='y', labelsize=15)

    

    plt.legend(['Not Survived', 'Survived'], loc='upper center', prop={'size': 15})

    plt.title('Count of Survival vs {} '.format(feature), size=15, y=1)



plt.show()
#drop the columns from test and train dataset



PassengerId = test_data['PassengerId']



train_data.drop(['PassengerId','Name','Cabin','Ticket','SibSp','Parch','Age'], axis=1, inplace=True)

test_data.drop(['PassengerId','Name','Cabin','Ticket','SibSp','Parch','Age'], axis=1, inplace=True)
train_data = pd.get_dummies(train_data, 

                    columns=['Pclass','Sex','Embarked','FamilySize','Title','AgeGroup'], drop_first=False)

test_data = pd.get_dummies(test_data, 

                    columns=['Pclass','Sex','Embarked','FamilySize','Title','AgeGroup'], drop_first=False)



##
plt.figure(figsize=(14,14))

plt.title('Corelation Matrix', size=8)

sns.heatmap(train_data.astype(float).corr(method='pearson').abs(),linewidths=0.1,vmax=1.0, 

            square=True, cmap='coolwarm', linecolor='white', annot=True)

plt.show()
# separating our independent and dependent variable

X = train_data.drop(['Survived'], axis = 1)

y = train_data["Survived"]

column_names = X.columns



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = .20, random_state=0)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



pd.DataFrame(X_train, columns=column_names).head()
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit

from sklearn.metrics import mean_absolute_error, accuracy_score

from sklearn.metrics import classification_report, balanced_accuracy_score, confusion_matrix

## Logistic Regression



## Choosing penalties(Lasso(l1) or Ridge(l2))

penalties = ['l1','l2']



max_iters = [1000,2500,3000,5000]



## setting param for param_grid in GridSearchCV. 

param = {'penalty': penalties, 'C': np.logspace(-4, 4, 20), 'max_iter': max_iters, 'solver' : ['liblinear']}



cv = StratifiedShuffleSplit(n_splits = 10, test_size = .25)





## Calling on GridSearchCV object. 

grid = GridSearchCV(estimator=LogisticRegression(), 

                            param_grid = param,

                            scoring = 'accuracy',

                            n_jobs =-1,

                            cv = 5)





## Fitting the model

grid.fit(X_train, y_train)



model_logreg = grid.best_estimator_



model_logreg.fit(X_train, y_train)



y_pred = model_logreg.predict(X_test)



log_reg_accuracy = accuracy_score(y_test,y_pred)



print ("Model Accuracy:\t{}".format(log_reg_accuracy))



print("-"*10)



print(classification_report(y_test, y_pred))



pd.DataFrame(confusion_matrix(y_test,y_pred),\

            columns=["Predicted Not-Survived", "Predicted Survived"],\

            index=["Not-Survived","Survived"] )
from sklearn.tree import DecisionTreeClassifier



max_depth = range(1,30)

max_feature = [15,20,25,30,35,'auto']



criterion=["entropy", "gini"]



param = {'max_depth':max_depth, 

         'max_features':max_feature, 

         'criterion': criterion}



model_dtc = DecisionTreeClassifier()



grid = GridSearchCV(estimator=DecisionTreeClassifier(), 

                                  param_grid = param, 

                                  verbose=False, 

                                  cv=5,

                                  n_jobs = -1)

## Fitting the model



grid.fit(X_train, y_train)



model_dtc = grid.best_estimator_



model_dtc.fit(X_train, y_train)



y_pred = model_dtc.predict(X_test)



model_dtc_accuracy = accuracy_score(y_test,y_pred)



print ("Model Accuracy:\t{}".format(model_dtc_accuracy))



print("-"*10)



print(classification_report(y_test, y_pred))



pd.DataFrame(confusion_matrix(y_test,y_pred),\

            columns=["Predicted Not-Survived", "Predicted Survived"],\

            index=["Not-Survived","Survived"] )
from sklearn.svm import SVC



Cs = [0.001, 0.01, 0.1, 1,1.5,2,2.5,3,4,5, 10] ## penalty parameter C for the error term. 

gammas = [0.0001,0.001, 0.01, 0.1, 1]

param_grid = {'C': Cs, 'gamma' : gammas}





grid = GridSearchCV(SVC(kernel = 'rbf', probability=True), param_grid, cv=5) ## 'rbf' stands for gaussian kernel



grid.fit(X_train, y_train)



model_svc = grid.best_estimator_



model_svc.fit(X_train, y_train)



y_pred = model_svc.predict(X_test)



model_svc_accuracy = accuracy_score(y_test,y_pred)



print ("Model Accuracy:\t{}".format(model_svc_accuracy))



print("-"*10)



print(classification_report(y_test, y_pred))



pd.DataFrame(confusion_matrix(y_test,y_pred),\

            columns=["Predicted Not-Survived", "Predicted Survived"],\

            index=["Not-Survived","Survived"] )
from sklearn.ensemble import RandomForestClassifier



n_estimators = [120,125,130,135,140,150,160,180,200];

max_depth = range(1,10);

criterions = ['gini', 'entropy'];



parameters = {'n_estimators':n_estimators,

              'max_depth':max_depth,

              'criterion': criterions}



grid = GridSearchCV(estimator=RandomForestClassifier(max_features = 'auto'),

                                 param_grid=parameters,

                                 cv=5,

                                 n_jobs = -1)

grid.fit(X_train,y_train)



model_rfc = grid.best_estimator_



model_rfc.fit(X_train,y_train)





y_pred = model_rfc.predict(X_test)



model_rfc_accuracy = accuracy_score(y_test,y_pred)



print ("Model Accuracy:\t{}".format(model_rfc_accuracy))



print("-"*10)



print(classification_report(y_test, y_pred))



pd.DataFrame(confusion_matrix(y_test,y_pred),\

            columns=["Predicted Not-Survived", "Predicted Survived"],\

            index=["Not-Survived","Survived"] )

#from xgboost import XGBClassifier



from xgboost.sklearn import XGBClassifier



parameters = {'objective':['binary:logistic'],

              'learning_rate': [0.05], #so called `eta` value

              'max_depth': [5,6,7],

              'subsample': [0.6,0.7,0.8],

              'n_estimators': [200,400,600,800,1000]}



grid = GridSearchCV(XGBClassifier(), parameters, scoring='accuracy', n_jobs=-1, cv=5)



grid.fit(X_train,y_train)



model_xgb = grid.best_estimator_



model_xgb.fit(X_train,y_train)



y_pred = model_xgb.predict(X_test)



model_xgb_accuracy = accuracy_score(y_test,y_pred)



print ("Model Accuracy:\t{}".format(model_xgb_accuracy))



print("-"*10)



print(classification_report(y_test, y_pred))



pd.DataFrame(confusion_matrix(y_test,y_pred),\

            columns=["Predicted Not-Survived", "Predicted Survived"],\

            index=["Not-Survived","Survived"] )

# predict probabilities

pred_prob1 = model_logreg.predict_proba(X_test)

pred_prob2 = model_dtc.predict_proba(X_test)

pred_prob3 = model_svc.predict_proba(X_test)

pred_prob4 = model_rfc.predict_proba(X_test)

pred_prob5 = model_xgb.predict_proba(X_test)





from sklearn.metrics import roc_curve



# roc curve for models

fpr1, tpr1, thresh1 = roc_curve(y_test, pred_prob1[:,1], pos_label=1)

fpr2, tpr2, thresh2 = roc_curve(y_test, pred_prob2[:,1], pos_label=1)

fpr3, tpr3, thresh3 = roc_curve(y_test, pred_prob3[:,1], pos_label=1)

fpr4, tpr4, thresh4 = roc_curve(y_test, pred_prob4[:,1], pos_label=1)

fpr5, tpr5, thresh5 = roc_curve(y_test, pred_prob5[:,1], pos_label=1)





# roc curve for tpr = fpr 

random_probs = [0 for i in range(len(y_test))]

p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)

from sklearn.metrics import roc_auc_score



# auc scores

logreg_auc_score = roc_auc_score(y_test, pred_prob1[:,1])

dtc_auc_score = roc_auc_score(y_test, pred_prob2[:,1])

svc_auc_score = roc_auc_score(y_test, pred_prob3[:,1])

rfc_auc_score = roc_auc_score(y_test, pred_prob4[:,1])

xgb_auc_score = roc_auc_score(y_test, pred_prob5[:,1])







print(logreg_auc_score,dtc_auc_score,svc_auc_score,rfc_auc_score,xgb_auc_score)
results = pd.DataFrame({

    'Model Key': ['Logistic Regression', 'Decision Tree', 'SVC', 'Random Forest', 'XGBClassifier'],

    'Model Name': [model_logreg, model_dtc, model_svc, model_rfc, model_xgb],

    'Model Accuracy': [ log_reg_accuracy, model_dtc_accuracy, model_svc_accuracy, model_rfc_accuracy, model_xgb_accuracy],

    'Model Auc_Score': [ logreg_auc_score,dtc_auc_score,svc_auc_score,rfc_auc_score,xgb_auc_score]})



result_df = results.sort_values(by=['Model Accuracy'], ascending=False)

result_df.reset_index(drop=True)

import matplotlib.pyplot as plt

plt.style.use('seaborn')



# plot roc curves

plt.plot(fpr1, tpr1, linestyle='--',color='orange', label='Logistic Regression')

plt.plot(fpr2, tpr2, linestyle='-.',color='green', label='Decision Tree')

plt.plot(fpr3, tpr3, linestyle=':',color='brown', label='SVC')

plt.plot(fpr4, tpr4, linestyle='-',color='purple', label='Random Forest Classifier')

plt.plot(fpr5, tpr5, linestyle='-.',color='black', label='XGB')



plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')

# title

plt.title('ROC curve')

# x label

plt.xlabel('False Positive Rate')

# y label

plt.ylabel('True Positive rate')



plt.legend(loc='best')

plt.savefig('ROC',dpi=300)

plt.show();
model_xgb.fit(X,y)



test_prediction = model_xgb.predict(test_data)



survivors = pd.DataFrame(test_prediction, columns = ['Survived'])



len(survivors)



survivors.insert(0, 'PassengerId', PassengerId, True)



survivors

survivors.to_csv('first_submission_xgb.csv', index = False)