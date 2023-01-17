# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import math

import re

import os



import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn import linear_model

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.feature_selection import SelectFromModel



from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.metrics import roc_curve,auc

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory





print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
original_train=pd.read_csv("../input/train.csv")

original_test=pd.read_csv("../input/test.csv")
original_train.describe(include='all')
original_test.describe(include='all')
original_train.corr()
def get_title(name):

    title=re.search(r'[\s][a-zA-Z]*[.]',name).group()

    return title



def replace_title(df):

    title,sex=df

    if(title==' Don.' or title==' Sir.'):

        return ' Mr.'

    elif(title==' Mme.' or title==' Lady.' or title ==' Dona.'):

        return ' Mrs.'

    elif(title==' Ms.' or title==' Mlle.'):

        return ' Miss.'

    elif(title== ' Master.'):

        return ' Master.'

    elif(sex=='male'):

        return ' Mr.'

    elif(sex=='female'):

        return ' Mrs.'
original_train['Title']=original_train['Name'].apply(get_title)

original_train['Title']=original_train[['Title','Sex']].apply(replace_title,axis=1)

original_test['Title']=original_test['Name'].apply(get_title)

original_test['Title']=original_test[['Title','Sex']].apply(replace_title,axis=1)  
def alone_check(df):

    sibsp,parch=df

    if(sibsp >=1 or parch >=1):

        return 0

    else:

        return 1
original_train['IsAlone']=original_train[['SibSp','Parch']].apply(alone_check,axis=1)

original_test['IsAlone']=original_test[['SibSp','Parch']].apply(alone_check,axis=1)
def get_ticket(ticket):

    ticket_class=""

    for i in range(0,len(ticket)):

        if re.match(r'[0-9/.\s]',ticket[i]) is None:

            ticket_class += ticket[i]

        if ticket_class == "":

            ticket_class = "number"

    return ticket_class
def get_class(fare):

    income_class=""

    if fare >=70:

        income_class='upper'

    elif fare>=50 and fare<70:

        income_class='upper_middle'

    elif fare>=20 and fare<50:

        income_class='lower_middle'

    elif fare>=0 and fare<20:

        income_class='lower'

    return income_class

        
original_train['IncomeClass']=original_train['Fare'].apply(get_class)

original_test['IncomeClass']=original_test['Fare'].apply(get_class)
original_train['TicketClass']=original_train['Ticket'].apply(get_ticket)

original_test['TicketClass']=original_test['Ticket'].apply(get_ticket)
def get_passenger_category(df):

    age,sex=df

    p_cat=''

    if age>=0 and age<1:

        if sex=='male':

            p_cat='baby_boy'

        elif sex=='female':

            p_cat='baby_girl'

    elif age>=1 and age<=10:

        if sex=='male':

            p_cat='kid_boy'

        elif sex=='female':

            p_cat='kid_girl'

    elif age>=11 and age<20:

        if sex=='male':

            p_cat='teen_boy'

        elif sex=='female':

            p_cat='teen_girl'

    elif age>=20 and age<40:

        if sex=='male':

            p_cat='mid_male'

        elif sex=='female':

            p_cat='mid_female'

    elif age>=40 and age<60:

        if sex=='male':

            p_cat='adult_male'

        elif sex=='female':

            p_cat='adult_female'

    elif age>=60:

        if sex=='male':

            p_cat='old_male'

        elif sex=='female':

            p_cat='old_female'

    return p_cat

    
'''original_train['Passenger_category']=original_train[['Age','Sex']].apply(get_passenger_category,axis=1)

original_test['Passenger_category']=original_test[['Age','Sex']].apply(get_passenger_category,axis=1)'''
sns.set(rc={'figure.figsize':(15,15)})

sns.countplot(x='Sex',hue='Survived',data=original_train)
sns.countplot(x='Pclass',hue='Survived',data=original_train)
sns.countplot(x='Embarked',hue='Survived',data=original_train)
original_train.groupby(['Survived']).Fare.plot(kind='kde',legend=True)
original_train.groupby(['Survived']).Age.plot(kind='kde',legend=True)
sns.countplot(x='IsAlone',hue='Survived',data=original_train)
sns.countplot(x='Title',hue='Survived',data=original_train)
'''sns.countplot(x='IncomeClass',hue='Survived',data=original_train)'''
original_train['Embarked']=original_train.Embarked.fillna(original_train.Embarked.mode()[0])
original_test['Fare']=original_test.Fare.fillna(original_test.Fare.mean())
age_df=original_train[['Pclass','Title','Sex','Age','Embarked','IsAlone','Fare']]

age_df=age_df.append(original_test[['Pclass','Title','Sex','Age','Embarked','IsAlone','Fare']],ignore_index=True)
def data_processing(data_frame,align_frame,features,target,categorical,split,join):

    if split==1:

        X_train,X_val,y_train,y_val=train_test_split(data_frame[features],

                                                     data_frame[target],

                                                     random_state=42,shuffle=True)

        X_train=pd.get_dummies(X_train,columns=categorical)

        X_val=pd.get_dummies(X_val,columns=categorical)

        X_train_final,X_val_final=X_train.align(X_val,join=join,axis=1,fill_value=0)

        X_train_final=X_train_final.reindex(sorted(X_train_final.columns),axis=1)

        X_val_final=X_val_final.reindex(sorted(X_val_final.columns),axis=1)

        return X_train_final,X_val_final,y_train,y_val

    else:

        X_test=pd.get_dummies(data_frame,columns=categorical)

        X_test_final,align=X_test.align(align_frame,join=join,axis=1,fill_value=0)

        return X_test_final



def fill_age(model,model_df,original_df):

    age_predict=model.predict(model_df)

    indices=original_df[original_df['Age'].isnull()].index

    original_df['Age']=original_df['Age'].fillna(pd.Series(age_predict,index=indices))
age_features=['Pclass','Title','Sex','Embarked','IsAlone','Fare']

age_categorical=['Pclass','Title','Sex','Embarked']

age_target='Age'



X_train,X_val,y_train,y_val=data_processing(data_frame=age_df[age_df['Age'].isnull() == False],

                                            align_frame=None,

                                            features=age_features,

                                            target=age_target,

                                            categorical=age_categorical,

                                            split=1,join='outer')

print(X_train.shape)

print(y_train.shape)

print(X_val.shape)

print(y_val.shape)

print(X_train.columns)
age_model=linear_model.Ridge(alpha=0.0099,normalize=True)

age_model.fit(X_train,y_train)

print(age_model.score(X_train,y_train))

print(age_model.score(X_val,y_val))
X_test=data_processing(data_frame=original_train[original_train['Age'].isnull()][age_features],

                       align_frame=X_train,

                       features=None,

                       target=None,

                       categorical=age_categorical,split=0,join='right')
fill_age(model=age_model,model_df=X_test,original_df=original_train)
X_test_1=data_processing(data_frame=original_test[original_test['Age'].isnull()][age_features],

                        align_frame=X_train,

                        features=None,

                        target=None,

                        categorical=age_categorical,split=0,join='right')

fill_age(model=age_model,model_df=X_test_1,original_df=original_test)
original_train['IncomeClass']=original_train['Fare'].apply(get_class)

original_test['IncomeClass']=original_test['Fare'].apply(get_class)

original_train['PassengerCategory']=original_train[['Age','Sex']].apply(get_passenger_category,axis=1)

original_test['PassengerCategory']=original_test[['Age','Sex']].apply(get_passenger_category,axis=1)
model_features=['Pclass','Sex','Age','Fare','Embarked','Title','IsAlone','IncomeClass','PassengerCategory','TicketClass']

model_categorical=['Pclass','Sex','Embarked','Title','IncomeClass','PassengerCategory','TicketClass']

model_target='Survived'
'''titanic_X_train,titanic_X_val,titanic_y_train,titanic_y_val=data_processing(data_frame=original_train,

                                                                           features=model_features,

                                                                           target=model_target,

                                                                           categorical=model_categorical,

                                                                           split=1,

                                                                           align_frame=None,join='outer')'''



titanic_X_train=original_train[model_features]

titanic_X_train=pd.get_dummies(titanic_X_train,columns=model_categorical)

titanic_y_train=original_train['Survived']
titanic_X_train
print(titanic_X_train.shape)

'''print(titanic_X_val.shape)'''

print(titanic_y_train.shape)

'''print(titanic_y_val.shape)'''
titanic_X_test=data_processing(data_frame=original_test[model_features],

                              align_frame=titanic_X_train,

                              features=None,

                              categorical=model_categorical,

                              target=model_target,split=0,join='right')
titanic_X_test.shape
titanic_model=RandomForestClassifier()

titanic_model.fit(titanic_X_train,titanic_y_train)
imp_f=sorted(zip(titanic_model.feature_importances_,titanic_X_train.columns),reverse=True)

print("\n".join(['{}\t\t{}'.format(f,i) for i,f in imp_f]))

select_features=SelectFromModel(titanic_model,prefit=True,threshold='median')

titanic_train_reduced=select_features.transform(titanic_X_train)

titanic_test_reduced=select_features.transform(titanic_X_test)

print(titanic_train_reduced.shape)

print(titanic_test_reduced.shape)
titanic_X_train_red,titanic_X_val_red,titanic_y_train_red,titanic_y_val_red = train_test_split(titanic_train_reduced,titanic_y_train,random_state=42)

print(titanic_X_train_red.shape)

print(titanic_X_val_red.shape)

print(titanic_y_train_red.shape)

print(titanic_y_val_red.shape)
def evaluation_metrics(features,target,model):

    print("Accuracy: ",model.score(features,target))

    predict=model.predict(features)

    print("Confusion Matrix:\n ",confusion_matrix(target,predict))

    print("Classification Report:\n ",classification_report(target,predict))

    false_positive,true_positive,threshold=roc_curve(target,predict)

    print("AUC:",auc(false_positive,true_positive))

print("Training set Evaluation:\n")

'''evaluation_metrics(titanic_X_train,titanic_y_train,titanic_model)'''
print("Validation set Evaluation:\n")

'''evaluation_metrics(titanic_X_val,titanic_y_val,titanic_model)'''
param_grid={

    'n_estimators':[200,300,400,1000],

    'max_features':['sqrt'],

    'max_depth':[10,70,80,90],

    'min_samples_split':[3,5],

    'min_samples_leaf':[2]

}

grid_model=RandomForestClassifier()

grid_search_model=GridSearchCV(estimator=grid_model,param_grid=param_grid,cv=3,verbose=2,n_jobs=-1)

grid_search_model.fit(titanic_X_train_red,titanic_y_train_red)

print(grid_search_model.best_params_)

print("Training set:\n")

evaluation_metrics(titanic_X_train_red,titanic_y_train_red,grid_search_model)

print("Validation set:\n")

evaluation_metrics(titanic_X_val_red,titanic_y_val_red,grid_search_model)
best_model=grid_search_model.best_estimator_

best_model.fit(titanic_X_train_red,titanic_y_train_red)

print("BEST MODEL!!\nTraining set:\n")

evaluation_metrics(titanic_X_train_red,titanic_y_train_red,best_model)

print("Validation set:\n")

evaluation_metrics(titanic_X_val_red,titanic_y_val_red,best_model)
titanic_predict = best_model.predict(titanic_test_reduced)

print(titanic_predict.shape)

print(titanic_test_reduced.shape)
submit_optimised=pd.DataFrame(columns=['PassengerId','Survived'])

submit_optimised['PassengerId']=original_test['PassengerId']

submit_optimised['Survived']=pd.Series(titanic_predict)

submit_optimised.to_csv("optimized_submission_2.csv",index=False)