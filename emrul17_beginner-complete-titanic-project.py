import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

%matplotlib inline

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
# Load train and test data

train_data=pd.read_csv("../input/titanic/train.csv")

test_data=pd.read_csv("../input/titanic/test.csv")



# Store our passenger ID as we need them for making submission file.

PassengerId = test_data['PassengerId']
# merging the data

train_len = len(train_data) # created length of the train data so that after EDA is done we can seperate the train and test data

dataset = pd.concat(objs=[train_data, test_data], axis=0).reset_index(drop=True)

dataset.head()
# Primery information of the datasets

print('Total number of columns and rows in the combined dataset: ', dataset.shape)

print('Total number of columns and rows in the train data: ', train_data.shape)

print('Total number of columns and rows in the test data: ', test_data.shape)

print('----------############################-------------')

print('----------Basic information of the data--------')

print(dataset.info())

print('----------############################-------------')

print('Total number of duplicated data: ',dataset.duplicated().sum())
# A function for calculating the missing data

def missing_data(df):

    tot_missing=df.isnull().sum().sort_values(ascending=False)

    Percentage=tot_missing/len(train_data)*100

    missing_data=pd.DataFrame({'Missing Percentage': Percentage})

    

    return missing_data.head()



missing_data(dataset)
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Sex',data=dataset, palette='rainbow')
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Pclass',data=dataset, palette='rainbow')
sns.set_style("ticks", {"xtick.major.size": 12, "ytick.major.size": 12})

sns.countplot(x='Survived',hue='SibSp',data=dataset, palette='coolwarm')
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Parch',data=dataset, palette='coolwarm')
sns.heatmap(dataset.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.figure(figsize=(12,8))

dataset['Age'].hist(bins=15)
plt.figure(figsize=(12,7))

sns.boxplot(x='Pclass',y='Age', data=dataset, palette='viridis')
def impute_age(df):

    Age=df[0]

    Pclass=df[1]

    

    if pd.isnull(Age):

        if Pclass==1:

            return 37

        elif Pclass==2:

            return 29

        elif Pclass==3:

            return 24

    else:

        return Age

    



dataset['Age']=dataset[['Age','Pclass']].apply(impute_age,axis=1)
dataset.Embarked.value_counts()
dataset.fillna('S', inplace=True)

dataset.Embarked.value_counts()
dataset['Fare']=dataset.fillna(test_data['Fare'].mean())
dataset.drop(columns=['Cabin'], inplace=True)

sns.heatmap(dataset.isnull(), yticklabels=False, cbar=False, cmap='viridis')
def title(x):

    if 'Mr.' in x:

        return 'Mr'

    elif 'Mrs.' in x:

        return 'Mrs'

    elif 'Master' in x:

        return 'Master'

    elif 'Miss.' in x:

        return 'Miss'

    else:

        return 'Other'
dataset['title']=dataset['Name'].apply(title)

dataset['title'].value_counts()
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='title',data=dataset, palette='rainbow')
dataset['family_size']=dataset['SibSp']+dataset['Parch']+1
dataset['alone']=dataset['family_size'].apply(lambda x: 1 if x==1 else 0)

dataset['alone'].value_counts()
bins=[0,16,25,45,65,85]

label=['kid','teen', 'young', 'adult','elderly']

dataset['age_group']=pd.cut(dataset.Age,bins=bins, labels=label)
dataset_cat=dataset.select_dtypes(include=['object', 'category'])
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

dataset['age_group']=le.fit_transform(dataset['age_group'])

#dataset['age_group']=le.fit_transform(dataset['age_group'])
dataset.head()
df=pd.get_dummies(dataset,columns=['Sex','Embarked','title'])
# Droping columns that are not necessary for prediction

df.drop(['Name','Ticket','PassengerId'],axis=1, inplace=True)
# Let's separate the targets and features

X=df.drop(['Survived'],axis=1)

Y=df['Survived']



# Train and test data for modelling

X_train=X[:train_len]

Y_train=Y[:train_len]

X_test=X[train_len:]
X_train.shape,Y_train.shape,X_test.shape
# Converting object type to integer type

Y_train=Y_train.astype('int64')
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics #accuracy measure
# Machine learning classification models

ML_model=[

    LogisticRegression(),

    SVC(),

    RandomForestClassifier(),

    KNeighborsClassifier(),

    DecisionTreeClassifier()

]
# Function to fit models and calculating accuracy

def model_fit(models,x,y,z):  

    

    pred=[] # Y_pred

    model_accuracy=[]

    model_name=[]

    for model in models:

        model.fit(x,y)

        accuracy=round(model.score(x,y)*100,2)

        model_accuracy.append(accuracy)

        y_pred=model.predict(z)

        pred.append(y_pred)

        model_name.append(model.__class__.__name__)

    

    # Model accuracy Table

    accuracy_table=pd.DataFrame({'Model':model_name, 'Accuracy':model_accuracy})



    return accuracy_table, pred
# Calcualting accuracy

accuracy_table, Y_pred=model_fit(ML_model,X_train, Y_train,X_test)

accuracy_table.sort_values(by='Accuracy', ascending=False)
from sklearn.model_selection import cross_val_score
def kFold_cross_val(ml_models,x,y,kFold):

    mean_cross_val=[]

    cv_accuracy=[]

    model_name=[]

    for model in ml_models:

        cross_val= cross_val_score(model,x,y, cv=kFold, scoring = "accuracy")

        mean_cross_val.append(cross_val.mean())

        cv_accuracy.append(cross_val)

        model_name.append(model.__class__.__name__)

    

    cross_val_accuracy=pd.DataFrame({'Model':model_name,'KFoldCV_Mean':mean_cross_val})

    

    return cross_val_accuracy
kFold_cross_val(ML_model,X_train, Y_train,5)
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=10)

def StratifiedkFold_cross_val(ml_models,x,y,kfold):

    mean_cross_val=[]

    cv_accuracy=[]

    model_name=[]

    for model in ml_models:

        cross_val= cross_val_score(model,x,y, cv=kfold, scoring = "accuracy")

        mean_cross_val.append(cross_val.mean())

        cv_accuracy.append(cross_val)

        model_name.append(model.__class__.__name__)

    

    cross_val_accuracy=pd.DataFrame({'Model':model_name,'StratifiedKFoldCV_Mean':mean_cross_val})

    

    return cross_val_accuracy
# Stratified kfold score

StratifiedkFold_cross_val(ML_model,X_train, Y_train,kfold)
from sklearn.model_selection import GridSearchCV
C=[1, 10, 100, 1000]

gamma=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

kernel=['rbf']

hyper={'kernel':kernel,'C':C,'gamma':gamma}

gd=GridSearchCV(estimator=SVC(),param_grid=hyper,cv=5,verbose=True)

gd.fit(X_train,Y_train)

print(gd.best_estimator_)

print(gd.best_score_)
n_estimators=range(100,300)

hyper={'n_estimators':n_estimators}

gd=GridSearchCV(estimator=RandomForestClassifier(random_state=0),param_grid=hyper,cv=3,verbose=True)

gd.fit(X_train,Y_train)



print(gd.best_score_)

print(gd.best_estimator_)
hyper={'criterion': ['gini', 'entropy'],'max_depth': [2,4,6,8,10,None]}

gd=GridSearchCV(estimator=DecisionTreeClassifier(),param_grid=hyper,cv=5,verbose=True)

gd.fit(X_train,Y_train)



print(gd.best_score_)

print(gd.best_estimator_)
from sklearn.ensemble import VotingClassifier



Voting_class=VotingClassifier(estimators=[('LR',LogisticRegression() ),

                                        ('SVM',SVC(C=1.0,gamma=0.1, kernel='rbf',probability=True,)),

                                        ('KNN',KNeighborsClassifier(n_neighbors=10)),

                                        ("RF",RandomForestClassifier(n_estimators=146,random_state=0) ),

                                        ('DT',DecisionTreeClassifier(criterion='entropy', max_depth=6))])



Voting_class.fit(X_train,Y_train)
Voting_class.score(X_train, Y_train)
from sklearn.ensemble import BaggingClassifier

Bagged_Decision=BaggingClassifier(base_estimator=DecisionTreeClassifier(),random_state=0,n_estimators=146)



Bagged_Decision.fit(X_train,Y_train)
Bagged_Decision.score(X_train, Y_train)
from sklearn.ensemble import AdaBoostClassifier

ada=AdaBoostClassifier(n_estimators=200,random_state=0,learning_rate=0.1)
Adaboost_score=cross_val_score(ada,X_train,Y_train,cv=10,scoring='accuracy')

Adaboost_score.mean()
from sklearn.ensemble import GradientBoostingClassifier

Grad_boost=GradientBoostingClassifier(n_estimators=500,random_state=0,learning_rate=0.1)

Grad_boost_score=cross_val_score(Grad_boost,X_train,Y_train,cv=10,scoring='accuracy')

Grad_boost_score.mean()
test_pred = pd.Series(Bagged_Decision.predict(X_test), name="Survived")
submit= pd.concat([PassengerId,test_pred],axis=1)
submit.to_csv("submission_emrul.csv", index=False)
submit.head()