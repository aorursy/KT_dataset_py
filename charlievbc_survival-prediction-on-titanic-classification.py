# import libraries

import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

from collections import Counter

%matplotlib inline



# import datasets

try: 

    df_train = pd.read_csv('/kaggle/input/titanic/train.csv')

    print('File1 loading - Success!')

    df_test = pd.read_csv('/kaggle/input/titanic/test.csv')

    print('File2 loading - Success!')

except:

    print('File loading - Failed!')
# let us check our training dataset

df_train.info()
df_train.describe(include=['O'])
df_train[df_train.Ticket=='113803']
df_train.describe()
df_train[['Name','Ticket','Fare','Embarked']][df_train.Fare > 500]
print('Total Passengers aboard:',Counter(df_train.Sex))

print('Total Passengers survived:',Counter((df_train['Sex'][df_train['Survived']==1])))

sns.countplot(x='Survived',hue='Sex',data=df_train)
print('Total Passengers per Port:',Counter(df_train.Embarked))

print('Total Passengers survived per Port:',Counter((df_train['Embarked'][df_train['Survived']==1])))

print('Total Passengers died per Port:',Counter((df_train['Embarked'][df_train['Survived']==0])))

sns.countplot(x='Survived',hue='Embarked',data=df_train)
for i in ['Pclass','Age','SibSp','Parch','Fare']: 

    plot = sns.FacetGrid(df_train,col='Survived')

    plot.map(plt.hist,i,bins=20)
# basic statistics on the 'survived' group

df_train[['Pclass','Age','SibSp','Parch','Fare']][df_train.Survived == 1].describe()
# basic statistics on the 'did not survive' group

df_train[['Pclass','Age','SibSp','Parch','Fare']][df_train.Survived == 0].describe()
# Count per Pclass of those who survived

print('Survived Count per Pclass:',Counter(df_train['Pclass'][df_train['Survived']==1]))
# Count per SibSp 

print('SibSp count:',Counter(df_train['SibSp'][df_train['SibSp']>0]))

print('Survived SibSp count:',Counter(df_train['SibSp'][df_train['Survived']==1]))

print('Died SibSp count:',Counter(df_train['SibSp'][df_train['Survived']==0]))

print('Survived gender count for SibSp>0:',Counter(df_train['Sex'][(df_train['Survived']==1)&(df_train['SibSp']>0)]))

print('Survived class count for SibSp>0:',Counter(df_train['Pclass'][(df_train['Survived']==1)&(df_train['SibSp']>0)]))

print('Survived gender count for SibSp=0:',Counter(df_train['Sex'][(df_train['Survived']==1)&(df_train['SibSp']==0)]))

print('Survived class count for SibSp=0:',Counter(df_train['Pclass'][(df_train['Survived']==1)&(df_train['SibSp']==0)]))
# Count per Parch 

print('Parch count:',Counter(df_train['Parch'][df_train['Parch']>0]))

print('Survived Parch count:',Counter(df_train['Parch'][df_train['Survived']==1]))

print('Died Parch count:',Counter(df_train['Parch'][df_train['Survived']==0]))

print('Survived gender count for Parch>0:',Counter(df_train['Sex'][(df_train['Survived']==1)&(df_train['Parch']>0)]))

print('Survived class count for Parch>0:',Counter(df_train['Pclass'][(df_train['Survived']==1)&(df_train['Parch']>0)]))

print('Survived gender count for Parch=0:',Counter(df_train['Sex'][(df_train['Survived']==1)&(df_train['Parch']==0)]))

print('Survived class count for Parch=0:',Counter(df_train['Pclass'][(df_train['Survived']==1)&(df_train['Parch']==0)]))
print('Fare unique count:',len(df_train.Fare.unique()))
col_list=['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']

val_list=['drop','target','use','drop','use (to convert)','use (to complete)','use','use','drop','use','drop','use (to complete, to convert)']

pd.DataFrame(val_list,index=col_list,columns=['valid feature'])
df_train[(df_train.Embarked.isnull())]
df_train[(df_train.Pclass == 1)&(df_train.Fare>78)&(df_train.Fare<82)&(df_train['Cabin'].str[:1]=='B')]
def fillEmbarked(df,pclass,fare,cabin):

    """ Replace NaN with either S/C/Q, depending on Pclass, nearest Fare, & Cabin """

    temp = df[(df.Pclass == pclass)&(df.Fare>(fare-2))&(df_train.Fare<(fare+2))&(df_train['Cabin'].str[:1]==cabin)]

    return temp.Embarked.mode()[0]
df_train.Embarked.fillna(fillEmbarked(df_train,1,80,'B'),inplace=True)



# the 2 records with 'Embarked' == NaN has been updated.

df_train[df_train.Ticket=='113572']
df_train[(df_train.Age.isnull())]
def fillAge(df):

    """ Replace NaN values with the average age per title usage from name field """

    for i in ['Mr. ','Mrs.','Miss.','Ms. ','Master.','Dr. ']:

        filter1 = (df.Age.isnull())&(df.Name.str.contains(i))

        if i == 'Ms. ':

            replace1 = round(df['Age'][df.Name.str.contains('Miss.')].mean())

        else:

            replace1 = round(df['Age'][df.Name.str.contains(i)].mean())

        df['Age'][filter1] = df['Age'][filter1].fillna(replace1)

    return df
df_train[df_train.Ticket=='330877']
fillAge(df_train)



# sample record Ticket =='330877' with Age == NaN has been updated

df_train[df_train.Ticket=='330877']
def convGenCat(df):

    """ Converts 'Sex' alphanumeric category to numerical category.

        Male:1 ; Female:2

    """

    df['Sex'] = df['Sex'].map({'male':1,'female':2}).astype(int)

    return df



def convPortCat(df):

    """ Converts 'Embarked' alphanumeric category to numerical category.

        S:1 ; C:2; Q:3

    """

    df['Embarked'] = df['Embarked'].map({'S':1,'C':2,'Q':3}).astype(int)

    return df
convGenCat(df_train)

convPortCat(df_train)



# cleaned training data

df_train.head()
# FINAL FEATURES AND TARGET VARIABLES

X = df_train[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]

y = df_train.Survived



# FEATURE VARIABLES

X
from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
from sklearn.model_selection import cross_val_score



def crossValScore(algorithm,X,y):

    """ Input: classifier algorithm, X features, y target

        Returns: mean accuracy of the classifier algorithm via cross validation scoring

    """

    clf = algorithm

    return cross_val_score(clf,X,y,cv=10,scoring='accuracy').mean() 
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

from sklearn.tree import DecisionTreeClassifier



tree = DecisionTreeClassifier()



# fit and predict using train-test data

tree.fit(X_train,y_train)

y_pred = tree.predict(X_test)



# evaluate train-test-split prediction

print('CLF REPORT:\n',classification_report(y_test,y_pred))

print('CONFUSION MATRIX:\n',confusion_matrix(y_test,y_pred))



tts_tree_score = accuracy_score(y_test,y_pred)

print('TTS ACCURACY SCORE:',tts_tree_score)



# fit predict evaluate using cross-val data

cv_tree_score = crossValScore(tree,X,y)

print('CV ACCURACY SCORE:',cv_tree_score)
from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression(solver='liblinear')



# fit and predict using train test data 

logreg.fit(X_train,y_train)

y_pred = logreg.predict(X_test)



# evaluate train-test-split prediction

print('CLF REPORT:\n',classification_report(y_test,y_pred))

print('CONFUSION MATRIX:\n',confusion_matrix(y_test,y_pred))



tts_logreg_score = accuracy_score(y_test,y_pred)

print('TTS ACCURACY SCORE:',tts_logreg_score)



# fit predict evaluate using cross-val data

cv_logreg_score = crossValScore(logreg,X,y)

print('CV ACCURACY SCORE:',cv_logreg_score)
from sklearn.svm import SVC



svm = SVC()



# fit and predict using train test data 

svm.fit(X_train,y_train)

y_pred = svm.predict(X_test)



# evaluate train-test-split prediction

print('CLF REPORT:\n',classification_report(y_test,y_pred))

print('CONFUSION MATRIX:\n',confusion_matrix(y_test,y_pred))



tts_svm_score = accuracy_score(y_test,y_pred)

print('TTS ACCURACY SCORE:',tts_svm_score)



# fit predict evaluate using cross-val data

cv_svm_score = crossValScore(svm,X,y)

print('CV ACCURACY SCORE:',cv_svm_score)
from sklearn.naive_bayes import GaussianNB



nb = GaussianNB()



# fit and predict using train test data 

nb.fit(X_train,y_train)

y_pred = nb.predict(X_test)



# evaluate train-test-split prediction

print('CLF REPORT:\n',classification_report(y_test,y_pred))

print('CONFUSION MATRIX:\n',confusion_matrix(y_test,y_pred))



tts_nb_score = accuracy_score(y_test,y_pred)

print('TTS ACCURACY SCORE:',tts_nb_score)



# fit predict evaluate using cross-val data

cv_nb_score = crossValScore(nb,X,y)

print('CV ACCURACY SCORE:',cv_nb_score)
models = pd.DataFrame({

    'Model': ['Decision Tree','Logistics Regression','SVM','Naive Bayes'],

    'TTS Score': [tts_tree_score,tts_logreg_score,tts_svm_score,tts_nb_score],

    'CV Score' : [cv_tree_score,cv_logreg_score,cv_svm_score,cv_nb_score]})

models
df_test.head()
df_test.info()
# complete the 'Age' feature

fillAge(df_test)



# complete the 'Fare' feature

df_test['Fare'].fillna(df_test['Fare'].dropna().median(),inplace=True)



# convert 'Sex' and 'Embarked' to categorical numeric features

convGenCat(df_test)

convPortCat(df_test)
# FINAL FEATURE VARIABLES

X = df_test[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]

X
y_pred = logreg.predict(X)
submission = pd.DataFrame({

        "PassengerId": df_test['PassengerId'],

        "Survived": y_pred

    })

submission.to_csv('titanic.csv',index=False)