# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np #

import pandas as pd 

import string

import json

from patsy import dmatrices

from operator import itemgetter

from sklearn.ensemble import RandomForestClassifier 

from sklearn.model_selection import cross_val_score 

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold

from sklearn.pipeline import Pipeline

from sklearn import preprocessing

from sklearn.metrics import classification_report

from sklearn.externals import joblib



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

train_file="train.csv"

test_file="test.csv"

seed= 0

print (train_file,seed)



def report(grid_scores, n_top=3):

    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]

    for i, score in enumerate(top_scores):

        print("Model with rank: {0}".format(i + 1))

        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(

              score.mean_validation_score,

              np.std(score.cv_validation_scores)))

        print("Parameters: {0}".format(score.parameters))

        print("")



def substrings_in_string(big_string, substrings):

    for substring in substrings:

        if big_string.find( substring) != -1:

            return substring

    print (big_string)

    return np.nan



le = preprocessing.LabelEncoder()

enc=preprocessing.OneHotEncoder()



def clean_and_munge_data(df):

    df.Fare = df.Fare.map(lambda x: np.nan if x==0 else x)

    #title list 

    title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',

                'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',

                'Don', 'Jonkheer']

    df['Title']=df['Name'].map(lambda x: substrings_in_string(x, title_list))



    def replace_titles(x):

        title=x['Title']

        if title in ['Mr','Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:

            return 'Mr'

        elif title in ['Master']:

            return 'Master'

        elif title in ['Countess', 'Mme','Mrs']:

            return 'Mrs'

        elif title in ['Mlle', 'Ms','Miss']:

            return 'Miss'

        elif title =='Dr':

            if x['Sex']=='Male':

                return 'Mr'

            else:

                return 'Mrs'

        elif title =='':

            if x['Sex']=='Male':

                return 'Master'

            else:

                return 'Miss'

        else:

            return title



    df['Title']=df.apply(replace_titles, axis=1)



    #family size

    df['Family_Size']=df['SibSp']+df['Parch']

    df['Family']=df['SibSp']*df['Parch']





    df.loc[ (df.Fare.isnull())&(df.Pclass==1),'Fare'] =np.median(df[df['Pclass'] == 1]['Fare'].dropna())

    df.loc[ (df.Fare.isnull())&(df.Pclass==2),'Fare'] =np.median( df[df['Pclass'] == 2]['Fare'].dropna())

    df.loc[ (df.Fare.isnull())&(df.Pclass==3),'Fare'] = np.median(df[df['Pclass'] == 3]['Fare'].dropna())



    df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)



    df['AgeFill']=df['Age']

    mean_ages = np.zeros(4)

    mean_ages[0]=np.average(df[df['Title'] == 'Miss']['Age'].dropna())

    mean_ages[1]=np.average(df[df['Title'] == 'Mrs']['Age'].dropna())

    mean_ages[2]=np.average(df[df['Title'] == 'Mr']['Age'].dropna())

    mean_ages[3]=np.average(df[df['Title'] == 'Master']['Age'].dropna())

    df.loc[ (df.Age.isnull()) & (df.Title == 'Miss') ,'AgeFill'] = mean_ages[0]

    df.loc[ (df.Age.isnull()) & (df.Title == 'Mrs') ,'AgeFill'] = mean_ages[1]

    df.loc[ (df.Age.isnull()) & (df.Title == 'Mr') ,'AgeFill'] = mean_ages[2]

    df.loc[ (df.Age.isnull()) & (df.Title == 'Master') ,'AgeFill'] = mean_ages[3]



    df['AgeCat']=df['AgeFill']

    df.loc[ (df.AgeFill<=10) ,'AgeCat'] = 'child'

    df.loc[ (df.AgeFill>60),'AgeCat'] = 'aged'

    df.loc[ (df.AgeFill>10) & (df.AgeFill <=30) ,'AgeCat'] = 'adult'

    df.loc[ (df.AgeFill>30) & (df.AgeFill <=60) ,'AgeCat'] = 'senior'



    df.Embarked = df.Embarked.fillna('S')





    df.loc[ df.Cabin.isnull()==True,'Cabin'] = 0.5

    df.loc[ df.Cabin.isnull()==False,'Cabin'] = 1.5



    df['Fare_Per_Person']=df['Fare']/(df['Family_Size']+1)



    #Age times class

    df['AgeClass']=df['AgeFill']*df['Pclass']

    df['ClassFare']=df['Pclass']*df['Fare_Per_Person']





    df['HighLow']=df['Pclass']

    df.loc[ (df.Fare_Per_Person<8) ,'HighLow'] = 'Low'

    df.loc[ (df.Fare_Per_Person>=8) ,'HighLow'] = 'High'



    le.fit(df['Sex'] )

    x_sex=le.transform(df['Sex'])

    df['Sex']=x_sex.astype(np.float)



    le.fit( df['Ticket'])

    x_Ticket=le.transform( df['Ticket'])

    df['Ticket']=x_Ticket.astype(np.float)



    le.fit(df['Title'])

    x_title=le.transform(df['Title'])

    df['Title'] =x_title.astype(np.float)



    le.fit(df['HighLow'])

    x_hl=le.transform(df['HighLow'])

    df['HighLow']=x_hl.astype(np.float)



    le.fit(df['AgeCat'])

    x_age=le.transform(df['AgeCat'])

    df['AgeCat'] =x_age.astype(np.float)



    le.fit(df['Embarked'])

    x_emb=le.transform(df['Embarked'])

    df['Embarked']=x_emb.astype(np.float)



    df = df.drop(['PassengerId','Name','Age','Cabin'], axis=1) #remove Name,Age and PassengerId

    return df



traindf=pd.read_csv("../input/train.csv")

df=clean_and_munge_data(traindf)



formula_ml='Survived~Pclass+C(Title)+Sex+C(AgeCat)+Fare_Per_Person+Fare+Family_Size' 



y_train, x_train = dmatrices(formula_ml, data=df, return_type='dataframe')

y_train = np.asarray(y_train).ravel()

print (y_train.shape,x_train.shape)



X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=0.2,random_state=seed)



clf=RandomForestClassifier(n_estimators=500, criterion='entropy', max_depth=5, min_samples_split=2,

  min_samples_leaf=1, max_features='auto',    bootstrap=False, oob_score=False, n_jobs=1, random_state=seed,

  verbose=0)

param_grid = dict( )

pipeline=Pipeline([ ('clf',clf) ])

strarifiedCV = StratifiedShuffleSplit(n_splits=10,test_size=0.2, random_state=0)

grid_search = GridSearchCV(pipeline,param_grid=param_grid,verbose=3,scoring='accuracy',

                           cv =strarifiedCV,).fit(X_train, Y_train)
print("Best score: %0.3f" % grid_search.best_score_)

print(grid_search.best_estimator_)

report(grid_search.grid_scores_)

 

print('-----grid search end------------')

print ('on all train set')

scores = cross_val_score(grid_search.best_estimator_, x_train, y_train,cv=3,scoring='accuracy')

print (scores.mean(),scores)

print ('on test set')

scores = cross_val_score(grid_search.best_estimator_, X_test, Y_test,cv=3,scoring='accuracy')

print (scores.mean(),scores)

print(classification_report(Y_train, grid_search.best_estimator_.predict(X_train) ))

print('test data')

print(classification_report(Y_test, grid_search.best_estimator_.predict(X_test) ))

data_test = pd.read_csv("../input/test.csv")

df_test=clean_and_munge_data(data_test)

print(df_test.shape)
from patsy import dmatrix

formula_ml='Pclass+C(Title)+Sex+C(AgeCat)+Fare_Per_Person+Fare+Family_Size' 

df_test1 = dmatrix(formula_ml, data=df_test, return_type='dataframe')
predictions = grid_search.best_estimator_.predict(df_test1)

result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})

result.to_csv("RF_cv.csv", index=False)