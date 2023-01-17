import numpy as np #

import pandas as pd 

import string

import json

from patsy import dmatrices

from operator import itemgetter

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier, ExtraTreesClassifier,AdaBoostClassifier

from sklearn.model_selection import cross_val_score 

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold

from sklearn.pipeline import Pipeline

from sklearn import preprocessing

from sklearn.metrics import classification_report

from sklearn.externals import joblib

import xgboost as xgb

train_df=pd.read_csv("../input/train.csv")

test_df=pd.read_csv("../input/test.csv")

seed= 42
#report grid search score for finding the best parameters 

def report(grid_scores, n_top=3):

    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]

    for i, score in enumerate(top_scores):

        print("Model with rank: {0}".format(i + 1))

        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(

              score.mean_validation_score,

              np.std(score.cv_validation_scores)))

        print("Parameters: {0}".format(score.parameters))

        print("")
#substring function for finding titles in name columns 

def substrings_in_string(big_string, substrings):

    for substring in substrings:

        if big_string.find(substring) != -1:

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

    

    #replace mapped title into different catogories 

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

    

    #new feature title 

    df['Title']=df.apply(replace_titles, axis=1)



    #new feature family size

    df['Family_Size']=df['SibSp']+df['Parch']

    df['Family']=df['SibSp']*df['Parch']



    #Handling missing value in Fare 

    #fill in missing fare with median value based on which class they are 

    df.loc[ (df.Fare.isnull())&(df.Pclass==1),'Fare'] =np.median(df[df['Pclass'] == 1]['Fare'].dropna())

    df.loc[ (df.Fare.isnull())&(df.Pclass==2),'Fare'] =np.median( df[df['Pclass'] == 2]['Fare'].dropna())

    df.loc[ (df.Fare.isnull())&(df.Pclass==3),'Fare'] = np.median(df[df['Pclass'] == 3]['Fare'].dropna())

    

    #mapping set to gender 

    df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    

    #fill age with mean age base on different title 

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

    

    #new feature age category 

    #better to transform continuse age value into different age bin 

    df['AgeCat']=df['AgeFill']

    df.loc[ (df.AgeFill<=10) ,'AgeCat'] = 'child'

    df.loc[ (df.AgeFill>60),'AgeCat'] = 'aged'

    df.loc[ (df.AgeFill>10) & (df.AgeFill <=30) ,'AgeCat'] = 'adult'

    df.loc[ (df.AgeFill>30) & (df.AgeFill <=60) ,'AgeCat'] = 'senior'



    df.Embarked = df.Embarked.fillna('S')



    df.loc[ df.Cabin.isnull()==True,'Cabin'] = 0.5

    df.loc[ df.Cabin.isnull()==False,'Cabin'] = 1.5



    df['Fare_Per_Person']=df['Fare']/(df['Family_Size']+1)



    #new feature based on two highly relevant feature age and pclass 

    #create new features 

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

train_df_feature = clean_and_munge_data(train_df)
formula_ml='Survived~Pclass+C(Title)+Sex+C(AgeCat)+Fare_Per_Person+Fare+Family_Size' 



y_train, x_train = dmatrices(formula_ml, data=train_df_feature, return_type='dataframe')

y_train = np.asarray(y_train).ravel()

print (y_train.shape,x_train.shape)
# feature_train = pd.concat([x_train,y_train], axis=1)

# print(feature_train.shape)
#feature_train.to_csv("data/baseline_feature.csv", index=False)
X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=0.2,random_state=seed)
print("x_trian shape",X_train.shape)

print("Y_train shape",Y_train.shape)

print("X_test shape",X_test.shape)

print("Y_test shape",Y_test.shape)
#Regression Tree 

rf_clf=RandomForestClassifier(n_estimators=500, criterion='entropy', max_depth=5, min_samples_split=2,

                           min_samples_leaf=1, max_features='auto',    bootstrap=False, oob_score=False, 

                           n_jobs=1, random_state=seed,verbose=0)





#Ada Boosting use gridsearch find best parameters

ada_clf = AdaBoostClassifier(random_state=seed, n_estimators=50,algorithm='SAMME',learning_rate=0.75 )



#Extra Trees 

et_clf = ExtraTreesClassifier(n_estimators=500, max_features= 'sqrt',max_depth=8,criterion='entropy',

                              n_jobs = 50,random_state =seed, verbose =0)





#Gradient Boosting 

gbm_clf = GradientBoostingClassifier(learning_rate=0.1,n_estimators=50,min_samples_split=2,max_depth=5,

                                     min_samples_leaf=5,max_features='sqrt',

                                     loss='exponential',random_state=42,verbose=0)







#SVC

# svc_params = {

#     'kernel' : ['linear'],

#     'C' : [0.025],

#     'gamma':[0.001, 0.01, 0.1]

#     }

# svc_clf=SVC()



%pdb
stratifiedCV = StratifiedShuffleSplit(n_splits = 10, test_size=0.2, random_state =0)

param_grid = dict( )

def grid_cv(clf,name):

    grid_search = GridSearchCV(clf,verbose = 3, param_grid = param_grid,scoring ='accuracy', cv = stratifiedCV)

    grid_search.fit(X_train,Y_train)

    #print(name, "Best Params:" + str(grid_search.best_params_))

    print(name, "Best Score:" + str(grid_search.best_score_))

    print('-----grid search end------------')

    print ('on all train set')

    scores = cross_val_score(grid_search.best_estimator_, x_train, y_train,cv=3,scoring='accuracy')

    print (scores.mean(),scores)

    print ('on test set')

    scores = cross_val_score(grid_search.best_estimator_, X_test, Y_test,cv=3,scoring='accuracy')

    print (scores.mean(),scores)

#     predictions

#     predictions = grid_search.best_estimator_.predict(feature_test)

    

    return grid_search.best_estimator_

feature_test=clean_and_munge_data(test_df)

print(feature_test.shape)
from patsy import dmatrix

formula_ml='Pclass+C(Title)+Sex+C(AgeCat)+Fare_Per_Person+Fare+Family_Size' 

feature_test = dmatrix(formula_ml, data=feature_test, return_type='dataframe')

print (feature_test.shape)
#feature_test.to_csv("data/baseline_feature_test.csv", index=False)
print (feature_test.shape)

print (x_train.shape)
print (y_train.shape)
#Ramdom Forest 

rf_estimator = grid_cv(rf_clf, 'randomForest')  
rf_train_predict = rf_estimator.predict(x_train).reshape(-1, 1)

rf_predict = rf_estimator.predict(feature_test).reshape(-1, 1)

print(rf_train_predict.shape)

print(rf_predict.shape)
ada_estimator = grid_cv(ada_clf, 'AdaBoosting')  
ada_train_predict = ada_estimator.predict(x_train).reshape(-1, 1)

ada_predict = ada_estimator.predict(feature_test).reshape(-1, 1)

print(ada_train_predict.shape)

print(ada_predict.shape)
gbm_estimator=grid_cv(gbm_clf,"GradientBoosting")
gbm_train_predict = gbm_estimator.predict(x_train).reshape(-1, 1)

gbm_predict = gbm_estimator.predict(feature_test).reshape(-1, 1)

print(gbm_train_predict.shape)

print(gbm_predict.shape)
et_estimator = grid_cv(et_clf,"ExtraTree")
et_train_predict = et_estimator.predict(x_train).reshape(-1, 1)

et_predict = et_estimator.predict(feature_test).reshape(-1, 1)

print(et_train_predict.shape)

print(et_predict.shape)
ada_predict_change = ada_predict.ravel()
x_train = np.concatenate((rf_train_predict, gbm_train_predict,ada_train_predict, et_train_predict), axis=1)

x_test = np.concatenate(( rf_predict, gbm_predict,ada_predict, et_predict), axis=1)



xgb_clf = xgb.XGBClassifier(n_estimators=2000,max_depth=4,min_child_weight=2,gamma=0.9,colsample_bytree=0.8,

                              objective='binary:logistic', nthread=-1,scale_pos_weight=1).fit(x_train,y_train)

xgb_prediction = xgb_clf.predict(x_test)
PassengerId = test_df['PassengerId']

StackingSubmission = pd.DataFrame({ 'PassengerId': PassengerId,

                            'Survived': xgb_prediction.astype(np.int32) })

StackingSubmission.to_csv("baselineCVSubmission.csv", index=False)