import pandas as pd

import numpy as np

from pandas import set_option

from pandas.plotting import scatter_matrix

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.linear_model import Perceptron

from xgboost import XGBClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.linear_model import SGDClassifier

from matplotlib import pyplot

from sklearn.model_selection import RandomizedSearchCV

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.linear_model import ElasticNet

from sklearn import linear_model

from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.svm import SVR

import featuretools as ft

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.ensemble import ExtraTreesRegressor

def GetBasedModel():

    models = []

   #models.append(('LN', linear_model.LinearRegression())) #(Regression - Supervised)

    #models.append(('RID', Ridge())) #(Regression - Supervised)

    #models.append(('LSO', Lasso())) #(Regression - Supervised)

    #models.append(('EN',  ElasticNet())) #(Regression - Supervised)

    #models.append(('KNNR', KNeighborsRegressor())) #(Regression - Supervised)

    #models.append(('CARTR', DecisionTreeRegressor())) #(Regression - Supervised)

    #models.append(('SVR', SVR())) #(Regression - Supervised)

    models.append(('PER', Perceptron( max_iter=1000,tol=1e-3,n_jobs=-1)))  #(Binary-Classification-Supervised)

    models.append(('XGB', XGBClassifier( n_jobs=-1))) #(Regression-Classification-binary-multi-Supervised)

    models.append(('LSVC',  LinearSVC(max_iter=10000))) #(Classification-Supervised)

    models.append(('SGDC',  SGDClassifier(max_iter=1000,tol=1e-3,n_jobs=-1))) #(Classification-Supervised)

    models.append(('LR', LogisticRegression(solver='lbfgs',max_iter=1000, multi_class='auto',n_jobs=-1))) #(Classification-Binary-Supervised)

    models.append(('LDA', LinearDiscriminantAnalysis())) #(Classification-binary-multi-Supervised)

    models.append(('KNN', KNeighborsClassifier(n_jobs=-1))) #(Classification-Supervised)

    models.append(('CART', DecisionTreeClassifier())) #(Clasification-Supervised)

    models.append(('NB', GaussianNB())) #(Clasification-Supervised)

    models.append(('SVM', SVC(gamma='scale',max_iter=1000))) #(Clasification-Supervised)

    models.append(('AB', AdaBoostClassifier())) #(Clasification-Supervised)

    models.append(('GBM', GradientBoostingClassifier())) #(Clasification-Supervised)

    models.append(('RF', RandomForestClassifier(n_estimators=100,n_jobs=-1))) #(Clasification-Supervised)

    models.append(('ET', ExtraTreesClassifier(n_estimators=100,n_jobs=-1))) #(Clasification-Supervised)



    

    return models
def BasedLine2(X_train, y_train,models):

    # Test options and evaluation metric

    num_folds = 10

    scoring = 'accuracy'



    results = []

    names = []

    for name, model in models:

        kfold = KFold(n_splits=num_folds, random_state=42)

        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)

        results.append(cv_results)

        names.append(name)

        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

        print(msg)

        

    return names, results
class PlotBoxR(object):

    

    

    def __Trace(self,nameOfFeature,value): 

    

        trace = go.Box(

            y=value,

            name = nameOfFeature,

            marker = dict(

                color = 'rgb(0, 128, 128)',

            )

        )

        return trace



    def PlotResult(self,names,results):

        

        data = []



        for i in range(len(names)):

            data.append(self.__Trace(names[i],results[i]))





        py.iplot(data)
def ScoreDataFrame(names,results):

    def floatingDecimals(f_val, dec=3):

        prc = "{:."+str(dec)+"f}" 

    

        return float(prc.format(f_val))



    scores = []

    for r in results:

        scores.append(floatingDecimals(r.mean(),4))



    scoreDataFrame = pd.DataFrame({'Model':names, 'Score': scores})

    return scoreDataFrame
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import RobustScaler

from sklearn.preprocessing import Normalizer





def GetScaledModel(nameOfScaler):

    

    if nameOfScaler == 'standard':

        scaler = StandardScaler()

    elif nameOfScaler =='minmax':

        scaler = MinMaxScaler()

    elif nameOfScaler == 'Robust':

        scaler= RobustScaler()

    elif nameOfScaler == 'normalizer':

        scaler= Normalizer()

        

    pipelines = []

    #pipelines.append((nameOfScaler+'LN'  , Pipeline([('Scaler', scaler),('LN',  linear_model.LinearRegression())])))

    #pipelines.append((nameOfScaler+'RID'  , Pipeline([('Scaler', scaler),('RID',  Ridge())])))

    #pipelines.append((nameOfScaler+'LSO'  , Pipeline([('Scaler', scaler),('LSO',  Lasso())])))

    #pipelines.append((nameOfScaler+'EN'  , Pipeline([('Scaler', scaler),('EN',  ElasticNet())])))

    #pipelines.append((nameOfScaler+'kNNR'  , Pipeline([('Scaler', scaler),('KNNR',  KNeighborsRegressor())])))

    #pipelines.append((nameOfScaler+'CARTR'  , Pipeline([('Scaler', scaler),('CARTR',  DecisionTreeRegressor())])))

    #pipelines.append((nameOfScaler+'SVR'  , Pipeline([('Scaler', scaler),('SVR',  SVR())])))

    pipelines.append((nameOfScaler+'PER'  , Pipeline([('Scaler', scaler),('PER', Perceptron( max_iter=1000,tol=1e-3,n_jobs=-1))])))

    pipelines.append((nameOfScaler+'XGB'  , Pipeline([('Scaler', scaler),('XGB', XGBClassifier(n_jobs=-1))])))

    pipelines.append((nameOfScaler+'LSVC'  , Pipeline([('Scaler', scaler),('LSVC',  LinearSVC(max_iter=1000))])))

    pipelines.append((nameOfScaler+'SGDC'  , Pipeline([('Scaler', scaler),('SGDC',  SGDClassifier(max_iter=1000,tol=1e-3,n_jobs=-1))])))

    pipelines.append((nameOfScaler+'LR'  , Pipeline([('Scaler', scaler),('LR'  , LogisticRegression(solver='lbfgs',max_iter=1000, multi_class='auto',n_jobs=-1))])))

    pipelines.append((nameOfScaler+'LDA' , Pipeline([('Scaler', scaler),('LDA' , LinearDiscriminantAnalysis())])))

    pipelines.append((nameOfScaler+'KNN' , Pipeline([('Scaler', scaler),('KNN' , KNeighborsClassifier(n_jobs=-1))])))

    pipelines.append((nameOfScaler+'CART', Pipeline([('Scaler', scaler),('CART', DecisionTreeClassifier())])))

    pipelines.append((nameOfScaler+'NB'  , Pipeline([('Scaler', scaler),('NB'  , GaussianNB())])))

    pipelines.append((nameOfScaler+'SVM' , Pipeline([('Scaler', scaler),('SVM' , SVC(gamma='scale',max_iter=1000))])))

    pipelines.append((nameOfScaler+'AB'  , Pipeline([('Scaler', scaler),('AB'  , AdaBoostClassifier())])  ))

    pipelines.append((nameOfScaler+'GBM' , Pipeline([('Scaler', scaler),('GMB' , GradientBoostingClassifier())])  ))

    pipelines.append((nameOfScaler+'RF'  , Pipeline([('Scaler', scaler),('RF'  , RandomForestClassifier(n_estimators=100,n_jobs=-1))])))

    pipelines.append((nameOfScaler+'ET'  , Pipeline([('Scaler', scaler),('ET'  , ExtraTreesClassifier(n_estimators=100,n_jobs=-1))])))

    

    return pipelines 
class RandomSearch(object):

    

    def __init__(self,X_train,y_train,model,hyperparameters):

        

        self.X_train = X_train

        self.y_train = y_train

        self.model = model

        self.hyperparameters = hyperparameters

        

    def RandomSearch(self):

        # Create randomized search 10-fold cross validation and 100 iterations

        cv = 10

        clf = RandomizedSearchCV(self.model,

                                 self.hyperparameters,

                                 random_state=1,

                                 n_iter=100,

                                 cv=cv,

                                 verbose=0,

                                 n_jobs=-1,

                                 )

        # Fit randomized search

        best_model = clf.fit(self.X_train, self.y_train)

        message = (best_model.best_score_, best_model.best_params_)

        print("Best: %f using %s" % (message))



        return best_model,best_model.best_params_

    

    def BestModelPridict(self,X_test):

        

        best_model,_ = self.RandomSearch()

        pred = best_model.predict(X_test)

        return pred
class GridSearch(object):

    

    def __init__(self,X_train,y_train,model,hyperparameters):

        

        self.X_train = X_train

        self.y_train = y_train

        self.model = model

        self.hyperparameters = hyperparameters

        

    def GridSearch(self):

        # Create randomized search 10-fold cross validation and 100 iterations

        cv = 10

        clf = GridSearchCV(self.model,

                                 self.hyperparameters,

                                 cv=cv,

                                 verbose=0,

                                 n_jobs=-1,

                                 )

        # Fit randomized search

        best_model = clf.fit(self.X_train, self.y_train)

        message = (best_model.best_score_, best_model.best_params_)

        print("Best: %f using %s" % (message))



        return best_model,best_model.best_params_

    

    def BestModelPridict(self,X_test):

        

        best_model,_ = self.GridSearch()

        pred = best_model.predict(X_test)

        return pred
df_train= pd.read_csv("../input//train.csv")

df_validation= pd.read_csv("../input/test.csv")
df_train.head(5)
df_validation.head(5)
df_train.shape, df_validation.shape
missing_data=df_train.isnull()

missing_data.sum()
df_train.describe(include='O')
agmean= df_train["Age"].median()

agmean
df_train["Age"].replace(np.nan,agmean,inplace=True)
df_train["Embarked"].replace(np.nan,"S",inplace=True)

df_train=df_train.drop(['Cabin'],axis=1)
missing_data2= df_train.isnull()

missing_data2.sum()
df_train.columns
df_train=df_train[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp',

       'Parch', 'Fare', 'Embarked']]
df_train.hist(bins=10, figsize=(20,15))

plt.show()
df_train.plot(kind='box', subplots=True, layout=(5,5), sharex=False, sharey=False, figsize=(20,20))

plt.show()
scatter_matrix(df_train,figsize=(20,20))

plt.show()
df_train = pd.get_dummies(df_train)
df_train.corr()
corr_matrix= df_train.corr()
corr_matrix['Survived'].sort_values(ascending=False)
sns.heatmap(df_train.corr(), vmin=-1, vmax=1.0, annot=True)

plt.show()
X =  df_train.drop(['Survived'],axis=1)

y = df_train['Survived']  
model = ExtraTreesClassifier(n_estimators=100,n_jobs=-1)

model.fit(X,y)

print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers

#plot graph of feature importances for better visualization

feat_importances = pd.Series(model.feature_importances_, index=X.columns)

feat_importances.nlargest(20).plot(kind='barh')

plt.show()
feat_importances.nlargest(75)
feat_importances
df_clean_train = df_train
agmean= df_validation["Age"].median()

faremean=df_validation['Fare'].median()



df_validation["Age"].replace(np.nan,agmean,inplace=True)

df_validation["Fare"].replace(np.nan,faremean,inplace=True)

df_validation["Embarked"].replace(np.nan,"S",inplace=True)





df_validation=df_validation.drop(['Cabin'],axis=1)



missing_data2= df_validation.isnull()

missing_data2.sum()
df_clean_test=df_validation
df_clean_test=pd.get_dummies(df_clean_test)
X =  df_clean_train.drop(['Survived'],axis=1).values

y = df_clean_train['Survived'].values
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.20, random_state=42)
X_train.shape, X_test.shape , y_train.shape, y_test.shape
models = GetBasedModel()

names,results = BasedLine2(X_train, y_train,models)

fig = pyplot.figure()

fig.suptitle( ' Algorithm Comparison ' )

ax = fig.add_subplot(111)

pyplot.boxplot(results)

ax.set_xticklabels(names)

pyplot.show()
basedLineScore = ScoreDataFrame(names,results)

basedLineScore.sort_values(by='Score', ascending=False)
models = GetScaledModel('standard')

names,results = BasedLine2(X_train, y_train,models)



scaledScoreStandard = ScoreDataFrame(names,results)

compareModels = pd.concat([basedLineScore,

                           scaledScoreStandard], axis=1)

compareModels
models = GetScaledModel('minmax')

names,results = BasedLine2(X_train, y_train,models)





scaledScoreMinMax = ScoreDataFrame(names,results)

compareModels = pd.concat([basedLineScore,

                           scaledScoreStandard,

                          scaledScoreMinMax], axis=1)

compareModels
models = GetScaledModel('normalizer')

names,results = BasedLine2(X_train, y_train,models)





scaledScoreNormal = ScoreDataFrame(names,results)

compareModels = pd.concat([basedLineScore,

                           scaledScoreStandard,

                          scaledScoreMinMax,scaledScoreNormal], axis=1)

compareModels
models = GetScaledModel('Robust')

names,results = BasedLine2(X_train, y_train,models)





scaledScoreRobust = ScoreDataFrame(names,results)

compareModels = pd.concat([basedLineScore,

                           scaledScoreStandard,

                          scaledScoreMinMax,scaledScoreNormal,scaledScoreRobust], axis=1)

compareModels
def floatingDecimals(f_val, dec=10):

        prc = "{:."+str(dec)+"f}" #first cast decimal as str

    #     print(prc) #str format output is {:.3f}

        return float(prc.format(f_val))
X= df_clean_train[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_female','Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S']].values

y= df_clean_train['Survived'].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state =0)
scaler = StandardScaler().fit(X_train)

rescaledX = scaler.transform(X_train)
rescaledValidationX = scaler.transform(X_test)
model = XGBClassifier()
max_depth=[3] 

learning_rate=[0.1,0.2,0.3] 

n_estimators=[100,200,300] 

#verbosity=[2]

#silent=[None]

objective=['binary:logistic']

booster=['gbtree', 'gblinear','dart']

n_jobs=[-1]

nthread=[None]

gamma=[0,0.1,0.2,1]

min_child_weight=[1,2,3]

max_delta_step=[0,1,2,3]

subsample=[0,1]

#colsample_bytree=[0,1]

#colsample_bylevel=[0,1]

#colsample_bynode=[0,1]

reg_alpha=[0,1,2]

reg_lambda=[1,2,3]

scale_pos_weight=[1,2]

base_score=[0.5,0.6,0.7,0.8,0.9]

random_state=[42]

seed=[None] 

missing=[None]





hyperparameters= dict(max_depth=max_depth, 

learning_rate=learning_rate, 

n_estimators=n_estimators,

#verbosity=verbosity,

#silent=silent,

objective=objective,

booster=booster,

n_jobs=n_jobs,

nthread=nthread,

gamma=gamma,

min_child_weight=min_child_weight,

max_delta_step=max_delta_step,

subsample=subsample,

#colsample_bytree=colsample_bytree,

#colsample_bylevel=colsample_bylevel,

#colsample_bynode=colsample_bynode,

reg_alpha=reg_alpha,

reg_lambda=reg_lambda,

scale_pos_weight=scale_pos_weight,

base_score=base_score,

random_state=random_state,

seed=seed, 

missing=missing

)
RandSearch = RandomSearch(rescaledX,y_train,model,hyperparameters)

# LR_best_model,LR_best_params = LR_RandSearch.RandomSearch()

Prediction =RandSearch.BestModelPridict(rescaledValidationX)
X= df_clean_train[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_female','Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S']].values

y= df_clean_train['Survived'].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state =0)
scaler = StandardScaler().fit(X_train)

rescaledX = scaler.transform(X_train)

model = XGBClassifier(subsample= 1, seed= None, scale_pos_weight=1, reg_lambda= 1, reg_alpha=2, random_state=42, objective= 'binary:logistic', nthread=None, n_jobs=-1, n_estimators= 200, missing=None, min_child_weight= 2, max_depth= 3, max_delta_step= 0, learning_rate=0.2, gamma= 0.2, booster='gbtree', base_score=0.9)

model.fit(rescaledX, y_train)

# estimate accuracy on validation dataset

rescaledValidationX = scaler.transform(X_test)

predictions = model.predict(rescaledValidationX)

model.score(rescaledX, y_train)

model.score(rescaledValidationX, y_test)
final=df_clean_test[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_female','Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S']].values
rescaledFinal = scaler.transform(final)
predictions = model.predict(rescaledFinal)
output = pd.DataFrame({'PassengerId': df_validation.PassengerId,'Survived': predictions})

output.to_csv('submission.csv', index=False)

output.head()
