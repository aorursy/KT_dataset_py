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

from sklearn.metrics import accuracy_score

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.ensemble import AdaBoostRegressor

from sklearn.linear_model import Perceptron

from sklearn.svm import SVC, LinearSVC

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

from boruta import BorutaPy

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import Pipeline

from mlxtend.regressor import StackingRegressor

from sklearn.pipeline import make_pipeline

from sklearn import preprocessing

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.externals import joblib 

from  xgboost import XGBRegressor
def GetBasedModel():

    models = []

    models.append(('LN', linear_model.LinearRegression())) #(Regression - Supervised)

    models.append(('RID', Ridge())) #(Regression - Supervised)

    models.append(('LSO', Lasso(max_iter=4000))) #(Regression - Supervised)

    models.append(('EN',  ElasticNet())) #(Regression - Supervised)

    models.append(('KNNR', KNeighborsRegressor(n_jobs=-1))) #(Regression - Supervised)

    models.append(('CARTR', DecisionTreeRegressor())) #(Regression - Supervised)

    models.append(('AB', AdaBoostRegressor())) #(Regression - Supervised)

    models.append(('GBM',GradientBoostingRegressor())) #(Regression - Supervised)

    models.append(('RFR', RandomForestRegressor(n_jobs=-1,n_estimators=100))) #(Regression - Supervised)

    models.append(('ETR', ExtraTreesRegressor(n_jobs=-1,n_estimators= 100))) #(Regression - Supervised) 

    #models.append(('SVR', SVR(gamma='scale'))) #(Regression - Supervised)

    models.append(('XGB', XGBRegressor(n_jobs=-1))) #(Regression - Supervised)



    

    return models
def BasedLine2(X_train, y_train,models):

    # Test options and evaluation metric

    num_folds = 2

    scoring = 'neg_mean_absolute_error'



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

    #elif nameOfScaler == 'Robust':

        #scaler= RobustScaler()

    #elif nameOfScaler == 'normalizer':

        #scaler= Normalizer()

        

    pipelines = []

    pipelines.append((nameOfScaler+'LN'  , Pipeline([('Scaler', scaler),('LN',  linear_model.LinearRegression())])))

    pipelines.append((nameOfScaler+'RID'  , Pipeline([('Scaler', scaler),('RID',  Ridge())])))

    pipelines.append((nameOfScaler+'LSO'  , Pipeline([('Scaler', scaler),('LSO',  Lasso(max_iter=4000))])))

    pipelines.append((nameOfScaler+'EN'  , Pipeline([('Scaler', scaler),('EN',  ElasticNet())])))

    pipelines.append((nameOfScaler+'kNNR'  , Pipeline([('Scaler', scaler),('KNNR',  KNeighborsRegressor())])))

    pipelines.append((nameOfScaler+'CARTR'  , Pipeline([('Scaler', scaler),('CARTR',  DecisionTreeRegressor())])))

    pipelines.append((nameOfScaler+'AB'  , Pipeline([('Scaler', scaler),('AB', AdaBoostRegressor())])))

    pipelines.append((nameOfScaler+'GBM'  , Pipeline([('Scaler', scaler),('GBM',GradientBoostingRegressor())])))

    pipelines.append((nameOfScaler+'RFR'  , Pipeline([('Scaler', scaler),('RFR', RandomForestRegressor(n_jobs=-1,n_estimators= 100))])))

    pipelines.append((nameOfScaler+'ETR'  , Pipeline([('Scaler', scaler),('ETR', ExtraTreesRegressor(n_jobs=-1,n_estimators= 100))])))

    #pipelines.append((nameOfScaler+'SVR'  , Pipeline([('Scaler', scaler),('SVR',  SVR(gamma='scale'))])))

    pipelines.append((nameOfScaler+'XGB'  , Pipeline([('Scaler', scaler),('XGB',  XGBRegressor(n_jobs=-1))])))

    return pipelines
class RandomSearch(object):

    

    def __init__(self,X_train,y_train,model,hyperparameters):

        

        self.X_train = X_train

        self.y_train = y_train

        self.model = model

        self.hyperparameters = hyperparameters

        

    def RandomSearch(self):

        # Create randomized search 10-fold cross validation and 100 iterations

        cv = 2

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

        cv = 2

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
df= pd.read_csv('../input/weatherHistory.csv')

df.head(5)
df.shape
df.columns
df=df[['Summary', 'Precip Type', 'Temperature (C)',

       'Apparent Temperature (C)', 'Humidity', 'Wind Speed (km/h)',

       'Wind Bearing (degrees)', 'Visibility (km)', 'Loud Cover',

       'Pressure (millibars)', 'Daily Summary']]
missing_data=df.isnull()

missing_data.sum()
df.describe(include='O')
df.dropna(inplace=True)
df.hist(bins=10, figsize=(20,15))

plt.show()
df.plot(kind='box', subplots=True, layout=(5,5), sharex=False, sharey=False, figsize=(20,20))

plt.show()
#scatter_matrix(df,figsize=(20,20))

#plt.show()
df= pd.get_dummies(df)

#df.to_csv('df.csv',index=False)
df.corr()
corr_matrix= df.corr()
corr_matrix['Humidity'].sort_values(ascending=False)
#sns.heatmap(df.corr(), vmin=-1, vmax=1.0, annot=True)

#plt.show()
X =  df.drop(['Humidity'],axis=1)

y = df['Humidity']  
# load X and y

# NOTE BorutaPy accepts numpy arrays only, hence the .values attribute

X = X.values

y = y.values

y = y.ravel()



# define random forest classifier, with utilising all cores and

# sampling in proportion to y labels

rf = RandomForestRegressor(n_jobs=-1)



# define Boruta feature selection method

feat_selector = BorutaPy(rf, n_estimators=5, verbose=2, random_state=42)



# find all relevant features - 5 features should be selected

feat_selector.fit(X, y)
print ('\n Initial features: ',  df.drop(['Humidity'],axis=1).columns.tolist() )

print ('\n Number of selected features:')

print (feat_selector.n_features_)

feature_df = pd.DataFrame( df.drop(['Humidity'],axis=1).columns.tolist(), columns=['features'])

feature_df['rank']=feat_selector.ranking_

feature_df = feature_df.sort_values('rank', ascending=True).reset_index(drop=True)
print ('\n Top %d features:' % feat_selector.n_features_)

print (feature_df.head(feat_selector.n_features_))
selected =  df.drop(['Humidity'],axis=1).columns[feat_selector.support_]

selected
df_selected=df[selected]

df_selected.columns
X = df_selected.values

y = df['Humidity'].values  
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
X = df_selected.values

y = df['Humidity'].values  
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state =42)
#Create Pipeline

estimators=[]

estimators.append(('Standardize',StandardScaler()))

estimators.append(('rfr',RandomForestRegressor(n_jobs=-1)))

pipeline=Pipeline(estimators)
# Declare Hyperparameters

hyperparameters={

'rfr__criterion':['mse','mae'], 

'rfr__n_estimators':np.array([100]), 

'rfr__max_features':[ 'auto','sqrt','log2' ]

}
# Tune model using cross-validation pipeline

rf=GridSearchCV(pipeline,param_grid=hyperparameters,cv=2,scoring = 'neg_mean_absolute_error',n_jobs=-1)

results=rf.fit(X_train,y_train)
#Check best score and best params for the model 

print("Best: %f using %s" % (results.best_score_, results.best_params_))

#means = results.cv_results_['mean_test_score']

#stds = results.cv_results_['std_test_score']

#params = results.cv_results_['params

#for mean, stdev, param in zip(means, stds, params):

    #print("%f (%f) with: %r" % (mean, stdev, param))
final_model=results.best_estimator_

final_model.fit(X_train,y_train)
#Evaluate model on test data

pred=final_model.predict(X_test)



print(mean_absolute_error(y_test, pred))

#print (r2_score(y_test, pred))

#print (mean_squared_error(y_test, pred))