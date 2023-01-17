!ls /kaggle/input/results
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
#get the dataset

import os

import tarfile

from six.moves import urllib

DOWNLOAD_ROOT="https://raw.githubusercontent.com/ageron/handson-ml/master/"

HOUSING_PATH="datasets/housing"

HOUSING_URL=DOWNLOAD_ROOT+HOUSING_PATH+"/housing.tgz"





#download the data

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):

    """

    create datasets/housing directory

    download housing.tgz file

    extract housing.csv

    

    """

    if not os.path.isdir(housing_path):

        os.makedirs(housing_path)

    

    tgz_path=os.path.join(housing_path,"housing.tgz")

    urllib.request.urlretrieve(housing_url,tgz_path)

    housing_tgz=tarfile.open(tgz_path)

    housing_tgz.extractall(path=housing_path)

    housing_tgz.close()



#load the data from csv into pandas df

def load_housing_data(housing_path=HOUSING_PATH):

    csv_path=os.path.join(housing_path,"housing.csv")

    return pd.read_csv(csv_path)
fetch_housing_data()

housing=load_housing_data()

housing.head()
housing.info()
housing.describe()
%matplotlib inline

import matplotlib.pyplot as plt

housing.hist(bins=50,figsize=(20,15))

plt.show()
housing['income_cat']=np.ceil(housing['median_income']/1.5)

housing['income_cat'].where(housing['income_cat']<5,5,inplace=True)  #everything that belongs to cat 6 and beyond is put into cat 5

housing.hist(['income_cat'],figsize=(5,5))
from sklearn.model_selection import StratifiedShuffleSplit



strat_split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=13)

for train_ind,test_ind in strat_split.split(housing,housing['income_cat']):

    strat_train_set=housing.loc[train_ind]

    strat_test_set=housing.loc[test_ind]



    

f,ax=plt.subplots(2,2,sharey=True,sharex=True)

s=housing['income_cat'].value_counts()/len(housing)

ax[0,0].hist(s)

ax[0,0].set_title('total data')



ss=strat_test_set['income_cat'].value_counts()/len(strat_test_set)

ax[1,1].hist(ss)

ax[1,1].set_title('test set')

ss=strat_train_set['income_cat'].value_counts()/len(strat_train_set)

ax[1,0].hist(ss)

ax[1,0].set_title('train set')



#delete the axes[0,1] as it holds no plots

f.delaxes(ax[0,1])

plt.show()
for s in (strat_train_set,strat_test_set):

    s.drop('income_cat',axis=1,inplace=True)

housing=strat_train_set.copy()  #make a copy of the training data for EDA
housing.plot(kind='scatter',x='longitude',y='latitude',alpha=0.4, 

             s=housing['population']/100, #circles of radius s indicates population size

             label='population',

             c=housing['median_house_value'],  #color indicating median_income

             cmap=plt.get_cmap('jet'),

             colorbar=True,

             figsize=(20,10)

            )

plt.legend()
from pandas.plotting import scatter_matrix

housing.columns

attributes=['median_house_value','median_income','total_rooms','housing_median_age']

scatter_matrix(housing[attributes],figsize=(20,10))

plt.show()
housing.plot(kind='scatter',x='median_house_value',y='median_income',alpha=0.1)
housing=strat_train_set.drop('median_house_value',axis=1)  # housing is now just predictors. (X)

housing_labels=strat_train_set['median_house_value'].copy() #labels (Y)
housing.isna().sum()
from sklearn.impute import SimpleImputer

imputer=SimpleImputer(missing_values=np.nan,strategy='median')

#drop the ocean_proximity feature which is categorical

housing_numer=housing.drop('ocean_proximity',axis=1)

#impute the numerical data. Remember that this is all been done on the training set

imputer.fit(housing_numer)

print(imputer.statistics_)

XX=imputer.transform(housing_numer)  #XX is a numpy array

housing_tr=pd.DataFrame(XX,columns=housing_numer.columns) # convert XX to a pandas df

housing_tr.head()
""" Commented out. since this process can now directly be done using the OneHotEncoder

from sklearn.preprocessing import LabelEncoder

encoder=LabelEncoder()

housing_cat=housing['ocean_proximity']

housing_cat_encoded=encoder.fit_transform(housing_cat)

print(housing_cat_encoded)

print(encoder.classes_)    #<1H mapped to 0, INLAND mapped to 1,..

"""
#housing_cat_encoded.reshape(-1,1).shape
from sklearn.preprocessing import OneHotEncoder

ohe=OneHotEncoder()

housing_cat=housing['ocean_proximity']

#housing_cat_ohe=ohe.fit_transform(housing_cat.values.reshape(-1,1)) 

# the reshape is because fit_transform expects a 2D array. Our housing_cat_encoded array is of shape 

# (n,) i.e a 1D array. We reshape it to (n,1) . We specify the '1' in the (-1,1) the '-1' causes numpy 

# to automagically infer 'n'

housing_cat_ohe=ohe.fit_transform(housing_cat.values.reshape(-1,1)) 

housing_cat_ohe

housing_cat_ohe.toarray()
#Custom Transformers

from sklearn.base import BaseEstimator,TransformerMixin



#Including the BaseEstimator provides the class with get_params() and set_params()

#Including the TransformerMixin as a base provides the fit_transform() method



room_ix,bedrooms_ix,population_ix,household_ix= 3,4,5,6  #indices in the df



class CombinedAttributesAdder(BaseEstimator,TransformerMixin): 

    def __init__(self,add_bedrooms_per_room=True):

        self.add_bedrooms_per_room=add_bedrooms_per_room

    

    def fit(self,X,y=None):

        return self   #nothing to do except return self in this method

    

    def transform(self,X,y=None):

        rooms_per_household=X[:,room_ix]/X[:,household_ix]

        population_per_household=X[:,population_ix]/X[:,household_ix]

        if self.add_bedrooms_per_room:

            bedrooms_per_room=X[:,bedrooms_ix]/X[:,room_ix]

            return np.c_[X,rooms_per_household,population_per_household,bedrooms_per_room]  #include new features         

        else:

            return np.c_[X,rooms_per_household,population_per_household]





        

attribute_adder=CombinedAttributesAdder(add_bedrooms_per_room=True)

housing_new_attribs= attribute_adder.transform(housing.values)

print(housing_new_attribs.shape)

print(housing.shape)
class DataFrameSelector(BaseEstimator,TransformerMixin):

    def __init__(self,attribute_names):

        self.attribute_names=attribute_names

    def fit(self,X,y=None):

        return self

    def transform(self,X,y=None):

        return X[self.attribute_names].values    #return the np values from the df for the selected attributes
#the entire data cleaning steps in one go

housing=strat_train_set.drop('median_house_value',axis=1)   #this is our X

housing_labels=strat_train_set['median_house_value'].copy() # our Y

housing_num=housing.drop('ocean_proximity',axis=1)  #X=X_num+X_cat. This is our X_num

numerical_attribs=list(housing_num)

categorical_attrib=['ocean_proximity']  #X_cat



from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler



#pipeline 1  load_df->impute_nans->add_new_attribs->perform standardization

pipe_num=Pipeline([

    ('selector',DataFrameSelector(numerical_attribs)),                    #load data

    ('imputer',SimpleImputer(missing_values=np.nan,strategy='median')),   #impute 

    ('attribs_adder',CombinedAttributesAdder()),                          #add_new_attribs

    ('std_scaler',StandardScaler())                                       #perform standardization

])



#pipeline 2 load_df->ohe

pipe_cat=Pipeline([

    ('selector',DataFrameSelector(categorical_attrib)),                   #load_df

    ('ohe',OneHotEncoder())                                               #perform ohe

])

#test pipelines

num_res=pipe_num.fit_transform(housing)

print(num_res.shape)



ohe_res=pipe_cat.fit_transform(housing)

print(ohe_res.toarray())
from sklearn.pipeline import FeatureUnion

full_pipe=FeatureUnion(

    [

        ('pipe_num',pipe_num),   

        ('pipe_cat',pipe_cat)

    ]

)

#load the dataframe into the full pipe.

housing_prep=full_pipe.fit_transform(housing)  #the pipe_num will produce its output (num_op) and the pipe_cat will produce it's output (cat_op)

#housing_prep=concat(num_op,cat_op)

print(housing_prep)  # a sparse matrix (ohe introduces zeros)

print(housing_prep.shape)
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score



forest_reg=RandomForestRegressor()



#cross validation expects a utility function (greater is better) rather than a cost function (lower is better)

scores=cross_val_score(forest_reg,housing_prep,housing_labels,scoring='neg_mean_squared_error',cv=2)

scores





rmse=np.sqrt(-scores)
def display_scores(scores):

    print("scores: ",scores)

    print("mean: ",scores.mean())

    print('std_dev:',scores.std())

display_scores(rmse)
#forest_reg.feature_importances_
RESULTS='/kaggle/input/results'

if not os.path.isdir(RESULTS):

    os.makedirs(RESULTS)
#using Grid Search





from sklearn.externals import joblib

from sklearn.model_selection import GridSearchCV



param_grid=[

    {'n_estimators':[20,30,40],'max_features':[6,8,9,10]},             #3X4 combinations of params

    {'bootstrap':[False],'n_estimators':[20,30],'max_features':[9,10]} #1X2X2 combinations of params

    

]

reg=RandomForestRegressor()



GRID_SEARCH_RESULT_LOC="/kaggle/input/results/gridsearch_randforestreg.pkl"





try :

    gs_cv=joblib.load(GRID_SEARCH_RESULT_LOC)

    print('loaded previous instance of gridsearch with best_params: ',gs_cv.best_params_)

except(FileNotFoundError):

    print('no saved file found. continuing with gridsearch')

    gs_cv=GridSearchCV(reg,param_grid,cv=5,scoring='neg_mean_squared_error',n_jobs=-1)  #n_jobs=-1 ==>parallel execution

    gs_cv.fit(housing_prep,housing_labels)

#save gridsearch

    joblib.dump(gs_cv,GRID_SEARCH_RESULT_LOC)

    

    

#best parameters and the best estimator that grid search found

cvres=gs_cv.cv_results_

cvres

#cdvresdf=pd.DataFrame(cvres.values,columns=cvres.keys())
print(gs_cv.best_estimator_.feature_importances_)


""" 

Just to recap. these were the pipelines we had so far



#pipeline 1  load_df->impute_nans->add_new_attribs->perform standardization

pipe_num=Pipeline([

    ('selector',DataFrameSelector(numerical_attribs)),                    #load data

    ('imputer',SimpleImputer(missing_values=np.nan,strategy='median')),   #impute 

    ('attribs_adder',CombinedAttributesAdder()),                          #add_new_attribs

    ('std_scaler',StandardScaler())                                       #perform standardization

])



#pipeline 2 load_df->ohe

pipe_cat=Pipeline([

    ('selector',DataFrameSelector(categorical_attrib)),                   #load_df

    ('ohe',OneHotEncoder())                                               #perform ohe

])



full_pipe=FeatureUnion(

    [

        ('pipe_num',pipe_num),

        ('pipe_cat',pipe_cat)

    ]

)



"""



pipe_reg=Pipeline([

    ('full_pipe',full_pipe),

    ('reg',RandomForestRegressor())

])

#the parameters are referenced using reg__  where reg refers to the RandomForestRegressor.

#we can further nest these calls. full_pipe->pipe_num-->imputer-->strategy can take either 'mean' or median



parameter_grid=[

    {

     'reg__n_estimators':[20,30,40],'reg__max_features':[6,8,9,10],

     'full_pipe__pipe_num__imputer__strategy':['mean','median']

    },#3X4X2 combinations of params

    #{'reg__bootstrap':[False],'reg__n_estimators':[20,30],'reg__max_features':[9,10]}  #1X2X2 combinations of params

]



GRID_SEARCH_PIPL_LOC="/kaggle/input/results/gridsearch_over_pipe_randforestreg"

try:

    gs_p_cv=joblib.load(GRID_SEARCH_PIPL_LOC)

    print('loaded previous instance of gridsearch with best_params: ',gs_p_cv.best_params_)

except(FileNotFoundError):

    print('File not found. continuing with GridSearch')

    gs_p_cv=GridSearchCV(pipe_reg,cv=5,param_grid=parameter_grid,scoring='neg_mean_squared_error',n_jobs=-1)

    gs_p_cv.fit(housing,housing_labels)

    joblib.dump(gs_p_cv,GRID_SEARCH_PIPL_LOC)

#best params

print(gs_p_cv.best_params_)

#best scores

bs=gs_p_cv.best_score_

rmsbs=np.sqrt(-bs)

print(rmsbs)
#all params , all scores

cvsres=gs_p_cv.cv_results_

cvsresdf=pd.DataFrame.from_dict(cvsres)

cvsresdf.sort_values(by=['rank_test_score'])
#gs_p_cv.best_estimator_.named_steps.full_pipe.get_params()  This is how you would access the params under full_pipe
feature_importances=gs_p_cv.best_estimator_.named_steps.reg.feature_importances_

extra_attribs=['rooms_per_house','pop_per_house','bedrooms_per_house']

ohe_attrs=list(ohe.get_feature_names())

attributes=numerical_attribs+extra_attribs+ohe_attrs        #if you look up the code, this is the order in which we split the data columnwise

sorted_list_of_important_features=sorted(zip(feature_importances,attributes),reverse=True)

plt.figure()

plt.barh(attributes,feature_importances)

plt.show()

from sklearn.metrics import mean_squared_error

final_model=gs_p_cv.best_estimator_

X_test=strat_test_set.drop('median_house_value',axis=1)

y_test=strat_test_set['median_house_value'].copy()



final_predictions=final_model.predict(X_test)  #the pipeline exposes the methods of its final estimator which in our case is the RandomForestRegressor

                                                #If we were to open up the pipe, this is what will be seen

                                                #X_test_tr=full_pipe.transform(X_test)

                                                #final_predictions=res.predict(X_test_tr)

#final error

final_mse=mean_squared_error(final_predictions,y_test)

final_rmse=np.sqrt(final_mse)

final_rmse



from sklearn.svm import SVR

pipe_svr=Pipeline(

    [

        ('data_pipe',full_pipe),

        ('svr',SVR())

    ]

)



para_grid=[

    {

        'data_pipe__pipe_num__imputer__strategy':['mean','median'],

        'svr__kernel':['linear','rbf'],

        'svr__C':[0.3,0.5,0.7,1.0,1.2,1.4,1.6,1.8,2.0],

        'svr__gamma':['auto','scale']

    }

]



GRID_SEARCH_PIPL_SVR_LOC="/kaggle/input/results/gridsearch_over_pipe_svr"

try:

    gs_p_svr_cv=joblib.load(GRID_SEARCH_PIPL_SVR_LOC)

    print('loaded previous instance of gridsearch with best_params: ',gs_p_svr_cv.best_params_)

except(FileNotFoundError):

    print('File not found. continuing with GridSearch')

    gs_p_svr_cv=GridSearchCV(pipe_svr,cv=5,param_grid=para_grid,scoring='neg_mean_squared_error',n_jobs=-1)

    gs_p_svr_cv.fit(housing,housing_labels)

    joblib.dump(gs_p_svr_cv,GRID_SEARCH_PIPL_SVR_LOC)



svr_model=gs_p_svr_cv.best_estimator_

print(gs_p_svr_cv.best_params_)

svr_preds=svr_model.predict(X_test)

svr_mse=mean_squared_error(svr_preds,y_test)

svr_rmse=np.sqrt(svr_mse)

svr_rmse
from sklearn.model_selection import RandomizedSearchCV

import scipy



#the parameter grid for RandomizedSearchCV is a dictionary

para_grid={

     'svr__gamma': ['auto','scale'],

     'data_pipe__pipe_num__imputer__strategy':['mean','median'],

      'svr__kernel':['linear','rbf'],

      'svr__C':[0.5,0.7,0.8,1.0,1.2,1.4]

        

    

}



RND_SEARCH_PIPL_SVR_LOC="/kaggle/input/results/rndsearch_over_pipe_svr"

#RND_SEARCH_PIPL_SVR_LOC="rndsearch_over_pipe_svr"

try:

    rnd_best_model=joblib.load(RND_SEARCH_PIPL_SVR_LOC)

    print('loaded previous instance of random search with best_params: ',rnd_best_model.best_params_)

except(FileNotFoundError):

    print('File not found. continuing with RandomizedSearchCV')

    rnd_p_svr_cv=RandomizedSearchCV(pipe_svr,cv=5,

                                    param_distributions=para_grid,

                                    n_iter=5,

                                    scoring='neg_mean_squared_error',

                                    n_jobs=-1)

    rnd_best_model=rnd_p_svr_cv.fit(housing,housing_labels)  #https://chrisalbon.com/machine_learning/model_selection/hyperparameter_tuning_using_random_search/

    joblib.dump(rnd_best_model,RND_SEARCH_PIPL_SVR_LOC)



    

print(rnd_best_model.best_params_)

svr_rnd_preds=rnd_best_model.predict(X_test)

svr_rnd_mse=mean_squared_error(svr_rnd_preds,y_test)

svr_rnd_rmse=np.sqrt(svr_rnd_mse)

svr_rnd_rmse
def indices_of_first_k_features(ar,k):

    print(k)

    return np.sort(np.argpartition(np.array(ar),-k)[-k:])   #get the indices of the k-highest values of the features

     







class ImportantFeatures(BaseEstimator,TransformerMixin):

    def __init__(self,feature_importances=None,kk=None):

        self.feature_importances=feature_importances

        self.kk=kk

        

    

    def fit(self,X,y=None):

        self.feature_indices=indices_of_first_k_features(self.feature_importances,self.kk)

        return self

    

    def transform(self,X):

        return X[:,self.feature_indices]

    

        

        
k=5



pipe_pick_features=Pipeline(

    [

        ('data_pipe',full_pipe),

        ('pick_k_important_features',ImportantFeatures(feature_importances,k))

    ]

)



housing_imp_feat=pipe_pick_features.fit_transform(housing)







housing_imp_feat
print(housing_imp_feat[0:3].toarray())
cols=[]

for i,j in sorted_list_of_important_features[0:k]:

    cols.append(j)



#print(housing_prep[0:3].toarray())

imp_df=pd.DataFrame(housing_prep.toarray(),columns=attributes)

print(imp_df[0:3][cols])



data_prep_predict=Pipeline(

    [

        ('pipe_pick_features',pipe_pick_features),    #here k=5 . In the next exercise we will run a grid search over k

        ('svr',SVR())                                 #Will run a grid search for the best parameters in exercise 5

    

    ]

)



data_prep_predict.fit(housing,housing_labels)









#get some data.

da=housing.iloc[0:3]

da_lab=housing_labels.iloc[0:3]





#run predictions

da_preds=data_prep_predict.predict(da)



print('rmse : ',np.sqrt(mean_squared_error(da_lab,da_preds)))



from scipy.stats import reciprocal,expon



pipe_pick_features=Pipeline(

    [

        ('data_pipe',full_pipe),

        ('pick_important_features',ImportantFeatures()),

        ('svr',SVR())

    ]

)





param_grid=[{

            'data_pipe__pipe_num__imputer__strategy':['mean','median','most_frequent'],  #nesting to reach the imputer 

            'pick_important_features__feature_importances':[feature_importances],    #fixed feature_importances from our RandomForestRegressor()

            'pick_important_features__kk':list(range(1,len(feature_importances)+1)),

            'svr__kernel': ['linear', 'rbf'],

            'svr__C': np.linspace(1,20,num=5),

            'svr__gamma': np.linspace(0.1,1,num=5)

           }]



GRID_SEARCH_EX_5_LOC="/kaggle/input/results/gridsearch_over_full_pipe"

try:

    gs_5=joblib.load(GRID_SEARCH_EX_5_LOC)

    print('loaded previous instance of gridsearch with best_params: ',gs_5.best_params_)

except(FileNotFoundError):

    print('File not found. continuing with GridSearch')

    gs_5=GridSearchCV(pipe_pick_features,

                                     param_grid,

                                     cv=2,

                                     scoring='neg_mean_squared_error',    

                                     verbose=2,

                                     n_jobs=4

                                    )

    gs_5.fit(housing,housing_labels)

    joblib.dump(gs_5,GRID_SEARCH_EX_5_LOC)

    







#predict

best_model=gs_5.best_estimator_

preds=best_model.predict(X_test)

mse=mean_squared_error(preds,y_test)

rmse=np.sqrt(mse)

print(rmse)
