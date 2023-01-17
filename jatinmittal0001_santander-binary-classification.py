import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split,GridSearchCV  

from xgboost import XGBClassifier

from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

from sklearn.model_selection import StratifiedKFold

from bayes_opt import BayesianOptimization

from lightgbm import LGBMModel,LGBMClassifier

from catboost import CatBoostClassifier

import gc 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))
traindata = pd.read_csv("../input/train.csv")

testdata = pd.read_csv("../input/test.csv")

print(traindata.shape)

print(testdata.shape)
traindata.head()
testdata.head()


master_test_id = testdata['ID_code']



traindata.drop(['ID_code'],axis=1,inplace=True)

testdata.drop(['ID_code'],axis=1,inplace=True)
a = traindata[traindata.target == 1].target.sum()

print('Percentage of target variables with label =1 is: ',a*100/traindata.shape[0])
uni = (traindata.nunique()).sort_values()

print(uni)



#here we see that there is no variable which is binary.
#checked correlation for all but no luck



'''

corrmat = traindata.iloc[:,1:199].corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);

'''
#checked for duplicated rows but no duplicates found



'''



sam = traindata.append(testdata,sort=False)

sam.drop('target',axis=1,inplace=True)



df = sam[sam.duplicated()]  # checks duplicate rows considering all columns

df

'''
traindata.head(3)
testdata.head(2)
y_master_train = traindata['target']

#traindata.drop(['target'],axis=1,inplace=True)

testdata['target'] = 'test_data'

totaldata = pd.concat([traindata, testdata])  #concatenation will automatically match columns and append

totaldata.head()
totaldata.shape
y_master_train.value_counts(normalize=True)   #checking proportion of different ratings
#Plotting boxplots of 5 variables

m=1

plt.figure(figsize = (20,20))

for i in totaldata.columns[1:6]:

    plt.subplot(3,4,m)

    sns.boxplot(totaldata[i])

    m = m+1
a=list(totaldata.columns)

a.remove("target")



def outlier_treatment(data,cols):

    data_X = data.copy()

    

    for i in cols:

        a = data_X[data_X['target']!='test_data'][i].quantile([0.25,0.75]).values  #doing only on train data

        p_cap = a[1] + 1.5*(a[1]-a[0])

        p_clip = a[0] - 1.5*(a[1]-a[0])

        data_X[i][data_X[i] <= p_clip] = p_clip

        data_X[i][data_X[i] >= p_cap] = p_cap

    

  

    return data_X



#totaldata = outlier_treatment(totaldata,a)
from scipy.stats import skew

def skew_treatment(data):

    data_X = data.copy()

    #finding skewness of all variables

    col = data_X.columns

    skewed_feats = data_X[col].apply(lambda x: skew(x.dropna()))

    #adjusting features having skewness >0.75

    skewed_feats = skewed_feats[skewed_feats > 0.75]

    skewed_feats = skewed_feats.index

    data_X[skewed_feats] = np.log1p(data_X[skewed_feats])

    

    return data_X
#totaldata.iloc[:,1:] = skew_treatment(totaldata.iloc[:,1:])
from sklearn.model_selection import train_test_split



train_data = (totaldata[totaldata['target']!='test_data']).drop(['target'],axis=1)

test_data = (totaldata[totaldata['target']=='test_data']).drop(['target'],axis=1)

#  split X between training and testing set

x_train, x_test, y_train, y_test = train_test_split(train_data,y_master_train, test_size=0.25, shuffle=True,stratify=y_master_train)
'''



#PCA

from sklearn.decomposition import PCA

pca = PCA(n_components=50)

pca = pca.fit(x_train)

principalComponents_train = pca.transform(x_train)

principalComponents_test = pca.transform(x_test)

x_train_pca = pd.DataFrame(principalComponents_train)

x_test_pca = pd.DataFrame(principalComponents_test)



'''



#Not using PCA as it decreased the AUC score from 89.4 to 72 with all other things kept constant
del totaldata

del traindata

del testdata

del uni

gc.collect();
del train_data

del test_data

gc.collect();
y_train.value_counts(normalize=True)
def model2_lgbm(num_leaves,  # int

    learning_rate,  

    feature_fraction,

    lambda_l1,

    lambda_l2,

    min_gain_to_split,

    max_depth,n_estimators):

    

    clf = LGBMClassifier( n_estimators=int(n_estimators),

                         num_leaves = int(num_leaves),

                         learning_rate=learning_rate,

                         feature_fraction=feature_fraction,

                         lambda_l1=lambda_l1,

                        lambda_l2=lambda_l2,

                        min_gain_to_split=min_gain_to_split,

                        max_depth=int(max_depth),

                     eval_metric='auc'

                     

            )        



    clf.fit(x_train, y_train, 

                eval_set=[(x_test,y_test)],early_stopping_rounds=200,eval_metric='auc',verbose=False

               )

    

    a = clf.best_score_['valid_0']['auc']

    

    return a
bounds_lgbm = {

    'max_depth': (10, 15),

    'num_leaves':(5,40),

    'learning_rate':(0.01,0.1),

    'feature_fraction':(0.7,1),

    'lambda_l1': (0, 8.0), 

    'lambda_l2': (0, 8.0), 

    'min_gain_to_split': (0, 1.0),

    'n_estimators':(2000,5000)

}
from bayes_opt import BayesianOptimization



LGB_BO = BayesianOptimization(model2_lgbm, bounds_lgbm)



init_points = 5

n_iter = 15





print('-' * 130)



LGB_BO.maximize(init_points=init_points, n_iter=n_iter)
print(LGB_BO.max['target'])

print(LGB_BO.max['params'])
sdsd
def model_1_catb( iterations,learning_rate,depth,l2_leaf_reg):

        catb = CatBoostClassifier(

            iterations=int(iterations),

           #cat_features=cat_col,

            learning_rate=learning_rate,

            depth = int(depth),

            l2_leaf_reg = l2_leaf_reg,

            #subsample=subsample,  #can't be trained for catboost using bayesian opt

            early_stopping_rounds=50,

          #  colsample_bylevel = colsample_bylevel,  #can't be trained on GPU(but only on CPU) for catboost using bayesian opt

          # max_leaves = int(max_leaves),  can't be trained on CPU

            eval_metric='AUC',

           task_type='GPU',

           #verbose=30

        )

        catb.fit(

            x_train, y_train,

            eval_set=(x_test, y_test),verbose=10

        )

       # print('Model is fitted: ' + str(catb.is_fitted()))

        #print('Model params:')

        #print(catb.get_params())

        a = catb.get_best_score()

        return a['validation_0']['AUC']
bounds_catb = {

    'iterations': (30, 150), 

    'learning_rate': (0.05, 0.9),  

    'depth': (6, 15),

    'l2_leaf_reg': (0,5),    

  # 'subsample': (0.75,1),

   # 'colsample_bylevel': (0.75, 1), 

   # 'max_leaves': (5,40)

}
from bayes_opt import BayesianOptimization



CATB_BO = BayesianOptimization(model_1_catb, bounds_catb)



init_points = 3

n_iter = 10





print('-' * 130)



#CATB_BO.maximize(init_points=init_points, n_iter=n_iter)



print(CATB_BO.max['target'])

print(CATB_BO.max['params'])
# NOTE: Here I have shown how to make prediction on test set made from train data split

# To make prediction on unknown test set just replace X_train with full train data and 

# X_test with unknown test data

from sklearn.metrics import roc_auc_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from vecstack import stacking     #this is a library made for stacking by a Kaggler

from catboost import CatBoostClassifier



models = [

    #KNeighborsClassifier(n_neighbors=5,

                       # n_jobs=-1),

    CatBoostClassifier(

    learning_rate=0.05,

    depth = 10,

    rsm = 0.7, loss_function = 'Logloss', logging_level='Verbose', eval_metric='AUC',iterations = 300,),

        

    #RandomForestClassifier(random_state=0, n_jobs=-1, 

                           #n_estimators=100, max_depth=3),

        

    XGBClassifier(n_estimators=2000, reg_alpha = 0.01, objective= 'rank:pairwise',silent=False)

]





S_train, S_test = stacking(models,                   

                           x_train, y_train, x_test,   

                           regression=False, 

     

                           mode='oof_pred_bag', 

       

                           needs_proba=False,

         

                           save_dir=None, 

            

                           metric=roc_auc_score, 

    

                           n_folds=4, 

                 

                           stratified=True,

            

                           shuffle=True)



#meta model

model = XGBClassifier(random_state=0, n_jobs=-1, learning_rate=0.1, 

                      n_estimators=100, max_depth=3)

    

model = model.fit(S_train, y_train)

y_pred = model.predict(S_test)

print('Final prediction score: [%.8f]' % roc_auc_score(y_test, y_pred))
a = np.ravel(y_test)

b = np.ravel(y_pred)

roc_auc_score(a,b)
from catboost import CatBoostClassifier



model = CatBoostClassifier(

    random_seed=63,

    iterations=300,

    learning_rate=0.05,

    depth=10,

    loss_function='Logloss',

    rsm = 0.7,

    od_type='Iter',

    od_wait=20,

    eval_metric = 'AUC',

)

model.fit(

    x_train, y_train,

    logging_level='Silent',

    eval_set=(x_test, y_test),

    plot=True

)
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC

from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import make_pipeline

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold

lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))

xgb = XGBClassifier(n_estimators=2000, reg_alpha = 0.01)

rf = RandomForestClassifier()

extraT = ExtraTreesClassifier()



class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, base_models, meta_model, n_folds=5):

        self.base_models = base_models

        self.meta_model = meta_model

        self.n_folds = n_folds

   

    # We again fit the data on clones of the original models

    def fit(self, X, y):

        self.base_models_ = [list() for x in self.base_models]

        self.meta_model_ = clone(self.meta_model)

        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        

        # Train cloned base models then create out-of-fold predictions

        # that are needed to train the cloned meta-model

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))

        for i, model in enumerate(self.base_models):

            for train_index, holdout_index in kfold.split(X, y):

                instance = clone(model)

                self.base_models_[i].append(instance)

                instance.fit(X[train_index], y[train_index])

                y_pred = instance.predict(X[holdout_index])

                out_of_fold_predictions[holdout_index, i] = y_pred

                

        # Now train the cloned  meta-model using the out-of-fold predictions as new feature

        self.meta_model_.fit(out_of_fold_predictions, y)

        return self

   

    #Do the predictions of all base models on the test data and use the averaged predictions as 

    #meta-features for the final prediction which is done by the meta-model

    def predict(self, X):

        meta_features = np.column_stack([

            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)

            for base_models in self.base_models_ ])

        return self.meta_model_.predict(meta_features)

    

    

stacked_averaged_models = StackingAveragedModels(base_models = (extraT, rf, xgb),

                                                 meta_model = lasso)
#stacked_averaged_models.fit(x_train, y_train)

#y_pred = stacked_averaged_models.predict(x_test)
def Stacking(model,train,y,test,n_fold):

    folds=StratifiedKFold(n_splits=n_fold,random_state=1)

    test_pred=np.empty((test.shape[0],1),float)

    train_pred=np.empty((0,1),float)

    for train_indices,val_indices in folds.split(train,y.values):

        x_train,x_val=train.iloc[train_indices],train.iloc[val_indices]

        y_train,y_val=y.iloc[train_indices],y.iloc[val_indices]



        model.fit(X=x_train,y=y_train)

        train_pred=np.append(train_pred,model.predict(x_val))

        test_pred=np.append(test_pred,model.predict(test))

    return test_pred.reshape(-1,1),train_pred
#lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))

xgb = XGBClassifier(n_estimators=100, reg_alpha = 0.01)

rf = RandomForestClassifier()





test_pred1 ,train_pred1=Stacking(model=xgb,n_fold=5, train=x_train,test=x_test,y=y_train)



train_pred1=pd.DataFrame(train_pred1)

test_pred1=pd.DataFrame(test_pred1)
test_pred2 ,train_pred2=Stacking(model=rf,n_fold=10,train=x_train,test=x_test,y=y_train)



train_pred2=pd.DataFrame(train_pred2)

test_pred2=pd.DataFrame(test_pred2)
df = pd.concat([train_pred1, train_pred2], axis=1)

df_test = pd.concat([test_pred1, test_pred2], axis=1)



model = ExtraTreesClassifier()

model.fit(df,y_train)

model.score(df_test, y_test)
roc_auc_score(y_test, lgb_pred)
y_pred = clf.predict(test_data)
sub = pd.DataFrame(data = testid,columns =['ID_code'])

sub['target'] = y_pred

sub.to_csv('submission.csv', index=False)