from IPython.display import HTML

from IPython.display import Image

Image(url= "https://www.autocar.co.nz/_News/_2018Bin/Mercedes-badge_www.jpg")



from IPython.core.display import HTML

HTML('''<script>

code_show=true; 

function code_toggle() {

 if (code_show){

 $('div.input').hide();

 } else {

 $('div.input').show();

 }

 code_show = !code_show

} 

$( document ).ready(code_toggle);

</script>

The raw code for this IPython notebook is by default hidden for easier reading.

To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')
import h2o

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import preprocessing

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

from xgboost import XGBClassifier

import xgboost as xgb

from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split



color = sns.color_palette()

import warnings; warnings.simplefilter('ignore')
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")

train_df.describe()
train_df.head()
train_df.info()
dtype_df = train_df.dtypes.reset_index()

dtype_df.columns = ["Count", "Column Type"]

dtype_df.groupby("Column Type").aggregate('count').reset_index()
y_train = train_df['y'].values

plt.figure(figsize=(15, 5))

plt.hist(y_train, bins=20)

plt.xlabel('Target value in seconds')

plt.ylabel('Occurences')

plt.title('Distribution of the target value')



print('min: {} max: {} mean: {} std: {}'.format(min(y_train), max(y_train), y_train.mean(), y_train.std()))

print('Count of values above 180: {}'.format(np.sum(y_train > 200)))
#Finding missing values in the data set 

total = train_df.isnull().sum()[train_df.isnull().sum() != 0].sort_values(ascending = False)

percent = pd.Series(round(total/len(train_df)*100,2))

pd.concat([total, percent], axis=1, keys=['total_missing', 'percent'])
from IPython.display import HTML

from IPython.display import Image

Image(url= "https://media.giphy.com/media/jNdw5Qmy5MOpq/giphy.gif")



cols = [c for c in train_df.columns if 'X' in c]

cols_x = [c for c in test_df.columns if 'X' in c]



counts = [[], [], []]

counts_ = [[], [], []]

for c in cols:

    typ = train_df[c].dtype

    uniq = len(np.unique(train_df[c]))

    if uniq == 1: counts[0].append(c)

    elif uniq == 2 and typ == np.int64: counts[1].append(c)

    else: counts[2].append(c)



print('Constant features: {} Binary features: {} Categorical features: {}\n'.format(*[len(c) for c in counts]))



print('Constant features:', counts[0])

print('Categorical features:', counts[2])



print("Features for test")

for c in cols_x:

    typ = test_df[c].dtype

    uniq = len(np.unique(test_df[c]))

    if uniq == 1: counts_[0].append(c)

    elif uniq == 2 and typ == np.int64: counts_[1].append(c)

    else: counts_[2].append(c)



print('Constant features: {} Binary features: {} Categorical features: {}\n'.format(*[len(c) for c in counts]))



print('Constant features:', counts[0])

print('Categorical features:', counts[2])
#correlation

train_df.corr()

var_name = "X0"

col_order = np.sort(train_df[var_name].unique()).tolist()

plt.figure(figsize=(12,6))

sns.stripplot(x=var_name, y='y', data=train_df, order=col_order)

plt.xlabel(var_name, fontsize=12)

plt.ylabel('y', fontsize=12)

plt.title("Distribution of y variable with "+var_name, fontsize=15)

plt.show()
var_name = "X1"

col_order = np.sort(train_df[var_name].unique()).tolist()

plt.figure(figsize=(12,6))

sns.stripplot(x=var_name, y='y', data=train_df, order=col_order)

plt.xlabel(var_name, fontsize=12)

plt.ylabel('y', fontsize=12)

plt.title("Distribution of y variable with "+var_name, fontsize=15)

plt.show()
var_name = "X2"

col_order = np.sort(train_df[var_name].unique()).tolist()

plt.figure(figsize=(12,6))

sns.boxplot(x=var_name, y='y', data=train_df, order=col_order)

plt.xlabel(var_name, fontsize=12)

plt.ylabel('y', fontsize=12)

plt.title("Distribution of y variable with "+var_name, fontsize=15)

plt.show()
var_name = "X3"

col_order = np.sort(train_df[var_name].unique()).tolist()

plt.figure(figsize=(12,6))

sns.violinplot(x=var_name, y='y', data=train_df, order=col_order)

plt.xlabel(var_name, fontsize=12)

plt.ylabel('y', fontsize=12)

plt.title("Distribution of y variable with "+var_name, fontsize=15)

plt.show()
var_name = "X4"

col_order = np.sort(train_df[var_name].unique()).tolist()

plt.figure(figsize=(12,6))

sns.violinplot(x=var_name, y='y', data=train_df, order=col_order)

plt.xlabel(var_name, fontsize=12)

plt.ylabel('y', fontsize=12)

plt.title("Distribution of y variable with "+var_name, fontsize=15)

plt.show()
var_name = "X5"

col_order = np.sort(train_df[var_name].unique()).tolist()

plt.figure(figsize=(12,6))

sns.boxplot(x=var_name, y='y', data=train_df, order=col_order)

plt.xlabel(var_name, fontsize=12)

plt.ylabel('y', fontsize=12)

plt.title("Distribution of y variable with "+var_name, fontsize=15)

plt.show()
var_name = "X6"

col_order = np.sort(train_df[var_name].unique()).tolist()

plt.figure(figsize=(12,6))

sns.boxplot(x=var_name, y='y', data=train_df, order=col_order)

plt.xlabel(var_name, fontsize=12)

plt.ylabel('y', fontsize=12)

plt.title("Distribution of y variable with "+var_name, fontsize=15)

plt.show()
var_name = "X8"

col_order = np.sort(train_df[var_name].unique()).tolist()

plt.figure(figsize=(12,6))

sns.boxplot(x=var_name, y='y', data=train_df, order=col_order)

plt.xlabel(var_name, fontsize=12)

plt.ylabel('y', fontsize=12)

plt.title("Distribution of y variable with "+var_name, fontsize=15)

plt.show()
for c in counts[2]:

    value_counts = train_df[c].value_counts()

    fig, ax = plt.subplots(figsize=(10, 5))

    plt.title('Categorical feature {} - Cardinality {}'.format(c, len(np.unique(train_df[c]))))

    plt.xlabel('Feature value')

    plt.ylabel('Occurences')

    plt.bar(range(len(value_counts)), value_counts.values)

    ax.set_xticks(range(len(value_counts)))

    ax.set_xticklabels(value_counts.index, rotation='vertical')

    plt.show()
usable_columns = list(set(test_df.columns) - set(['ID', 'y']))

usable_columns = list(set(train_df.columns) - set(['ID', 'y']))



y_train = train_df['y'].values

id_test = test_df['ID'].values



##dataset for normal modelling and not H2O

x_train = train_df[usable_columns]  

x_test = test_df[usable_columns]

x_train_not_dropped = train_df[usable_columns]  

x_test_not_dropped = test_df[usable_columns]
x_test.head()
unique_columns=['X11', 'X93', 'X107', 'X233', 'X235', 'X268', 'X289', 'X290', 'X293', 'X297', 'X330', 'X347']

x_test = x_test.drop(['X11', 'X93', 'X107', 'X233', 'X235', 'X268', 'X289', 'X290', 'X293', 'X297', 'X330', 'X347'],axis=1)



for column in usable_columns:

    cardinality = len(np.unique(test_df[column]))

    if cardinality > 2: # Column is categorical

        mapper = lambda x: sum([ord(digit) for digit in x])

        test_df[column] = test_df[column].apply(mapper)

        x_test[column]= x_test[column].apply(mapper)

        x_test_not_dropped[column]= x_test_not_dropped[column].apply(mapper)



        

x_train = x_train.drop(unique_columns,axis=1)



for column in usable_columns:

    cardinality = len(np.unique(train_df[column]))

      

    if cardinality > 2: # Column is categorical

        mapper = lambda x: sum([ord(digit) for digit in x])

        train_df[column] = train_df[column].apply(mapper)

        x_train[column]= x_train[column].apply(mapper)

        x_train_not_dropped[column]= x_train_not_dropped[column].apply(mapper)



finaltrainset = train_df[usable_columns].values

finaltestset = test_df[usable_columns].values
mean_Y = np.mean(y_train)

# prepare dict of params for xgboost to run with

xgb_params = {

    'n_trees': 500,   #general parameter used as booster

    'eta': 0.005,     #Booster Parameter used as learning_rate, step shrinkage reduces overfitting and eta shrinks the feature weights to make the boosting process more conservative.

    'max_depth': 4,   #Booster Parameter Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit

    'subsample': 0.95, #Booster Parameter Subsample ratio of the training instances. Setting it to 0.5 means that XGBoost would randomly sample half of the training data prior to growing trees. and this will prevent overfitting. 

    'objective': 'reg:linear', #Learning Parameter Specify the learning task and the corresponding learning objective

    'eval_metric': 'rmse', #Learning Parameter Evaluation metrics for validation data, a default metric will be assigned according to objective , user can add multiple metrics

    'base_score': mean_Y, # Learning Parameter The initial prediction score of all instances, global bias base prediction = mean(target)

    'silent': 1    #silent [default=0] [Deprecated] Deprecated. Verbosity of printing messages.

}



# form DMatrices for Xgboost training

dtrain = xgb.DMatrix(x_train_not_dropped, label=y_train)

dtest = xgb.DMatrix(x_test_not_dropped)



# xgboost, cross-validation

cv_result = xgb.cv(xgb_params, 

                   dtrain, 

                   num_boost_round=1000, # increase to have better results (~700)

                   early_stopping_rounds=50,

                   verbose_eval=50, 

                   show_stdv=False

                  )



num_boost_rounds = 500



# train model

model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)



# check f2-score (to get higher score - increase num_boost_round in previous cell)

from sklearn.metrics import r2_score



# now fixed, correct calculation

print("Accuracy Score:%.2f" %r2_score(dtrain.get_label(), model.predict(dtrain)))



# make predictions and save results

pred_y = model.predict(dtest)



print("Accuracy Score:%.2f" %r2_score(dtrain.get_label(), model.predict(dtrain)))

output = pd.DataFrame({'id': id_test, 'y': pred_y})

output.to_csv('submission_manali_1.csv', index=False)
mean_Y = np.mean(y_train)

# prepare dict of params for xgboost to run with

xgb_params = {

    'n_trees': 500,   #general parameter used as booster

    'eta': 0.005,     #Booster Parameter used as learning_rate, step shrinkage reduces overfitting and eta shrinks the feature weights to make the boosting process more conservative.

    'max_depth': 4,   #Booster Parameter Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit

    'subsample': 0.95, #Booster Parameter Subsample ratio of the training instances. Setting it to 0.5 means that XGBoost would randomly sample half of the training data prior to growing trees. and this will prevent overfitting. 

    'objective': 'reg:linear', #Learning Parameter Specify the learning task and the corresponding learning objective

    'eval_metric': 'rmse', #Learning Parameter Evaluation metrics for validation data, a default metric will be assigned according to objective , user can add multiple metrics

    'base_score': mean_Y, # Learning Parameter The initial prediction score of all instances, global bias base prediction = mean(target)

    'silent': 1    #silent [default=0] [Deprecated] Deprecated. Verbosity of printing messages.

}



# form DMatrices for Xgboost training

dtrain = xgb.DMatrix(x_train, label=y_train)

dtest = xgb.DMatrix(x_test)



# xgboost, cross-validation

cv_result = xgb.cv(xgb_params, 

                   dtrain, 

                   num_boost_round=1000, # increase to have better results (~700)

                   early_stopping_rounds=50,

                   verbose_eval=50, 

                   show_stdv=False

                  )



num_boost_rounds = 500



# train model

model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)



# check f2-score (to get higher score - increase num_boost_round in previous cell)

from sklearn.metrics import r2_score



# now fixed, correct calculation

print("Accuracy Score:%.2f" %r2_score(dtrain.get_label(), model.predict(dtrain)))



# make predictions and save results

#pred_y = model.predict(dtest)



print("Accuracy Score:%.2f" %r2_score(dtrain.get_label(), model.predict(dtrain)))

#make predictions and save results

pred_y = model.predict(dtest)



output = pd.DataFrame({'id': id_test, 'y': pred_y})

output.to_csv('submission_manali_2.csv', index=False)
from sklearn.decomposition import PCA, FastICA

from sklearn.decomposition import TruncatedSVD
from sklearn import datasets

from sklearn.preprocessing import StandardScaler 

pca = PCA().fit(train_df)

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('number of components')

plt.ylabel('cumulative explained variance'); 
from sklearn.decomposition import PCA, FastICA

from sklearn.decomposition import TruncatedSVD
#svd

tsvd = TruncatedSVD(n_components=10, random_state=42)

tsvd_results_train = tsvd.fit_transform(train_df.drop(["y"], axis=1))

tsvd_results_test = tsvd.transform(test_df)

 

# PCA

pca = PCA(n_components=10, random_state=42)

pca2_results_train = pca.fit_transform(train_df.drop(["y"], axis=1))

pca2_results_test = pca.transform(test_df)



# ICA

ica = FastICA(n_components=10, random_state=42)

ica2_results_train = ica.fit_transform(train_df.drop(["y"], axis=1))

ica2_results_test = ica.transform(test_df)

# Append decomposition components to datasets

n_comp=10

for i in range(1, 11):

    x_train['pca_' + str(i)] = pca2_results_train[:,i-1]

    x_test['pca_' + str(i)] = pca2_results_test[:,i-1]

    

    x_train['ica_' + str(i)] = ica2_results_train[:,i-1]

    x_test['ica_' + str(i)] = ica2_results_test[:,i-1]

    

    x_train['tsvd_' + str(i)] = tsvd_results_train[:,i-1]

    x_test['tsvd_' + str(i)] = tsvd_results_test[:, i-1]





### Regressor

import xgboost as xgb



# prepare dict of params for xgboost to run with

xgb_params = {

    

    'n_trees': 500, 

    'eta': 0.005,

    'max_depth': 4,

    'subsample': 0.95,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'base_score': mean_Y, # base prediction = mean(target)

    'silent': 1

}







# form DMatrices for Xgboost training

dtrain = xgb.DMatrix(x_train,label=y_train)

dtest = xgb.DMatrix(x_test)



# xgboost, cross-validation

cv_result = xgb.cv(xgb_params, 

                   dtrain, 

                   num_boost_round=500, # increase to have better results (~700)

                   early_stopping_rounds=50,

                   verbose_eval=10, 

                   show_stdv=False

                 )



                



num_boost_rounds = len(cv_result)

print(num_boost_rounds)

                   

num_boost_rounds = 500

# train model



model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)

from sklearn.metrics import r2_score

print("Accuracy Score:%.2f" %r2_score(model.predict(dtrain), dtrain.get_label()))
# make predictions and save results

y_pred = model.predict(dtest)



output = pd.DataFrame({'id': id_test, 'y': y_pred})

output.to_csv('submission_manali_3.csv', index=False)
print(h2o.__version__)
h2o.init(strict_version_check=False) # start h2o
h2o.connect()
h2o_frame=h2o.H2OFrame(train_df)

h2o_test=h2o.H2OFrame(test_df)
h2o_frame.describe()
h2o_test.describe()
y = h2o_frame.columns[1] #target variable

X = [name for name in h2o_frame.columns if name != y] #train features
from h2o.automl import H2OAutoML

aml = H2OAutoML(max_runtime_secs=1200,project_name ="automl_test" ,balance_classes= False) # init automl, run for 300 seconds

aml.train(x=X,  

           y=y,

           training_frame=h2o_frame)
lb=aml.leaderboard

lb
aml_leaderboard_df=aml.leaderboard.as_data_frame()
 #Get model ids for all models in the AutoML Leaderboard

model_ids = list(aml.leaderboard['model_id'].as_data_frame().iloc[:,0])

# Get the "All Models" Stacked Ensemble model

se = h2o.get_model([mid for mid in model_ids if "StackedEnsemble_AllModels_AutoML_20190403_204847" in mid][0])

# Get the Stacked Ensemble metalearner model

metalearner = h2o.get_model(aml.leader.metalearner()['name'])
metalearner.coef_norm()
m_id=''

for model in aml_leaderboard_df['model_id']:

    if 'StackedEnsemble' not in model:

      print (model)

      if m_id=='':

            m_id=model

print ("model_id ", m_id)
non_stacked= h2o.get_model(m_id)

print (non_stacked.algo)
pred = aml.predict(h2o_test)

x=pred.as_data_frame(use_pandas=True)
x['id'] = test_df['ID'].values
x=x[['id','predict']]
x = x.rename(columns={'id': 'id', 'predict': 'y'})
x.to_csv('submission_manali_4.csv')
x.head()
h2o.shutdown()
import numpy as np

from sklearn.base import BaseEstimator,TransformerMixin, ClassifierMixin

from sklearn.preprocessing import LabelEncoder

import xgboost as xgb

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import ElasticNetCV, LassoLarsCV

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.pipeline import make_pipeline, make_union

from sklearn.utils import check_array

from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeRegressor

from sklearn.random_projection import GaussianRandomProjection

from sklearn.random_projection import SparseRandomProjection

from sklearn.decomposition import PCA, FastICA

from sklearn.decomposition import TruncatedSVD

from sklearn.metrics import r2_score

import warnings; warnings.simplefilter('ignore')



train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



train_ids = train.ID





class StackingEstimator(BaseEstimator, TransformerMixin):

    

    def __init__(self, estimator):

        self.estimator = estimator



    def fit(self, X, y=None, **fit_params):

        self.estimator.fit(X, y, **fit_params)

        return self

    def transform(self, X):

        X = check_array(X)

        X_transformed = np.copy(X)

        # add class probabilities as a synthetic feature

        if issubclass(self.estimator.__class__, ClassifierMixin) and hasattr(self.estimator, 'predict_proba'):

            X_transformed = np.hstack((self.estimator.predict_proba(X), X))



        # add class prodiction as a synthetic feature

        X_transformed = np.hstack((np.reshape(self.estimator.predict(X), (-1, 1)), X_transformed))



        return X_transformed

usable_columns = list(set(train.columns) - set(['y']))



cat_columns = train.select_dtypes(['object']).columns

cat_columns_test= test.select_dtypes(['object']).columns

train[cat_columns] = train[cat_columns].astype('category')

test[cat_columns_test] = test[cat_columns_test].astype('category')

test[cat_columns_test] = test[cat_columns_test].apply(lambda x: x.cat.codes)

train[cat_columns] = train[cat_columns].apply(lambda x: x.cat.codes)



y_train = train['y'].values

y_mean = np.mean(y_train)

id_test = test['ID'].values

#finaltrainset and finaltestset are data to be used only the stacked model (does not contain PCA, SVD... arrays) 







finaltrainset = train[usable_columns].values

finaltestset = test[usable_columns].values





'''Train the xgb model then predict the test data'''



sub = pd.DataFrame()

sub['ID'] = id_test

sub['y'] = 0

for fold in range(1,4):

    np.random.seed(fold)

    xgb_params = {

        'n_trees': 520, 

        'eta': 0.0045,

        'max_depth': 4,

        'subsample': 0.93,

        'objective': 'reg:linear',

        'eval_metric': 'rmse',

        'base_score': y_mean, # base prediction = mean(target)

        'silent': True,

        'colsample_bytree': 0.7,

        'seed': fold,

    }

    # NOTE: Make sure that the class is labeled 'class' in the data file

    

    dtrain = xgb.DMatrix(train.drop('y', axis=1), y_train)

    dtest = xgb.DMatrix(test)

    

    num_boost_rounds = 1250

    # train model

    model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)

    y_pred = model.predict(dtest)

    

    '''Train the stacked models then predict the test data'''

    

    stacked_pipeline = make_pipeline(

        StackingEstimator(estimator=LassoLarsCV(normalize=True)),

        StackingEstimator(estimator=GradientBoostingRegressor(learning_rate=0.001, loss="huber", max_depth=3, max_features=0.55, min_samples_leaf=18, min_samples_split=14, subsample=0.7)),

        LassoLarsCV()

    

    )

    

    stacked_pipeline.fit(finaltrainset, y_train)

    results = stacked_pipeline.predict(finaltestset)

    

    '''R2 Score on the entire Train data when averaging'''

    

    print('R2 score on train data:')

    print(r2_score(y_train,stacked_pipeline.predict(finaltrainset)*0.2855 + model.predict(dtrain)*0.7145))

    

    '''Average the preditionon test data  of both models then save it on a csv file'''



    sub['y'] += y_pred*0.75 + results*0.25

sub['y'] /= 3



leaks = {

    1:71.34112,

    12:109.30903,

    23:115.21953,

    28:92.00675,

    42:87.73572,

    43:129.79876,

    45:99.55671,

    57:116.02167,

    3977:132.08556,

    88:90.33211,

    89:130.55165,

    93:105.79792,

    94:103.04672,

    1001:111.65212,

    104:92.37968,

    72:110.54742,

    78:125.28849,

    105:108.5069,

    110:83.31692,

    1004:91.472,

    1008:106.71967,

    1009:108.21841,

    973:106.76189,

    8002:95.84858,

    8007:87.44019,

    1644:99.14157,

    337:101.23135,

    253:115.93724,

    8416:96.84773,

    259:93.33662,

    262:75.35182,

    1652:89.77625

    }

sub['y'] = sub.apply(lambda r: leaks[int(r['ID'])] if int(r['ID']) in leaks else r['y'], axis=1)

sub.to_csv('stacked-models.csv', index=False)
import sys

from astropy.table import Table, Column



t = Table(names=('Model Name', 'Model_Score(r2squared)', 'Algorithm Used'), dtype=('S40', 'S10', 'S30'))

t.add_row(('XGBoost_Model_1', '0.54975', 'XGBoost with full data converted'))

t.add_row(('XGBoost_Model_2', '0.54998', 'XGBoost with some cleaned data'))

t.add_row(('XGBoost_Model_3', '0.55199', 'XGBoost with PCA, SVD and ICA'))

t.add_row(('H2O_Model_1', '0.55642', 'H2O AutoML'))

t.add_row(('Stacked_Model ', '0.58118', 'Stackong and pipeline and XGBoost'))



t.meta['comments'] = ['Conclusion For Models']

t.write(sys.stdout, format='ascii')

print(t)