# Python ≥3.5 is required

import sys

assert sys.version_info >= (3, 5)



# Scikit-Learn ≥0.20 is required

import sklearn

assert sklearn.__version__ >= "0.20"



# Common imports

import numpy as np

import os

import gc



import matplotlib.pyplot as plt

import seaborn as sns

palette = sns.color_palette('Paired', 10)



import numpy as np

import pandas as pd

# Pandas display options

pd.set_option('display.float_format', lambda x: '%.3f' % x)





#setting fontsize and style for all the plots

plt.style.use('fivethirtyeight')

plt.rcParams['font.size'] = 18

plt.rcParams['figure.figsize'] = (16,5)



%matplotlib inline 

#plotting directly without requering the plot()



import warnings

warnings.filterwarnings(action="ignore") #ignoring most of warnings, cleaning up the notebook for better visualization



pd.set_option('display.max_columns', 500) #fixing the number of rows and columns to be displayed

pd.set_option('display.max_rows', 500)



print(os.listdir("../input")) #showing all the files in the ../input directory



# Set random seed 

randomseed = 42



# Any results you write to the current directory are saved as output. Kaggle message :D
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
print(train.shape, test.shape)
train.head()
##Merging the train and test dataset in order to have more data to train our model.



train['source']='train' #creating a label for the training and testing set

test['source']='test'



data = pd.concat([train, test],ignore_index=True)

print (train.shape, test.shape, data.shape) #printing the shape
data.head()
data.describe()
data.info()
data.nunique()
#let's create a function to check for null values, calculate the percentage relative to the total size

#only shows null values.

def missing_values_calculate(trainset): 

    nulldata = (trainset.isnull().sum() / len(trainset)) * 100

    nulldata = nulldata.drop(nulldata[nulldata == 0].index).sort_values(ascending=False)

    ratio_missing_data = pd.DataFrame({'Ratio' : nulldata})

    return ratio_missing_data.head(30)



missing_values_calculate(data)
data['v18q1'] = data['v18q1'].fillna(0)



# Fill in households that own the house with 0 rent payment

data.loc[(data['tipovivi1'] == 1), 'v2a1'] = 0



# Create missing rent payment column

data['v2a1-missing'] = data['v2a1'].isnull()



# If individual is over 19 or younger than 7 and missing years behind, set it to 0

data.loc[((data['age'] > 19) | (data['age'] < 7)) & (data['rez_esc'].isnull()), 'rez_esc'] = 0



# Add a flag for those between 7 and 19 with a missing value

data['rez_esc-missing'] = data['rez_esc'].isnull()



data.loc[data['rez_esc'] > 5, 'rez_esc'] = 5

# Difference between people living in house and household size

data['hhsize-diff'] = data['tamviv'] - data['hhsize']



elec = []



# Assign values

for i, row in data.iterrows():

    if row['noelec'] == 1:

        elec.append(0)

    elif row['coopele'] == 1:

        elec.append(1)

    elif row['public'] == 1:

        elec.append(2)

    elif row['planpri'] == 1:

        elec.append(3)

    else:

        elec.append(np.nan)

        

# Record the new variable and missing flag

data['elec'] = elec

data['elec-missing'] = data['elec'].isnull()



# Remove the electricity columns

# data = data.drop(columns = ['noelec', 'coopele', 'public', 'planpri'])



# Wall ordinal variable

data['walls'] = np.argmax(np.array(data[['epared1', 'epared2', 'epared3']]),

                           axis = 1)



# data = data.drop(columns = ['epared1', 'epared2', 'epared3'])



# Roof ordinal variable

data['roof'] = np.argmax(np.array(data[['etecho1', 'etecho2', 'etecho3']]),

                           axis = 1)

# data = data.drop(columns = ['etecho1', 'etecho2', 'etecho3'])



# Floor ordinal variable

data['floor'] = np.argmax(np.array(data[['eviv1', 'eviv2', 'eviv3']]),

                           axis = 1)

# data = data.drop(columns = ['eviv1', 'eviv2', 'eviv3'])



# Create new feature

data['walls+roof+floor'] = data['walls'] + data['roof'] + data['floor']



# No toilet, no electricity, no floor, no water service, no ceiling

data['warning'] = 1 * (data['sanitario1'] + 

                         (data['elec'] == 0) + 

                         data['pisonotiene'] + 

                         data['abastaguano'] + 

                         (data['cielorazo'] == 0))



# Owns a refrigerator, computer, tablet, and television

data['bonus'] = 1 * (data['refrig'] + 

                      data['computer'] + 

                      (data['v18q1'] > 0) + 

                      data['television'])



# Per capita features

data['phones-per-capita'] = data['qmobilephone'] / data['tamviv']

data['tablets-per-capita'] = data['v18q1'] / data['tamviv']

data['rooms-per-capita'] = data['rooms'] / data['tamviv']

data['rent-per-capita'] = data['v2a1'] / data['tamviv']



# Create one feature from the `instlevel` columns

data['inst'] = np.argmax(np.array(data[[c for c in data if c.startswith('instl')]]), axis = 1)

# data = data.drop(columns = [c for c in data if c.startswith('instlevel')])



data['escolari/age'] = data['escolari'] / data['age']

data['inst/age'] = data['inst'] / data['age']

data['tech'] = data['v18q'] + data['mobilephone']



print('Data shape: ', data.shape)
mapping = {"yes": 1, "no": 0}



# Fill in the values with the correct mapping

data['dependency'] = data['dependency'].replace(mapping).astype(np.float64)

data['edjefa'] = data['edjefa'].replace(mapping).astype(np.float64)

data['edjefe'] = data['edjefe'].replace(mapping).astype(np.float64)
# Create correlation matrix

corr_matrix = data.corr()



# Select upper triangle of correlation matrix

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))



# Find index of feature columns with correlation greater than 0.95

to_drop = [column for column in upper.columns if any(abs(upper[column]) > 0.975)]



print(f'There are {len(to_drop)} correlated columns to remove.')

print(to_drop)
import featuretools.variable_types as vtypes



#Household variables boolean type

hh_bool = ['hacdor', 'hacapo', 'v14a', 'refrig', 'paredblolad', 'paredzocalo', 

           'paredpreb','pisocemento', 'pareddes', 'paredmad',

           'paredzinc', 'paredfibras', 'paredother', 'pisomoscer', 'pisoother', 

           'pisonatur', 'pisonotiene', 'pisomadera',

           'techozinc', 'techoentrepiso', 'techocane', 'techootro', 'cielorazo', 

           'abastaguadentro', 'abastaguafuera', 'abastaguano',

            'public', 'planpri', 'noelec', 'coopele', 'sanitario1', 

           'sanitario2', 'sanitario3', 'sanitario5',   'sanitario6',

           'energcocinar1', 'energcocinar2', 'energcocinar3', 'energcocinar4', 

           'elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4', 

           'elimbasu5', 'elimbasu6', 'epared1', 'epared2', 'epared3',

           'etecho1', 'etecho2', 'etecho3', 'eviv1', 'eviv2', 'eviv3', 

           'tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5', 

           'computer', 'television', 'lugar1', 'lugar2', 'lugar3',

           'lugar4', 'lugar5', 'lugar6', 'area1', 'area2', 'v2a1-missing', 'elec-missing']



#household ordered type

hh_ordered = [ 'rooms', 'r4h1', 'r4h2', 'r4h3', 'r4m1','r4m2','r4m3', 'r4t1',  'r4t2', 

              'r4t3', 'v18q1', 'tamhog','tamviv','hhsize','hogar_nin','hhsize-diff',

              'elec',  'walls', 'roof', 'floor', 'walls+roof+floor', 'warning', 'bonus',

              'hogar_adul','hogar_mayor','hogar_total',  'bedrooms', 'qmobilephone']



#household continuous type

hh_cont = ['v2a1', 'dependency', 'edjefe', 'edjefa', 'meaneduc', 'overcrowding',

          'phones-per-capita', 'tablets-per-capita', 'rooms-per-capita', 'rent-per-capita']



#individual boolean type

ind_bool = ['v18q', 'dis', 'male', 'female', 'estadocivil1', 'estadocivil2', 'estadocivil3', 

            'estadocivil4', 'estadocivil5', 'estadocivil6', 'estadocivil7', 

            'parentesco1', 'parentesco2',  'parentesco3', 'parentesco4', 'parentesco5', 

            'parentesco6', 'parentesco7', 'parentesco8',  'parentesco9', 'parentesco10', 

            'parentesco11', 'parentesco12', 'instlevel1', 'instlevel2', 'instlevel3', 

            'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 

            'instlevel9', 'mobilephone', 'rez_esc-missing']



#individual ordered type

ind_ordered = ['age', 'escolari', 'rez_esc', 'inst', 'tech']



#individual continuous type

ind_cont = ['escolari/age', 'inst/age']

to_remove = []

for l in [hh_ordered, hh_bool, hh_cont, ind_bool, ind_ordered, ind_cont]:

    for c in l:

        if c not in data:

            to_remove.append(c)
for l in [hh_ordered, hh_bool, hh_cont, ind_bool, ind_ordered, ind_cont]:

    for c in to_remove:

        if c in l:

            l.remove(c)
for variable in (hh_bool + ind_bool):

    data[variable] = data[variable].astype('bool')
for variable in (hh_cont + ind_cont):

    data[variable] = data[variable].astype(float)
for variable in (hh_ordered + ind_ordered):

    try:

        data[variable] = data[variable].astype(int)

    except Exception as e:

        print(f'Could not convert {variable} because of missing values.')
fig = plt.figure(figsize=(10,5))

sns.countplot(data[data['source'] == 'train']['Target'])

plt.title("Target distribution")
fig = plt.figure(figsize=(10,5))

sns.countplot(data['parentesco1'])

plt.title("Head of the household distribution")
fig = plt.figure(figsize=(16,5))

plt.subplot(2,1,1)

sns.distplot(data['v2a1'].dropna())

plt.title("Monthly rent payment distribution")

plt.subplot(2,1,2)

sns.distplot(np.log1p(data['v2a1'].dropna()))

plt.title("Monthly rent payment distribution in log + 1")

plt.tight_layout(h_pad=0.5)
data['SQBdependency'].value_counts()
fig = plt.figure(figsize=(16,10))

for i,col in enumerate(['agesq','SQBmeaned','SQBdependency','SQBovercrowding']):

    plt.subplot(4,1,i+1)

    sns.distplot(np.log1p(data[col].dropna()))

plt.tight_layout(h_pad=0.5)
corr = data.dropna().corr() #Let's take a look at the pearson's corr, just to have an overall view of how the attributes influence the price.

corr = corr[(corr >=0.2) | (corr <=-0.1)]

corr['Target'].dropna().sort_values(ascending=False)
def kde_target(var_name, df):

    

    # Calculate the correlation coefficient between the new variable and the target

    corr = df['Target'].corr(df[var_name])

        

    # Plot the distribution

    sns.kdeplot(df.ix[df['Target'] == 1.0, var_name], label = 'Target == 1')

    sns.kdeplot(df.ix[df['Target'] == 2.0, var_name], label = 'Target == 2')

    sns.kdeplot(df.ix[df['Target'] == 3.0, var_name], label = 'Target == 3')

    sns.kdeplot(df.ix[df['Target'] == 4.0, var_name], label = 'Target == 4')



    

    # label the plot

    plt.xlabel(var_name); plt.ylabel('Density'); plt.title('%s Distribution' % var_name)

    plt.legend();

    

    # print out the correlation

    print('The correlation between %s and the Target: %0.4f' % (var_name, corr))
#plotting negativaly correlated distributions

fig = plt.figure(figsize=(16,10))

for i,col in enumerate(['eviv2','dependency','etecho2']):

    plt.subplot(3,1,i+1)

    kde_target(col,data)

plt.tight_layout(h_pad=0.5)
data = data[[x for x in data if not x.startswith('SQB')]]

data = data.drop(columns = ['agesq'])

data.shape
train = data[data['source'] == 'train']

test = data[data['source'] == 'test']

train_labels = train['Target'].copy()

train.drop(["source","Id",'idhogar',"Target"], axis=1, inplace=True) #this is really important, separate our target Y from our X

test.drop(["source","Id",'idhogar',"Target"], axis=1, inplace=True)

print(train.shape, train_labels.shape, test.shape)
num_attribs = train.select_dtypes(exclude=['object', 'bool']) #selecting all the numerical data to use in our function DataFrameSelector

cat_attribs = train.select_dtypes(exclude=['int64','float64']) #selecting non numerical data to use in our function DataFrameSelector
# Inspired from stackoverflow.com/questions/25239958

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler,RobustScaler, MinMaxScaler

from sklearn.preprocessing import OrdinalEncoder

from sklearn.preprocessing import OneHotEncoder #gonna try this one later



class MostFrequentImputer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):

        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],

                                        index=X.columns)

        return self

    def transform(self, X, y=None):

        return X.fillna(self.most_frequent_)

    



#this pipeline is gonna be use for numerical atributes and standard scaler    

num_pipeline = Pipeline([

        ('imputer', SimpleImputer(strategy="median")),

        #('std_scaler', StandardScaler()),

        #('robust_scaler', RobustScaler()),

        ('minmaxscaler', MinMaxScaler()),

    ])



#this is gonna be used to imput categorical values

cat_pipeline = Pipeline([

        ("imputer", MostFrequentImputer()),

        ("cat_encoder", OrdinalEncoder()),

    ])
from sklearn.compose import ColumnTransformer



full_pipeline = ColumnTransformer([

        ("num", num_pipeline, list(num_attribs)),

        ("cat", cat_pipeline, list(cat_attribs)),

    ])

from sklearn.model_selection import train_test_split



train_final,train_validation, train_y, train_val_y  = train_test_split(train, train_labels,test_size=0.2, shuffle = True, random_state=randomseed)
print(train_final.shape, train_y.shape)
from sklearn.metrics import precision_score, recall_score, accuracy_score,confusion_matrix,classification_report,f1_score, roc_auc_score

import time #implementing in this function the time spent on training the model

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV,cross_val_score,train_test_split, KFold

from catboost import CatBoostClassifier, Pool

import lightgbm as lgb

import xgboost as xgb



#Generic function for making a classification model and accessing performance:

def classification_model(X_train,x_val, y_train,y_val,test_set, model_type='lgb',

                         params={},plot_confusion_matrix=False,

                         plot_features_importances=False):

    

    time_start = time.perf_counter() #start counting the time

    

    test_sub = np.zeros(test_set.shape[0])

    

    predict_val = np.zeros(X_train.shape[0])

    score = {}



     

    if model_type == 'lgb':

        

        model = lgb.LGBMClassifier(**params, n_estimators = 1500,class_weight = 'balanced', nthread = 4, n_jobs = -1)

        

        model.fit(X_train, y_train, eval_set=[(X_train, y_train), (x_val, y_val)], eval_metric='multiclass', 

                  early_stopping_rounds=200,

                    verbose=50)

            

        predict_val = model.predict(x_val, num_iteration=model.best_iteration_)

        

        #predicting using the model that has been trained above

        

        predict_val = model.predict(x_val)

        score['Accuracy'] = (accuracy_score(y_val, predict_val))

        score['Precision'] = (precision_score(y_val, predict_val,average='micro'))

        score['F1 score'] = (f1_score(y_val, predict_val,average='micro'))

        

        print("Model Report")



        print("Accuracy: "+ str(score["Accuracy"]))

        print("Precision: "+ str(score["Precision"]))

        print("F1 score: "+ str(score["F1 score"]))

        print('\n')

        

        print("         -------Classification Report----------")

        print(classification_report(y_val, predict_val))

    

        test_sub = model.predict(test_set) 

            

    if model_type == 'xgb':



        model = xgb.XGBClassifier(**params, nthread = 4, n_jobs = -1)



        model.fit(X_train, y_train, 

                      eval_set=[(X_train, y_train), (x_val, y_val)],

                          early_stopping_rounds=100,

                             verbose=50)

        

        #predicting using the model that has been trained above

        

        predict_val = model.predict(x_val, ntree_limit=model.best_ntree_limit)

        score['Accuracy'] = (accuracy_score(y_val, predict_val))

        score['Precision'] = (precision_score(y_val, predict_val,average='micro'))

        score['F1 score'] = (f1_score(y_val, predict_val,average='micro'))

        

        print("Model Report")



        print("Accuracy: "+ str(score["Accuracy"]))

        print("Precision: "+ str(score["Precision"]))

        print("F1 score: "+ str(score["F1 score"]))

        print('\n')

        

        print("         -------Classification Report----------")

        print(classification_report(y_val, predict_val))

    

        test_sub = model.predict(test_set) 



        

        if plot_features_importances:

            # feature importance

            fig, ax = plt.subplots(figsize=(12,10))

            xgb.plot_importance(model, max_num_features=50, height=0.8,color='c', ax=ax)

            ax.grid(False)

            plt.title("XGBoost - Feature Importance", fontsize=15)

            

    if model_type == 'cat':

        model = CatBoostClassifier(**params)

        model.fit(X_train, y_train, eval_set=(x_val, y_val), 

                  cat_features=[], use_best_model=True)



        #predicting using the model that has been trained above

        

        predict_val = model.predict(x_val)

        score['Accuracy'] = (accuracy_score(y_val, predict_val))

        score['Precision'] = (precision_score(y_val, predict_val,average='micro'))

        score['F1 score'] = (f1_score(y_val, predict_val,average='micro'))

        

        print("Model Report")



        print("Accuracy: "+ str(score["Accuracy"]))

        print("Precision: "+ str(score["Precision"]))

        print("F1 score: "+ str(score["F1 score"]))

        print('\n')

        

        print("         -------Classification Report----------")

        print(classification_report(y_val, predict_val))

    

        test_sub = model.predict(test_set)

    #################### PLOTTING FEATURES IMPORTANCE #################### 

    if plot_features_importances:

        plt.figure(figsize=(20, 20)) #figure size

        feature_importance = model.feature_importances_[:30]

        # make importances relative to max importance

        feature_importance = 100.0 * (feature_importance / feature_importance.max()) #making it a percentage relative to the max value

        sorted_idx = np.argsort(feature_importance)

        pos = np.arange(sorted_idx.shape[0]) + .5

        plt.barh(pos, feature_importance[sorted_idx], align='center')

        plt.yticks(pos, train.columns[sorted_idx], fontsize=15) #used train_drop here to show the name of each feature instead of our train_prepared 

        plt.xlabel('Relative Importance', fontsize=20)

        plt.ylabel('Features', fontsize=20)

        plt.title('Variable Importance', fontsize=30)



    #################### PLOTTING CONFUSION MATRIX #######################

    

    if plot_confusion_matrix:

        fig, ax = plt.subplots(figsize=(8,8)) #setting the figure size and ax

        mtx = confusion_matrix(y_val, predict_val)

        sns.heatmap(mtx, annot=True, fmt='d', linewidths=.5,  cbar=True, ax=ax) #create a heatmap with the values of our confusion matrix

        plt.ylabel('true label')

        plt.xlabel('predicted label')



    

    time_end = time.perf_counter() #end of counting the time

    

    total_time = time_end-time_start #total time spent during training and cross_validation

    

    print("Amount of time spent during training the model and cross validation: %4.3f seconds" % (total_time))

    

    # Clean up memory

    gc.enable()

    del model,score,total_time, time_end, time_start,predict_val,test_set

    gc.collect()

                        

    return test_sub
train_prepared = full_pipeline.fit_transform(train_final)

train_validation_prepared = full_pipeline.fit_transform(train_validation)

test_prepared = full_pipeline.fit_transform(test)
params_lgb = {

        "metric" : "multiclass",

        "num_leaves" : 30,

        "min_child_weight" : 50,

        "learning_rate" : 0.05,

        "bagging_fraction" : 0.7,

        "feature_fraction" : 0.7,

        "bagging_frequency" : 5,

        "bagging_seed" : 42,

        'reg_aplha': 1,

        'reg_lambda': 0.001

}



preds_lgb = classification_model(X_train=train_prepared,x_val=train_validation_prepared,y_train=train_y,y_val=train_val_y, 

                                 test_set=test_prepared,params=params_lgb, plot_features_importances=True,plot_confusion_matrix=True)
sub = pd.read_csv('../input/sample_submission.csv')

sub['Target'] = np.array(preds_lgb).astype(int)

sub.to_csv('lgb.csv', index = False)

sns.countplot(sub['Target'])