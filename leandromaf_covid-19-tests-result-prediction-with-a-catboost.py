!jupyter nbextension enable --py widgetsnbextension



from catboost import CatBoostClassifier, Pool, cv

#from dataprep.eda import plot

from matplotlib import pyplot as plt

from numpy.random import RandomState

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score



import catboost

import hyperopt

import json

import numpy as np

import pandas as pd

import seaborn as sns

import shap 







print(catboost.__version__)

#print(dataprep.__version__)

print(shap.__version__)



!python --version
data_path = '../input/covid19/dataset.xlsx'

raw_data_einstein = pd.read_excel(data_path)

print(raw_data_einstein.shape)

raw_data_einstein.head()


# Columns to explicitely drop

vars_to_drop = [

  'Patient ID',

  'Relationship (Patient/Normal)',

  'International normalized ratio (INR)',

  'Urine - pH',

]





# Variables related to type of admission that won't be used in a first approach

admition_vars = [

  'Patient addmited to regular ward (1=yes, 0=no)',

  'Patient addmited to semi-intensive unit (1=yes, 0=no)',

  'Patient addmited to intensive care unit (1=yes, 0=no)'

]



# Results of others tests

other_tests_vars = [

  # just the 25% of the dataset have values for these

  'Respiratory Syncytial Virus',

  'Influenza A',

  'Influenza B',

  'Parainfluenza 1',

  'CoronavirusNL63',

  'Rhinovirus/Enterovirus',

  'Coronavirus HKU1',

  'Parainfluenza 3',

  'Chlamydophila pneumoniae',

  'Adenovirus',

  'Parainfluenza 4',

  'Coronavirus229E',

  'CoronavirusOC43',

  'Inf A H1N1 2009',

  'Bordetella pertussis',

  'Metapneumovirus',

  'Parainfluenza 2',

  # just the 15% of the dataset have values for these

  'Influenza B, rapid test',

  'Influenza A, rapid test',



]



# name of the column that holds the results of the test

target_variable = 'SARS-Cov-2 exam result'



# list to variables to work with, dropping what we don't want from the total columns of the raw dataset

vars_to_work = list(set(raw_data_einstein.columns) - set(vars_to_drop) - set(admition_vars))

working_data = raw_data_einstein[vars_to_work]



print(f'New dataset size:{working_data.shape}')

# get the number of exampeles with null values by column

def get_null_counts(df):

  null_columns=df.columns[df.isnull().any()]

  nan_count_df = pd.DataFrame(data=df[null_columns].isnull().sum(),columns=['NaN count'])

  nan_count_df.sort_values('NaN count',ascending=False,inplace=True)

  return nan_count_df





null_count = get_null_counts(working_data)



print('\nAmount of rows with Null values by column')

print(null_count.to_string())
nrows_limit = working_data.shape[0]



cols_empty = null_count.loc[null_count['NaN count'] >= nrows_limit].index.values

print('More columns to drop :\n{}'.format(cols_empty))



vars_to_work = list(set(working_data.columns) - set(cols_empty))



working_data = working_data[vars_to_work]

print(working_data.shape)

print(working_data.columns)

working_data.head(3)
working_data['y'] = 0

working_data.loc[working_data[target_variable]=='positive',['y']] = 1



print(working_data.loc[working_data.y == 1][[target_variable,'y']].sample(3))

print(working_data.loc[working_data.y == 0][[target_variable,'y']].sample(3))

print(working_data['y'].value_counts())

print(working_data.shape)
#col = target_variable

col = 'y'



# this value parametrizes how much % more of negatives we have from the positive count

sample_weight = 1.30



print(f'Classes proportions before sub-sample')

print(working_data[col].value_counts())



# split the dataset in positive and negative cases

working_data_pos = working_data.loc[working_data[col] == 1]

working_data_neg = working_data.loc[working_data[col] == 0]



# build a new one with only a sample of the negative cases, considering the weight we set before

working_data = pd.concat([

  working_data_pos,

  working_data_neg.sample(int(working_data_pos.shape[0] * sample_weight),random_state=123)

])



print(f'Umbalance after sub-sample')

print(working_data[col].value_counts())

print(working_data.shape)

null_count = get_null_counts(working_data)

nrows_limit = working_data.shape[0]



cols_empty = null_count.loc[null_count['NaN count'] >= nrows_limit].index.values

print('More columns to drop :\n{}'.format(cols_empty))



# armamos una lista con las columnas que quedaran: el total menos las que tienen todas las filas en nulo

vars_to_work = list(set(working_data.columns) - set(cols_empty))



# obtenemos solo las columnas de interés

working_data = working_data[vars_to_work]

print(working_data.shape)

print(working_data.columns)

working_data.head(3)
X = working_data.drop(target_variable, axis=1)

X = X.drop('y', axis=1)

y = working_data['y']

#y = working_data[target_variable]



print(f'Input (X) shape : {X.shape}')

print(f'Output (y) shape : {y.shape}')
X = X.fillna(-999)

X.sample(3)
categorical_features_indices = np.where(X.dtypes != np.float)[0]

categorical_features_indices = list(set(categorical_features_indices))

print(f'Categorical columns: \n{X.columns[categorical_features_indices]}')
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.1)

X_train, X_validation, y_train, y_validation = train_test_split(X_train,y_train,test_size = 0.1)

print(X_train.shape)

X_train.head(3)
# which metric do we want to optimize? 

#metric = 'Precision' 

metric = 'Recall'

#metric = 'F1'

#metric = 'BalancedAccuracy'



def hyperopt_objective(params):

    

    model = CatBoostClassifier(

        l2_leaf_reg=int(params['l2_leaf_reg']),

        learning_rate=float(params['learning_rate']),

        max_depth=int(params['max_depth']),

        subsample=float(params['subsample']),

        colsample_bylevel=float(params['colsample_bylevel']),

        bootstrap_type = 'Bernoulli',

        iterations=500,

        eval_metric=metric,

        random_seed=123,

        verbose=False,

        loss_function='CrossEntropy',

        use_best_model = True,

        #task_type='GPU',

        task_type='CPU',

        early_stopping_rounds=100,

    )

    

    cv_data = cv(

        Pool(X_validation, y_validation, cat_features=categorical_features_indices),

        model.get_params()

    )

    best_metric = np.max(cv_data[f'test-{metric}-mean'])

    

    return 1 - best_metric # as hyperopt minimises


# the parameters to choose are sampled with uniform destribution between the ranges specified

params_space = {

    'l2_leaf_reg': hyperopt.hp.uniform('l2_leaf_reg',2,5),

    'learning_rate': hyperopt.hp.uniform('learning_rate',0.001,0.01),

    'max_depth' : hyperopt.hp.uniform('max_depth',4,8),

    'subsample' : hyperopt.hp.uniform('subsample',0.8,1.0),

    'colsample_bylevel' : hyperopt.hp.uniform('colsample_bylevel',0.7,0.9)

}



trials = hyperopt.Trials()



best = hyperopt.fmin(

    hyperopt_objective,

    space=params_space,

    algo=hyperopt.tpe.suggest,

    max_evals=5,

    trials=trials,

    rstate=RandomState(123)

)



with open(f'best_{metric}.json', 'w') as fp:

    json.dump(best, fp)

    

print(best)


params = {

    'iterations': 500,

    'eval_metric': metric,

    'random_seed' : 123,

    'use_best_model': True,

    'loss_function': 'CrossEntropy',

    #'task_type':'GPU',

    'task_type':'CPU',

    'early_stopping_rounds' : 100,

    'bootstrap_type' : 'Bernoulli',

    'verbose' : True,

    'learning_rate': float(best['learning_rate']),

    'l2_leaf_reg' : float(best['l2_leaf_reg']),

    'max_depth' : int(best['max_depth']),

    'subsample' : float(best['subsample']),

    'colsample_bylevel' : float(best['colsample_bylevel']),

    #'class_weights' : train_class_weights

}



model = CatBoostClassifier(**params)



print(model)



model.fit(

    X_train, y_train,

    cat_features=categorical_features_indices,

    eval_set=(X_validation, y_validation),

    logging_level='Verbose',  # you can uncomment this for text output

    #verbose=True,

    plot=True

)



model.save_model(f'catboost_model_{metric}.bin')

model.save_model(f'catboost_model_{metric}.json', format='json')

'''

  Function to plot and print a confusion matrix and common classification metrics

'''

def plot_cm(labels, predictions, p=0.5):



  cm = confusion_matrix(labels, predictions > p)

  plt.figure(figsize=(5,5))

  sns.heatmap(cm, annot=True, fmt="d")

  plt.title('Confusion matrix @{:.2f}'.format(p))

  plt.ylabel('Actual label')

  plt.xlabel('Predicted label')



  print('True Negatives: ', cm[0][0])

  print('False Positives: ', cm[0][1])

  print('False Negatives: ', cm[1][0])

  print('True Positives: ', cm[1][1])

  print('Total : ', np.sum(cm[1]))





metric = 'Recall'

best_model = CatBoostClassifier()

best_model.load_model(f'catboost_model_{metric}.bin')

print(f'Using model with parameters : \n {best_model.get_params()}')



train_predictions = best_model.predict(X_train)



print('Train report')

print(classification_report(y_train, train_predictions))



print('Train Confusion matrix')

print()

plot_cm(y_train, train_predictions)

plt.show()







test_predictions = best_model.predict(X_test)



print('Test report')

print(classification_report(y_test, test_predictions))



print('Test Confusion matrix')

print()

plot_cm(y_test, test_predictions)

plt.show()

feature_importances = model.get_feature_importance(prettified=True)

important_features = feature_importances.loc[feature_importances['Importances'] > 0.0]

important_features


test_df = working_data



f,ax = plt.subplots(int((important_features.shape[0]+1)/2),2,figsize=(20,35))



row = 0

col = 0



for i in range(important_features.shape[0]):

    #print(f'{row}-{col}')

    

    the_ax = ax[row,col]



    if test_df[important_features['Feature Id'].values[i]].dtypes != np.float :

        sns.countplot(data=test_df,y=important_features['Feature Id'].values[i],hue='y', ax = the_ax)

    else:

        sns.violinplot(data=test_df,x='y',y=important_features['Feature Id'].values[i] , ax = the_ax)

        

    if col == 0 :

        col += 1

    else :

        row += 1

        col = 0



plt.show()

        

positive_cases_index = y_test.iloc[np.where(y_test == 1)].index



true_datapoints = X_test.loc[positive_cases_index]

print(true_datapoints.shape)



postive_datapoints_predictions = best_model.predict(true_datapoints)



true_positives_index = np.where(postive_datapoints_predictions == 1)[0]

true_positives_datapoints = true_datapoints.iloc[true_positives_index,:]

print(true_positives_datapoints.shape)



pool_tp = Pool(data=true_positives_datapoints, label=pd.Series(np.ones(true_positives_datapoints.shape[0])), cat_features=categorical_features_indices)

shap_values_tp = model.get_feature_importance(pool_tp, type='ShapValues')

expected_value = shap_values_tp[0,-1]

print(expected_value)

shap_values_tp = shap_values_tp[:,:-1]

print(shap_values_tp.shape)



shap.summary_plot(shap_values_tp, true_positives_datapoints)
plt.figure(figsize=(10,5))

sns.distplot(best_model.predict_proba(true_datapoints)[:,1])
good_true_positives_index = np.where(best_model.predict_proba(true_datapoints)[:,1] >= 0.5)[0]

good_true_positives_datapoints = true_datapoints.iloc[good_true_positives_index,:]

print(good_true_positives_datapoints.shape)

pool_gtp = Pool(data=good_true_positives_datapoints, label=pd.Series(np.ones(good_true_positives_datapoints.shape[0])), cat_features=categorical_features_indices)

shap_values_gtp = model.get_feature_importance(pool_gtp, type='ShapValues')

expected_value = shap_values_gtp[0,-1]

print(expected_value)

shap_values_gtp = shap_values_gtp[:,:-1]

print(shap_values_gtp.shape)

for i in range(min([good_true_positives_datapoints.shape[0],5])):

    shap.force_plot(expected_value, shap_values_gtp[i,:], good_true_positives_datapoints.iloc[i,:],matplotlib=True,text_rotation=45)
negative_cases_index = y_test.iloc[np.where(y_test == 0)].index

false_datapoints = X_test.loc[negative_cases_index]

print(false_datapoints.shape)



negative_datapoints_predictions = model.predict(false_datapoints)

true_negative_index = np.where(negative_datapoints_predictions == 0)[0]

true_negative_datapoints = false_datapoints.iloc[true_negative_index,:]

print(true_negative_datapoints.shape)



pool_tn = Pool(data=true_negative_datapoints, label=pd.Series(np.zeros(true_negative_datapoints.shape[0])), cat_features=categorical_features_indices)

shap_values_tn = model.get_feature_importance(pool_tn, type='ShapValues')

expected_value_tn = shap_values_tn[0,-1]

print(expected_value_tn)

shap_values_tn = shap_values_tn[:,:-1]

print(shap_values_tn.shape)



shap.summary_plot(shap_values_tn, true_negative_datapoints)
plt.figure(figsize=(10,5))

sns.distplot(best_model.predict_proba(false_datapoints)[:,0])
good_true_negatives_index = np.where(best_model.predict_proba(false_datapoints)[:,0] >= 0.505)[0]

good_true_negatives_datapoints = false_datapoints.iloc[good_true_negatives_index,:]

print(good_true_negatives_datapoints.shape[0])

pool_gtn = Pool(data=good_true_negatives_datapoints, label=pd.Series(np.ones(good_true_negatives_datapoints.shape[0])), cat_features=categorical_features_indices)

shap_values_gtn = model.get_feature_importance(pool_gtn, type='ShapValues')

expected_value = shap_values_gtn[0,-1]

print(expected_value)

shap_values_gtn = shap_values_gtn[:,:-1]

print(shap_values_gtn.shape)

for i in range(min([good_true_negatives_datapoints.shape[0],5])):

    shap.force_plot(expected_value, shap_values_gtn[i,:], good_true_negatives_datapoints.iloc[i,:],matplotlib=True,text_rotation=45)