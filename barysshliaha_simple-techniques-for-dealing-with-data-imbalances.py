import pandas as pd

import numpy as np

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

from sklearn.preprocessing import StandardScaler

from sklearn.utils import shuffle

from sklearn.metrics import roc_curve

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

rndd = 12345



df = pd.read_csv('/kaggle/input/bank-customer-churn-modeling/Churn_Modelling.csv')

df.info()
df.head()
df.tail()
df.drop('RowNumber', axis = 1, inplace = True)

df.head()
df.Gender.unique()
df.Gender = df.Gender.map({'Female': 0, 'Male':1})

df.head()
encoder = OrdinalEncoder()

data = encoder.fit_transform(df)

df_trans = pd.DataFrame(data, columns = df.columns)

df_trans.head()
df_trans.info()
df_trans = df_trans.astype({

    'CustomerId'    : 'int32',

    'Surname'       : 'int32',

    'Geography'     : 'int32',

    'Gender'        : 'int32',

    'Age'           : 'int32',

    'Tenure'        : 'int32',

    'NumOfProducts' : 'int32',

    'HasCrCard'     : 'int32',

    'IsActiveMember': 'int32',

    'Exited'        : 'int32'})

df_trans.info()
target = df_trans['Exited']

train = df_trans.drop('Exited', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.25, random_state=rndd)
#create a model for prediction

rand_Forest = RandomForestClassifier(random_state = rndd)



# define model parameters and values for tuning

parameters = {

    'n_estimators':np.arange(1,300, 50),

    'max_depth' : np.arange(2, 30, 2),

    'min_samples_split': np.arange(2, 30, 2),

    'min_samples_leaf': np.arange(2, 30, 2)    

}

#create a searchCV to cycle through the possible values

rand_Forest_grid = RandomizedSearchCV(

    estimator = rand_Forest,

    param_distributions  = parameters,

    scoring='f1',

    n_jobs=2,

    cv = 5,

    n_iter = 150,

    verbose=True, refit=True, return_train_score = True, random_state = rndd)

    

#fit the model    

rand_Forest_grid.fit(X_train, y_train)

#check scores result

f1_train = rand_Forest_grid.best_score_

print('Best Estimator: ', rand_Forest_grid.best_estimator_)

print('Best Params: ', rand_Forest_grid.best_params_)

print('f1 =', f1_train)

predicted_train = rand_Forest_grid.predict(X_train)

accuracy_train = accuracy_score(y_train, predicted_train)

print('accuracy =', accuracy_train)

roc_auc_score_train =  roc_auc_score(y_train, predicted_train)

print('roc_auc_score',  roc_auc_score_train)
#predict values on previously trained model

y_predicted = rand_Forest_grid.predict(X_test)



f1_test = f1_score(y_test, y_predicted)

accuracy_test = accuracy_score(y_test, y_predicted)

roc_auc_score_test =  roc_auc_score(y_test, y_predicted)

print('TEST       f1      =', f1_test)

print('TEST accuracy      =', accuracy_test)

print('TEST roc_auc_score =', roc_auc_score_test)
#Create empty dataframe with columns

results = pd.DataFrame(columns=['expirement', 'f1_train', 'f1_test', 'accuracy_train', 'accuracy_test', 'roc_auc_train', 'roc_auc_test'])

#add values to columns accordingly

results = results.append([{'expirement':'simple model',

                           'f1_train':f1_train, 'f1_test': f1_test,

                           'accuracy_train': accuracy_train, 'accuracy_test':accuracy_test,

                           'roc_auc_train':roc_auc_score_train, 'roc_auc_test':roc_auc_score_test}])

results
X_train.describe()
#create scaler

scaler = StandardScaler()

#fit and transform data

X_train_scaled = scaler.fit_transform(X_train)

#transform data based on previous fit process

X_test_scaled = scaler.transform(X_test)



#put transformed data for pretty print

d = pd.DataFrame(columns=X_train.columns, data=X_train_scaled).describe()

print('order of values', abs(d.loc['mean','EstimatedSalary']/ d.loc['mean','CreditScore']))
#create model with parameters vased on previous training result

rand_Forest = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

                       max_depth=18, max_features='auto', max_leaf_nodes=None,

                       min_impurity_decrease=0.0, min_impurity_split=None,

                       min_samples_leaf=2, min_samples_split=20,

                       min_weight_fraction_leaf=0.0, n_estimators=101,

                       n_jobs=None, oob_score=False, random_state=12345,

                       verbose=0, warm_start=False)



#define function for reducing code duplication

def checkModel(X_train, y_train, X_test, y_test, model = rand_Forest):

    

    model.fit(X_train, y_train)

    y_train_predicted = rand_Forest.predict(X_train)

    f1_train = f1_score(y_train, y_train_predicted)

    accuracy_train = accuracy_score(y_train, y_train_predicted)

    roc_auc_score_train =  roc_auc_score(y_train, y_train_predicted)

    

    print('roc_auc_score',  roc_auc_score_train)

    print('f1 =', f1_train)

    print('accuracy =', accuracy_train)

    

    y_test_predicted = rand_Forest.predict(X_test)

    f1_test = f1_score(y_test, y_test_predicted)

    accuracy_test = accuracy_score(y_test, y_test_predicted)

    roc_auc_score_test =  roc_auc_score(y_test, y_test_predicted)

    

    print('TEST       f1 =', f1_test)

    print('TEST accuracy =', accuracy_test)

    print('TEST roc_auc_score =', roc_auc_score_test)

    

    return f1_train, accuracy_train, roc_auc_score_train, f1_test, accuracy_test, roc_auc_score_test



#call function

f1_train, accuracy_train, roc_auc_score_train, f1_test, accuracy_test, roc_auc_score_test = checkModel(X_train_scaled, y_train, X_test_scaled, y_test)
results = results.append([{'expirement':'scaled data model',

                           'f1_train':f1_train, 'f1_test': f1_test,

                           'accuracy_train': accuracy_train, 'accuracy_test':accuracy_test,

                           'roc_auc_train':roc_auc_score_train, 'roc_auc_test':roc_auc_score_test}])

results
y_train.value_counts()
def upsample_1(features, target, repeat):

    #array only with 0 values from features

    features_zeros = features[target == 0]

    #array only with 1 values from features

    features_ones = features[target == 1]

    

    #array only with 0 values from target

    target_zeros = target[target == 0]

    #array only with 1 values from target

    target_ones = target[target == 1]

    

    #create new data frame with features 0 values and features 1 value repeated Repeat(incoming parameters in functions) times

    features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat)

    

    #create new data frame with target 0 values and target 1 value repeated Repeat(incoming parameters in functions) times

    target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)

    

    #just shuffle values in dataframe

    features_upsampled, target_upsampled = shuffle(features_upsampled, target_upsampled, random_state=rndd)

    

    return features_upsampled, target_upsampled
X_train_u, y_train_u = upsample_1(X_train, y_train, 4)

X_test_u, y_test_u = upsample_1(X_test, y_test, 4)

y_train_u.value_counts()
f1_train, accuracy_train, roc_auc_score_train, f1_test, accuracy_test, roc_auc_score_test = checkModel(X_train_u, y_train_u, X_test_u, y_test_u)
results = results.append([{'expirement':'upsmpled data model',

                           'f1_train':f1_train, 'f1_test': f1_test,

                           'accuracy_train': accuracy_train, 'accuracy_test':accuracy_test,

                           'roc_auc_train':roc_auc_score_train, 'roc_auc_test':roc_auc_score_test}])

results
scaler = StandardScaler()

X_train_u_scaled = scaler.fit_transform(X_train_u)

X_test_u_scaled = scaler.transform(X_test_u)



f1_train, accuracy_train, roc_auc_score_train, f1_test, accuracy_test, roc_auc_score_test = checkModel(X_train_u_scaled, y_train_u, X_test_u_scaled, y_test_u)
results = results.append([{'expirement':'upsmpled scaled data model',

                           'f1_train':f1_train, 'f1_test': f1_test,

                           'accuracy_train': accuracy_train, 'accuracy_test':accuracy_test,

                           'roc_auc_train':roc_auc_score_train, 'roc_auc_test':roc_auc_score_test}])

results