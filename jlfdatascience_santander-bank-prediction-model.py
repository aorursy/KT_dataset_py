from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2, f_classif

from sklearn.linear_model import LinearRegression #Selección VIF de características

from sklearn.linear_model.logistic import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from catboost import CatBoostClassifier, Pool, cv

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error

from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import roc_auc_score

from sklearn.metrics import recall_score

from sklearn.metrics import make_scorer

from sklearn.metrics import roc_curve

from sklearn.metrics import f1_score

from sklearn.metrics import auc

import matplotlib.pyplot as plt

from hyperopt import hp

import lightgbm as lgb

import numpy as np

import seaborn as sns

import pandas as pd

import warnings

import imblearn

import zipfile

import time



%pylab

%matplotlib inline

%config InlineBackend.figure_format = 'retina'

warnings.simplefilter("ignore")
#zf = zipfile.ZipFile('datos\santander-customer-transaction-prediction.zip') 

df_train = pd.read_csv('../input/santander-customer-transaction-prediction/train.csv')

df_test = pd.read_csv('../input/santander-customer-transaction-prediction/test.csv')
df_train.head()
df_train.info()
df_train.describe()
#We note that the dependent/target variable is very unbalanced.

df_train.target.value_counts().plot.bar() #.plot(kind="bar")
print("There is {}% of values 1 in the target variable".format(100*df_train['target'].value_counts()[1]/df_train.shape[0], 2))
df_train.isnull().sum()
#We have many variables, we look for a method to specifically locate null values

null_columns=df_train.columns[df_train.isnull().any()]

df_train[null_columns].isnull().sum()

print(df_train[df_train.isnull().any(axis=1)][null_columns].head())

print('It can´t find null values throughout the df')
#Separation of the target variable and the explanatory

target = 'target'

features = list(df_train.columns)

features.remove('target')

features.remove('ID_code')

#Separating the labels from the target variable

t0 = df_train[df_train['target'] == 0]

t1 = df_train[df_train['target'] == 1]

plt.figure(figsize=(16,6))

plt.title("Distribución de la media por fila")

sns.distplot(t0[features].mean(axis=1),color="red", kde=True,bins=120, label='target = 0')

sns.distplot(t1[features].mean(axis=1),color="blue", kde=True,bins=120, label='target = 1', hist_kws={'alpha':0.3})

plt.legend(); plt.show()
corr_matrix = df_train.corr().abs()

high_corr_var=np.where(corr_matrix>0.5)

high_corr_var=[(corr_matrix.columns[x],corr_matrix.columns[y]) for x,y in zip(*high_corr_var) if x!=y and x<y]

if len(high_corr_var)==0:

    print('There are no correlated variables')
#Generate two variables with the number of records in each class

count_class_0, count_class_1 = df_train.target.value_counts()



#Divide into two df with each class

df_class_0 = df_train[df_train['target'] == 0]

df_class_1 = df_train[df_train['target'] == 1]



#Undersampling with the 'sample' pandas property

df_class_0_under = df_class_0.sample(count_class_1)

df_train_under = pd.concat([df_class_0_under, df_class_1], axis=0)



print('Undersampling is in a number of records:')

print(df_train_under.target.value_counts())



df_train_under.target.value_counts().plot(kind='bar', title='Count (target)');
#Separation of the target variable and the explanatory

target = 'target'

features = list(df_train_under.columns)

features.remove('target')

features.remove('ID_code')

x_train = df_train_under[features]

y_train = df_train_under[target]



#Divide dataset into training and validation

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.75, random_state = 0, stratify=y_train)
#Check result

print(y_train.value_counts())
import imblearn

from imblearn.under_sampling import RandomUnderSampler



subm = RandomUnderSampler(return_indices=True)

x_subm, y_subm, id_subm = subm.fit_sample(df_train[features], df_train[target])



y_subm_plot = pd.DataFrame(y_subm)

y_subm_plot[0].value_counts().plot(kind='bar', title='Count (target)');
#Check the result

print(y_subm_plot[0].value_counts())
var_sk = SelectKBest(f_classif, k = 50)

x_sk = var_sk.fit_transform(x_train, y_train)



print(u"Number of final features:", x_sk.shape[1])

print(u"List of final features: \n", x_train.columns[var_sk.get_support()])

x_train_50best = x_train[x_train.columns[var_sk.get_support()]]
var_pc = SelectPercentile(f_classif, percentile = 50)

x_pc = var_pc.fit_transform(x_train, y_train)



print(u"Number of final features:", x_pc.shape[1])

print(u"List of final features: \n", x_train.columns[var_pc.get_support()])

x_train_100best = x_train[x_train.columns[var_pc.get_support()]]
def metricas(y_true, y_pred):

    print(u'La matriz de confusión es ')

    print(confusion_matrix(y_true, y_pred))



    print(u'Precisión:', accuracy_score(y_true, y_pred))

    print(u'Exactitud:', precision_score(y_true, y_pred))

    print(u'Exhaustividad:', recall_score(y_true, y_pred))

    print(u'F1:', f1_score(y_true, y_pred))



    false_positive_rate, recall, thresholds = roc_curve(y_true, y_pred)

    roc_auc = auc(false_positive_rate, recall)



    print(u'AUC:', roc_auc)



    plot(false_positive_rate, recall, 'b');

    plot([0, 1], [0, 1], 'r--');

    title(u'AUC = %0.2f' % roc_auc);
%%time

lr_classifier = LogisticRegression().fit(x_train, y_train)

y_train_pred = lr_classifier.predict(x_train)



print('Métricas de entrenamiento:')

metricas(y_train, y_train_pred);
%%time

#See the overfitting in the test dataset

y_test_pred  = lr_classifier.predict(x_test)

print('Métricas de validación:')

metricas(y_test, y_test_pred);
%%time

rf_classifier = RandomForestClassifier(n_estimators = 5,

                                       max_depth = 7, #Without limiting the depth of the tree is overfitting is even greater AUC:0.93

                                       random_state = 1)

rf_classifier.fit(x_train, y_train)

y_pred = rf_classifier.predict(x_train)



print('Métricas de entrenamiento:')

metricas(y_train, y_pred);
#Check that this method performs well but we will check the overfitting

#Comprobamos que este método tiene buen rendimiento pero comprobaremos el sobreajuste

y_test_pred = rf_classifier.predict(x_test)

print('Métricas de validación:')

metricas(y_test, y_test_pred);
#Definimos the parameter of the index of categorical properties although 

#in this case as we said by the previous data exploration, we do not have such variables

categorical_features_indices = np.where(x_train.dtypes != np.float)[0]
%%time

#Apply the model by setting some initial generic parameters

cat_model = CatBoostClassifier(

        depth=4,

        custom_loss=['AUC'],

        learning_rate=0.3,

        verbose=50,

        iterations=None,

        od_type='Iter',

        early_stopping_rounds=10

)



cat_model.fit(x_train,y_train,eval_set=(x_test,y_test),use_best_model=True)#para mejorar el procesado paramos en el mejor ajuste



pred = cat_model.predict_proba(x_test)[:,1]

y_train_pred = cat_model.predict(x_train)

#print('AUC de validación: ',roc_auc_score(y_test, pred))

print('Métricas de entrenamiento:')

metricas(y_train, y_train_pred); #Probamos el rendimiento del modelo
y_test_pred  = cat_model.predict(x_test)

print('Métricas de validación:')

metricas(y_test, y_test_pred); #Test on the validation sample to see if it shows overfitting
#Define the parameter of the index of categorical properties although

#in this case as we said by the previous data exploration, we do not have such variables

categorical_features_indices = np.where(x_train.dtypes != np.float)[0]

feature_names = x_train.columns.tolist()
# LightGBM dataset formatting 

lgtrain = lgb.Dataset(x_train, y_train,

                feature_name=feature_names,

                categorical_feature = categorical_features_indices)

lgvalid = lgb.Dataset(x_test, y_test,

                feature_name=feature_names,

                categorical_feature = categorical_features_indices)
#Set some generic initial parameters

params = {

    'objective' : 'binary',

    #'metric' : 'rmse',

    'num_leaves' : 200,

    'max_depth': 10,

    'learning_rate' : 0.01,

    #'feature_fraction' : 0.6,

    'verbosity' : -1

}

params['metric']=['auc', 'binary_logloss']
%%time

#Apply the model

lgb_clf = lgb.train(

    params,

    lgtrain,

    #num_iterations=2000,

    valid_sets=[lgtrain, lgvalid],

    valid_names=["train", "valid"],

    early_stopping_rounds=500,

    verbose_eval=500

)



#Training Prediction::

y_train_pred = lgb_clf.predict(x_train)



#Convert to binary values because in this model it gives us probabilities

for i in range(len(y_train_pred)):

    if y_train_pred[i]>=.5:       # setting threshold to .5

        y_train_pred[i]=1

    else:

        y_train_pred[i]=0

print('Métricas de entrenamiento:')      

metricas(y_train, y_train_pred);
#Validation prediction:

y_test_pred = lgb_clf.predict(x_test)



#Convert to binary values because in this model it gives us probabilities

for i in range(len(y_test_pred)):

    if y_test_pred[i]>=.5:       # setting threshold to .5

        y_test_pred[i]=1

    else:

        y_test_pred[i]=0

print('Métricas de validación:')

metricas(y_test, y_test_pred);
#%%time

#cat_model = CatBoostClassifier(verbose=50)

#Definition parameters space

#params = {'depth'         : [3,4,5,6],

#          'learning_rate' : [0.01,0.05,0.1,0.35,0.4],

#          'iterations'    : [30,50,125,150],

#          'l2_leaf_reg': [3,1,2,5,10]

#          }

#grid = GridSearchCV(estimator=cat_model, param_grid = params, cv = 3, n_jobs=-1)

#grid.fit(x_train,y_train)



#print("\n La mejor métrica de validación cruzada:\n", grid.best_score_)

#print("\n Los mejores parámetros:\n", grid.best_params_)
%%time

#Applying these "optimal paramenters" we would get the following model in Catboost

cat_model = CatBoostClassifier(

        depth= 3,

        learning_rate=0.4,

        iterations=150,

        l2_leaf_reg=10,

        custom_loss=['AUC'],

        verbose=50,

        random_seed=501

        )

cat_model.fit(x_train,y_train,eval_set=(x_test,y_test),use_best_model=True)



y_train_pred = cat_model.predict_proba(x_train)[:,1]

y_test_pred = cat_model.predict_proba(x_test)[:,1]

print('AUC_train: ',roc_auc_score(y_train, y_train_pred))

print('AUC_test: ',roc_auc_score(y_test, y_test_pred))
#Complete but very demanding parameters for processing

#lgb_grid_params = {

#    'objetive':['binary'],

#    'boosting_type' : ['gbdt'],

#    'learning_rate':  [0.05, 0.1 , 0.15, 0.2 , 0.255, 0.3], 

#    'max_depth': [ 5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15],

#    'min_child_weight': [1, 2, 3, 4, 5, 6, 7],

#    'num_leaves': [20, 30, 40],

#    'min_child_samples': [20, 33, 40, 50],

#    'colsample_bytree': [0.3, 0.4, 0.5, 0.6, 0.7],

#    'n_estimators': [50, 100, 118, 130],

#    'subsample' : [0.7,0.75],

#    'reg_alpha' : [1,1.2],

#    'reg_lambda' : [1,1.2,1.4],

#    'random_state' : [501]

#}
#%%time

#mdl = lgb.LGBMClassifier(boosting_type= 'gbdt',

#          objective = 'binary',

#          metric= ['binary_logloss', 'auc'],

          #n_jobs = 3, # Updated from 'nthread'

#          silent = True)



#grid = GridSearchCV(estimator=mdl, param_grid = lgb_grid_params, cv = 3, n_jobs=-1)

#grid.fit(x_train,y_train)



#print("\n El mejor estimador:\n", grid.best_estimator_)

#print("\n La mejor métrica de validación cruzada:\n", grid.best_score_)

#print("\n Los mejores parámetros:\n", grid.best_params_)
%%time

#LightGBM dataset formatting 

lgtrain = lgb.Dataset(x_train, y_train)

lgvalid = lgb.Dataset(x_test, y_test)



#Applying these optimal paramentros we would get the following model in LightGBM

params = {

    'objective' : 'binary',

    'num_leaves' : 20,

    #'max_depth': 10,

    'learning_rate' : 0.255,

    #'feature_fraction' : 0.6,

    'min_child_samples': 33,

    'n_estimators': 118,

    'verbosity' : 50,

    'random_state':501

}

params['metric']=['auc', 'binary_logloss']



#Training model

lgb_clf = lgb.train(

    params,

    lgtrain,

    #num_iterations=20000,

    valid_sets=[lgtrain, lgvalid],

    valid_names=["train", "valid"],

    early_stopping_rounds=500,

    verbose_eval=500,

    feature_name='auto', 

    categorical_feature='auto'

)



print("RMSE of train:", np.sqrt(mean_squared_error(y_train, lgb_clf.predict(x_train))));

y_train_pred = lgb_clf.predict(x_train);

print('AUC of train: ',roc_auc_score(y_train, y_train_pred ));

y_test_pred = lgb_clf.predict(x_test);

print('AUC of test: ',roc_auc_score(y_test, y_test_pred ));
feature_importances = lgb_clf.feature_importance()

feature_names = x_train.columns

for score, name in sorted(zip(feature_importances, feature_names), reverse=True):

    print('{}: {}'.format(name, score))
fea_imp = pd.DataFrame({'imp': feature_importances, 'col': feature_names})

fea_imp = fea_imp.sort_values(['imp', 'col'], ascending=[True, False]).iloc[-30:]

fea_imp.plot(kind='barh', x='col', y='imp', figsize=(10, 7), legend=None)

plt.title('LightGBM - Feature Importance')

plt.ylabel('Features')

plt.xlabel('Importance');
df_test.head()
target = 'target'

features = list(df_test.columns)

features.remove('ID_code')

X_test = df_test[features]



#Prediction with choose model LGBM

Y_prediction = lgb_clf.predict(X_test);



#Convert to binary values because in this model it gives us probabilities

for i in range(len(Y_prediction)):

    if Y_prediction[i]>=.5:       # setting threshold to .5

        Y_prediction[i]=1

    else:

        Y_prediction[i]=0



sub_df = pd.DataFrame({"ID_code":df_test["ID_code"].values})

sub_df["target"] = Y_prediction

sub_df["target"] = sub_df["target"].astype(int)

sub_df.to_csv("submission.csv", index=False)
pd.read_csv("submission.csv").head()