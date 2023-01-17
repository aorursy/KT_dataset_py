#import useful stuff

import numpy as np

import pandas as pd



# figures 

%matplotlib inline

import matplotlib as mpl

import matplotlib.pyplot as plt

mpl.rc('axes', labelsize=14)

mpl.rc('xtick', labelsize=12)

mpl.rc('ytick', labelsize=12)
#some evaluation functions I have made, may not be used 

from sklearn.metrics import mean_squared_error



def model_rmse(x):

    """put predicted array as x"""

    return (np.sqrt(mean_squared_error(training_data_labels, \

                           x)))



from sklearn.metrics import confusion_matrix



def conf_matrix(x):

    """put model name as x"""

    return confusion_matrix(training_data_labels, x)



from sklearn.metrics import precision_score, recall_score



def model_precision(x):

    return precision_score(training_data_labels, x)



def model_recall(x):

    return recall_score(training_data_labels, x)



from sklearn.metrics import accuracy_score



def model_accuracy(x):

    """put model name as x"""

    return accuracy_score(training_data_labels, x)



def all_metrics(x):

    print('RMSE = ' + str(model_rmse(x)))

    print('Confusion matrix: ' + str(conf_matrix(x)))

    print('Precision: ' + str(model_precision(x)))

    print('Recall: ' + str(model_recall(x)))

    print('Accuracy: ' + str(model_accuracy(x)))

    

def feat_imp(model,cols):

    """give the model you want feature importance for

    specify the prepared dataframe you want columns from"""

    feat_df = pd.DataFrame(model.feature_importances_).T

    feat_df.columns = cols.columns

    return feat_df
# import the data to a pandas dataframe

training_data = pd.read_csv("../input/titanic/train.csv")

training_data_original = training_data.copy() # make a copy of the original data as a back up
training_data.head()
training_data.info()
training_data.describe()
#how many NaN/missing values in each feature?

training_data.isnull().sum()
#how many unique values in each feature?

training_data.nunique()
#quick histogram plots

training_data.hist(bins=25, figsize=(20,20))
#psuedocode: if the age/0.5 is odd = data is estimated

training_data.loc[((training_data['Age'] // 0.5) % 2 != 0) & \

                  (training_data['Age'] >= 1.0)].count() # don't count the under 1 as these are decimals
training_data.loc[training_data['Age'] < 1].count()
training_data.dtypes
dummy = pd.get_dummies(training_data['Sex'])

training_data = pd.concat([training_data, dummy['male']], axis=1)                                     

# male column only, if 1 = male, if 0 = female
training_data.head()
#have a quick look at sex and survival

training_data['Survived'].loc[(training_data['male'] == 1)].hist()

training_data['Survived'].loc[(training_data['male'] == 0)].hist(alpha=0.5)
corr_matrix = training_data.corr()

corr_matrix['Survived']
dummy_embark = pd.get_dummies(training_data['Embarked'])

training_data = pd.concat([training_data, dummy_embark], axis=1)                                     
training_data.head()
corr_matrix = training_data.corr()

corr_matrix['Survived']
dummy_class = pd.get_dummies(training_data['Pclass'])

training_data = pd.concat([training_data, dummy_class], axis=1)                                     
corr_matrix = training_data.corr()

corr_matrix['Survived']
training_data = training_data.drop(columns=['male','C','Q','S',1,2,3])

training_data.head()
training_data.isnull().sum()
training_data_numerical = training_data[['Pclass','Age','SibSp','Parch','Fare']].copy()

training_data_categorical = training_data[['Sex','Embarked']].copy()
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")

imputer.fit(training_data_numerical)

numerical_imputed = pd.DataFrame((imputer.transform(training_data_numerical)),\

                          columns=training_data_numerical.columns,\

                          index=training_data_numerical.index)
training_data_numerical.head(6)
numerical_imputed.head(6)
imputer.statistics_
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(numerical_imputed)

numerical_imputed_scaled = pd.DataFrame((scaler.transform(numerical_imputed)),\

                          columns=numerical_imputed.columns,\

                          index=numerical_imputed.index)
numerical_imputed_scaled.head()
from sklearn.pipeline import Pipeline



numerical_pipeline = Pipeline([

        ('imputer', SimpleImputer(strategy="median")),

        ('scaler', StandardScaler()),

    ])
num_pip_test =  numerical_pipeline.fit_transform(training_data_numerical)
#make the pipeline output a df for visualisation

num_pip_test_df =  pd.DataFrame(num_pip_test ,\

                          columns=training_data_numerical.columns,\

                          index=training_data_numerical.index)
num_pip_test_df.head()
cat_imputer = SimpleImputer(strategy="most_frequent")

cat_imputer.fit(training_data_categorical)
cat_imputer.statistics_
cat_imputer = SimpleImputer(strategy="most_frequent")

categorical_imputed = pd.DataFrame(cat_imputer.fit_transform(training_data_categorical),\

                          columns=training_data_categorical.columns,\

                          index=training_data_categorical.index)
categorical_imputed.head()
categorical_imputed.isnull().sum()
from sklearn.preprocessing import OneHotEncoder

onehot_cat = OneHotEncoder()

onehot_cat.fit_transform(categorical_imputed)
onehot_cat.categories_
categorical_pipeline = Pipeline([

        ('imputer', SimpleImputer(strategy="most_frequent")),

        ('onehot', OneHotEncoder(categories='auto',sparse=False))

    ])

# sparse is set to false only so I can make the dataframe below and to make things easier to visualise
cat_pip_test = categorical_pipeline.fit_transform(training_data_categorical)
#make the pipeline output a df for visualisation

cat_pip_test_df =  pd.DataFrame(cat_pip_test ,\

                          columns=['female','male','C','Q','S'],\

                          index=training_data_categorical.index)
cat_pip_test_df.head()
from sklearn.compose import ColumnTransformer



full_pipeline = ColumnTransformer([

        ("num", numerical_pipeline, training_data_numerical.columns),

        ("cat", categorical_pipeline, training_data_categorical.columns),

    ])
prepared_data = full_pipeline.fit_transform(training_data)
prepared_data.shape
prepared_data[0]
#output a df for visualisation

prep_data_df =  pd.DataFrame(prepared_data ,\

                          columns=['Pclass','Age','SibSp','Parch','Fare','female','male','C','Q','S'],\

                          index=training_data_categorical.index)
prep_data_df.head()
training_data_labels = training_data['Survived'].values

training_data_labels[0:5]
labels_df = pd.DataFrame(training_data_labels,\

                        columns=['Survived'],\

                        index=training_data.index)
labels_df.head()
#specify input data, I might want to change the feature I use later

input_data = prep_data_df\

[['Pclass','Age','SibSp','Parch','Fare','female','male','C','Q','S']]
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(penalty='l2', dual=False, tol=0.00001, C=0.1,\

                         fit_intercept=True, intercept_scaling=1,\

                         class_weight=None, random_state=42, solver='saga',\

                         max_iter=100, multi_class='warn', verbose=0,\

                         warm_start=False, n_jobs=None)

lr.fit(input_data, training_data_labels)
from sklearn.model_selection import cross_val_score

lr_cv_score = cross_val_score(lr, input_data, training_data_labels, cv=10, scoring="accuracy")

lr_cv_score
lr_cv_score.mean()
lr_pred = lr.predict(input_data)

all_metrics(lr_pred)
from sklearn.svm import SVC 

svm = SVC(C=0.1, kernel='linear', gamma='auto',\

           shrinking=True, probability=True,\

           tol=0.001, class_weight=None,\

           verbose=False, max_iter=-1, decision_function_shape='ovr'\

           , random_state=42) 



svm.fit(input_data, training_data_labels)
svm_score = cross_val_score(svm, input_data, training_data_labels, cv=10, scoring="accuracy")

svm_score
svm_score.mean()
svm_pred = svm.predict(input_data)

all_metrics(svm_pred)
from sklearn.model_selection import GridSearchCV



svm = SVC(shrinking=True, probability=True,\

           tol=0.001, class_weight=None,\

           verbose=False, max_iter=-1, decision_function_shape='ovr'\

           , random_state=42) 



param_grid = [

  {'C': [0.1, 1, 10, 100, 1000], 'kernel': ['linear']},

  {'C': [0.1, 1, 10, 100, 1000], 'gamma': [10, 1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']},

 ]



gs = GridSearchCV(svm, param_grid, scoring='accuracy', refit=True, cv=3,  n_jobs=-1, return_train_score=True)





gs.fit(input_data, training_data_labels)
gs_cv_df = pd.DataFrame(gs.cv_results_)
gs_cv_df
gs.best_estimator_
gs.best_score_
gs.best_params_
from sklearn.svm import SVC 

svm = SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,\

          decision_function_shape='ovr', degree=3, gamma=0.1, kernel='rbf',\

          max_iter=-1, probability=True, random_state=42, shrinking=True,\

          tol=0.001, verbose=False)



svm.fit(input_data, training_data_labels)
svm_score = cross_val_score(svm, input_data, training_data_labels, cv=10, scoring="accuracy")

svm_score
svm_score.mean()
svm_pred = svm.predict(input_data)

all_metrics(svm_pred)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=42, n_estimators=1000, max_features='auto', \

                            min_samples_split=10,max_leaf_nodes=10, \

                            min_samples_leaf=10,max_depth=5, warm_start=False, oob_score=True)

rf.fit(input_data, training_data_labels)



#added some regularisation with min split etc.
rf_score = cross_val_score(rf, input_data, training_data_labels, cv=10, scoring="accuracy")

rf_score
rf_score.mean()
rf_pred = rf.predict(input_data)
all_metrics(rf_pred)
rf.oob_score_
feat_imp(rf,prep_data_df)
training_data['Age'].loc[(training_data['Survived'] == 1)].hist(bins=16,density=1)

training_data['Age'].loc[(training_data['Survived'] == 0)].hist(alpha=0.5,bins=16,density=1)
def child_cat(X,drop=False):

    X['Child'] = pd.cut(X['Age'], [0, 15, 80],\

                           labels=['1','0'])

# child = 1 (is a child under age 15)

child_cat(training_data)
training_data[['Age','Child']].head(10)
training_data = training_data.drop(columns='Child')

# drop the new column to ut training data back to normal
from sklearn.preprocessing import FunctionTransformer



child_add = FunctionTransformer(child_cat,validate=False)
child_add.fit_transform(training_data)
training_data.head()
training_data = training_data.drop(columns='Child')

training_data.head()
#redifine all of the pipelines here for clarity, also chane how the columns are selected



num_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

cat_cols = ['Sex', 'Embarked','Child']



numerical_pipeline = Pipeline([

        ('imputer', SimpleImputer(strategy="median")),

        ('child_add', FunctionTransformer(child_cat(training_data),validate=False)),

        ('scaler', StandardScaler())

    ])



categorical_pipeline = Pipeline([

        ('imputer', SimpleImputer(strategy="most_frequent")),

        ('onehot', OneHotEncoder(categories='auto',sparse=False))

    ])



full_pipeline = ColumnTransformer([

        ("num", numerical_pipeline, num_cols),

        ("cat", categorical_pipeline, cat_cols)

    ])

full_pipeline.fit(training_data)
prepared_data = full_pipeline.fit_transform(training_data)
prep_data_df =  pd.DataFrame(prepared_data ,\

                          columns=['Pclass','Age','SibSp','Parch','Fare','female','male','C','Q','S','Child','Not-Child'],\

                          index=training_data.index)

                          

prep_data_df.head(10)
input_data2 = prep_data_df[['Pclass','Age','SibSp','Parch','Fare','female','male','C','Q','S','Child','Not-Child']]
lr.fit(input_data2, training_data_labels)
lr_score = cross_val_score(lr, input_data2, training_data_labels, cv=10, scoring="accuracy")

lr_score
lr_score.mean()
lr_pred = lr.predict(input_data2)
all_metrics(lr_pred)
svm.fit(input_data2, training_data_labels)

svm_score = cross_val_score(svm, input_data2, training_data_labels, cv=10, scoring="accuracy")

svm_score
svm_score.mean()
svm_pred = svm.predict(input_data2)
all_metrics(svm_pred)
rf.fit(input_data2, training_data_labels)

rf_score = cross_val_score(rf, input_data2, training_data_labels, cv=10, scoring="accuracy")

rf_score
rf_score.mean()
rf_pred = rf.predict(input_data2)
all_metrics(rf_pred)
feat_imp(rf,input_data2)
from sklearn.ensemble import VotingClassifier



vote = VotingClassifier(estimators=[('lr', lr), ('rf', rf), ('svm',svm)], voting='hard')
vote.fit(input_data2, training_data_labels)
vote_pred = vote.predict(input_data2)
all_metrics(vote_pred)
vote = VotingClassifier(estimators=[('lr', lr), ('rf', rf), ('svm',svm)], voting='soft')

vote.fit(input_data2, training_data_labels)

vote_pred = vote.predict(input_data2)

all_metrics(vote_pred)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=10, weights='uniform', algorithm='auto',\

                           leaf_size=30, p=2, metric='minkowski',\

                           metric_params=None, n_jobs=None)



knn.fit(input_data2, training_data_labels)
knn_score = cross_val_score(knn, input_data2, training_data_labels, cv=10, scoring="accuracy")

knn_score
knn_score.mean()
knn_pred = knn.predict(input_data2)
all_metrics(knn_pred)
vote = VotingClassifier(estimators=[('lr', lr), ('rf', rf), ('svm',svm), ('knn',knn)], voting='soft')

vote.fit(input_data2, training_data_labels)

vote_pred = vote.predict(input_data2)

all_metrics(vote_pred)
# import the data to a pandas dataframe

test_data = pd.read_csv("../input/titanic/test.csv")

test_data_original = test_data.copy() # make a copy of the original data as a back up
test_data.head()
training_data.head()
num_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

cat_cols = ['Sex', 'Embarked','Child']



numerical_pipeline = Pipeline([

        ('imputer', SimpleImputer(strategy="median")),

        ('child_add', FunctionTransformer(child_cat(test_data),validate=False)),

        ('scaler', StandardScaler())

    ])



categorical_pipeline = Pipeline([

        ('imputer', SimpleImputer(strategy="most_frequent")),

        ('onehot', OneHotEncoder(categories='auto',sparse=False))

    ])



full_pipeline = ColumnTransformer([

        ("num", numerical_pipeline, num_cols),

        ("cat", categorical_pipeline, cat_cols)

    ])
# full_pipeline.fit(test_data)

prepared_test_data = full_pipeline.fit_transform(test_data)

prep_test_data_df =  pd.DataFrame(prepared_test_data ,\

                          columns=['Pclass','Age','SibSp','Parch','Fare','female','male','C','Q','S','Child','Not-Child'],\

                          index=test_data.index)

                          

prep_test_data_df.head(10)
input_test_data2 = prep_test_data_df[['Pclass','Age','SibSp','Parch','Fare','female','male','C','Q','S','Child','Not-Child']]
vote_pred_test = vote.predict(input_test_data2)
test_out_df = pd.DataFrame(vote_pred_test,columns=['Survived'])
test_out_df.head()
cat = pd.concat([test_data['PassengerId'], test_out_df['Survived']], axis=1, sort=False)

cat.head()
cat = cat.set_index('PassengerId')

cat.head()
cat.to_csv('test_predictions.csv')
!ls -ltr titanic/

rf_pred_test = rf.predict(input_test_data2)

test_out_df = pd.DataFrame(rf_pred_test,columns=['Survived'])

cat = pd.concat([test_data['PassengerId'], test_out_df['Survived']], axis=1, sort=False)

cat = cat.set_index('PassengerId')

cat.to_csv('test_predictions_rf.csv')
lr_pred_test = lr.predict(input_test_data2)

test_out_df = pd.DataFrame(lr_pred_test,columns=['Survived'])

cat = pd.concat([test_data['PassengerId'], test_out_df['Survived']], axis=1, sort=False)

cat = cat.set_index('PassengerId')

cat.to_csv('test_predictions_logr.csv')
svm_pred_test = svm.predict(input_test_data2)

test_out_df = pd.DataFrame(svm_pred_test,columns=['Survived'])

cat = pd.concat([test_data['PassengerId'], test_out_df['Survived']], axis=1, sort=False)

cat = cat.set_index('PassengerId')

cat.to_csv('test_predictions_svm.csv')
knn_pred_test = knn.predict(input_test_data2)

test_out_df = pd.DataFrame(knn_pred_test,columns=['Survived'])

cat = pd.concat([test_data['PassengerId'], test_out_df['Survived']], axis=1, sort=False)

cat = cat.set_index('PassengerId')

cat.to_csv('test_predictions_knn.csv')