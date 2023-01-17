import numpy as np 

import pandas as pd 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import statsmodels.api as sm

#from statsmodels.nonparametric.kde import KDEUnivariate

#from statsmodels.nonparametric import smoothers_lowess

#from pandas import Series, DataFrame

#from patsy import dmatrices

from sklearn import datasets, svm

#from KaggleAux import predict as ka # see github.com/agconti/kaggleaux for more details





from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.metrics import f1_score, roc_auc_score

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

#from sklearn.model_selection import GridSearchCV

from sklearn.neural_network import MLPClassifier

from imblearn.pipeline import make_pipeline as make_pipeline_imb

from imblearn.over_sampling import SMOTE

from imblearn.metrics import classification_report_imbalanced

import xgboost as xgb

from xgboost import XGBClassifier



from collections import Counter

import time



from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

#from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold
pd.options.display.max_rows = 4000
df_train = pd.read_csv("../input/titanic/train.csv") 

df_test = pd.read_csv("../input/titanic/test.csv")

df_gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")



print('df_train.shape: ', df_train.shape)

print('df_test.shape: ', df_test.shape)

print('df_gender_submission.shape: ', df_gender_submission.shape)
df_train.info()
df_train.describe()
df_train.head(3)
df_train['Survived'].value_counts()
#Variable	Definition	Key

#survival	Survival	0 = No, 1 = Yes

#pclass	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd

#sex	Sex	

#Age	Age in years	

#sibsp	# of siblings / spouses aboard the Titanic	

#parch	# of parents / children aboard the Titanic	

#ticket	Ticket number	

#fare	Passenger fare	

#cabin	Cabin number	

#embarked	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton
df_train_copy = df_train.copy() # duplicate dataframe

# df_train_copy['Cabin_abc'] = 'X' # create column

# cabin_abc_dict = {'A':'A', 'B':'B', 'C':'C', 'D':'D', 'E':'E', 'F':'F', 'G':'G' }



# import re

# df_train_copy['Cabin_123'] = 0 # create column



# for ind in range(len(df_train_copy['Cabin'])):

#     for key in cabin_abc_dict:

#         if str(df_train_copy.loc[ind, 'Cabin'])[:1] == key:

#             df_train_copy.loc[ind, 'Cabin_abc'] = cabin_abc_dict[key]

#     try:

#         df_train_copy.loc[ind, 'Cabin_123'] = int( re.findall(r'\d+', str(df_train_copy.loc[ind, 'Cabin']) )[0]) # extract number only and select the 1st array 

#     except: # if error in regex extraction

#         df_train_copy.loc[ind, 'Cabin_123'] = 0

    

df_train_copy.head(3)
#df_train_copy['Fare'].value_counts().sort_index()
#df_train_copy.sort_values('Fare')
#df_train_copy.groupby(['Survived', 'Fare']).mean()
#len(df_train_copy[df_train_copy['Age'].isnull()])

#np.random.seed(1001)

#int( np.random.normal(df_train_copy['Age'].mean(), df_train_copy['Age'].std(), 1) )
### fill na

Age_median = np.median(df_train_copy['Age'].dropna())

df_train_copy['Age'] = df_train_copy['Age'].fillna(Age_median) # fill median

df_train_copy['Age_cat'] = np.floor(df_train_copy['Age']/10).astype('int').astype('object')



Embarked_mode = df_train_copy['Embarked'].mode()[0]

df_train_copy['Embarked'] = df_train_copy['Embarked'].fillna(Embarked_mode) #df_train_copy['Embarked'].value_counts().idxmax()[0]) # fill mode



iqr = df_train_copy['Fare'].quantile(.75) - df_train_copy['Fare'].quantile(.25)#df_train_copy['Fare'].median()

df_train_copy['Fare'].quantile(.75) + (iqr * 3)

df_train_copy['Fare'] = np.where(df_train_copy['Fare'] > df_train_copy['Fare'].quantile(.75) + (iqr * 3), df_train_copy['Fare'].median(), df_train_copy['Fare'] ) # assign median if outlier

bin_border = []

bin_size = 10

for i in np.arange(0,bin_size,1): #array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    bin_border.append( df_train_copy['Fare'].quantile(0.1 * i) )

df_train_copy['Fare_cat'] = np.where( df_train_copy['Fare'] > bin_border[bin_size-1], bin_size-1, 

                                     (np.where( df_train_copy['Fare'] > bin_border[bin_size-2], bin_size-2, 

                                               (np.where( df_train_copy['Fare'] > bin_border[bin_size-3], bin_size-3, 

                                                         (np.where( df_train_copy['Fare'] > bin_border[bin_size-4], bin_size-4,

                                                                   (np.where( df_train_copy['Fare'] > bin_border[bin_size-5], bin_size-5, 

                                                                             (np.where( df_train_copy['Fare'] > bin_border[bin_size-6], bin_size-6,

                                                                                       (np.where( df_train_copy['Fare'] > bin_border[bin_size-7], bin_size-7, 

                                                                                                 (np.where( df_train_copy['Fare'] > bin_border[bin_size-8], bin_size-8, 

                                                                                                           (np.where( df_train_copy['Fare'] > bin_border[bin_size-9], bin_size-9, 

                                                                                                                     (np.where( df_train_copy['Fare'] > bin_border[bin_size-10], bin_size-10, 1)))))))))))))))))))



df_train_copy.head(10)
NUM_COLS = ['SibSp','Parch']

CAT_COLS = ['Pclass', 'Sex', 'Embarked', 'Age_cat', 'Fare_cat' ]

NECCESSARY_COLUMNS = ['Survived'] + NUM_COLS + CAT_COLS
for col in NECCESSARY_COLUMNS:

    print( df_train_copy.groupby([col, 'Survived']).count()['PassengerId'] )
### factorize categorical variables

for column in ['Sex', 'Embarked']:

    labels, uniques = pd.factorize(df_train_copy[column])

    df_train_copy[column] = labels

    print(column, ' ; ', uniques)

    

df_train_copy.head(3) 
sns.set(style="ticks")



#df = sns.load_dataset("iris")

sns.pairplot(df_train_copy.iloc[:,1:], hue="Survived",  palette={0: 'red',1: 'green'}) # kind="reg",
for col in NECCESSARY_COLUMNS[1:]: #df_train_copy.columns:

    print('----------', col)

    print( df_train_copy[col].nunique() )

    print( df_train_copy[col].value_counts() )

    #print( df_train_copy[col].unique() )
# numerical_col = ['Age', 'Fare', 'SibSp','Parch']

# categorical_col = ['Pclass', 'Sex', 'Embarked', ]
# stacked bar chart https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/bar_stacked.html#sphx-glr-gallery-lines-bars-and-markers-bar-stacked-py



# for col in categorical_col:

#     print(col)

#     plt.bar( df_train_copy.loc[:,col].unique(), np.array( df_train_copy.groupby(['Survived',col]).count()['PassengerId'][0] ), color='red' ) # NOT survived

#     plt.bar( df_train_copy.loc[:,col].unique(), np.array( df_train_copy.groupby(['Survived',col]).count()['PassengerId'][1] ), color='green' ) # survived

#     plt.legend()

#     plt.show()
# for col in numerical_col:

#     print(col)

#     fig = plt.figure(figsize=(12, 4))

#     ax = fig.add_subplot(111)

#     ax.hist(x=np.array(df_train_copy[col][df_train_copy.loc[:, 'Survived'] == 0]), bins=10, alpha=0.6, color="red")

#     ax.hist(x=np.array(df_train_copy[col][df_train_copy.loc[:, 'Survived'] == 1]), bins=10, alpha=0.6, color="green")

#     ax.set_title(col)

#     ax.legend(prop={'size': 18})

#     plt.show()
df_train_copy = df_train_copy[NECCESSARY_COLUMNS]

df_train_copy.head(10)
#plt.hist( df_train_copy['Fare'] ) #.value_counts()

#plt.show()
## one hot encoding https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html

# from sklearn import preprocessing

# #from sklearn.preprocessing import OneHotEncoder



# le = preprocessing.LabelEncoder()

# X = X[1:].apply(le.fit_transform)



# #enc = preprocessing.LabelEncoder().fit_transform(X[1:]).reshape(-1,1)

# #X = OneHotEncoder().fit_transform(enc).toarray()

# #enc = OneHotEncoder(handle_unknown='ignore')

# #enc.fit(X[1:])
X = np.array( df_train_copy.iloc[:, 1:] )

y = np.array( df_train_copy.iloc[:, 0] ) # 'Survived'



print(X.shape)

print(y.shape)
### scaling https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html  https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html



#from sklearn.preprocessing import MinMaxScaler

#scaler = MinMaxScaler()



from sklearn.preprocessing import RobustScaler

scaler = RobustScaler().fit(X)



#print(scaler.fit_transform(X[:,:len(NUM_COLS)]))

#print(scaler.data_max_)

#print(scaler.transform(data))



X_scaled = scaler.fit_transform(X[:,:len(NUM_COLS)])

X = np.concatenate([X_scaled, X[:,len(NUM_COLS):]], 1)

print(X[:3,:]) # show head 3



for col in range(len(NUM_COLS)):

    print( 'col ', col, ', max: ', X[:,col].max(), ', min: ', X[:,col].min() )
### one hot encoding for categorical variables

from sklearn.preprocessing import OneHotEncoder



enc = OneHotEncoder(handle_unknown='ignore')

X_ohm = enc.fit_transform(X[:,len(NUM_COLS):]) # except Numerical variables row



X = np.concatenate([X[:,:len(NUM_COLS)], X_ohm.toarray()], 1)

print(X[:3,:]) # show head 3
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



X_train, X_cv, y_train, y_cv = train_test_split(X, y,

                                                    test_size=0.3,

                                                    shuffle=True,

                                                    random_state=42)

print(X_train.shape)

print(X_cv.shape)

print(y_train.shape)

print(y_cv.shape)
# https://www.kaggle.com/tilii7/hyperparameter-grid-search-with-xgboost

# https://www.kaggle.com/syd359/train-a-xgboost-classifier
def logistic_regression():

    #X_train, X_test, y_train, y_test = data_processor()

    

    parameters={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}# l1 lasso l2 ridge

    

    clf = LogisticRegression()#solver='lbfgs')#, class_weight = 'balanced')

    clf = GridSearchCV(clf, parameters, cv=10, verbose=10)

    clf.fit(X_train, y_train)

    print("Score: ", clf.score(X_cv, y_cv))

    

#     y_pred = clf.predict(X_cv)

#     y_pred_proba = clf.predict_proba(X_cv)[:, 1]

#     print("Accuracy is: {}".format(accuracy_score(y_cv, y_pred)))

#     print("F1 score is: {}".format(f1_score(y_cv, y_pred)))

#     print("AUC Score is: {}".format(roc_auc_score(y_cv, y_pred_proba)))
def logistic_with_smote():

    #X_train, X_test, y_train, y_test = data_processor()

    parameters={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}# l1 lasso l2 ridge



    clf = LogisticRegression(C=1e5)

    clf = GridSearchCV(clf, parameters, cv=10, verbose=10)

    clf.fit(X_train, y_train)

    # build model with SMOTE imblearn

    smote_pipeline = make_pipeline_imb(SMOTE(random_state=4), clf)



    smote_model = smote_pipeline.fit(X_cv, y_cv)

    print("Score: ", smote_model.score(X_cv, y_cv))

    

#     smote_prediction = smote_model.predict(X_test)

#     smote_prediction_proba = smote_model.predict_proba(X_test)[:, 1]



#     print(classification_report_imbalanced(y_test, smote_prediction))

#     print('SMOTE Pipeline Score {}'.format(smote_pipeline.score(X_test, y_test)))

#     print("SMOTE AUC score: ", roc_auc_score(y_test, smote_prediction_proba))

#     print("SMOTE F1 Score: ", f1_score(y_test, smote_prediction))
def randomForest():

    #X_train, X_test, y_train, y_test = data_processor()

    parameters = {'n_estimators': [10, 20, 30, 50], 'max_depth': [2, 3, 4]}



    clf = RandomForestClassifier(class_weight='balanced')

    clf = GridSearchCV(clf, parameters, n_jobs=10, verbose=10)

    clf.fit(X_train, y_train)

    

    print("Score: ", clf.score(X_cv, y_cv))

    

#     y_pred = clf.predict(X_cv)

#     y_pred_proba = clf.predict_proba(X_cv)[:, 1]



#     print("F1 score is: {}".format(f1_score(y_cv, y_pred)))

#     print("AUC Score is: {}".format(roc_auc_score(y_cv, y_pred_proba)))



#     print("The Features Importance are: ")  # for feature, value in zip(X_train.columns, clf.feature_importances_):

#     print(feature, value)

#     print(clf.best_estimator_)

#     print(clf.best_params_)

#     print(clf.best_score_)
def neural_nets():

    #X_train, X_test, y_train, y_test = data_processor()

    clf = MLPClassifier(hidden_layer_sizes=(100, 100, 100,))



    clf.fit(X_train, y_train)

    print("Score: ", clf.score(X_cv, y_cv))

    

#     y_pred = clf.predict(X_test)

#     y_pred_proba = clf.predict_proba(X_test)[:, 1]

#     print("F1 score is: {}".format(f1_score(y_test, y_pred)))

#     print("AUC Score is: {}".format(roc_auc_score(y_test, y_pred_proba)))
def xgb():

    # A parameter grid for XGBoost

    params = {

            'min_child_weight': [1, 5, 10],

            'gamma': [0.5, 1, 1.5, 2, 5],

            'subsample': [0.3, 0.8, 1.0],

            'colsample_bytree': [0.6, 0.8, 1.0],

            'max_depth': [3, 4, 5]

            }

    

    xgb = XGBClassifier(learning_rate=0.02, n_estimators=600, objective='binary:logistic',

                    silent=True, nthread=1)



    folds = 3

    param_comb = 5



    skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)



    global random_search

    random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring='roc_auc', n_jobs=10, cv=skf.split(X_train,y_train), verbose=10, random_state=1001 )



    # Here we go

    #start_time = timer(None) # timing starts from this point for "start_time" variable

    random_search.fit(X_train, y_train)

    #timer(start_time) # timing ends here for "start_time" variable

    print("Score: ", random_search.score(X_cv, y_cv))

    

    

#     print('\n All results:')

#     print(random_search.cv_results_)

#     print('\n Best estimator:')

#     print(random_search.best_estimator_)

#     print('\n Best normalized gini score for %d-fold search with %d parameter combinations:' % (folds, param_comb))

#     print(random_search.best_score_ * 2 - 1)

#     print('\n Best hyperparameters:')

#     print(random_search.best_params_)
start = time.time()



#logistic_regression()

#logistic_with_smote()

#randomForest()

#neural_nets()

xgb()

print("Total Time is: ", (time.time() - start)/60)
# https://blog.amedama.jp/entry/2019/01/29/235642



# dtrain = xgb.DMatrix(X_train, label=y_train)

# dtest = xgb.DMatrix(X_test, label=y_test)



# xgb_params = {

#     'objective': 'binary:logistic',

#     'eval_metric': 'error', # 'logloss'

# }



# bst = xgb.train(xgb_params,

#                 dtrain,

#                 num_boost_round=50,  

#                 )



# y_pred_proba = bst.predict(dtest)
# y_pred = np.where(y_pred_proba > 0.5, 1, 0)



# acc = accuracy_score(y_test, y_pred)

# print('Accuracy:', acc)
df_test.info()
df_test['Age'] = df_test['Age'].fillna(Age_median) # fill median

df_test['Age_cat'] = np.floor(df_test['Age']/10).astype('int').astype('object')



df_test['Embarked'] = df_test['Embarked'].fillna(Embarked_mode) #df_train_copy['Embarked'].value_counts().idxmax()[0]) # fill mode







df_test['Fare_cat'] = np.where( df_test['Fare'] > bin_border[bin_size-1], bin_size-1, 

                                     (np.where( df_test['Fare'] > bin_border[bin_size-2], bin_size-2, 

                                               (np.where( df_test['Fare'] > bin_border[bin_size-3], bin_size-3, 

                                                         (np.where( df_test['Fare'] > bin_border[bin_size-4], bin_size-4,

                                                                   (np.where( df_test['Fare'] > bin_border[bin_size-5], bin_size-5, 

                                                                             (np.where( df_test['Fare'] > bin_border[bin_size-6], bin_size-6,

                                                                                       (np.where( df_test['Fare'] > bin_border[bin_size-7], bin_size-7, 

                                                                                                 (np.where( df_test['Fare'] > bin_border[bin_size-8], bin_size-8, 

                                                                                                           (np.where( df_test['Fare'] > bin_border[bin_size-9], bin_size-9, 

                                                                                                                     (np.where( df_test['Fare'] > bin_border[bin_size-10], bin_size-10, 1)))))))))))))))))))



df_test.head(10)
for col in NECCESSARY_COLUMNS[1:]:

    print('----------', col)

    print( df_test[col].nunique() )

    print( df_test[col].value_counts() )

    #print( df_test[col].unique() )
df_test_copy = df_test[NECCESSARY_COLUMNS[1:]] # except for 'Survived'

print( df_test_copy.head(3) )     

    

### factorize categorical variables

for column in ['Sex', 'Embarked']:

    labels, uniques = pd.factorize(df_test_copy[column])

    df_test_copy[column] = labels

    print(column, ' ; ', uniques)

    

print( df_test_copy.head(3) )   

    

X_game = np.array( df_test_copy.iloc[:, :] )



### scaling

X_game_scaled = scaler.transform(X_game[:,:len(NUM_COLS)])

X_game = np.concatenate([X_game_scaled, X_game[:,len(NUM_COLS):]], 1)

print(X_game[:3,:]) # show head 3



### one hot encoding

X_game_ohm = enc.transform(X_game[:,len(NUM_COLS):]) # except Numerical variables row

X_game = np.concatenate([X_game[:,:len(NUM_COLS)], X_game_ohm.toarray()], 1)

print( X_game[:3, :] ) # show head 3

print( X_game.shape)



### predict

y_game_pred = random_search.predict(X_game)





# X_game =xgb.DMatrix(X_game) # convert to xgboost type

# X_game_pred_proba = bst.predict(X_game)

# y_game_pred = np.where(X_game_pred_proba > 0.5, 1, 0)
submission_csv = pd.DataFrame({'PassengerId': np.array( df_test.loc[:,'PassengerId'] ), 'Survived': y_game_pred})

submission_csv.head()
import datetime

filename = str(datetime.datetime.now())[:-7].replace('-','').replace(':','').replace(' ','')

print( filename)

submission_csv.to_csv('submission' + filename + '.csv', index=False)