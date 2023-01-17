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
df = pd.read_csv('/kaggle/input/home-credit-default-risk-pk/application_train.csv')
df.shape
columns_count_dict = (df.count()/df.shape[0]).to_dict()

columns_null_rem = {k: v for k, v in columns_count_dict.items() if v < .55}

columns_count_dict
columns_null_rem.pop('EXT_SOURCE_1')
df = df.drop(columns_null_rem.keys(),axis=1)
df.shape
valid_cols = ['TARGET','NAME_CONTRACT_TYPE','CODE_GENDER','FLAG_OWN_CAR', 'FLAG_OWN_REALTY','CNT_CHILDREN','AMT_INCOME_TOTAL',\

              'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE','NAME_TYPE_SUITE','NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', \

              'NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','REGION_POPULATION_RELATIVE','DAYS_BIRTH','DAYS_EMPLOYED','DAYS_REGISTRATION',\

              'DAYS_ID_PUBLISH','CNT_FAM_MEMBERS','REGION_RATING_CLIENT','REGION_RATING_CLIENT_W_CITY','REG_REGION_NOT_LIVE_REGION','EXT_SOURCE_1', \

              'EXT_SOURCE_2','EXT_SOURCE_3','DEF_60_CNT_SOCIAL_CIRCLE','AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_YEAR','OCCUPATION_TYPE',\

              'ORGANIZATION_TYPE'] 



imp_cols = ['TARGET','NAME_CONTRACT_TYPE','FLAG_OWN_CAR', 'FLAG_OWN_REALTY','AMT_INCOME_TOTAL','AMT_CREDIT', \

            'AMT_ANNUITY', 'AMT_GOODS_PRICE','NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS','NAME_HOUSING_TYPE',\

           'DAYS_BIRTH','DAYS_EMPLOYED','REGION_RATING_CLIENT','EXT_SOURCE_1', 'EXT_SOURCE_2','EXT_SOURCE_3','OCCUPATION_TYPE','ORGANIZATION_TYPE']





print(len(valid_cols))

print(len(imp_cols))

df = df[imp_cols]
from operator import add



df['annuity_ratio'] = df['AMT_ANNUITY']/df['AMT_INCOME_TOTAL']

df['credit_ratio'] = df['AMT_CREDIT']/df['AMT_INCOME_TOTAL']

df['perc_age_employed'] = df['DAYS_EMPLOYED']/df['DAYS_BIRTH']

df['number_of_nulls'] = list(map(add,[(1 if k else 0) for k in list(df['EXT_SOURCE_1'].isnull())],\

                        list(map(add,[(1 if k else 0) for k in list(df['EXT_SOURCE_2'].isnull())],\

                                  [(1 if k else 0) for k in list(df['EXT_SOURCE_3'].isnull())]))))
obj_dict = df[df.select_dtypes(include=['object']).columns].nunique().to_dict()



onehot_cols = [k for k, v in obj_dict.items() if v < 10]

cust_encode_cols = [k for k, v in obj_dict.items() if v >= 10]
encoder_dict = {}



for i in cust_encode_cols:

    temp_dict = {}

    temp = df[[i,'TARGET']]

    for j in df[i].unique():

        temp_2 = temp[temp[i]==j]

        if len(temp_2) != 0:

            temp_dict[j] = len(temp_2[temp_2['TARGET']== 1])/len(temp_2)

    encoder_dict[i] =  temp_dict

for i in cust_encode_cols:

    temp_dict = encoder_dict[i]

#     print(i)

    df[i+'_float'] = df[i].apply(lambda x: temp_dict.get(x,0))

    df = df.drop(i,axis=1)
df.shape
df = pd.get_dummies(df, prefix=onehot_cols,drop_first=True)

df.shape
columns_count_dict = (df.count()/df.shape[0]).to_dict()

# columns_count_dict
# df[df['EXT_SOURCE_1'].isnull()]['TARGET'].value_counts(normalize = True)
# df[~(df['EXT_SOURCE_1'].isnull())]['TARGET'].value_counts(normalize = True)
df['EXT_SOURCE_1'] = df['EXT_SOURCE_1'].fillna(df['EXT_SOURCE_1'].mean())
df = df.dropna()

df.shape
from sklearn import preprocessing



print(df.shape)

temp_ = df['TARGET']

scaler = preprocessing.StandardScaler()

scaled_data = scaler.fit_transform(df.drop(['TARGET'],axis=1))

df = pd.DataFrame(scaled_data,columns=df.drop(['TARGET'],axis=1).columns)

df['TARGET'] = list(temp_)

print(df.shape)
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn import metrics
x_train, x_test, y_train, y_test = train_test_split(df.select_dtypes(exclude='object').drop(['TARGET'],axis=1), df['TARGET'], test_size=0.2, random_state=2)
# from sklearn.model_selection import GridSearchCV



# grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}# l1 lasso l2 ridge

# logreg=LogisticRegression()

# logreg_cv=GridSearchCV(logreg,grid,cv=10)

# logreg_cv.fit(x_train,y_train)



# print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)

# print("accuracy :",logreg_cv.best_score_)
%%time



from sklearn import pipeline

from sklearn import preprocessing



pipe = pipeline.Pipeline([

#     ("scaler", preprocessing.StandardScaler()),

    ("est", LogisticRegression(class_weight = 'balanced',solver='sag',max_iter=100))

])



pipe.fit(x_train,y_train)

y_test_pred = pipe.predict(x_test)

print(metrics.confusion_matrix(y_test,y_test_pred))

print(metrics.accuracy_score(y_test,y_test_pred))

print(metrics.classification_report(y_test,y_test_pred))
import sklearn.metrics as metrics

# calculate the fpr and tpr for all thresholds of the classification

probs = pipe.predict_proba(x_test)

preds = probs[:,1]

fpr, tpr, threshold = metrics.roc_curve(y_test, preds)

roc_auc = metrics.auc(fpr, tpr)



# method I: plt

import matplotlib.pyplot as plt

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
# temp = {}



# for i in range(len(x_train.columns)):

#     temp[x_train.columns[i]] = pipe[1].coef_[0][i]
# from imblearn.over_sampling import SMOTE



# print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))

# print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))



# sm = SMOTE(random_state=2)

# X_train_res, y_train_res = sm.fit_sample(x_train, y_train.ravel())



# print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))

# print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))
# X_train_res = pd.DataFrame(X_train_res,columns=x_train.columns)
# print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))

# print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))
%%time



from sklearn import pipeline

from sklearn import preprocessing

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier





pipe2 = pipeline.Pipeline([

#     ("scaler", preprocessing.StandardScaler()),

    ("rf", DecisionTreeClassifier(class_weight='balanced',max_depth= 3))

])



pipe2.fit(x_train,y_train)

y_test_pred = pipe2.predict(x_test)

print(metrics.confusion_matrix(y_test,y_test_pred))

print(metrics.accuracy_score(y_test,y_test_pred))

print(metrics.classification_report(y_test,y_test_pred))
import sklearn.metrics as metrics

# calculate the fpr and tpr for all thresholds of the classification

probs = pipe2.predict_proba(x_test)

preds = probs[:,1]

fpr, tpr, threshold = metrics.roc_curve(y_test, preds)

roc_auc = metrics.auc(fpr, tpr)



# method I: plt

import matplotlib.pyplot as plt

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
# def get_code(tree, feature_names):

#         left      = tree.tree_.children_left

#         right     = tree.tree_.children_right

#         threshold = tree.tree_.threshold

#         features  = [feature_names[i] for i in tree.tree_.feature]

        

#         value = tree.tree_.value



#         def recurse(left, right, threshold, features, node):

#                 if (threshold[node] != -2):

#                         print ("if ( " + features[node] + " <= " + str(threshold[node]) + " ) {")

#                         if left[node] != -1:

#                                 recurse (left, right, threshold, features,left[node])

#                         print ("} else {")

#                         if right[node] != -1:

#                                 recurse (left, right, threshold, features,right[node])

#                         print ("}")

#                 else:

#                         print ("return " +  str(value[node]))



#         recurse(left, right, threshold, features, 0)

        

# get_code(pipe2[0],X_train_res.columns)
estimator_limited = pipe2[0]





from sklearn.tree import export_graphviz

export_graphviz(estimator_limited, out_file='tree_limited.dot', feature_names =x_train.columns ,

                 class_names = [str(k) for k in pipe2[0].classes_],

                rounded = True, proportion = False, precision = 2, filled = True)



!dot -Tpng tree_limited.dot -o tree_limited.png -Gdpi=600
from IPython.display import Image

Image(filename = 'tree_limited.png')
# from sklearn.model_selection import RandomizedSearchCV

# import pprint



# n_estimators = [int(x) for x in np.linspace(start =10, stop = 100, num = 10)]

# # max_features = ['auto', 'sqrt']

# max_depth = [int(x) for x in np.linspace(10, 25, num = 5)]

# max_depth.append(None)

# # min_samples_split = [2, 5, 10]

# # min_samples_leaf = [1, 2, 4]

# # bootstrap = [True, False]



# random_grid = {'n_estimators': n_estimators,

# #                'max_features': max_features,

#                'max_depth': max_depth,

# #                'min_samples_split': min_samples_split,

# #                'min_samples_leaf': min_samples_leaf,

# #                'bootstrap': bootstrap

#               }



# # pprint(random_grid)



# rf = RandomForestClassifier(random_state = 42)



# rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

# rf_random.fit(X_train_res,y_train_res)

# rf_random.best_params_