%reset -f
from time import time

import pandas as pd

import numpy as np

from sklearn.random_projection import SparseRandomProjection as sr  # Projection features

from sklearn.cluster import KMeans                    # Cluster features

from sklearn.preprocessing import PolynomialFeatures  # Interaction features

from sklearn.preprocessing import StandardScaler, MinMaxScaler, scale

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.decomposition import PCA

import os, time, gc

from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier 

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics

from sklearn.metrics import auc, roc_curve

from eli5.sklearn import PermutationImportance

from sklearn.svm import SVC, LinearSVC, NuSVC

from sklearn.linear_model import LogisticRegression, SGDClassifier

import h2o

from h2o.estimators.deeplearning import H2ODeepLearningEstimator

from xgboost.sklearn import XGBClassifier

from sklearn.pipeline import Pipeline

from sklearn.pipeline import make_pipeline

from xgboost import plot_importance

from sklearn.model_selection import cross_val_score

import eli5

from eli5.sklearn import PermutationImportance

import time

import os

import gc

import random

from scipy.stats import uniform

from sklearn.ensemble import GradientBoostingClassifier as gbm

from bayes_opt import BayesianOptimization

from sklearn.model_selection import GridSearchCV

from skopt import BayesSearchCV

import os

import pandas as pd

os.chdir("../input")

print(os.listdir("../input"))
train = pd.read_csv("train.csv")

test = pd.read_csv("test.csv")
train.shape, test.shape


train['Target'].value_counts().sort_index().plot(kind='bar')
train.dtypes.value_counts().plot(kind='bar')
from pandas.plotting import scatter_matrix

scatter_matrix(train.select_dtypes('float'), alpha=0.2, figsize=(26, 20), diagonal='kde')
sns.countplot(data=train,

             x='Target',

             hue='r4t3')
trainPV = pd.pivot_table(train,index="Target", values=["r4t3"],aggfunc=np.mean)

trainPV
trainPVdf = trainPV.reset_index()

trainPVdf
sns.boxplot(data=trainPV.reset_index(),

             x='Target',

             hue='r4t3')
train.dtypes.value_counts()
train.select_dtypes('float').hist(bins=50,figsize=(20,16))
train.select_dtypes('object').head()
from collections import OrderedDict



plt.figure(figsize = (20, 16))

plt.style.use('fivethirtyeight')



# Color mapping

colors = OrderedDict({1: 'cyan', 2: 'magenta', 3: 'orange', 4: 'green'})

poverty_mapping = OrderedDict({1: 'extreme', 2: 'moderate', 3: 'vulnerable', 4: 'non vulnerable'})



# Iterate through the float columns

for i, col in enumerate(train.select_dtypes('float')):

    ax = plt.subplot(4, 2, i + 1)

    # Iterate through the poverty levels

    for poverty_level, color in colors.items():

        # Plot each poverty level as a separate line

        sns.kdeplot(train.loc[train['Target'] == poverty_level, col].dropna(), 

                    ax = ax, color = color, label = poverty_mapping[poverty_level])

        

    plt.title(f'{col.capitalize()} Distribution'); plt.xlabel(f'{col}'); plt.ylabel('Density')



plt.subplots_adjust(top = 2)


train['dependency']=train['dependency'].replace(['yes','no'],[1,0]).astype(np.float64)

train['edjefe']=train['edjefe'].replace(['yes','no'],[1,0]).astype(np.float64)

train['edjefa']=train['edjefa'].replace(['yes','no'],[1,0]).astype(np.float64)

train.drop(columns = ['Id','idhogar'] , inplace = True)



test['dependency']=test['dependency'].replace(['yes','no'],[1,0]).astype(np.float64)

test['edjefe']=test['edjefe'].replace(['yes','no'],[1,0]).astype(np.float64)

test['edjefa']=test['edjefa'].replace(['yes','no'],[1,0]).astype(np.float64)

test.drop(columns = ['Id','idhogar'] , inplace = True)

train.dtypes.value_counts().plot(kind='bar')
train[['dependency','edjefe','edjefa']].hist(bins=10)
# Number of missing data in each column

missing = pd.DataFrame(train.isnull().sum()).rename(columns = {0: 'total'})

# Create a percentage missing

missing['percent'] = (missing['total'] / len(train)*100)

missing.sort_values('percent', ascending = False).head(10)
imp = SimpleImputer(strategy="mean") 

train['rez_esc'] = imp.fit_transform(train[['rez_esc']])

train['v18q1'] = imp.fit_transform(train[['v18q1']])

train['v2a1'] = imp.fit_transform(train[['v2a1']])

train['SQBmeaned'] = imp.fit_transform(train[['SQBmeaned']])

train['meaneduc'] = imp.fit_transform(train[['meaneduc']])

test['rez_esc'] = imp.fit_transform(test[['rez_esc']])

test['v18q1'] = imp.fit_transform(test[['v18q1']])

test['v2a1'] = imp.fit_transform(test[['v2a1']])

test['SQBmeaned'] = imp.fit_transform(test[['SQBmeaned']])

test['meaneduc'] = imp.fit_transform(test[['meaneduc']])
# Number of missing data in each column

missing = pd.DataFrame(test.isnull().sum()).rename(columns = {0: 'total'})

# Create a percentage missing

missing['percent'] = (missing['total'] / len(test)*100)

missing.sort_values('percent', ascending = False).head(10)
train.head()
train.shape, test.shape
train.isnull().sum().sum() , test.isnull().sum().sum()   # 0



target=train['Target']
train.drop('Target', inplace = True, axis = 1)

train.shape, test.shape
train.head()
feat = [ "var", "median", "mean", "std", "max", "min"]

for i in feat:

    train[i] = train.aggregate(i,  axis =1)

    test[i]  = test.aggregate(i,axis = 1)
train.shape, test.shape
tmp = pd.concat([train,test],

                axis = 0,            # Stack one upon another (rbind)

                ignore_index = True

                )
tmp.head()
#tmp.drop(['idhogar'],axis=1,inplace=True)

#tmp.drop(['Id'],axis=1,inplace=True)
# Let us create 10 random projections/columns

NUM_OF_COM = 10 

# 13.1 Create an instance of class

rp_instance = sr(n_components = NUM_OF_COM)

rp = rp_instance.fit_transform(tmp.iloc[:, :147])
# Transfrom resulting array to pandas dataframe

# Also assign column names

rp = pd.DataFrame(rp, columns = ['r1','r2','r3','r4', 'r5', 'r6', 'r7', 'r8', 'r9', 'r10'] )

rp.head()   # Default column names are 0,1,2...9
# 13.4 Concat the new columns to tmp

tmp = np.hstack((tmp,rp))

tmp.shape
poly = PolynomialFeatures(2,                      # Degree 2

                          interaction_only=True,  # Avoid e.g. square(a)

                          include_bias = False   # No constant term

                          )
df =  poly.fit_transform(tmp[:, : 6])
df.shape
c_names = []

p = "p"

for i in range(21):

    t = p + str(i)

    c_names.append(t)

tmp = np.hstack([tmp,df])

tmp.shape     
# Separate train and test

X = tmp[: train.shape[0], : ]

test = tmp[train.shape[0] :, : ]



X.shape , test.shape                            
del tmp

gc.collect()
X_copy=X

X_copy.shape
X_train, X_test, y_train, y_test = train_test_split(X,target,test_size = 0.2)



X_train.shape  ,X_test.shape, y_train.shape, y_test.shape
scale = StandardScaler()

X_train_scaled = scale.fit_transform(X_train)

X_test_scaled = scale.fit_transform(X_test)
type(X_train_scaled), X_train_scaled.shape
pca=PCA().fit(X_train_scaled)

pca
type(pca)
len(pca.components_[0])

np.sum(pca.components_[0]**2)
print(len(pca.explained_variance_ratio_))
def pca_summary(pca, standardized_data, out=True):

    names = ["PC"+str(i) for i in range(1, len(pca.explained_variance_ratio_)+1)]

    a = list(np.std(pca.transform(standardized_data), axis=0))

    b = list(pca.explained_variance_ratio_)

    c = [np.sum(pca.explained_variance_ratio_[:i]) for i in range(1, len(pca.explained_variance_ratio_)+1)]

    columns = pd.MultiIndex.from_tuples([("sdev", "Standard deviation"), ("varprop", "Proportion of Variance"), ("cumprop", "Cumulative Proportion")])

    summary = pd.DataFrame(list(zip(a, b, c)), index=names, columns=columns)

    

    if out:

        print("Importance of components:")

        display(summary)

    return summary
Summary = pca_summary(pca, X_train_scaled)
def screeplot(pca, standardized_values):

    y = np.std(pca.transform(standardized_values), axis=0)**2

    x = np.arange(len(y)) + 1

    plt.figure(figsize=(20,10))

    plt.plot(x, y, "o-")

    plt.xticks(x, ["Comp."+str(i) for i in x], rotation=60)

    plt.ylabel("Variance")

    plt.show()
screeplot(pca, X_train_scaled)
cumsum = np.cumsum(pca.explained_variance_ratio_)

d = np.argmax(cumsum >= 0.95) + 1

d
pca=PCA(n_components=d)

X_train_reduced=pca.fit_transform(X_train_scaled)

X_test_reduced=pca.fit_transform(X_test_scaled)
X_train_reduced.shape, X_train_scaled.shape, X_test_reduced.shape, X_test_scaled.shape
X_train_reduced
%matplotlib inline
plt.figure(figsize=(20,10))

plt.plot(cumsum)
X_train_reduced.shape,y_train.shape
h2o.init()
train_h2o =h2o.import_file("train.csv")

test_h2o =h2o.import_file("test.csv")
type(train_h2o), type(test_h2o)
train_h2o['Target'] = train_h2o['Target'].asfactor()
train_h2o['Target'].levels()
trainh,testh = train_h2o.split_frame(ratios= [0.7])

trainh.shape, testh.shape
trainh_columns = trainh.columns[0:142] 

y_columns = trainh.columns[142] 
trainh['Target'] = trainh['Target'].asfactor()
h2o_model = H2ODeepLearningEstimator(epochs=1000,

                                    distribution = 'bernoulli',                 # Response has two levels

                                    missing_values_handling = "MeanImputation", # Not needed by us

                                    variable_importances=True,

                                    nfolds = 2,                           # CV folds

                                    fold_assignment = "Stratified",       # Each fold must be sampled carefully

                                    keep_cross_validation_predictions = True,  # For analysis

                                    balance_classes=False,                # SMOTE is not provided by h2o

                                    standardize = True,                   # z-score standardization

                                    activation = 'RectifierWithDropout',  # Default dropout is 0.5

                                    hidden = [100,100],                  # ## more hidden layers -> more complex interactions

                                    stopping_metric = 'logloss',

                                    loss = 'CrossEntropy')
import time

from time import time
start = time()

h2o_model.train(trainh_columns,

               y_columns,

               training_frame = trainh)

end = time()

(end - start)/60
print(h2o_model)
# 5.3 Column importance:

h2o_model_imp = pd.DataFrame(h2o_model.varimp(),

             columns=["Variable", "Relative Importance", "Scaled Importance", "Percentage"])
h2o_model_imp.head(20)
XGBCLF= XGBClassifier(silent = False,n_jobs=2)
X_train_reduced.shape,y_train.shape, X_test_reduced.shape, y_test.shape
XGBCLF.fit(X_train_reduced,y_train)
XGBCLF.get_params
XGB_pred = XGBCLF.predict(X_test_reduced)
##Accuracy of model.

(XGB_pred==y_test).sum()/(y_test.size)*100
from sklearn.metrics import classification_report
print(classification_report(y_test,XGB_pred))
XGBCLF.feature_importances_
plot_importance(XGBCLF,max_num_features=15)
pd.DataFrame({'Feature Importance':XGBCLF.feature_importances_}).sort_values(by = "Feature Importance", ascending=False)
GBM=gbm()

start = time()

GBM.fit(X_train_reduced,y_train)

end = time()

(end-start)/60
GBM_pred = GBM.predict(X_test_reduced) 
##Accuracy of model.

(GBM_pred ==y_test).sum()/(y_test.size)*100
print(classification_report(y_test,GBM_pred))
RanForCLF = RandomForestClassifier(random_state=42)

start = time()

RanForCLF.fit(X_train_reduced,y_train)

end = time()

(end-start)/60
RanForpred = RanForCLF.predict(X_test_reduced)

##Accuracy of model.

(RanForpred==y_test).sum()/(y_test.size)*100
print(classification_report(y_test,RanForpred))

#pd.DataFrame({'Feature Importance':RanForCLF.feature_importances_}).sort_values(by = "Feature Importance", ascending=False)
knnclf = KNeighborsClassifier(n_neighbors=4,p=2,metric='minkowski',n_jobs=-1)
start = time()

knnclf.fit(X_train_reduced,y_train)

end = time()

(end-start)/60
knn_pred = knnclf.predict(X_test_reduced)
##Accuracy of model.

(knn_pred==y_test).sum()/(y_test.size)*100
print(classification_report(y_test,knn_pred))
from sklearn.tree import DecisionTreeClassifier

treeclf = DecisionTreeClassifier(max_depth=4, random_state=1)

treeclf.fit(X_train_reduced,y_train)

treeclfpred = treeclf.predict(X_test_reduced) 
##Accuracy of model.

(treeclfpred==y_test).sum()/(y_test.size)*100
print(classification_report(y_test,treeclfpred))
from bayes_opt import BayesianOptimization
         

para_set = {

           'learning_rate':  (0, 1),                 # any value between 0 and 1

           'n_estimators':   (50,300),               # any number between 50 to 300

           'max_depth':      (3,10),                 # any depth between 3 to 10

           'n_components' :  (20,30)                 # any number between 20 to 30

            }
def xg_eval(learning_rate,n_estimators, max_depth,n_components):

    # 12.1 Make pipeline. Pass parameters directly here

    pipe_xg1 = (XGBClassifier             (

                                           silent = False,

                                           n_jobs=2,

                                           learning_rate=learning_rate,

                                           max_depth=int(round(max_depth)),

                                           n_estimators=int(round(n_estimators))

                                           )

                )



    # 12.2 Now fit the pipeline and evaluate

    cv_result = cross_val_score(estimator = pipe_xg1,

                                X= X_train_reduced,

                                y = y_train,

                                cv = 2,

                                n_jobs = 2,

                                scoring = 'accuracy'

                                ).mean()             # take the average of all results





    # 12.3 Finally return maximum/average value of result

    return cv_result

           
xgBO = BayesianOptimization(xg_eval,para_set)

                         
gp_params = {"alpha": 1e-5} 
start = time()

xgBO.maximize(init_points=5,    # Number of randomly chosen points to

                                 # sample the target function before

                                 #  fitting the gaussian Process (gp)

                                 #  or gaussian graph

               n_iter=25,      # Total number of times the

               #acq="ucb",       # ucb: upper confidence bound

                                 #   process is to be repeated

                                 # ei: Expected improvement

               # kappa = 1.0     # kappa=1 : prefer exploitation; kappa=10, prefer exploration

               **gp_params

               )

end = time()

(end-start)/60
xgBO.res

xgBO.max
params_new = xgBO.max['params']

params_new 