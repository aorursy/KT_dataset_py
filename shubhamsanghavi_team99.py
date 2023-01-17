import numpy as np

import pandas as pd



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



import seaborn as sns

%matplotlib inline



from collections import Counter



from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV,RandomizedSearchCV, cross_val_score, StratifiedKFold, learning_curve, train_test_split, KFold

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score



# roc curve and auc

from sklearn.datasets import make_classification

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

from sklearn.metrics import precision_score

from matplotlib import pyplot



from sklearn.decomposition import PCA

from collections import defaultdict

import matplotlib

import xgboost

import fancyimpute

sns.set(style='white', context='notebook', palette='deep')
# MODEL REPRODUCIBLITY PARAMTERS

import random

REPRODUCABLE_RANDOM_STATE = 9

random.seed(REPRODUCABLE_RANDOM_STATE)
df = pd.read_csv("/kaggle/input/equipfails/equip_failures_training_set.csv",na_values=["na"])



# drop id

df = df.loc[:,df.columns!='id']

df_raw = df.copy(deep=True)
# seperate the positive dataframe for data exploration

df_pos = df[df['target']==1]

print(df_pos.shape)
# features and target

X = df.drop('target',1)

y=df['target']
# for testing purposes, work on 20% of the data

from sklearn.model_selection import train_test_split



# stratified spliting of dataset

# X_sample, _, y_sample, _ = train_test_split(X, y,

#                                 stratify=y, 

#                                 test_size=0.90,random_state=REPRODUCABLE_RANDOM_STATE)



# print(y_sample.value_counts())

# print(y_sample.value_counts(normalize=True))

# print(y_sample.value_counts())

# print(y_sample.value_counts(normalize=True))

X_sample = X

y_sample = y
from sklearn.model_selection import train_test_split



# stratified spliting of dataset

X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample,

                                                    stratify=y_sample, 

                                                    test_size=0.25,random_state=REPRODUCABLE_RANDOM_STATE)

X_train =  X_train.copy(deep=True)

X_test = X_test.copy(deep=True)

y_train = y_train.copy(deep=True)

y_test = y_test.copy(deep=True)

print(y_train.value_counts())

print(y_train.value_counts(normalize=True))

print(y_test.value_counts())

print(y_test.value_counts(normalize=True))
# NA values

# get the na counts in another notebook

df_na_count = X_train.isna().sum()

df_na_count = pd.DataFrame({'feature':df_na_count.index, 'NAcount':df_na_count.values})

df_na_count['percent'] = df_na_count['NAcount']/df.shape[0]

cut = [6000*i for i in range(0,10)]

label = [str(_) for _ in cut]

df_na_count['na_bin'] = pd.cut(df_na_count['NAcount'],cut,label)

df_na_count['na_bin'].value_counts(sort=False)
sns.heatmap(df.corr(),annot=True,fmt='.2f')
print(df['target'].value_counts())

sns.countplot(df['target'])
from collections import defaultdict
# preliminary handling of NA values for model comparision

missing_values = defaultdict(int)



for column in list(X_train):

    if len(X_train[column].unique()) > 100:

        val = X_train[X_train[column] != np.nan][column].mean()

        print(column,' -> mean : ',val)

    else:

        val = X_train[X_train[column] != np.nan][column].mode()

        if(val.shape):

            print(column, ' -> NaN')

            val = 0.0

        else:

            print(column,' -> mode : ',val)

    

    missing_values[column] = val 

    X_train[column].fillna(val,inplace=True)

# fill na for the test data

for column in list(X_test):

    X_test[column].fillna(missing_values[column],inplace=True)



# X_test.head()

# helper functions

class model_eval:

    def __init__(self):

        self.columns = ['Model','F1','Recall','Precision','auc_score','log_loss','Brier_loss']

        new = []

        self.results = pd.DataFrame(new,columns=self.columns)

        self.precision_recall_curves = defaultdict(list)

        self.roc_curves = defaultdict(list)

        self.count = 0

        



    def add_pred(self, model_name,y_test,y_pred,y_pred_probs=[]):

        self.y_test = y_test

        self.count += 1

        values=[]

        values.append(model_name)

        values.append(metrics.f1_score(self.y_test,y_pred))

        values.append(metrics.recall_score(self.y_test,y_pred))

        values.append(metrics.precision_score(self.y_test,y_pred))

        if len(y_pred_probs):

            values.append(metrics.roc_auc_score(self.y_test,y_pred_probs[:,1]))

            values.append(metrics.log_loss(self.y_test,y_pred_probs[:,1]))

            values.append(metrics.brier_score_loss(self.y_test,y_pred_probs[:,1]))

            self.precision_recall_curves[model_name] = metrics.precision_recall_curve(self.y_test,y_pred_probs[:,1])

            self.roc_curves[model_name] = metrics.roc_curve(self.y_test,y_pred_probs[:,1])

        else:

            values.append(np.nan)

            values.append(np.nan)

            values.append(np.nan)

            

        print(values)

        self.results = self.results.append(pd.Series(values, index=self.columns, name=self.count))

    

    def show_results(self):

        return self.results

    

    def plot_roc(self):

        

        for name in self.roc_curves.keys():

            # plot the precision-recall curves

            vals = self.roc_curves[name]

            pyplot.plot(vals[0], vals[1], label=name)



        pyplot.xlabel('Specifivity')

        pyplot.ylabel('Sensitivity')

        pyplot.title('ROC')

        # show the legend

        pyplot.legend()

        # show the plot

        pyplot.show()



    def plot_precision_recall(self):

        

        for name in self.precision_recall_curves.keys():

            # plot the precision-recall curves

            vals = self.precision_recall_curves[name]

            pyplot.plot(vals[0], vals[1], label=name)



        pyplot.xlabel('Precision')

        pyplot.ylabel('Recall')

        # show the legend

        pyplot.legend()

        # show the plot

        pyplot.show()





    def plot_scores(self):

        sns.scatterplot(x=self.results['F1'],y=self.results['auc_score'],hue=self.results['Model'],s =500)

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.datasets import make_moons, make_circles, make_classification

from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import xgboost
# LETS PEE_SEE_AEYYY

X_for_pca = X_sample.fillna(X_sample.mean())

# X_for_pca = X.fillna(X.mean())



# Checking number of components, X_train should be fine

pca = PCA().fit(X_for_pca)

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('number of components')

plt.ylabel('cumulative explained variance');
# visualize data in 2-dimensions



pca = PCA(n_components=2)

principalComponents = pca.fit_transform(X_for_pca)

principalDf = pd.DataFrame(data = principalComponents

             , columns = ['pc1','pc2'])



print(pca.explained_variance_ratio_)

principalDf['target'] = y

principalDf.head()

sns.distplot(principalDf['pc1'])

colors = ['red','blue']

principalDf = principalDf[principalDf['pc1'] < 10]

principalDf = principalDf[principalDf['pc2'] < 10]
plt.scatter(principalDf['pc1'],principalDf['pc2'],c=principalDf['target'],alpha =0.32, cmap=matplotlib.colors.ListedColormap(colors))
## START DATA MODELLING FROM HERE

## WE ARE GONNA TRY AND TRAIN WITH REDUCED FEATURES USING PCA

eval_data_proc = model_eval()



# add the baseline train-test for comparision

xgb_model_base = xgboost.XGBClassifier(learning_rate=0.1,n_estimators=1000, max_depth=170,min_child_weight=0,colsample_bytree=0.8,

                             subsample=0.8,colsample_bylevel=1,base_score=0.5,scale_pos_weight=1

                            ,max_delta_step=0,reg_alpha=1e-5,reg_lambda=0,gamma=1e-5)

xgb_model_base = xgb_model_base.fit(X_train, y_train)

y_pred_base = xgb_model_base.predict(X_test)

y_pred_proba_base = xgb_model_base.predict_proba(X_test)



eval_data_proc.add_pred('Baseline_with_XGB',y_test,y_pred_base,y_pred_proba_base)

# XGB with PCA 50 componenets 

pca = PCA(n_components=50)

principalComponents = pca.fit_transform(X_for_pca)

X_pca = principalComponents

X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y_sample,

                                                    stratify=y_sample, 

                                                    test_size=0.25,random_state=REPRODUCABLE_RANDOM_STATE)

xgb_model_pca = xgboost.XGBClassifier(learning_rate=0.1,n_estimators=1000, max_depth=170,min_child_weight=0,colsample_bytree=0.8,

                             subsample=0.8,colsample_bylevel=1,base_score=0.5,scale_pos_weight=1

                            ,max_delta_step=0,reg_alpha=1e-5,reg_lambda=0,gamma=1e-5)



xgb_model_pca = xgb_model_pca.fit(X_train_pca, y_train_pca)

y_pred_pca = xgb_model_pca.predict(X_test_pca)

y_pred_proba_pca = xgb_model_pca.predict_proba(X_test_pca)





#add evaluation for comparision

eval_data_proc.add_pred('PCA_50_with_XGB',y_test_pca,y_pred,y_pred_proba)

eval_data_proc.add_pred('PCA_50_with_XGB',y_test,y_pred,y_pred_proba)

## imputing missing values



X_numeric = X_sample.select_dtypes(include=[np.float]).as_matrix()

X_filled = pd.DataFrame(fancyimpute.SoftImpute().fit_transform(X_numeric))

X_train_soft, X_test_soft, y_train_soft, y_test_soft = train_test_split(X_filled, y_sample,

                                                    stratify=y_sample, 

                                                    test_size=0.25,random_state=REPRODUCABLE_RANDOM_STATE)

xgb_model_soft = xgboost.XGBClassifier(learning_rate=0.1,n_estimators=1000, max_depth=170,min_child_weight=0,colsample_bytree=0.8,

                             subsample=0.8,colsample_bylevel=1,base_score=0.5,scale_pos_weight=1

                            ,max_delta_step=0,reg_alpha=1e-5,reg_lambda=0,gamma=1e-5)



xgb_model_soft = xgb_model_soft.fit(X_train_soft, y_train_soft)

y_pred_soft = xgb_model_soft.predict(X_test_soft)

y_pred_proba_soft = xgb_model_soft.predict_proba(X_test_soft)





#add evaluation for comparision

eval_data_proc.add_pred('Soft_Impute_NA_with_XGB',y_test_soft,y_pred_soft,y_pred_proba_soft)
print(eval_data_proc.show_results())

eval_data_proc.plot_roc()

eval_data_proc.plot_precision_recall()

eval_data_proc.plot_scores()
# Oversampling for imbalanced class

from imblearn.over_sampling import RandomOverSampler

X_for_ros = X_sample.fillna(X_sample.mean())



ros = RandomOverSampler()

X_ros, y_ros = ros.fit_sample(X_for_ros, y_sample)

print(y_train.value_counts())

np.unique(y_ros,return_counts=True)
X_train_ros, X_test_ros, y_train_ros, y_test_ros = train_test_split(X_ros, y_ros,

                                                    stratify=y_ros, 

                                                    test_size=0.25,random_state=REPRODUCABLE_RANDOM_STATE)

xgb_model_ros = xgboost.XGBClassifier(learning_rate=0.1,n_estimators=1000, max_depth=170,min_child_weight=0,colsample_bytree=0.8,

                             subsample=0.8,colsample_bylevel=1,base_score=0.5,scale_pos_weight=1

                            ,max_delta_step=0,reg_alpha=1e-5,reg_lambda=0,gamma=1e-5)



xgb_model_ros = xgb_model_ros.fit(X_train_ros, y_train_ros)

y_pred_ros = xgb_model_ros.predict(X_test_ros)

y_pred_proba_ros = xgb_model_ros.predict_proba(X_test_ros)





#add evaluation for comparision

eval_data_proc.add_pred('Oversampling_with_XGB',y_test_soft,y_pred_soft,y_pred_proba_soft)
print(eval_data_proc.show_results())

eval_data_proc.plot_roc()

eval_data_proc.plot_precision_recall()

eval_data_proc.plot_scores()


# do a prelimiary classification and comparision of evaluations



names = ["Nearest Neighbors", "Gaussian Process",

         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",

         "Naive Bayes", "QDA","Log reg","XGBoost"]



names = ["XGBoost"]





classifiers = [

    xgboost.XGBClassifier(random_state=REPRODUCABLE_RANDOM_STATE)

    ]





eval_pre = model_eval()

for name, clf in zip(names, classifiers):

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    if "predict_proba" in dir(clf):

        y_pred_proba = clf.predict_proba(X_test)

    else:

        y_pred_proba = []

    

    eval_pre.add_pred(name,y_test,y_pred,y_pred_proba)





print(eval_pre.show_results())

eval_pre.plot_roc()

eval_pre.plot_precision_recall()
# increase the plot size to be more visible

plt.rcParams['figure.figsize'] = [15, 8]
eval_pre.plot_roc()

eval_pre.plot_precision_recall()
results_df = eval_pre.show_results()



sns.scatterplot(x=results_df['F1'],y=results_df['auc_score'],hue=results_df['Model'],s =500)

eval_pre.plot_scores()



# doing a grid search on log_reg and XGBoost
import graphviz
from xgboost import plot_tree

from xgboost import plot_importance
import os

os.environ["PATH"] += os.pathsep + r"C:\Users\shubh\Anaconda3\Library\bin\graphviz"



# print(os.environ["PATH"])
fig, ax = plt.subplots(figsize=(90, 60))

plot_tree(classifiers[-1], num_trees=4, ax=ax)

plot_importance(classifiers[-1],max_num_features=20)



# plot_tree(classifiers[-1])

# fig = plt.gcf()

# fig.set_size_inches(18.5, 10.5)

# plt.show()
xgboost.plot_importance(classifiers[-1],max_num_features=20)

plt.title("xgboost.plot_importance(model)")

plt.show()
xgboost.plot_importance(classifiers[-1],max_num_features=20)

plt.title("xgboost.plot_importance(model)")

plt.show()
xgboost.plot_importance(classifiers[-1], importance_type="cover",max_num_features=20)

plt.title('xgboost.plot_importance(model, importance_type="cover")')

plt.show()
xgboost.plot_importance(classifiers[-1], importance_type="gain",max_num_features=20)

plt.title('xgboost.plot_importance(model, importance_type="cover")')

plt.show()
import shap

shap.initjs()

explainer = shap.TreeExplainer(classifiers[-1])

shap_values = explainer.shap_values(X_train)
shap.force_plot(explainer.expected_value, shap_values[0,:], X_train.iloc[0,:])
shap.force_plot(explainer.expected_value, shap_values[:10,:], X_test.iloc[:10,:])
shap.summary_plot(shap_values, X_test, plot_type="bar")
param_test = {

#  'max_delta_step':[i for i in range(0,10,1)],

#     'reg_lambda':[0,1e-3,0.1],

#     'reg_alpha':[0,1e-3,0.1],

#     'gamma':[0.00001,0.00002,0.00003,0.00004,0.00005,0.00006,0.00007,0.00008,0.00009,0.0001]

    'max_depth' :[120,170,190],

    'min_child_weight':[0,1] 

#     'colsample_bytree':[0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8],

#     'subsample':[0.6,0.63,0.66,0.69,0.71,0.74,0.77,0.8]

#  'subsample':[0.84,0.85,0.86,0.87],

#  'colsample_bytree':[0.28,0.29,0.30,0.31,0.32]

}

gsearch = GridSearchCV(estimator = xgboost.XGBClassifier(eval_metric='auc',learning_rate=0.1,n_estimators=1000,

                                  colsample_bytree=0.8,subsample=0.8,colsample_bylevel=1,reg_alpha=1e-5

                                  ,base_score=0.5,scale_pos_weight=1,max_delta_step=0,reg_lambda=0

                                  ,max_depth=170, min_child_weight=0,gamma=1e-5), 

param_grid = param_test, scoring='roc_auc',iid=False, cv=3, verbose=50,n_jobs=-1)



gsearch.fit(X_train,y_train.values.ravel())
xgb_gridsearch = pd.DataFrame(gsearch.cv_results_)
xgb_gridsearch
gsearch.best_params_
param_test = {

 'max_delta_step':[i for i in range(0,10,1)],

    'reg_lambda':[0,1e-3,0.1],

    'reg_alpha':[0,1e-3,0.1],

    'gamma':[0.00001,0.00002,0.00003,0.00004,0.00005,0.00006,0.00007,0.00008,0.00009,0.0001],

#     'max_depth' :[120,170,190],

#     'min_child_weight':[0,1] 

    'colsample_bytree':[0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8],

    'subsample':[0.6,0.63,0.66,0.69,0.71,0.74,0.77,0.8],

 'subsample':[0.84,0.85,0.86,0.87],

 'colsample_bytree':[0.28,0.29,0.30,0.31,0.32]

}

Rsearch = RandomizedSearchCV(estimator = xgboost.XGBClassifier(eval_metric='auc',learning_rate=0.1,n_estimators=1000,

                                  colsample_bytree=0.8,subsample=0.8,colsample_bylevel=1,reg_alpha=1e-5

                                  ,base_score=0.5,scale_pos_weight=1,max_delta_step=0,reg_lambda=0

                                  ,max_depth=170, min_child_weight=0,gamma=1e-5), 

param_distributions = param_test, scoring='roc_auc',iid=False, cv=3, verbose=50,n_jobs=-1)



Rsearch.fit(X_train,y_train.values.ravel())
Rsearch.best_score_