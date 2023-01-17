# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD

from sklearn.manifold import TSNE

import matplotlib.patches as mpatches

import time

import collections

from collections import Counter



from sklearn.model_selection import (GridSearchCV,train_test_split, cross_val_predict, cross_val_score, 

                                     KFold, StratifiedKFold, ShuffleSplit,learning_curve,

                                     RandomizedSearchCV)

from sklearn.metrics import (confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,

                             make_scorer,classification_report,roc_auc_score,roc_curve,

                             average_precision_score,precision_recall_curve)



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,VotingClassifier



from sklearn.pipeline import make_pipeline



from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline

from imblearn.metrics import classification_report_imbalanced

from imblearn.over_sampling import SMOTE

from imblearn.over_sampling import BorderlineSMOTE

from imblearn.over_sampling import ADASYN

from imblearn.over_sampling import RandomOverSampler



#from imblearn.under_sampling import ClusterCentroids

from imblearn.under_sampling import OneSidedSelection

from imblearn.under_sampling import NearMiss

from imblearn.under_sampling import EditedNearestNeighbours

from imblearn.under_sampling import TomekLinks

from imblearn.under_sampling import RandomUnderSampler



from imblearn.combine import SMOTETomek

from imblearn.combine import SMOTEENN



pd.set_option('display.max_columns', None)



import warnings

warnings.filterwarnings("ignore")



RANDOM_SEED = 101



colors = ["#0101DF", "#DF0101"]



import collections

from mpl_toolkits import mplot3d
df = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")

df.head()
df.describe()
df.columns
# The classes are heavily imbalanced

print('Normal', round(df['Class'].value_counts()[0]/len(df) * 100,2), '% of the dataset')

print('Anomalies', round(df['Class'].value_counts()[1]/len(df) * 100,2), '% of the dataset')
sns.countplot('Class', data=df, palette=colors)

plt.title('Class Distributions \n (0: Normal || 1: Anomaly)', fontsize=14)
df.isnull().sum()
def check_impute_variance(ser,var):

    mean = ser.mean()

    median = ser.median()

    

    mean_ser = ser.fillna(mean)

    median_ser = ser.fillna(median)

    

    var_original = ser.std()**2

    var_mean = mean_ser.std()**2

    var_median = median_ser.std()**2

    

    fig = plt.figure(figsize = (10,5))

    ax = fig.add_subplot(111)

    ax = sns.kdeplot(ser.dropna(), color="Red", shade = True, label="Original Variance : %.2f"%(var_original))

    ax = sns.kdeplot(mean_ser, color="Blue", shade= True, label="Mean Variance : %.2f"%(var_mean))

    ax = sns.kdeplot(median_ser, color="Green", shade = True, label="Median Variance : %.2f"%(var_median))

    ax.set_xlabel(var)

    ax.set_ylabel("Frequency")

    ax.legend(loc="best")

    ax.set_title('Frequency Distribution of {}'.format(var), fontsize = 15)
for col in df.columns[:-1]:

    check_impute_variance(df[col], col)
g = sns.distplot(df["Time"].dropna(), color="m", label="Skewness : %.2f"%(df["Time"].skew()))

g = g.legend(loc="best")
def impute_na_num(ser,var):

    mean = ser.mean()

    median = ser.median()

    

    mean_ser = ser.fillna(mean)

    median_ser = ser.fillna(median)

    

    var_original = ser.std()**2

    var_mean = mean_ser.std()**2

    var_median = median_ser.std()**2

    

    if((var_mean < var_original) | (var_median < var_original)):

        if(var_mean < var_median):

            return mean_ser

        else:

            return median_ser

    else:

        return median_ser
na_cols = df.isnull().sum()[df.isnull().sum().values > 0].index
for col in na_cols:

    df[col] = impute_na_num(df[col],col)
df.isnull().sum()
for col in df.columns[:-1]:

    fig = plt.figure(figsize = (12,5))

    ax = fig.add_subplot(111)

    ax = sns.kdeplot(df[col][(df["Class"] == 0)], color="Red", shade = True)

    ax = sns.kdeplot(df[col][(df["Class"] == 1)], color="Blue", shade= True)

    ax.set_xlabel(col)

    ax.set_ylabel("Frequency")

    ax.legend(df.Class.unique(),loc="best")

    ax.set_title('Frequency Distribution of {}'.format(col), fontsize = 15)
X = df.drop('Class', axis=1)

y = df['Class']



sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)



for train_index, test_index in sss.split(X, y):

    print("Train:", train_index, "Test:", test_index)

    original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]

    original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]



# Turn into an array

original_Xtrain = original_Xtrain.values

original_Xtest = original_Xtest.values

original_ytrain = original_ytrain.values

original_ytest = original_ytest.values



# See if both the train and test label distribution are similarly distributed

train_unique_label, train_counts_label = np.unique(original_ytrain, return_counts=True)

test_unique_label, test_counts_label = np.unique(original_ytest, return_counts=True)

print('-' * 100)



print('Label Distributions: \n')

print(train_counts_label/ len(original_ytrain))

print(test_counts_label/ len(original_ytest))
rus = RandomUnderSampler()

X_rus, y_rus = rus.fit_resample(original_Xtrain,original_ytrain)



new_df = pd.concat([pd.DataFrame(X_rus, columns=df.columns[:-1]),

                    pd.DataFrame(y_rus, columns=["Class"])],

                   axis=1)



# Shuffle dataframe rows

new_df = new_df.sample(frac=1, random_state=RANDOM_SEED)



new_df.head()
print('Distribution of the Classes in the subsample dataset')

print(new_df['Class'].value_counts()/len(new_df))







sns.countplot('Class', data=new_df, palette=colors)

plt.title('Equally Distributed Classes', fontsize=14)

plt.show()
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(24,20))



# Entire DataFrame

corr = df.corr()

sns.heatmap(corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax1)

ax1.set_title("Imbalanced Correlation Matrix \n (won't use for reference)", fontsize=14)





sub_sample_corr = new_df.corr()

sns.heatmap(sub_sample_corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax2)

ax2.set_title('Under Sample Correlation Matrix \n (will use for reference)', fontsize=14)

plt.show()
f, axes = plt.subplots(ncols=4, figsize=(20,4))



# Negative Correlations with our Class (The lower our feature value the more likely it will be a fraud transaction)

sns.boxplot(x="Class", y="V17", data=new_df, palette=colors, ax=axes[0])

axes[0].set_title('V17 vs Class Negative Correlation')



sns.boxplot(x="Class", y="V14", data=new_df, palette=colors, ax=axes[1])

axes[1].set_title('V14 vs Class Negative Correlation')





sns.boxplot(x="Class", y="V12", data=new_df, palette=colors, ax=axes[2])

axes[2].set_title('V12 vs Class Negative Correlation')





sns.boxplot(x="Class", y="V10", data=new_df, palette=colors, ax=axes[3])

axes[3].set_title('V10 vs Class Negative Correlation')



plt.show()
f, axes = plt.subplots(ncols=4, figsize=(20,4))



# Positive correlations (The higher the feature the probability increases that it will be a fraud transaction)

sns.boxplot(x="Class", y="V11", data=new_df, palette=colors, ax=axes[0])

axes[0].set_title('V11 vs Class Positive Correlation')



sns.boxplot(x="Class", y="V4", data=new_df, palette=colors, ax=axes[1])

axes[1].set_title('V4 vs Class Positive Correlation')





sns.boxplot(x="Class", y="V2", data=new_df, palette=colors, ax=axes[2])

axes[2].set_title('V2 vs Class Positive Correlation')





sns.boxplot(x="Class", y="V19", data=new_df, palette=colors, ax=axes[3])

axes[3].set_title('V19 vs Class Positive Correlation')



plt.show()
us_outlier_cols = ['V19','V17','V16','V14','V12','V11','V10','V4','V2']
from scipy.stats import norm



for col in us_outlier_cols:

    fig = plt.figure(figsize = (10,5))

    ax = fig.add_subplot(111)

    anoms = new_df[col].loc[new_df['Class'] == 1].values

    sns.distplot(anoms,ax=ax, fit=norm, color='#FB8861')

    ax.set_title('{} Distribution \n (for Anamalies)'.format(col), fontsize=14)

plt.show()
def handle_outliers(df,var,target,target_val,tol):

    anoms = df[var].loc[df[target] == target_val].values

    q25, q75 = np.percentile(anoms, 25), np.percentile(anoms, 75)

    print('Outliers handling for: {}'.format(var))

    print('Quartile 25: {} | Quartile 75: {}'.format(q25, q75))

    iqr = q75 - q25

    print('IQR {}'.format(iqr))

    

    cut_off = iqr * tol

    lower, upper = q25 - cut_off, q75 + cut_off

    print('Cut Off: {}'.format(cut_off))

    print('{} Lower: {}'.format(var,lower))

    print('{} Upper: {}'.format(var,upper))

    

    outliers = [x for x in anoms if x < lower or x > upper]



    print('Number of Outliers in feature {} for Anomalies: {}'.format(var,len(outliers)))



    print('{} outliers:{}'.format(var,outliers))





    df = df.drop(df[(df[var] > upper) | (df[var] < lower)].index)



    print('----' * 25)

    print('\n')

    print('\n')
for col in us_outlier_cols:

    handle_outliers(new_df,col,'Class',1,1.5)
f,(ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,6))



colors = ['#B3F9C5', '#f9c5b3']

# Boxplots with outliers removed

# Feature V14

sns.boxplot(x="Class", y="V14", data=new_df,ax=ax1, palette=colors)

ax1.set_title("V14 Feature \n Reduction of outliers", fontsize=14)

ax1.annotate('Fewer extreme \n outliers', xy=(0.98, -17.5), xytext=(0, -12),

            arrowprops=dict(facecolor='black'),

            fontsize=14)



# Feature 12

sns.boxplot(x="Class", y="V12", data=new_df, ax=ax2, palette=colors)

ax2.set_title("V12 Feature \n Reduction of outliers", fontsize=14)

ax2.annotate('Fewer extreme \n outliers', xy=(0.98, -17.3), xytext=(0, -12),

            arrowprops=dict(facecolor='black'),

            fontsize=14)



# Feature V10

sns.boxplot(x="Class", y="V10", data=new_df, ax=ax3, palette=colors)

ax3.set_title("V10 Feature \n Reduction of outliers", fontsize=14)

ax3.annotate('Fewer extreme \n outliers', xy=(0.95, -16.5), xytext=(0, -12),

            arrowprops=dict(facecolor='black'),

            fontsize=14)





plt.show()
# New_df is from the random undersample data (fewer instances)

X = new_df.drop('Class', axis=1)

y = new_df['Class']





# T-SNE Implementation

t0 = time.time()

X_reduced_tsne = TSNE(n_components=2, random_state=42).fit_transform(X.values)

t1 = time.time()

print("T-SNE took {:.2} s".format(t1 - t0))



# PCA Implementation

t0 = time.time()

X_reduced_pca = PCA(n_components=2, random_state=42).fit_transform(X.values)

t1 = time.time()

print("PCA took {:.2} s".format(t1 - t0))



# TruncatedSVD

t0 = time.time()

X_reduced_svd = TruncatedSVD(n_components=2, algorithm='randomized', random_state=42).fit_transform(X.values)

t1 = time.time()

print("Truncated SVD took {:.2} s".format(t1 - t0))
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24,6))

# labels = ['No Fraud', 'Fraud']

f.suptitle('Clusters using Dimensionality Reduction', fontsize=14)





blue_patch = mpatches.Patch(color='#0A0AFF', label='No Fraud')

red_patch = mpatches.Patch(color='#AF0000', label='Fraud')





# t-SNE scatter plot

ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)

ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)

ax1.set_title('t-SNE', fontsize=14)



ax1.grid(True)



ax1.legend(handles=[blue_patch, red_patch])





# PCA scatter plot

ax2.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)

ax2.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)

ax2.set_title('PCA', fontsize=14)



ax2.grid(True)



ax2.legend(handles=[blue_patch, red_patch])



# TruncatedSVD scatter plot

ax3.scatter(X_reduced_svd[:,0], X_reduced_svd[:,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)

ax3.scatter(X_reduced_svd[:,0], X_reduced_svd[:,1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)

ax3.set_title('Truncated SVD', fontsize=14)



ax3.grid(True)



ax3.legend(handles=[blue_patch, red_patch])



plt.show()
# Undersampling before cross validating (prone to overfit)

X = new_df.drop('Class', axis=1)

y = new_df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)



X_train = X_train.values

X_test = X_test.values

y_train = y_train.values

y_test = y_test.values
classifiers = {

    "LogisiticRegression": LogisticRegression(),

    "KNearest": KNeighborsClassifier(),

    "DecisionTreeClassifier": DecisionTreeClassifier(),

    "Random Forest Classifier": RandomForestClassifier(),

    "AdaBoost Classifier": AdaBoostClassifier()

}
for key, classifier in classifiers.items():

    classifier.fit(X_train, y_train)

    training_score = cross_val_score(classifier, X_train, y_train, cv=5)

    print("Classifiers: ", classifier.__class__.__name__, "Has a training score of", round(training_score.mean(), 2) * 100, "% accuracy score")
# Logistic Regression Parameters

log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params, n_jobs=-1)



t0 = time.time()

grid_log_reg.fit(X_train, y_train)



t1 = time.time()

print("Grid Search took {:.2} s for Logistic Regression".format(t1 - t0))



log_reg = grid_log_reg.best_estimator_



grid_log_reg.best_params_
# kNeighbours Classifier Parameters

knn_params = {"n_neighbors": list(range(2,15,1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}

grid_knn = GridSearchCV(KNeighborsClassifier(), knn_params, n_jobs=-1)



t0 = time.time()

grid_knn.fit(X_train, y_train)



t1 = time.time()

print("Grid Search took {:.2} s for kNN".format(t1 - t0))



knn = grid_knn.best_estimator_



grid_knn.best_params_
# DecisionTree Classifier Parameters

tree_params = {"criterion": ["gini", "entropy"], 

               "max_depth": list(range(2,6,1)),

               "min_samples_leaf": list(range(5,7,1)),

               'max_features': ['auto', 'sqrt', 'log2']}

grid_tree = GridSearchCV(DecisionTreeClassifier(), tree_params, n_jobs=-1)



t0 = time.time()

grid_tree.fit(X_train, y_train)



t1 = time.time()

print("Grid Search took {:.2} s for Decision Tree".format(t1 - t0))



tree_clf = grid_tree.best_estimator_



grid_tree.best_params_
# Random Forest Classifier Parameters

rf_params = {'n_estimators' : [50,100,150,200],

             'criterion' : ["gini","entropy"],

             'max_features': ['auto', 'sqrt', 'log2'],

             'class_weight' : ["balanced", "balanced_subsample"]}

grid_rf = GridSearchCV(RandomForestClassifier(), rf_params, n_jobs=-1)



t0 = time.time()

grid_rf.fit(X_train, y_train)



t1 = time.time()

print("Grid Search took {:.2} s for Random Forest".format(t1 - t0))



rf_clf = grid_rf.best_estimator_



grid_rf.best_params_
# AdaBoost Classifier Parameters

adb_params = {'n_estimators' : [25,50,75,100],

              'learning_rate' : [0.001,0.01,0.05,0.1,1,10],

              'algorithm' : ['SAMME', 'SAMME.R']}

grid_adb = GridSearchCV(AdaBoostClassifier(), adb_params, n_jobs=-1)



t0 = time.time()

grid_adb.fit(X_train, y_train)



t1 = time.time()

print("Grid Search took {:.2} s for AdaBoost".format(t1 - t0))



adb_clf = grid_adb.best_estimator_



grid_adb.best_params_
estimators = [log_reg,knn,tree_clf,rf_clf,adb_clf]
# Check for Overfitting Case



for est in estimators:

    est_score = cross_val_score(est, X_train, y_train, cv=10, n_jobs=-1)

    print('{} Cross Validation Score: '.format(type(est).__name__), round(est_score.mean() * 100, 2).astype(str) + '%')
# Implementing NearMiss Technique 

# Distribution of NearMiss

X_nearmiss, y_nearmiss = NearMiss().fit_sample(original_Xtrain, original_ytrain)

print('NearMiss Label Distribution: {}'.format(Counter(y_nearmiss)))
us_clf_results = {}

us_clf_results['Time'] = {}

us_clf_results['Accuracy'] = {}

us_clf_results['Precision'] = {}

us_clf_results['Recall'] = {}

us_clf_results['F1_score'] = {}

us_clf_results['Auc_Roc'] = {}



sss = StratifiedKFold(n_splits=5, random_state=RANDOM_SEED, shuffle=False)



for key, classifier in classifiers.items():

    

    # We will undersample during cross validating

    undersample_accuracy = []

    undersample_precision = []

    undersample_recall = []

    undersample_f1 = []

    undersample_auc = []

    

    cv_time = []

    idx=1

    

    # Cross Validating the right way

    for train, test in sss.split(original_Xtrain, original_ytrain):

        print("Cross Validation Split - {} for {}".format(idx,key))

        undersample_pipeline = imbalanced_make_pipeline(NearMiss(sampling_strategy='majority'), classifier)

        

        start_time = time.time()

        undersample_model = undersample_pipeline.fit(original_Xtrain[train], original_ytrain[train])

        elapsed_time = time.time() - start_time

        

        cv_time.append(elapsed_time)

        undersample_prediction = undersample_model.predict(original_Xtest)

    

        undersample_accuracy.append(undersample_pipeline.score(original_Xtest, original_ytest))

        undersample_precision.append(precision_score(original_ytest, undersample_prediction))

        undersample_recall.append(recall_score(original_ytest, undersample_prediction))

        undersample_f1.append(f1_score(original_ytest, undersample_prediction))

        undersample_auc.append(roc_auc_score(original_ytest, undersample_prediction))

        

        idx = idx + 1

        print("---"*40)

        print("\n")

    

    us_clf_results['Time'][key] = cv_time

    us_clf_results['Accuracy'][key] = undersample_accuracy

    us_clf_results['Precision'][key] = undersample_precision

    us_clf_results['Recall'][key] = undersample_recall

    us_clf_results['F1_score'][key] = undersample_f1

    us_clf_results['Auc_Roc'][key] = undersample_auc
for key, us_res in us_clf_results.items():

    df_results = (pd.DataFrame(us_res).unstack().reset_index())

    plt.figure()

    sns.boxplot(y='level_0', x=0, data=df_results)

    sns.despine(top=True, right=True, left=True)

    plt.xlabel(key)

    plt.ylabel('')

    plt.title('Results for {} from cross-validation with under-sampling across multiple models'.format(key))
def plot_learning_curve(estimators, X, y, ylim=None, cv=None,

                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):

        

    for estimator in estimators:

        fig = plt.figure(figsize = (15,5))

        ax = fig.add_subplot(111)

        if ylim is not None:

            plt.ylim(*ylim)

        

        train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, 

                                                                cv=cv, n_jobs=n_jobs, 

                                                                train_sizes=train_sizes)

        train_scores_mean = np.mean(train_scores, axis=1)

        train_scores_std = np.std(train_scores, axis=1)

        test_scores_mean = np.mean(test_scores, axis=1)

        test_scores_std = np.std(test_scores, axis=1)

        ax1.fill_between(train_sizes, 

                         train_scores_mean - train_scores_std,

                         train_scores_mean + train_scores_std, 

                         alpha=0.1,

                         color="#ff9124")

        ax1.fill_between(train_sizes, 

                         test_scores_mean - test_scores_std,

                         test_scores_mean + test_scores_std, 

                         alpha=0.1, 

                         color="#2492ff")

        ax.plot(train_sizes, 

                train_scores_mean, 

                'o-', 

                color="#ff9124",

                label="Training score")

        ax.plot(train_sizes, 

                test_scores_mean, 

                'o-', 

                color="#2492ff",

                label="Cross-validation score")

        ax.set_title("{} Learning Curve".format(type(estimator).__name__), fontsize=14)

        ax.set_xlabel('Training size')

        ax.set_ylabel('Score')

        ax.grid(True)

        ax.legend(loc="best")

    plt.show()
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=RANDOM_SEED)

plot_learning_curve(estimators, X_train, y_train, (0.87, 1.01), cv=cv, n_jobs=-1)
# Create a DataFrame with all the scores and the classifiers names.



log_reg_pred = cross_val_predict(log_reg, X_train, y_train, cv=5, n_jobs=-1,

                             method="decision_function")



knn_pred = cross_val_predict(knn, X_train, y_train, cv=5, n_jobs=-1)



tree_pred = cross_val_predict(tree_clf, X_train, y_train, cv=5, n_jobs=-1)



rf_pred = cross_val_predict(rf_clf, X_train, y_train, cv=5, n_jobs=-1)



adb_pred = cross_val_predict(rf_clf, X_train, y_train, cv=5, n_jobs=-1)
from sklearn.metrics import roc_auc_score



print('Logistic Regression: ', roc_auc_score(y_train, log_reg_pred))

print('KNN: ', roc_auc_score(y_train, knn_pred))

print('Decision Tree Classifier: ', roc_auc_score(y_train, tree_pred))

print('Random Forest Classifier: ', roc_auc_score(y_train, rf_pred))

print('AdaBoost Classifier: ', roc_auc_score(y_train, adb_pred))
log_fpr, log_tpr, log_thresold = roc_curve(y_train, log_reg_pred)

knn_fpr, knn_tpr, knn_threshold = roc_curve(y_train, knn_pred)

tree_fpr, tree_tpr, tree_threshold = roc_curve(y_train, tree_pred)

rf_fpr, rf_tpr, rf_threshold = roc_curve(y_train, rf_pred)

adb_fpr, adb_tpr, adb_threshold = roc_curve(y_train, adb_pred)





def graph_roc_curve_multiple(log_fpr, log_tpr, 

                             knn_fpr, knn_tpr,

                             tree_fpr, tree_tpr,

                             rf_fpr, rf_tpr,

                             adb_fpr, adb_tpr):

    plt.figure(figsize=(15,5))

    plt.title('ROC Curve \n Classifiers (Under-Sampling)', fontsize=18)

    plt.plot(log_fpr, log_tpr, label='Logistic Regression Classifier Score: {:.4f}'.format(roc_auc_score(y_train, log_reg_pred)))

    plt.plot(knn_fpr, knn_tpr, label='KNN Classifier Score: {:.4f}'.format(roc_auc_score(y_train, knn_pred)))

    plt.plot(tree_fpr, tree_tpr, label='Decision Tree Classifier Score: {:.4f}'.format(roc_auc_score(y_train, tree_pred)))

    plt.plot(rf_fpr, rf_tpr, label='Random Forest Classifier Score: {:.4f}'.format(roc_auc_score(y_train, rf_pred)))

    plt.plot([0, 1], [0, 1], 'k--')

    plt.axis([-0.01, 1, 0, 1])

    plt.xlabel('False Positive Rate', fontsize=16)

    plt.ylabel('True Positive Rate', fontsize=16)

    plt.annotate('Minimum ROC Score of 50% \n (This is the minimum score to get)', xy=(0.5, 0.5), xytext=(0.6, 0.3),

                arrowprops=dict(facecolor='#6E726D', shrink=0.05),

                )

    plt.legend(loc='best')

    

graph_roc_curve_multiple(log_fpr, log_tpr, 

                             knn_fpr, knn_tpr, 

                             tree_fpr, tree_tpr,

                             rf_fpr, rf_tpr,

                             adb_fpr, adb_tpr)

plt.show()
os_clf_results = {}

os_clf_results['Time'] = {}

os_clf_results['Accuracy'] = {}

os_clf_results['Precision'] = {}

os_clf_results['Recall'] = {}

os_clf_results['F1_score'] = {}

os_clf_results['Auc_Roc'] = {}



sss = StratifiedKFold(n_splits=5, random_state=RANDOM_SEED, shuffle=False)



for key, classifier in classifiers.items():

    

    # We will undersample during cross validating

    oversample_accuracy = []

    oversample_precision = []

    oversample_recall = []

    oversample_f1 = []

    oversample_auc = []

    

    cv_time = []

    idx=1

    

    # Cross Validating the right way

    for train, test in sss.split(original_Xtrain, original_ytrain):

        print("Cross Validation Split - {} for {}".format(idx,key))

        pipeline = imbalanced_make_pipeline(SMOTE(sampling_strategy='minority',random_state=RANDOM_SEED), 

                                            classifier)

        

        start_time = time.time()

        model = pipeline.fit(original_Xtrain[train], original_ytrain[train])

        elapsed_time = time.time() - start_time

        

        cv_time.append(elapsed_time)

        oversample_prediction = model.predict(original_Xtest)

    

        oversample_accuracy.append(pipeline.score(original_Xtest, original_ytest))

        oversample_precision.append(precision_score(original_ytest, oversample_prediction))

        oversample_recall.append(recall_score(original_ytest, oversample_prediction))

        oversample_f1.append(f1_score(original_ytest, oversample_prediction))

        oversample_auc.append(roc_auc_score(original_ytest, oversample_prediction))

        

        idx = idx + 1

        print("---"*40)

        print("\n")

    

    os_clf_results['Time'][key] = cv_time

    os_clf_results['Accuracy'][key] = oversample_accuracy

    os_clf_results['Precision'][key] = oversample_precision

    os_clf_results['Recall'][key] = oversample_recall

    os_clf_results['F1_score'][key] = oversample_f1

    os_clf_results['Auc_Roc'][key] = oversample_auc
for key, os_res in os_clf_results.items():

    df_results = (pd.DataFrame(os_res).unstack().reset_index())

    plt.figure()

    sns.boxplot(y='level_0', x=0, data=df_results)

    sns.despine(top=True, right=True, left=True)

    plt.xlabel(key)

    plt.ylabel('')

    plt.title('Results for {} from cross-validation with over-sampling across multiple models'.format(key))
import keras

from keras import backend as K

from keras.models import Sequential

from keras.layers import Activation, Dense, Dropout, BatchNormalization

from keras.optimizers import Adam

from keras.metrics import categorical_crossentropy
from imblearn.keras import BalancedBatchGenerator

training_generator = BalancedBatchGenerator(X, y, sampler=NearMiss(), batch_size=10, random_state=(0))
def make_model(n_features):

    model = Sequential()

    model.add(Dense(200, input_shape=(n_features,),

              kernel_initializer='glorot_normal'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Dropout(0.5))

    model.add(Dense(100, kernel_initializer='glorot_normal', use_bias=False))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Dropout(0.25))

    model.add(Dense(50, kernel_initializer='glorot_normal', use_bias=False))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Dropout(0.15))

    model.add(Dense(25, kernel_initializer='glorot_normal', use_bias=False))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Dropout(0.1))

    model.add(Dense(1, activation='sigmoid'))



    model.compile(Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])



    return model
import time

from functools import wraps





def timeit(f):

    @wraps(f)

    def wrapper(*args, **kwds):

        start_time = time.time()

        result = f(*args, **kwds)

        elapsed_time = time.time() - start_time

        print('Elapsed computation time: {:.3f} secs'

              .format(elapsed_time))

        return (elapsed_time, result)

    return wrapper
@timeit

def fit_predict_imbalanced_model(X_train, y_train, X_test, y_test):

    model = make_model(X_train.shape[1])

    model.fit(X_train, y_train, epochs=2, verbose=1, batch_size=100)

    y_pred = model.predict_proba(X_test, batch_size=100)

    return roc_auc_score(y_test, y_pred)
@timeit

def fit_predict_balanced_us_model(X_train, y_train, X_test, y_test):

    model = make_model(X_train.shape[1])

    training_generator = BalancedBatchGenerator(X_train, y_train,

                                                batch_size=100,

                                                random_state=RANDOM_SEED)

    model.fit_generator(generator=training_generator, epochs=10, verbose=1)

    y_pred = model.predict_proba(X_test, batch_size=100)

    return roc_auc_score(y_test, y_pred)
@timeit

def fit_predict_balanced_os_model(X_train, y_train, X_test, y_test):

    model = make_model(X_train.shape[1])

    training_generator = BalancedBatchGenerator(X_train, y_train,

                                                sampler=RandomOverSampler(),

                                                batch_size=1000,

                                                random_state=RANDOM_SEED)

    model.fit_generator(generator=training_generator, epochs=5, verbose=1)

    y_pred = model.predict_proba(X_test, batch_size=1000)

    return roc_auc_score(y_test, y_pred)
skf = StratifiedKFold(n_splits=5)



cv_results_imbalanced = []

cv_time_imbalanced = []

cv_results_balanced_us = []

cv_time_balanced_us = []

cv_results_balanced_os = []

cv_time_balanced_os = []

idx=1



for train_idx, valid_idx in skf.split(original_Xtrain, original_ytrain):

    print("Cross Validation Split: {}".format(idx))

    print("---"*40)

    

    X_local_train = original_Xtrain[train_idx]

    y_local_train = original_ytrain[train_idx]

    X_local_test = original_Xtrain[valid_idx]

    y_local_test = original_ytrain[valid_idx]



    elapsed_time, roc_auc = fit_predict_imbalanced_model(

        X_local_train, y_local_train, original_Xtest, original_ytest)

    cv_time_imbalanced.append(elapsed_time)

    cv_results_imbalanced.append(roc_auc)



    elapsed_time, roc_auc = fit_predict_balanced_us_model(

        X_local_train, y_local_train, original_Xtest, original_ytest)

    cv_time_balanced_us.append(elapsed_time)

    cv_results_balanced_us.append(roc_auc)

    

    elapsed_time, roc_auc = fit_predict_balanced_os_model(

        X_local_train, y_local_train, original_Xtest, original_ytest)

    cv_time_balanced_os.append(elapsed_time)

    cv_results_balanced_os.append(roc_auc)

    

    idx = idx + 1

    print("---"*40)

    print("\n")
df_results = (pd.DataFrame({'Balanced model Under-Sampled': cv_results_balanced_us,

                            'Balanced model Over-Sampled': cv_results_balanced_os,

                            'Imbalanced model': cv_results_imbalanced})

              .unstack().reset_index())

df_time = (pd.DataFrame({'Balanced model Under-Sampled': cv_time_balanced_us,

                         'Balanced model Over-Sampled': cv_time_balanced_os,

                         'Imbalanced model': cv_time_imbalanced})

           .unstack().reset_index())



plt.figure()

sns.boxplot(y='level_0', x=0, data=df_time)

sns.despine(top=True, right=True, left=True)

plt.xlabel('time [s]')

plt.ylabel('')

plt.title('Computation time difference using a random under-sampling')



plt.figure()

sns.boxplot(y='level_0', x=0, data=df_results, whis=10.0)

sns.despine(top=True, right=True, left=True)

ax = plt.gca()

ax.xaxis.set_major_formatter(

    plt.FuncFormatter(lambda x, pos: "%i%%" % (100 * x)))

plt.xlabel('ROC-AUC')

plt.ylabel('')

plt.title('Difference in terms of ROC-AUC using a random under-sampling')