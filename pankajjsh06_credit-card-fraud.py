# Imported Libraries



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.manifold import TSNE

from sklearn.decomposition import PCA, TruncatedSVD

import matplotlib.patches as mpatches

import time



df = pd.read_csv('../input/creditcard.csv')

df.head()
df.describe()
%time

print('No fraud',round(df['Class'].value_counts()[0]/len(df['Class'])*100,2),'% of dataset' )

print('Fraud', round(df['Class'].value_counts()[1]/len(df)*100,2),'% of dataset')
sns.countplot(df.Class)

plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=14)
fig, ax = plt.subplots(1,2, figsize=(18,4))

sns.distplot(df['Amount'], ax = ax[0], color='r')

ax[0].set_title('Distribution of Transaction Amount', fontsize=14)

ax[0].set_xlim([min(df.Amount), max(df.Amount)])



sns.distplot(df['Time'], ax = ax[1], color='b')

ax[1].set_title('Distribution of Transaction Time', fontsize=14)

ax[1].set_xlim([min(df.Time), max(df.Time)])

plt.show()
from sklearn.preprocessing import RobustScaler, StandardScaler



std_scaler = StandardScaler()

rob_scaler = RobustScaler()



df['Scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))

df['Scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))



df.drop(['Amount','Time'], axis=1, inplace=True)
df.head()
scaled_amount = df.Scaled_amount

scaled_time = df.Scaled_time



df.drop(['Scaled_amount','Scaled_time'], axis=1, inplace=True)

df.insert(0, 'scaled_amount', scaled_amount)

df.insert(1, 'scaled_time', scaled_time)



df.head()
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold

X = df.drop(['Class'], axis=1)

y = df.Class



sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

for train_index, test_index in sss.split(X,y):

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
# Since our classes are highly skewed we should make them equivalent in order to have a normal distribution of the classes.



# Lets shuffle the data before creating the subsamples



df = df.sample(frac=1)



fraud_df = df.loc[df['Class']==1]

non_fraud_df =df.loc[df['Class']==0][:492]



normal_distributed_df = pd.concat([fraud_df, non_fraud_df])

# Shuffle dataframe rows

new_df = normal_distributed_df.sample(frac=1, random_state=42)

sns.countplot(new_df.Class)
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(24,20))



# Entire DataFrame

corr = df.corr()

sns.heatmap(corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax1)

ax1.set_title("Imbalanced Correlation Matrix \n (don't use for reference)", fontsize=14)





sub_sample_corr = new_df.corr()

sns.heatmap(sub_sample_corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax2)

ax2.set_title('SubSample Correlation Matrix \n (use for reference)', fontsize=14)

plt.show()
# Negative Correlations with our Class (The lower our feature value the more likely it will be a fraud transaction)

columns=['V17','V14','V12','V10']

i,axis=plt.subplots(ncols=4, figsize=(20,4))

for i, col in enumerate(new_df[columns]):

    plot=sns.boxplot(x=new_df.Class, y=new_df[col], ax=axis[i])

    axis[i].set_title(str(col+' vs Class Negative Correlation'))

plt.show()

i, axes = plt.subplots(ncols=4, figsize=(20,4))

columns = ['V11','V4','V2','V19']

for i,col in enumerate(new_df[columns]):

    plot=sns.boxplot(x=new_df.Class, y=new_df[col], ax=axes[i])

    axes[i].set_title(str(col+' vs Class Positive Correlation'))

plt.show()
from scipy.stats import norm



f, axes = plt.subplots(ncols=4, figsize=(20,4))

columns = ['V14','V12','V10','V19']

for i,col in enumerate(new_df[columns]):

        plot=sns.distplot(new_df[col].loc[new_df.Class==1].values,fit=norm, ax=axes[i])

        axes[i].set_title(str(col+' Distribution \n (Fraud Tranaction)'))

plt.show()
# # -----> V14 Removing Outliers (Highest Negative Correlated with Labels)

v14_fraud = new_df['V14'].loc[new_df['Class'] == 1].values

q25, q75 = np.percentile(v14_fraud, 25), np.percentile(v14_fraud, 75)

print('Quartile 25: {} | Quartile 75: {}'.format(q25, q75))

v14_iqr = q75 - q25

print('iqr: {}'.format(v14_iqr))



v14_cut_off = v14_iqr * 1.5

v14_lower, v14_upper = q25 - v14_cut_off, q75 + v14_cut_off

print('Cut Off: {}'.format(v14_cut_off))

print('V14 Lower: {}'.format(v14_lower))

print('V14 Upper: {}'.format(v14_upper))



outliers = [x for x in v14_fraud if x < v14_lower or x > v14_upper]

print('Feature V14 Outliers for Fraud Cases: {}'.format(len(outliers)))

print('V10 outliers:{}'.format(outliers))



new_df = new_df.drop(new_df[(new_df['V14'] > v14_upper) | (new_df['V14'] < v14_lower)].index)

print('----' * 44)



# -----> V12 removing outliers from fraud transactions

v12_fraud = new_df['V12'].loc[new_df['Class'] == 1].values

q25, q75 = np.percentile(v12_fraud, 25), np.percentile(v12_fraud, 75)

v12_iqr = q75 - q25



v12_cut_off = v12_iqr * 1.5

v12_lower, v12_upper = q25 - v12_cut_off, q75 + v12_cut_off

print('V12 Lower: {}'.format(v12_lower))

print('V12 Upper: {}'.format(v12_upper))

outliers = [x for x in v12_fraud if x < v12_lower or x > v12_upper]

print('V12 outliers: {}'.format(outliers))

print('Feature V12 Outliers for Fraud Cases: {}'.format(len(outliers)))

new_df = new_df.drop(new_df[(new_df['V12'] > v12_upper) | (new_df['V12'] < v12_lower)].index)

print('Number of Instances after outliers removal: {}'.format(len(new_df)))

print('----' * 44)





# Removing outliers V10 Feature

v10_fraud = new_df['V10'].loc[new_df['Class'] == 1].values

q25, q75 = np.percentile(v10_fraud, 25), np.percentile(v10_fraud, 75)

v10_iqr = q75 - q25



v10_cut_off = v10_iqr * 1.5

v10_lower, v10_upper = q25 - v10_cut_off, q75 + v10_cut_off

print('V10 Lower: {}'.format(v10_lower))

print('V10 Upper: {}'.format(v10_upper))

outliers = [x for x in v10_fraud if x < v10_lower or x > v10_upper]

print('V10 outliers: {}'.format(outliers))

print('Feature V10 Outliers for Fraud Cases: {}'.format(len(outliers)))

new_df = new_df.drop(new_df[(new_df['V10'] > v10_upper) | (new_df['V10'] < v10_lower)].index)

print('Number of Instances after outliers removal: {}'.format(len(new_df)))
i, axes = plt.subplots(ncols=3, figsize=(20,6))

columns = ['V14','V12','V10']

for i,col in enumerate(new_df[columns]):

    plot=sns.boxplot(x=new_df.Class, y=new_df[col], ax=axes[i])

    axes[i].set_title(str(col+' Feature \nReduction of outliers'))

    axes[i].annotate('Fewer extreme \n outliers', xy=(0.98, -17.5), xytext=(0, -12),

                     arrowprops=dict(facecolor='black'),fontsize=14)

plt.show()
from sklearn.manifold import t_sne

from sklearn.decomposition import PCA, truncated_svd



X = new_df.drop('Class', axis=1)

y = new_df.Class



# T-SNE Implementation

t0 = time.time()

X_reduced_tsne = TSNE(n_components=2, random_state=42).fit_transform(X.values)

t1 = time.time()

print("T-SNE took {:.2} s".format(t1 - t0))



# PCA Implementation

t0 = time.time()

X_reduced_PCA = PCA(n_components=2, random_state=42).fit_transform(X.values)

t1 = time.time()

print('PCA took {:.2} s'.format(t1-t0))



#truncated_svd implementation

t0 = time.time()

X_reduced_trucatedSVD = TruncatedSVD(n_components=2, random_state=42).fit_transform(X.values)

t1 = time.time()

print("TruncatedSVD took {:.2} s".format(t1 - t0))
f, ax = plt.subplots(1, 3, figsize=(24,6))

f.suptitle('Clusters using Dimensionality Reduction', fontsize=14)





blue_patch = mpatches.Patch(color='#0A0AFF', label='No Fraud')

red_patch = mpatches.Patch(color='#AF0000', label='Fraud')



models = [X_reduced_tsne,X_reduced_PCA, X_reduced_trucatedSVD]



for i, model in enumerate(models):

    ax[i].scatter(model[:,0], model[:,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)

    ax[i].scatter(model[:,0], model[:,1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)

    ax[i].set_title('Models', fontsize=14)



    ax[i].grid(True)



    ax[i].legend(handles=[blue_patch, red_patch])
# Undersampling before cross validating (prone to overfit)

X=new_df.drop('Class', axis=1)

y=new_df.Class
# Our data is already scaled we should split our training and test sets

from sklearn.model_selection import train_test_split



# This is explicitly used for undersampling.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Turn the values into an array for feeding the classification algorithms.

X_train=X_train.values

X_test=X_test.values

y_train=y_train.values

y_test=y_test.values
# Let's implement simple classifiers

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier



classifiers={

    "LogisticRegression":LogisticRegression(),

    "Support Vector classifier":SVC(),

    "kNeighbor":KNeighborsClassifier(),

    "DecisionTreeClassifier":DecisionTreeClassifier()

}
from sklearn.model_selection import cross_val_score



for key, classifier in classifiers.items():

    classifier.fit(X_train,y_train)

    training_score=cross_val_score(classifier,X_train,y_train)

    print("Classifiers: ",classifier.__class__.__name__,"has a training score of ", 

          round(training_score.mean(),2)*100," % of accuracy")
# Use GridSearchCV to find the best parameters.

from sklearn.model_selection import GridSearchCV



#LogisticRegression

log_reg_param={"penalty":['l1', 'l2'],'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000]}

grid_log_reg=GridSearchCV(LogisticRegression(), log_reg_param)

grid_log_reg.fit(X_train,y_train)

# We automatically get the logistic regression with the best parameters.

log_reg = grid_log_reg.best_estimator_



# KNears best Classifier

knears_params = {"n_neighbors": list(range(2,5,1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}

grid_knears = GridSearchCV(KNeighborsClassifier(), knears_params)

grid_knears.fit(X_train, y_train)

# KNears best estimator

knears_neighbors = grid_knears.best_estimator_



# Support Vector Classifier

svc_params = {'C': [0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}

grid_svc = GridSearchCV(SVC(), svc_params)

grid_svc.fit(X_train, y_train)



# SVC best estimator

svc = grid_svc.best_estimator_



# DecisionTree Classifier

tree_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(2,4,1)), 

              "min_samples_leaf": list(range(5,7,1))}

grid_tree = GridSearchCV(DecisionTreeClassifier(), tree_params)

grid_tree.fit(X_train, y_train)



# tree best estimator

tree_clf = grid_tree.best_estimator_
# Overfitting Case

log_reg_score = cross_val_score(log_reg, X_train, y_train, cv=5)

print('Logistic Regression Cross Validation Score: ', round(log_reg_score.mean() * 100, 2).astype(str) + '%')





knears_score = cross_val_score(knears_neighbors, X_train, y_train, cv=5)

print('Knears Neighbors Cross Validation Score', round(knears_score.mean() * 100, 2).astype(str) + '%')



svc_score = cross_val_score(svc, X_train, y_train, cv=5)

print('Support Vector Classifier Cross Validation Score', round(svc_score.mean() * 100, 2).astype(str) + '%')



tree_score = cross_val_score(tree_clf, X_train, y_train, cv=5)

print('DecisionTree Classifier Cross Validation Score', round(tree_score.mean() * 100, 2).astype(str) + '%')
# We will undersample during cross validating

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report

from imblearn.under_sampling import NearMiss

from collections import Counter

from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline



undersample_X = df.drop('Class', axis=1)

undersample_y = df['Class']



for train_index, test_index in sss.split(undersample_X, undersample_y):

    print("Train:", train_index, "Test:", test_index)

    undersample_Xtrain, undersample_Xtest = undersample_X.iloc[train_index], undersample_X.iloc[test_index]

    undersample_ytrain, undersample_ytest = undersample_y.iloc[train_index], undersample_y.iloc[test_index]

    

undersample_Xtrain = undersample_Xtrain.values

undersample_Xtest = undersample_Xtest.values

undersample_ytrain = undersample_ytrain.values

undersample_ytest = undersample_ytest.values 



undersample_accuracy = []

undersample_precision = []

undersample_recall = []

undersample_f1 = []

undersample_auc = []



# Implementing NearMiss Technique 

# Distribution of NearMiss (Just to see how it distributes the labels we won't use these variables)

X_nearmiss, y_nearmiss = NearMiss().fit_sample(undersample_X.values, undersample_y.values)

print('NearMiss Label Distribution: {}'.format(Counter(y_nearmiss)))

# Cross Validating the right way



for train, test in sss.split(undersample_Xtrain, undersample_ytrain):

    undersample_pipeline = imbalanced_make_pipeline(NearMiss(sampling_strategy='majority'), log_reg) # SMOTE happens during Cross Validation not before..

    undersample_model = undersample_pipeline.fit(undersample_Xtrain[train], undersample_ytrain[train])

    undersample_prediction = undersample_model.predict(undersample_Xtrain[test])

    

    undersample_accuracy.append(undersample_pipeline.score(original_Xtrain[test], original_ytrain[test]))

    undersample_precision.append(precision_score(original_ytrain[test], undersample_prediction))

    undersample_recall.append(recall_score(original_ytrain[test], undersample_prediction))

    undersample_f1.append(f1_score(original_ytrain[test], undersample_prediction))

    undersample_auc.append(roc_auc_score(original_ytrain[test], undersample_prediction))
# Let's Plot LogisticRegression Learning Curve



from sklearn.model_selection import ShuffleSplit, learning_curve



def plot_learning_curve(estimator1, estimator2, estimator3, estimator4, X, y, ylim=None, cv=None,

                       n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):

    f, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2,2, figsize=(20,14), sharey=True)

    if ylim is not None:

        plt.ylim(*ylim)

        #first estimator

        train_sizes, train_scores, test_scores = learning_curve(

            estimator1,X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes

        )

        train_scores_mean=np.mean(train_scores, axis=1)

        train_scores_std = np.std(train_scores, axis=1)

        test_scores_mean = np.mean(test_scores, axis=1)

        test_scores_std = np.std(test_scores, axis=1)

        

        ax1.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="#ff9124")

    ax1.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")

    ax1.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",

             label="Training score")

    ax1.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",

             label="Cross-validation score")

    ax1.set_title("Logistic Regression Learning Curve", fontsize=14)

    ax1.set_xlabel('Training size (m)')

    ax1.set_ylabel('Score')

    ax1.grid(True)

    ax1.legend(loc="best")

    

    # Second Estimator 

    train_sizes, train_scores, test_scores = learning_curve(

        estimator2, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    ax2.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="#ff9124")

    ax2.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")

    ax2.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",

             label="Training score")

    ax2.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",

             label="Cross-validation score")

    ax2.set_title("Knears Neighbors Learning Curve", fontsize=14)

    ax2.set_xlabel('Training size (m)')

    ax2.set_ylabel('Score')

    ax2.grid(True)

    ax2.legend(loc="best")

    

    # Third Estimator

    train_sizes, train_scores, test_scores = learning_curve(

        estimator3, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    ax3.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="#ff9124")

    ax3.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")

    ax3.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",

             label="Training score")

    ax3.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",

             label="Cross-validation score")

    ax3.set_title("Support Vector Classifier \n Learning Curve", fontsize=14)

    ax3.set_xlabel('Training size (m)')

    ax3.set_ylabel('Score')

    ax3.grid(True)

    ax3.legend(loc="best")

    

    # Fourth Estimator

    train_sizes, train_scores, test_scores = learning_curve(

        estimator4, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    ax4.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="#ff9124")

    ax4.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")

    ax4.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",

             label="Training score")

    ax4.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",

             label="Cross-validation score")

    ax4.set_title("Decision Tree Classifier \n Learning Curve", fontsize=14)

    ax4.set_xlabel('Training size (m)')

    ax4.set_ylabel('Score')

    ax4.grid(True)

    ax4.legend(loc="best")

    

    return plt
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=42)

plot_learning_curve(log_reg, knears_neighbors, svc, tree_clf, X_train, y_train, (0.87, 1.01),

                    cv=cv, n_jobs=4)
from sklearn.metrics import roc_curve

from sklearn.model_selection import cross_val_predict

# Create a DataFrame with all the scores and the classifiers names.



log_reg_pred = cross_val_predict(log_reg, X_train, y_train, cv=5,

                             method="decision_function")



knears_pred = cross_val_predict(knears_neighbors, X_train, y_train, cv=5)



svc_pred = cross_val_predict(svc, X_train, y_train, cv=5,

                             method="decision_function")



tree_pred = cross_val_predict(tree_clf, X_train, y_train, cv=5)
from sklearn.metrics import roc_auc_score



print('Logistic Regression: ', roc_auc_score(y_train, log_reg_pred))

print('KNears Neighbors: ', roc_auc_score(y_train, knears_pred))

print('Support Vector Classifier: ', roc_auc_score(y_train, svc_pred))

print('Decision Tree Classifier: ', roc_auc_score(y_train, tree_pred))
log_fpr, log_tpr, log_thresold = roc_curve(y_train, log_reg_pred)

knear_fpr, knear_tpr, knear_threshold = roc_curve(y_train, knears_pred)

svc_fpr, svc_tpr, svc_threshold = roc_curve(y_train, svc_pred)

tree_fpr, tree_tpr, tree_threshold = roc_curve(y_train, tree_pred)





def graph_roc_curve_multiple(log_fpr, log_tpr, knear_fpr, knear_tpr, svc_fpr, svc_tpr, tree_fpr, tree_tpr):

    plt.figure(figsize=(16,8))

    plt.title('ROC Curve \n Top 4 Classifiers', fontsize=18)

    plt.plot(log_fpr, log_tpr, label='Logistic Regression Classifier Score: {:.4f}'.format(roc_auc_score(y_train, log_reg_pred)))

    plt.plot(knear_fpr, knear_tpr, label='KNears Neighbors Classifier Score: {:.4f}'.format(roc_auc_score(y_train, knears_pred)))

    plt.plot(svc_fpr, svc_tpr, label='Support Vector Classifier Score: {:.4f}'.format(roc_auc_score(y_train, svc_pred)))

    plt.plot(tree_fpr, tree_tpr, label='Decision Tree Classifier Score: {:.4f}'.format(roc_auc_score(y_train, tree_pred)))

    plt.plot([0, 1], [0, 1], 'k--')

    plt.axis([-0.01, 1, 0, 1])

    plt.xlabel('False Positive Rate', fontsize=16)

    plt.ylabel('True Positive Rate', fontsize=16)

    plt.annotate('Minimum ROC Score of 50% \n (This is the minimum score to get)', xy=(0.5, 0.5), xytext=(0.6, 0.3),

                arrowprops=dict(facecolor='#6E726D', shrink=0.05),

                )

    plt.legend()

    

graph_roc_curve_multiple(log_fpr, log_tpr, knear_fpr, knear_tpr, svc_fpr, svc_tpr, tree_fpr, tree_tpr)

plt.show()

from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

y_pred = log_reg.predict(X_train)



# Overfitting Case

print('---' * 45)

print('Overfitting: \n')

print('Recall Score: {:.2f}'.format(recall_score(y_train, y_pred)))

print('Precision Score: {:.2f}'.format(precision_score(y_train, y_pred)))

print('F1 Score: {:.2f}'.format(f1_score(y_train, y_pred)))

print('Accuracy Score: {:.2f}'.format(accuracy_score(y_train, y_pred)))

print('---' * 45)



# How it should look like

print('---' * 45)

print('How it should be:\n')

print("Accuracy Score: {:.2f}".format(np.mean(undersample_accuracy)))

print("Precision Score: {:.2f}".format(np.mean(undersample_precision)))

print("Recall Score: {:.2f}".format(np.mean(undersample_recall)))

print("F1 Score: {:.2f}".format(np.mean(undersample_f1)))

print('---' * 45)