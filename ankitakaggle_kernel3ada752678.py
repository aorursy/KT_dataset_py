#Import libraries

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import statsmodels.api as sm

import scipy.stats as stats

import plotly

import chart_studio.plotly as py

import plotly.graph_objs as go

plotly.offline.init_notebook_mode(connected=True)

plt.rcParams.update({'figure.max_open_warning': 0})

from plotly import tools

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

# Machine Learning Library

from sklearn.preprocessing import LabelEncoder # Encode Categorical Variable to Numerical Variable

#from sklearn.preprocessing import Imputer # Imputer Class to replace missing values

from sklearn.metrics import confusion_matrix # Library for model evaluation

from sklearn.metrics import accuracy_score # Library for model evaluation

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold # Library to split datset into test and train

from imblearn.under_sampling import NearMiss

from sklearn.model_selection import StratifiedShuffleSplit

import time



from sklearn.linear_model  import LogisticRegression # Logistic Regression Classifier

from sklearn.linear_model import SGDClassifier # Stochastic Gradient Descent Classifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier # Decision Tree Classifier

from sklearn.ensemble  import RandomForestClassifier # Random Forest Classifier

from sklearn.neighbors import KNeighborsClassifier # K Nearest neighbors Classifier

from sklearn.naive_bayes import GaussianNB #Naive Bayes Classifier

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

from sklearn.model_selection import cross_val_score

from sklearn.metrics import precision_recall_curve

from sklearn.metrics import average_precision_score

from imblearn.over_sampling import SMOTE

from collections import Counter

from sklearn import pipeline

from sklearn.pipeline import make_pipeline

from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline



import pandas_profiling

%matplotlib inline

from sklearn.cluster import KMeans

# Supress Warnings



import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv(r"C:\Users\Ankita Rath\Downloads\CAPSTONE\creditcard.csv")

df.head()
df.describe()
df.isnull().sum().sort_values(ascending = False)
# The classes are heavily skewed we need to solve this issue later.

print('No Frauds', round(df['Class'].value_counts()[0]/len(df) * 100,2), '% of the dataset')

print('Frauds', round(df['Class'].value_counts()[1]/len(df) * 100,2), '% of the dataset')
f,ax=plt.subplots(1,2,figsize=(12,6))

df.Class.value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_ylabel('')

ax = sns.countplot('Class',data=df,ax=ax[1])

for p in ax.patches:

        x=p.get_bbox().get_points()[:,0]

        y=p.get_bbox().get_points()[1,1]

        ax.annotate('{:.2f}%'.format(100.*y/len(df)), (x.mean(), y), 

            ha='center', va='bottom',color='black',weight='bold')

plt.show()
target_defaulters=df.loc[df['Class'] == 1]

target_defaulters.head()
target_payers=df.loc[df['Class'] == 0]

target_payers.head()
#Function to get the Univariate analysis of Defaulters and non-defaulters segments using column as input.

def Univarite_numerical_val(column):

    trace0 = go.Box(

        y=target_defaulters[column].values,

        name = 'distribution for defaulters'

    )



    trace1 = go.Box(

        y=target_payers[column].values,

        name = 'distribution for non-defaulters'

    )



    layout = go.Layout(

        yaxis=dict(

            type='log',

            autorange=True

        ))



    data = [trace0, trace1]

    fig = go.Figure(data=data, layout=layout)

    plotly.offline.iplot(fig) 

#Function to get Outliers and all 10% to 100% values for defaulters

def univariate_default_numerical(column):

    return target_defaulters[column].quantile([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

#Function to get Outliers and all 10% to 100% values for payers

def univariate_payers_numerical(column):

    return target_payers[column].quantile([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
Univarite_numerical_val('Amount')
# Plot histograms of each parameter 

df.hist(figsize = (20, 20))

plt.show()
fig, ax = plt.subplots(1, 2, figsize=(18,4))



amount_val = df['Amount'].values

time_val = df['Time'].values



sns.distplot(amount_val, ax=ax[0], color='r')

ax[0].set_title('Distribution of Transaction Amount', fontsize=14)

ax[0].set_xlim([min(amount_val), max(amount_val)])



sns.distplot(time_val, ax=ax[1], color='b')

ax[1].set_title('Distribution of Transaction Time', fontsize=14)

ax[1].set_xlim([min(time_val), max(time_val)])







plt.show()
fraud = df.loc[df['Class'] == 1]



trace = go.Scatter(

    x = fraud['Time'],y = fraud['Amount'],

    name="Amount",

     marker=dict(

                color='rgb(238,23,11)',

                line=dict(

                    color='red',

                    width=1),

                opacity=0.5,

            ),

    text= fraud['Amount'],

    mode = "markers"

)

data = [trace]

layout = dict(title = 'Amount of fraudulent transactions',

          xaxis = dict(title = 'Time [s]', showticklabels=True), 

          yaxis = dict(title = 'Amount'),

          hovermode='closest'

         )

fig = dict(data=data, layout=layout)

iplot(fig, filename='fraud-amount')
plt.figure(figsize = (14,14))

plt.title('Credit Card Transactions features correlation plot (Pearson)')

corr = df.corr()

sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns,linewidths=.1,cmap="Reds")

plt.show()
s = sns.lmplot(x='V20', y='Amount',data=df, hue='Class', fit_reg=True,scatter_kws={'s':2})

s = sns.lmplot(x='V7', y='Amount',data=df, hue='Class', fit_reg=True,scatter_kws={'s':2})

plt.show()
s = sns.lmplot(x='V2', y='Amount',data=df, hue='Class', fit_reg=True,scatter_kws={'s':2})

s = sns.lmplot(x='V5', y='Amount',data=df, hue='Class', fit_reg=True,scatter_kws={'s':2})

plt.show()
var = df.columns.values



i = 0

t0 = df.loc[df['Class'] == 0]

t1 = df.loc[df['Class'] == 1]



sns.set_style('whitegrid')

plt.figure()

fig, ax = plt.subplots(8,4,figsize=(16,28))



for feature in var:

    i += 1

    plt.subplot(8,4,i)

    sns.kdeplot(t0[feature], bw=0.5,label="Class = 0")

    sns.kdeplot(t1[feature], bw=0.5,label="Class = 1")

    plt.xlabel(feature, fontsize=12)

    locs, labels = plt.xticks()

    plt.tick_params(axis='both', which='major', labelsize=12)

plt.show();
# Since most of our data has already been scaled we should scale the columns that are left to scale (Amount and Time)

from sklearn.preprocessing import StandardScaler, RobustScaler



# RobustScaler is less prone to outliers.



std_scaler = StandardScaler()

rob_scaler = RobustScaler()



df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))

df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))



df.drop(['Time','Amount'], axis=1, inplace=True)
scaled_amount = df['scaled_amount']

scaled_time = df['scaled_time']



df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)

df.insert(0, 'scaled_amount', scaled_amount)

df.insert(1, 'scaled_time', scaled_time)



# Amount and Time are Scaled!



df.head()
from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedShuffleSplit



print('No Frauds', round(df['Class'].value_counts()[0]/len(df) * 100,2), '% of the dataset')

print('Frauds', round(df['Class'].value_counts()[1]/len(df) * 100,2), '% of the dataset')



X = df.drop('Class', axis=1)

y = df['Class']



sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)



for train_index, test_index in sss.split(X, y):

    print("Train:", train_index, "Test:", test_index)

    original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]

    original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]



# We already have X_train and y_train for undersample data thats why I am using original to distinguish and to not overwrite these variables.

# original_Xtrain, original_Xtest, original_ytrain, original_ytest = train_test_split(X, y, test_size=0.2, random_state=42)



# Check the Distribution of the labels





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



# amount of fraud classes 492 rows.

fraud_df = df.loc[df['Class'] == 1]

non_fraud_df = df.loc[df['Class'] == 0][:492]



normal_distributed_df = pd.concat([fraud_df, non_fraud_df])



# Shuffle dataframe rows

new_df = normal_distributed_df.sample(frac=1, random_state=42)



new_df.head()
import matplotlib.gridspec as gridspec

columns = df.iloc[:,1:29].columns



frauds = df.Class == 1

normals = df.Class == 0



grid = gridspec.GridSpec(14, 2)

plt.figure(figsize=(15,20*4))



for n, col in enumerate(df[columns]):

    ax = plt.subplot(grid[n])

    sns.distplot(df[col][frauds], bins = 50, color='g') #Will receive the "semi-salmon" violin

    sns.distplot(df[col][normals], bins = 50, color='r') #Will receive the "ocean" color

    ax.set_ylabel('Density')

    ax.set_title(str(col))

    ax.set_xlabel('')

plt.show()
print('Distribution of the Classes in the subsample dataset')

print(new_df['Class'].value_counts()/len(new_df))

colors = ["#0101DF", "#DF0101"]

sns.countplot('Class', data=new_df, palette=colors)

plt.title('Equally Distributed Classes', fontsize=14)

plt.show()
# Make sure we use the subsample in our correlation



f, (ax1, ax2) = plt.subplots(2, 1, figsize=(24,20))



# Entire DataFrame

corr = df.corr()

sns.heatmap(corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax1)

ax1.set_title("Imbalanced Correlation Matrix \n (don't use for reference)", fontsize=14)





sub_sample_corr = new_df.corr()

sns.heatmap(sub_sample_corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax2)

ax2.set_title('SubSample Correlation Matrix \n (use for reference)', fontsize=14)

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
from scipy.stats import norm



f, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(20, 6))



v14_fraud_dist = new_df['V14'].loc[new_df['Class'] == 1].values

sns.distplot(v14_fraud_dist,ax=ax1, fit=norm, color='#FB8861')

ax1.set_title('V14 Distribution \n (Fraud Transactions)', fontsize=14)



v12_fraud_dist = new_df['V12'].loc[new_df['Class'] == 1].values

sns.distplot(v12_fraud_dist,ax=ax2, fit=norm, color='#56F9BB')

ax2.set_title('V12 Distribution \n (Fraud Transactions)', fontsize=14)





v10_fraud_dist = new_df['V10'].loc[new_df['Class'] == 1].values

sns.distplot(v10_fraud_dist,ax=ax3, fit=norm, color='#C5B3F9')

ax3.set_title('V10 Distribution \n (Fraud Transactions)', fontsize=14)



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
X = np.array(new_df.ix[:, new_df.columns != 'Class'])

y = np.array(new_df.ix[:, new_df.columns == 'Class'])

print('Shape of X: {}'.format(X.shape))

print('Shape of y: {}'.format(y.shape))
from imblearn.over_sampling import SMOTE



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)



print("Number transactions X_train dataset: ", X_train.shape)

print("Number transactions y_train dataset: ", y_train.shape)

print("Number transactions X_test dataset: ", X_test.shape)

print("Number transactions y_test dataset: ", y_test.shape)
print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))

print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))



sm = SMOTE(random_state=2)

X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())



print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))

print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))



print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))

print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve, recall_score, classification_report



parameters = {

    'C': np.linspace(1, 10, 10)

             }

lr = LogisticRegression()

clf = GridSearchCV(lr, parameters, cv=5, verbose=5, n_jobs=3)

clf.fit(X_train_res, y_train_res.ravel())
clf.best_params_
lr1 = LogisticRegression(C=4,penalty='l2', verbose=5)

lr1.fit(X_train_res, y_train_res.ravel())
import itertools



def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=0)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        #print("Normalized confusion matrix")

    else:

        1#print('Confusion matrix, without normalization')



    #print(cm)



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
y_train_pre = lr1.predict(X_train)



cnf_matrix_tra = confusion_matrix(y_train, y_train_pre)



print("Recall metric in the train dataset: {}%".format(100*cnf_matrix_tra[1,1]/(cnf_matrix_tra[1,0]+cnf_matrix_tra[1,1])))





class_names = [0,1]

plt.figure()

plot_confusion_matrix(cnf_matrix_tra , classes=class_names, title='Confusion matrix')

plt.show()
y_pre = lr1.predict(X_test)



cnf_matrix = confusion_matrix(y_test, y_pre)



print("Recall metric in the testing dataset: {}%".format(100*cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1])))

#print("Precision metric in the testing dataset: {}%".format(100*cnf_matrix[0,0]/(cnf_matrix[0,0]+cnf_matrix[1,0])))

# Plot non-normalized confusion matrix

class_names = [0,1]

plt.figure()

plot_confusion_matrix(cnf_matrix , classes=class_names, title='Confusion matrix')

plt.show()
tmp = lr1.fit(X_train_res, y_train_res.ravel())
y_pred_sample_score = tmp.decision_function(X_test)





fpr, tpr, thresholds = roc_curve(y_test, y_pred_sample_score)



roc_auc = auc(fpr,tpr)



# Plot ROC

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.1,1.0])

plt.ylim([-0.1,1.01])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
roc_auc
# Undersampling before cross validating (prone to overfit)

X = new_df.drop('Class', axis=1)

y = new_df['Class']
# Our data is already scaled we should split our training and test sets

# This is explicitly used for undersampling.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Turn the values into an array for feeding the classification algorithms.

X_train = X_train.values

X_test = X_test.values

y_train = y_train.values

y_test = y_test.values
# Let's implement simple classifiers



classifiers = {

    "LogisiticRegression": LogisticRegression(),

    "KNearest": KNeighborsClassifier(),

    "Support Vector Classifier": SVC(),

    "DecisionTreeClassifier": DecisionTreeClassifier()

}
# Wow our scores are getting even high scores even when applying cross validation.

from sklearn.model_selection import cross_val_score





for key, classifier in classifiers.items():

    classifier.fit(X_train, y_train)

    training_score = cross_val_score(classifier, X_train, y_train, cv=5)

    print("Classifiers: ", classifier.__class__.__name__, "Has a training score of", round(training_score.mean(), 2) * 100, "% accuracy score")
# Use GridSearchCV to find the best parameters.

from sklearn.model_selection import GridSearchCV





# Logistic Regression 

log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}







grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params)

grid_log_reg.fit(X_train, y_train)

# We automatically get the logistic regression with the best parameters.

log_reg = grid_log_reg.best_estimator_



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

from sklearn.model_selection import ShuffleSplit

from sklearn.model_selection import learning_curve

import numpy as np



def learning_curve_plot (estimator1, estimator2, estimator3, estimator4, X, y, ylim=None, cv=None,

                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(20,14), sharey=True)

    if ylim is not None:

        plt.ylim(*ylim)

    # First Estimator

    train_sizes, train_scores, test_scores = learning_curve(

        estimator1, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

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

    ax2.legend(loc="best")# Third Estimator

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
cv_Shuffle = ShuffleSplit(n_splits=100, test_size=0.2, random_state=42)

learning_curve_plot(log_reg, knears_neighbors, svc, tree_clf, X_train, y_train, (0.87, 1.01), cv=cv_Shuffle, n_jobs=4)
from sklearn.metrics import roc_curve

from sklearn.model_selection import cross_val_predict

# Create a DataFrame with all the scores and the classifiers names.



log_reg_predict = cross_val_predict(log_reg, X_train, y_train, cv=5,

                             method="decision_function")



knears_predict = cross_val_predict(knears_neighbors, X_train, y_train, cv=5)



svc_predict = cross_val_predict(svc, X_train, y_train, cv=5,

                             method="decision_function")



tree_predict = cross_val_predict(tree_clf, X_train, y_train, cv=5)
from sklearn.metrics import roc_auc_score

print('Logistic Regression: ', roc_auc_score(y_train, log_reg_predict))

print('KNears Neighbors: ', roc_auc_score(y_train, knears_predict))

print('Support Vector Classifier: ', roc_auc_score(y_train, svc_predict))

print('Decision Tree Classifier: ', roc_auc_score(y_train, tree_predict))
log_fpr, log_tpr, log_thresold = roc_curve(y_train, log_reg_predict)

knear_fpr, knear_tpr, knear_threshold = roc_curve(y_train, knears_predict)

svc_fpr, svc_tpr, svc_threshold = roc_curve(y_train, svc_predict)

tree_fpr, tree_tpr, tree_threshold = roc_curve(y_train, tree_predict)





def graph_roc_curve_multiple(log_fpr, log_tpr, knear_fpr, knear_tpr, svc_fpr, svc_tpr, tree_fpr, tree_tpr):

    plt.figure(figsize=(16,8))

    plt.title('ROC Curve \n Top 4 Classifiers', fontsize=18)

    plt.plot(log_fpr, log_tpr, label='Logistic Regression Classifier Score: {:.4f}'.format(roc_auc_score(y_train, log_reg_predict)))

    plt.plot(knear_fpr, knear_tpr, label='KNears Neighbors Classifier Score: {:.4f}'.format(roc_auc_score(y_train, knears_predict)))

    plt.plot(svc_fpr, svc_tpr, label='Support Vector Classifier Score: {:.4f}'.format(roc_auc_score(y_train, svc_predict)))

    plt.plot(tree_fpr, tree_tpr, label='Decision Tree Classifier Score: {:.4f}'.format(roc_auc_score(y_train, tree_predict)))

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
def roc_curve_logistic(log_fpr, log_tpr):

    plt.figure(figsize=(12,8))

    plt.title('Logistic Regression ROC Curve', fontsize=16)

    plt.plot(log_fpr, log_tpr, 'b-', linewidth=2)

    plt.plot([0, 1], [0, 1], 'r--')

    plt.xlabel('False Positive Rate', fontsize=16)

    plt.ylabel('True Positive Rate', fontsize=16)

    plt.axis([-0.01,1,0,1])

    

    

roc_curve_logistic(log_fpr, log_tpr)

plt.show()
from sklearn.metrics import precision_recall_curve

precision, recall, threshold = precision_recall_curve(y_train, log_reg_predict)
y_predicted = log_reg.predict(X_train)



# Overfitting Case

print('---' * 40)

print('Overfitting: \n')

print('Recall Score: {:.2f}'.format(recall_score(y_train, y_predicted)))

print('Precision Score: {:.2f}'.format(precision_score(y_train, y_predicted)))

print('F1 Score: {:.2f}'.format(f1_score(y_train, y_predicted)))

print('Accuracy Score: {:.2f}'.format(accuracy_score(y_train, y_predicted)))

print('---' * 40)



# How it should look like

print('---' * 40)

print('How it should be:\n')

print("Accuracy Score: {:.2f}".format(np.mean(undersample_accuracy)))

print("Precision Score: {:.2f}".format(np.mean(undersample_precision)))

print("Recall Score: {:.2f}".format(np.mean(undersample_recall)))

print("F1 Score: {:.2f}".format(np.mean(undersample_f1)))

print('---' * 40)
undersamples_yscore = log_reg.decision_function(original_Xtest)
average_undersample_precision = average_precision_score(original_ytest, undersamples_yscore)



print('Average precision-recall score: {0:0.2f}'.format(

      average_undersample_precision))
fig = plt.figure(figsize=(12,6))



precision, recall, _ = precision_recall_curve(original_ytest, undersamples_yscore)



plt.step(recall, precision, color='#004a93', alpha=0.2,

         where='post')

plt.fill_between(recall, precision, step='post', alpha=0.2,

                 color='#48a6ff')



plt.xlabel('Recall_values')

plt.ylabel('Precision_values')

plt.ylim([0.0, 1.05])

plt.xlim([0.0, 1.0])

plt.title('UnderSampling Precision-Recall curve: \n Average Precision-Recall Scores ={0:0.2f}'.format(

          average_undersample_precision), fontsize=16)
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split, RandomizedSearchCV





print('Length of X (train): {} | Length of y (train): {}'.format(len(original_Xtrain), len(original_ytrain)))

print('Length of X (test): {} | Length of y (test): {}'.format(len(original_Xtest), len(original_ytest)))



# List to append the score and then find the average

accuracy_lst = []

precision_lst = []

recall_lst = []

f1_lst = []

auc_lst = []



# Classifier with optimal parameters

# log_reg_sm = grid_log_reg.best_estimator_

log_reg_sm = LogisticRegression()

rand_log_reg = RandomizedSearchCV(LogisticRegression(), log_reg_params, n_iter=4)





# Implementing SMOTE Technique 

# Cross Validating the right way

# Parameters

log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

for train, test in sss.split(original_Xtrain, original_ytrain):

    pipeline = imbalanced_make_pipeline(SMOTE(sampling_strategy='minority'), rand_log_reg) # SMOTE happens during Cross Validation not before..

    model = pipeline.fit(original_Xtrain[train], original_ytrain[train])

    best_est = rand_log_reg.best_estimator_

    prediction = best_est.predict(original_Xtrain[test])

    

    accuracy_lst.append(pipeline.score(original_Xtrain[test], original_ytrain[test]))

    precision_lst.append(precision_score(original_ytrain[test], prediction))

    recall_lst.append(recall_score(original_ytrain[test], prediction))

    f1_lst.append(f1_score(original_ytrain[test], prediction))

    auc_lst.append(roc_auc_score(original_ytrain[test], prediction))

    

print('---' * 45)

print('')

print("accuracy: {}".format(np.mean(accuracy_lst)))

print("precision: {}".format(np.mean(precision_lst)))

print("recall: {}".format(np.mean(recall_lst)))

print("f1: {}".format(np.mean(f1_lst)))

print("auc: {}".format(np.mean(auc_lst)))

print('---' * 45)
labels_data = ['No Fraud', 'Fraud']

smote_predicted = best_est.predict(original_Xtest)

print(classification_report(original_ytest, smote_predicted, target_names=labels_data))
y_best_score = best_est.decision_function(original_Xtest)
avg_precision = average_precision_score(original_ytest, y_best_score)



print('Average precision-recall score: {0:0.2f}'.format(

      avg_precision))
fig = plt.figure(figsize=(12,6))



precision, recall, _ = precision_recall_curve(original_ytest, y_best_score)



plt.step(recall, precision, color='r', alpha=0.2,

         where='post')

plt.fill_between(recall, precision, step='post', alpha=0.2,

                 color='#F59B00')



plt.xlabel('Recall')

plt.ylabel('Precision')

plt.ylim([0.0, 1.05])

plt.xlim([0.0, 1.0])

plt.title('OverSampling Precision-Recall curve: \n Average Precision-Recall Score ={0:0.2f}'.format(

          avg_precision), fontsize=16)
# SMOTE Technique (OverSampling) After splitting and Cross Validating

sm_cross = SMOTE(ratio='minority', random_state=42)

# Xsm_train, ysm_train = sm.fit_sample(X_train, y_train)





# This will be the data were we are going to 

Xsmcross_train, ysmcross_train = sm_cross.fit_sample(original_Xtrain, original_ytrain)
# Logistic Regression

import time

t0 = time.time()

log_reg_sm = grid_log_reg.best_estimator_

log_reg_sm.fit(Xsmcross_train, ysmcross_train)

t1 = time.time()

print("Fitting oversample data took :{} sec".format(t1 - t0))
from sklearn.metrics import confusion_matrix



# Logistic Regression fitted using SMOTE technique

y_predicted_log_reg = log_reg_sm.predict(X_test)



# Other models fitted with UnderSampling

y_predicted_knear = knears_neighbors.predict(X_test)

y_predicted_svc = svc.predict(X_test)

y_predicted_tree = tree_clf.predict(X_test)





logreg_cf = confusion_matrix(y_test, y_predicted_log_reg)

kneighbors_cf = confusion_matrix(y_test, y_predicted_knear)

svc_cf = confusion_matrix(y_test, y_predicted_svc)

tree_cf = confusion_matrix(y_test, y_predicted_tree)



fig, ax = plt.subplots(2, 2,figsize=(22,12))





sns.heatmap(logreg_cf, ax=ax[0][0], annot=True, cmap=plt.cm.copper)

ax[0, 0].set_title("Logistic Regression \n Confusion Matrix", fontsize=14)

ax[0, 0].set_xticklabels(['', ''], fontsize=14, rotation=90)

ax[0, 0].set_yticklabels(['', ''], fontsize=14, rotation=360)



sns.heatmap(kneighbors_cf, ax=ax[0][1], annot=True, cmap=plt.cm.copper)

ax[0][1].set_title("KNearsNeighbors \n Confusion Matrix", fontsize=14)

ax[0][1].set_xticklabels(['', ''], fontsize=14, rotation=90)

ax[0][1].set_yticklabels(['', ''], fontsize=14, rotation=360)



sns.heatmap(svc_cf, ax=ax[1][0], annot=True, cmap=plt.cm.copper)

ax[1][0].set_title("Suppor Vector Classifier \n Confusion Matrix", fontsize=14)

ax[1][0].set_xticklabels(['', ''], fontsize=14, rotation=90)

ax[1][0].set_yticklabels(['', ''], fontsize=14, rotation=360)



sns.heatmap(tree_cf, ax=ax[1][1], annot=True, cmap=plt.cm.copper)

ax[1][1].set_title("DecisionTree Classifier \n Confusion Matrix", fontsize=14)

ax[1][1].set_xticklabels(['', ''], fontsize=14, rotation=90)

ax[1][1].set_yticklabels(['', ''], fontsize=14, rotation=360)





plt.show()
print('Logistic Regression:')

print(classification_report(y_test, y_predicted_log_reg))



print('KNears Neighbors:')

print(classification_report(y_test, y_predicted_knear))



print('Support Vector Classifier:')

print(classification_report(y_test, y_predicted_svc))



print('Support Vector Classifier:')

print(classification_report(y_test, y_predicted_tree))
# Logistic Regression with Under-Sampling

y_predicted = log_reg.predict(X_test)

undersampled_score = accuracy_score(y_test, y_predicted)







# Logistic Regression with SMOTE Technique (Better accuracy with SMOTE t)

y_predicted_sm = best_est.predict(original_Xtest)

oversampled_score = accuracy_score(original_ytest, y_predicted_sm)





d = {'Technique': ['Random UnderSampling', 'Oversampling (SMOTE)'], 'Score': [undersampled_score, oversampled_score]}

finalised_df = pd.DataFrame(data=d)



# Move column

score = finalised_df['Score']

finalised_df.drop('Score', axis=1, inplace=True)

finalised_df.insert(1, 'Score', score)



# Note how high is accuracy score it can be misleading! 

finalised_df
targeted = 'Class'

predictors_values = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',\

       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',\

       'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',\

       'Amount']
pd.set_option('display.max_columns', 100)





RFC_METRIC = 'gini'  #metric used for RandomForrestClassifier

NUM_ESTIMATORS = 100 #number of estimators used for RandomForrestClassifier

NO_JOBS = 4 #number of parallel jobs used for RandomForrestClassifier





#TRAIN/VALIDATION/TEST SPLIT

#VALIDATION

VALID_SIZE = 0.20 # simple validation using train_test_split

TEST_SIZE = 0.20 # test size using_train_test_split



#CROSS-VALIDATION

NUMBER_KFOLDS = 5 #number of KFolds for cross-validation







RANDOM_STATE = 2018



MAX_ROUNDS = 1000 #lgb iterations

EARLY_STOP = 50 #lgb early stop 

OPT_ROUNDS = 1000  #To be adjusted based on best validation rounds

VERBOSE_EVAL = 50 #Print out metric result



IS_LOCAL = False
# Reading again

df=pd.read_csv(r"C:\Users\Ankita Rath\Downloads\CAPSTONE\creditcard.csv")

df.head()
train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True )

train_df, validate_df = train_test_split(train_df, test_size=VALID_SIZE, random_state=RANDOM_STATE, shuffle=True )
clf_rf = RandomForestClassifier(n_jobs=NO_JOBS, 

                             random_state=RANDOM_STATE,

                             criterion=RFC_METRIC,

                             n_estimators=NUM_ESTIMATORS,

                             verbose=False)
clf_rf.fit(train_df[predictors_values], train_df[targeted].values)
predictions = clf_rf.predict(validate_df[predictors_values])
# Feature Importance

tmp = pd.DataFrame({'Feature': predictors_values, 'Feature importance': clf_rf.feature_importances_})

tmp = tmp.sort_values(by='Feature importance',ascending=False)

plt.figure(figsize = (7,4))

plt.title('Features importance',fontsize=14)

s = sns.barplot(x='Feature',y='Feature importance',data=tmp)

s.set_xticklabels(s.get_xticklabels(),rotation=90)

plt.show()  
# confusion matrix

cm = pd.crosstab(validate_df[targeted].values, predictions, rownames=['Actual'], colnames=['Predicted'])

fig, (ax1) = plt.subplots(ncols=1, figsize=(5,5))

sns.heatmap(cm, 

            xticklabels=['Not Fraud', 'Fraud'],

            yticklabels=['Not Fraud', 'Fraud'],

            annot=True,ax=ax1,

            linewidths=.2,linecolor="Darkblue", cmap="Blues")

plt.title('Confusion Matrix', fontsize=14)

plt.show()
TP = cm[1][1] # true positive 

TN = cm[0][0] # true negatives

FP = cm[0][1] # false positives

FN = cm[1][0] # false negatives



precision=TP/(TP+FP)

recall=TP/(TP+FN)

print('precision score',round(precision,2))

print('recall score',round(recall,2))
# Area under ROC Curve

from sklearn.metrics import roc_auc_score

roc_auc_score(validate_df[targeted].values, predictions)
from sklearn.ensemble import AdaBoostClassifier

clf_ada = AdaBoostClassifier(random_state=RANDOM_STATE,

                         algorithm='SAMME.R',

                         learning_rate=0.8,

                             n_estimators=NUM_ESTIMATORS)
clf_ada.fit(train_df[predictors_values], train_df[targeted].values)
predicted = clf_ada.predict(validate_df[predictors_values])
tmp = pd.DataFrame({'Feature': predictors_values, 'Feature importance': clf_ada.feature_importances_})

tmp = tmp.sort_values(by='Feature importance',ascending=False)

plt.figure(figsize = (7,4))

plt.title('Features importance',fontsize=14)

s = sns.barplot(x='Feature',y='Feature importance',data=tmp)

s.set_xticklabels(s.get_xticklabels(),rotation=90)

plt.show()
cm = pd.crosstab(validate_df[targeted].values, predicted, rownames=['Actual'], colnames=['Predicted'])

fig, (ax1) = plt.subplots(ncols=1, figsize=(5,5))

sns.heatmap(cm, 

            xticklabels=['Not Fraud', 'Fraud'],

            yticklabels=['Not Fraud', 'Fraud'],

            annot=True,ax=ax1,

            linewidths=.2,linecolor="Darkblue", cmap="Blues")

plt.title('Confusion Matrix', fontsize=14)

plt.show()
TP = cm[1][1] # true positive 

TN = cm[0][0] # true negatives

FP = cm[0][1] # false positives

FN = cm[1][0] # false negatives



precision=TP/(TP+FP)

recall=TP/(TP+FN)

print('precision score',round(precision,2))

print('recall score',round(recall,2))
# ROC Curve

roc_auc_score(validate_df[targeted].values, predicted)
import xgboost as xgb

# Prepare the train and valid datasets

d_train = xgb.DMatrix(train_df[predictors_values], train_df[targeted].values)

d_valid = xgb.DMatrix(validate_df[predictors_values], validate_df[targeted].values)

d_test = xgb.DMatrix(test_df[predictors_values], test_df[targeted].values)



#What to monitor (in this case, **train** and **valid**)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]



# Set xgboost parameters

params = {}

params['objective'] = 'binary:logistic'

params['eta'] = 0.039

params['silent'] = True

params['max_depth'] = 2

params['subsample'] = 0.8

params['colsample_bytree'] = 0.9

params['eval_metric'] = 'auc'

params['random_state'] = RANDOM_STATE
# Training the model

model_xg = xgb.train(params, 

                d_train, 

                MAX_ROUNDS, 

                watchlist, 

                early_stopping_rounds=EARLY_STOP, 

                maximize=True, 

                verbose_eval=VERBOSE_EVAL)
fig, (ax) = plt.subplots(ncols=1, figsize=(10,10))

xgb.plot_importance(model_xg, height=0.8, title="Features importance (XGBoost)", ax=ax, color="red") 

plt.show()
preds = model_xg.predict(d_test)
roc_auc_score(test_df[targeted].values, preds)