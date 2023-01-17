# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#importing libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, GridSearchCV

from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit, learning_curve

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_curve, f1_score, confusion_matrix, roc_curve, roc_auc_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, BaggingClassifier, GradientBoostingClassifier

from sklearn.neural_network import MLPClassifier

from sklearn import svm

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier

from sklearn.feature_selection import SelectFromModel

import lightgbm as lgb

from lightgbm import LGBMClassifier

import xgboost as xgb

import time

import itertools

from datetime import datetime

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.simplefilter("ignore")

pd.set_option('display.max_columns', 100)

%matplotlib inline

random = 40
# defining visualizaition functions

def format_spines(ax, right_border=True):

    

    ax.spines['bottom'].set_color('#666666')

    ax.spines['left'].set_color('#666666')

    ax.spines['top'].set_visible(False)

    if right_border:

        ax.spines['right'].set_color('#FFFFFF')

    else:

        ax.spines['right'].set_color('#FFFFFF')

    ax.patch.set_facecolor('#FFFFFF')

    



def count_plot(feature, df, colors='Blues_d', hue=False, ax=None, title=''):

    

    ncount = len(df)

    if hue != False:

        ax = sns.countplot(x=feature, data=df, palette=colors, hue=hue, ax=ax)

    else:

        ax = sns.countplot(x=feature, data=df, palette=colors, ax=ax)

        

    format_spines(ax)



    for p in ax.patches:

        x=p.get_bbox().get_points()[:,0]

        y=p.get_bbox().get_points()[1,1]

        ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 

                ha='center', va='bottom') # set the alignment of the text



    if not hue:

        ax.set_title(df[feature].describe().name + ' Analysis', size=13, pad=15)

    else:

        ax.set_title(df[feature].describe().name + ' Analysis by ' + hue, size=13, pad=15)  

    if title != '':

        ax.set_title(title)       

    plt.tight_layout()

    

    

def bar_plot(x, y, df, colors='Blues_d', hue=False, ax=None, value=False, title=''):



    try:

        ncount = sum(df[y])

    except:

        ncount = sum(df[x])

        

    if hue != False:

        ax = sns.barplot(x=x, y=y, data=df, palette=colors, hue=hue, ax=ax, ci=None)

    else:

        ax = sns.barplot(x=x, y=y, data=df, palette=colors, ax=ax, ci=None)



    format_spines(ax)



    for p in ax.patches:

        xp=p.get_bbox().get_points()[:,0]

        yp=p.get_bbox().get_points()[1,1]

        if value:

            ax.annotate('{:.2f}k'.format(yp/1000), (xp.mean(), yp), 

                    ha='center', va='bottom') # set the alignment of the text

        else:

            ax.annotate('{:.1f}%'.format(100.*yp/ncount), (xp.mean(), yp), 

                    ha='center', va='bottom') # set the alignment of the text

    if not hue:

        ax.set_title(df[x].describe().name + ' Analysis', size=12, pad=15)

    else:

        ax.set_title(df[x].describe().name + ' Analysis by ' + hue, size=12, pad=15)

    if title != '':

        ax.set_title(title)  

    plt.tight_layout()



def plot_roc_curve(fpr, tpr, label=None):

    """

    this function plots the ROC curve of a model

    

    input:

        fpr: false positive rate

        tpr: true positive rate

    returns:

        ROC curve

    """

    

    # Showing data

    plt.plot(fpr, tpr, linewidth=2, label=label)

    plt.plot([0, 1], [0, 1], 'k--')

    plt.axis([0, 1, 0, 1])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('ROC Curve')

    plt.show()



def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function set up and plot a Confusion Matrix

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title, fontsize=14)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes)

    plt.yticks(tick_marks, classes)

    

    # Format plot

    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 1.2

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j]),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    

def create_dataset():

    """

    This functions creates a dataframe to keep performance analysis

    """

    attributes = ['acc', 'prec', 'rec', 'f1', 'auc', 'total_time']

    model_performance = pd.DataFrame({})

    for col in attributes:

        model_performance[col] = []

    return model_performance



def model_analysis(classifiers, X, y, df_performance, cv=5, train=True):

    """

    This function brings up a full model evaluation and saves it in a DataFrame object.

    """

    for key, model in classifiers.items():

        t0 = time.time()

        

        # Accuracy, precision, recall and f1_score on training set using cv

        if train:

            acc = cross_val_score(model, X, y, cv=cv, scoring='accuracy').mean()

            prec = cross_val_score(model, X, y, cv=cv, scoring='precision').mean()

            rec = cross_val_score(model, X, y, cv=cv, scoring='recall').mean()

            f1 = cross_val_score(model, X, y, cv=cv, scoring='f1').mean()

        else:

            y_pred = model.predict(X)

            acc = accuracy_score(y, y_pred)

            prec = precision_score(y, y_pred)

            rec = recall_score(y, y_pred)

            f1 = f1_score(y, y_pred)

        

        # AUC score

        try:

            y_scores = cross_val_predict(model, X, y, cv=5, 

                                     method='decision_function')

        except:

            # Trees don't have decision_function but predict_proba

            y_probas = cross_val_predict(model, X, y, cv=5, 

                                         method='predict_proba')

            y_scores_tree = y_probas[:, 1]

            y_scores = y_scores_tree

        auc = roc_auc_score(y, y_scores)

        

        t1 = time.time()

        delta_time = t1-t0

        model_name = model.__class__.__name__

        

        # Saving on dataframe

        performances = {}

        performances['acc'] = round(acc, 4)

        performances['prec'] = round(prec, 4)

        performances['rec'] = round(rec, 4)

        performances['f1'] = round(f1, 4)

        performances['auc'] = round(auc, 4)

        performances['total_time'] = round(delta_time, 3)

        

        df_performance = df_performance.append(performances, ignore_index=True)

    df_performance.index = classifiers.keys()

    

    return df_performance



def model_confusion_matrix(classifiers, X, y, cmap=plt.cm.Blues):

    """

    This function computes predictions for all model and plots a confusion matrix

    for each one.

    """

    i = 1

    plt.figure(figsize=(19, 8))

    sns.set(style='white', palette='muted', color_codes=True)

    labels = ['Positive', 'Negative']

    

    # Ploting confusion matrix

    for key, model in classifiers.items(): 

        y_pred = model.predict(X)

        model_cf_mx = confusion_matrix(y, y_pred)



        # Plotando matriz

        model_name = model.__class__.__name__

        plt.subplot(2, 3, i)

        plot_confusion_matrix(model_cf_mx, labels, title=model_name + '\nConfusion Matrix', cmap=cmap)

        i += 1



    plt.tight_layout()

    plt.show()

    

def plot_roc_curve(fpr, tpr, y, y_scores, auc, label=None):

    """

    This function plots the ROC curve of a model

    """   

    # Showing data

    sns.set(style='white', palette='muted', color_codes=True)

    plt.plot(fpr, tpr, linewidth=2, label=f'{label}, auc={auc:.3f}')

    plt.plot([0, 1], [0, 1], 'k--')

    plt.axis([-0.02, 1.02, -0.02, 1.02])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title(f'ROC Curve', size=14)

    plt.annotate('Minimum ROC Score of 50% \n (This is the minimum score to get)', 

                 xy=(0.5, 0.5), xytext=(0.6, 0.4),

                 arrowprops=dict(facecolor='#6E726D', shrink=0.05),

                )

    plt.legend()

    

def plot_precision_vs_recall(precisions, recalls, label=None, color='b'):

    """

    This function plots precision versus recall curve.

    """

    sns.set(style='white', palette='muted', color_codes=True)

    if label=='LogisticRegression':

        plt.plot(recalls, precisions, 'r-', linewidth=2, label=label)

    else:

        plt.plot(recalls, precisions, color=color, linewidth=2, label=label)

    plt.title('Precision versus Recall', fontsize=14)

    plt.xlabel('Recall')

    plt.ylabel('Precision')

    plt.axis([0, 1, 0, 1])

    plt.legend()

    

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):

    """

    This function plots precision x recall among different thresholds

    """

    plt.plot(thresholds, precisions[:-1], 'b--', label='Precision')

    plt.plot(thresholds, recalls[:-1], 'g-', label='Recall')

    plt.xlabel('Threshold')

    plt.title('Precision versus Recall - Thresholds', size=14)

    plt.legend(loc='best')

    plt.ylim([0, 1])

    

def plot_learning_curve(trained_models, X, y, ylim=None, cv=5, n_jobs=1, 

                        train_sizes=np.linspace(.1, 1.0, 10)):

    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(14, 14))

    if ylim is not None:

        plt.ylim(*ylim)

    i = 0

    j = 0

    for key, model in trained_models.items():

        train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=cv, n_jobs=n_jobs, 

                                                                train_sizes=train_sizes)

        train_scores_mean = np.mean(train_scores, axis=1)

        train_scores_std = np.std(train_scores, axis=1)

        test_scores_mean = np.mean(test_scores, axis=1)

        test_scores_std = np.std(test_scores, axis=1)

        axs[i, j].fill_between(train_sizes, train_scores_mean - train_scores_std,

                               train_scores_mean + train_scores_std, alpha=0.1, color='blue')

        axs[i, j].fill_between(train_sizes, test_scores_mean - test_scores_std,

                               test_scores_mean + test_scores_std, alpha=0.1, color='crimson')

        axs[i, j].plot(train_sizes, train_scores_mean, 'o-', color="navy",

                       label="Training score")

        axs[i, j].plot(train_sizes, test_scores_mean, 'o-', color="red",

                       label="Cross-Validation score")

        axs[i, j].set_title(f'{key} Learning Curve', size=14)

        axs[i, j].set_xlabel('Training size (m)')

        axs[i, j].set_ylabel('Score')

        axs[i, j].grid(True)

        axs[i, j].legend(loc='best')

        j += 1

        if j == 2:

            i += 1

            j = 0
df = pd.read_csv('../input/train_technidus_clf.csv')

df_test = pd.read_csv('../input/test_technidus_clf.csv')
# displaying first three rows of dataset

df.head(3)
# displaying dataset information

df.info()
# displaying summary staticstics of columns

df.describe(include='all')
# displaying missing value counts and corresponding percentage against total observations

missing_values = df.isnull().sum().sort_values(ascending = False)

percentage = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)

pd.concat([missing_values, percentage], axis=1, keys=['Values', 'Percentage']).transpose()
# dropping non-important feature set

df = df.drop(["CustomerID", "Title", "FirstName", "MiddleName", "LastName", "Suffix", "AddressLine1", "AddressLine2", "PhoneNumber"], axis=1)
# dropping missing values

df.dropna(inplace=True)

df.isnull().values.any()
# rounding up floats

float_columns = ['HomeOwnerFlag', 'NumberCarsOwned', 'NumberChildrenAtHome', 'TotalChildren', 'BikeBuyer', 'YearlyIncome', 'AveMonthSpend']

for col in float_columns:

    df[col] = df[col].apply(np.ceil).astype('int64')
# converting birthdate to age

df['BirthDate'] = pd.to_datetime(df.BirthDate)

now = pd.to_datetime('now')

df['BirthDate'] = (now.year - df['BirthDate'].dt.year) - ((now.month - df['BirthDate'].dt.month) < 0)
# displaying total orders the disribution of each of the dependent(categorical) columns

fig, axs = plt.subplots(1, 4, figsize=(23, 5))

count_plot(feature='TotalChildren', df=df, ax=axs[0], title='Total Children Distribution')

count_plot(feature='NumberChildrenAtHome', df=df, ax=axs[1], title='Children At Home Distribution')

count_plot(feature='Occupation', df=df, ax=axs[2], title='Occupation Distribution')

count_plot(feature='Education', df=df, ax=axs[3], title='Education Distribution')

#format_spines(ax, right_border=False)

plt.suptitle('Feature Set Distribution', y=1.1)

plt.show()



fig, axs = plt.subplots(1, 5, figsize=(23, 5))

count_plot(feature='BikeBuyer', df=df, ax=axs[0], title='Bike Buyer Distribution')

count_plot(feature='Gender', df=df, ax=axs[1], title='Gender Distribution')

count_plot(feature='MaritalStatus', df=df, ax=axs[2], title='Marital Status Distribution')

count_plot(feature='HomeOwnerFlag', df=df, ax=axs[3], title='Home Owners Distribution')

count_plot(feature='NumberCarsOwned', df=df, ax=axs[4], title='Number of Cars Owned Distribution')

#format_spines(ax, right_border=False)

plt.suptitle('Feature Set Distribution', y=1.1)

plt.show()
# plotting the distribution of the continous feature set

sns.set(palette='muted', color_codes=True)

fig, axs = plt.subplots(1, 3, figsize=(23, 5))

sns.despine(left=True)

sns.distplot(df['YearlyIncome'], bins=20, ax=axs[0])

sns.distplot(df['AveMonthSpend'], bins=20, ax=axs[1])

sns.distplot(df['BirthDate'], bins=20, ax=axs[2])

plt.show()
# engineering new columns Ageband and Monthlyincome

df['AgeBand']=df['BirthDate'].apply(lambda x: 'Young' if x<=40 else ('Middleaged' if x>=41 and x<=70 else 'Old'))

df['MonthlyIncome']=df['YearlyIncome'].apply(lambda x: x/12).apply(np.ceil).astype('int64')
# plotting correlation heatmap

fig, ax = plt.subplots()

fig.set_size_inches(20, 8)

corr_matrix = df.corr()

ax=sns.heatmap(corr_matrix, vmin=-1, cmap="YlGnBu", annot=True)
# displaying comparative plot between independednt and dependent feature set

fig, axs = plt.subplots(1, 4, figsize=(22, 5))

count_plot(feature='TotalChildren', df=df, ax=axs[0], hue='BikeBuyer', title='Total Order Purchase by Year')

count_plot(feature='NumberChildrenAtHome', df=df, ax=axs[1], hue='BikeBuyer', title='Total Yearly order Purchase by Month')

count_plot(feature='Education', df=df, ax=axs[2], hue='BikeBuyer', title='Total Yearly order Purchase by Day of the Week')

count_plot(feature='Occupation', df=df, ax=axs[3], hue='BikeBuyer', title='Total Yearly order Purchase by Day of the Week')

#format_spines(ax, right_border=False)

plt.suptitle('Independednt and Dependent Feature Set Comparision Plot', y=1.1)

plt.show()



fig, axs = plt.subplots(1, 4, figsize=(23, 5))

count_plot(feature='NumberCarsOwned', df=df, ax=axs[0], hue='BikeBuyer', title='Bike Buyer Distribution')

count_plot(feature='Gender', df=df, ax=axs[1], hue='BikeBuyer', title='Gender Distribution')

count_plot(feature='MaritalStatus', df=df, ax=axs[2], hue='BikeBuyer', title='Marital Status Distribution')

count_plot(feature='HomeOwnerFlag', df=df, ax=axs[3], hue='BikeBuyer', title='Home Owners Distribution')

#format_spines(ax, right_border=False)

plt.suptitle('Feature Set Distribution', y=1.1)

plt.show()
# displaying comparative plot between independednt and dependent feature set

f, axes = plt.subplots(ncols=4, figsize=(23.5,4.5))

colors = 'cornflowerblue'

# Negative Correlations with our Class (The lower our feature value the more likely it will be a fraud transaction)

sns.stripplot(x="NumberChildrenAtHome", y="AveMonthSpend", hue='BikeBuyer', data=df, color=colors, ax=axes[0])

axes[0].set_title('NumberChildrenAtHome vs AveMonthSpend vs BikeBuyer')

sns.boxplot(x="NumberChildrenAtHome", y="AveMonthSpend", hue='BikeBuyer', data=df, color=colors, ax=axes[1])

axes[1].set_title('NumberChildrenAtHome vs AveMonthSpend vs BikeBuyer')

sns.stripplot(x="NumberChildrenAtHome", y="MonthlyIncome", data=df,  hue='BikeBuyer', jitter=True, color=colors, ax=axes[2])

axes[2].set_title('NumberChildrenAtHome vs MonthlyIncome vs BikeBuyer')

sns.boxplot(x="NumberChildrenAtHome", y="MonthlyIncome", hue='BikeBuyer', data=df, color=colors, ax=axes[3])

axes[3].set_title('NumberChildrenAtHome vs MonthlyIncome vs BikeBuyer')

plt.show()


# displaying comparative plot between independednt and dependent feature set

fig, axs = plt.subplots(1, 4, figsize=(23, 5))

count_plot(feature='CountryRegionName', df=df, ax=axs[0], title='Customer Location Distribution')

count_plot(feature='CountryRegionName', df=df, ax=axs[1], hue='BikeBuyer', title='Customer Location Distribution by BikeBuyer')

count_plot(feature='AgeBand', df=df, ax=axs[2], title='AgeBand Distribution')

count_plot(feature='AgeBand', df=df, ax=axs[3], hue='BikeBuyer', title='AgeBand Distribution by BikeBuyer')

plt.suptitle('Feature Set Distribution', y=1.1)

plt.show()
#Region_grouped = (df.groupby(['CountryRegionName', 'BikeBuyer'])[['CountryRegionName', 'BikeBuyer', 'payment_installments']]

                            # .agg({'payment_value':['count', 'mean', 'sum'], 'payment_installments': ['mean']})

               # ).sort_values(by=('payment_value','count'), ascending=False)

                 

#state_grouped
# displaying first rows of the dataset

df.head(3)
# mapping numbers to categorical columns

df['Gender'] = df['Gender'].apply(lambda x: 0 if x=='F' else 1)

df['MaritalStatus'] = df['MaritalStatus'].apply(lambda x: 0 if x=='S' else 1)

df['AgeBand'] = df['AgeBand'].apply(lambda x: 0 if x=='S' else 1)
#dropping non useful columns

df = df.drop(["City", "StateProvinceName", "PostalCode", "YearlyIncome"], axis=1)
# processing test_dataset for use in fiting onehotencoder due to data missing in both dataset

df_test['BirthDate'] = pd.to_datetime(df_test.BirthDate)

now = pd.to_datetime('now')

df_test['BirthDate'] = (now.year - df_test['BirthDate'].dt.year) - ((now.month - df_test['BirthDate'].dt.month) < 0)

df_test['AgeBand']=df_test['BirthDate'].apply(lambda x: 'Young' if x<=40 else ('Middleaged' if x>=41 and x<=70 else 'Old')).apply(lambda x: 0 if x=='S' else 1)

master_df = pd.concat([df, df_test], ignore_index=True, sort=False)
# OneHot encoding categorical colums

encode = OneHotEncoder().fit(master_df[['CountryRegionName', 'Education', 'Occupation','HomeOwnerFlag','AgeBand']])

df_trans = encode.transform(df[['CountryRegionName', 'Education', 'Occupation','HomeOwnerFlag','AgeBand']]).toarray()

cnames = encode.get_feature_names()

df_trans = pd.DataFrame(df_trans, columns=cnames)



df = df.join(df_trans)

df = df.drop(["CountryRegionName", "Education", "Occupation", "HomeOwnerFlag", "AgeBand"], axis=1)
# Robust scaling continous numerical colums due to presence of outliers and difference in column ranges

s_scal = master_df[['BirthDate', 'AveMonthSpend', 'MonthlyIncome']]

s_scale = RobustScaler(quantile_range=(25, 75)).fit(master_df[['BirthDate', 'AveMonthSpend', 'MonthlyIncome']])

df_scale = s_scale.transform(df[['BirthDate', 'AveMonthSpend', 'MonthlyIncome']])

df_scale = pd.DataFrame(df_scale, columns=s_scal.columns)



df = df.drop(["BirthDate", "AveMonthSpend", "MonthlyIncome"], axis=1)

df = df.join(df_scale)
# processing test dataset.

df_test = df_test.drop(["Title", "FirstName", "MiddleName", "LastName", "Suffix", "AddressLine1", "AddressLine2", "PhoneNumber"], axis=1)



float_columns = ['HomeOwnerFlag', 'NumberCarsOwned', 'NumberChildrenAtHome', 'TotalChildren', 'YearlyIncome', 'AveMonthSpend']

for col in float_columns:

    df_test[col] = df_test[col].apply(np.ceil).astype('int64')

    

df_test['BirthDate'] = pd.to_datetime(df_test.BirthDate)

now = pd.to_datetime('now')

df_test['BirthDate'] = (now.year - df_test['BirthDate'].dt.year) - ((now.month - df_test['BirthDate'].dt.month) < 0)



df_test['AgeBand']=df_test['BirthDate'].apply(lambda x: 'Young' if x<=40 else ('Middleaged' if x>=41 and x<=70 else 'Old'))

df_test['MonthlyIncome']=df_test['YearlyIncome'].apply(lambda x: x/12).apply(np.ceil).astype('int64')



df_test['Gender'] = df_test['Gender'].apply(lambda x: 0 if x=='F' else 1)

df_test['MaritalStatus'] = df_test['MaritalStatus'].apply(lambda x: 0 if x=='S' else 1)

df_test['AgeBand'] = df_test['AgeBand'].apply(lambda x: 0 if x=='S' else 1)



df_test = df_test.drop(["City", "StateProvinceName", "PostalCode", "YearlyIncome", "BikeBuyer"], axis=1)



df_test_trans = encode.transform(df_test[['CountryRegionName', 'Education', 'Occupation','HomeOwnerFlag','AgeBand']]).toarray()

df_test_trans = pd.DataFrame(df_test_trans, columns=cnames)



df_test = df_test.join(df_test_trans)

df_test = df_test.drop(["CountryRegionName", "Education", "Occupation", "HomeOwnerFlag", "AgeBand"], axis=1)



df_test_scale = s_scale.transform(df_test[['BirthDate', 'AveMonthSpend', 'MonthlyIncome']])

df_test_scale = pd.DataFrame(df_test_scale, columns=s_scal.columns)



df_test = df_test.drop(["BirthDate", "AveMonthSpend", "MonthlyIncome"], axis=1)

df_test = df_test.join(df_test_scale)
# dataset shape

print(df.shape)

print(df_test.shape)
# mapping and spliting train and test data

X = df.drop('BikeBuyer', axis=1)

y = df['BikeBuyer']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=random)



print(f'X_train dimension: {X_train.shape}')

print(f'X_test dimension: {X_test.shape}')

print(f'\ny_train dimension: {y_train.shape}')

print(f'y_test dimension: {y_test.shape}')
# defining and fitting classifiers

classifiers = {

    'log_reg': LogisticRegression(random_state = random),

    'rfc_clf': RandomForestClassifier(bootstrap=True, criterion='gini', max_depth=None, max_features='auto', n_estimators=200, random_state = random),

    'svc_clf': SVC(probability=True, kernel='linear', random_state = random),

    'mlp_clf': MLPClassifier(random_state = random),

    'knc_clf': KNeighborsClassifier(),

    'dtc_clf': DecisionTreeClassifier(random_state = random),

}



trained_models = {}



for key, model in classifiers.items():

    model.fit(X_train, y_train)

    trained_models[key] = model
# Creating dataframe to hold, evaluate and print metrics

train_performance = create_dataset()



train_performance = model_analysis(trained_models, X_train, y_train, train_performance)



cm = sns.light_palette("cornflowerblue", as_cmap=True)

train_performance.style.background_gradient(cmap=cm)
# computing and visualizing train confusion matrix

model_confusion_matrix(trained_models, X_train, y_train)
# computing and visualizing train learning curve

cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=42)

plot_learning_curve(trained_models, X_train, y_train, (0.37, 1.01), cv=5, n_jobs=4)

plt.tight_layout()
# computing and visualizing test confusion matrix

model_confusion_matrix(trained_models, X_test, y_test, cmap=plt.cm.Reds)
# computing and visualizing model set ROC Curve

plt.figure(figsize=(14, 7))



for key, model in trained_models.items():



    # Computing scores with cross_val_predict

    try:

        y_scores = cross_val_predict(model, X_test, y_test, cv=5, 

                                     method='decision_function')

    except:

        # Trees don't have decision_function but predict_proba

        y_probas = cross_val_predict(model, X_test, y_test, cv=5, 

                                     method='predict_proba')

        y_scores_tree = y_probas[:, 1]

        y_scores = y_scores_tree

        

    # ROC Curve

    model_name = model.__class__.__name__

    fpr, tpr, thresholds = roc_curve(y_test, y_scores)

    auc = roc_auc_score(y_test, y_scores)

    plot_roc_curve(fpr, tpr, y_test, y_scores, auc, label=model_name)

    plt.suptitle('Original Test Set')
# defining grid search parameter

param_grid_forest = [

    {

        'n_estimators': [5, 10, 20, 50],        

        'criterion': ['gini', 'entropy'],

        'min_samples_split': [2, 5],

        'max_depth': [None, 5, 10, 15],

        'min_samples_leaf': [1, 5],

        'bootstrap': [True, False]

    }

]



# defining anf fiting gridsearch on choosen classifier

forest_clf = RandomForestClassifier()

grid_search = GridSearchCV(forest_clf, param_grid_forest, cv=5,

                           scoring='roc_auc', verbose=1)



grid_search.fit(X_train, y_train)
#selecting gridsearch best model

best_model = grid_search.best_estimator_



# defining classifiers for voting classifier



log_reg = LogisticRegression(random_state = random)

svm_clf = SVC(probability=True)

knn_clf = KNeighborsClassifier()

mlp_clf = MLPClassifier()

tree_clf = DecisionTreeClassifier()



voting_clf = VotingClassifier(

    estimators=[('lr', log_reg), ('best_rf', best_model), ('svc', svm_clf), 

                ('knn', knn_clf), ('mlp', mlp_clf), ('tree', tree_clf)],

    voting='soft'

)



voting_clf.fit(X_train, y_train)



# Training a Bagging model with 500 Decision Trees

bag_clf = BaggingClassifier(

    DecisionTreeClassifier(), n_estimators=500,

    max_samples=100, bootstrap=True, n_jobs=-1, oob_score=True

)

bag_clf.fit(X_train, y_train)



# Training a Bagging model with 500 Decision Trees

ada_clf = AdaBoostClassifier(

    DecisionTreeClassifier(), n_estimators=500,

    learning_rate=0.5

)

ada_clf.fit(X_train, y_train)



# Training a Bagging model with 500 Decision Trees

gboost_clf = GradientBoostingClassifier(

     n_estimators=500, learning_rate=1.0

)

gboost_clf.fit(X_train, y_train)
voting = {

    'forest_grid': best_model,

    'voting_clf': voting_clf,

    'bag_clf': bag_clf,

    'ada_clf': ada_clf,

    'grad_boost': gboost_clf

}



trained_s_models = {}



for key, s_model in voting.items():

    s_model.fit(X_train, y_train)

    trained_s_models[key] = s_model



# Creating dataframe to hold metrics

train_s_performance = create_dataset()



# Evaluating models

train_s_performance = model_analysis(trained_s_models, X_train, y_train, train_s_performance)



# Result

cm = sns.light_palette("cornflowerblue", as_cmap=True)

train_s_performance.style.background_gradient(cmap=cm)
ensemble_performance = train_performance.append(train_s_performance)

ensemble_performance.index = ['LogisticRegression','RandomForest','SVMClassifier','MLPClassifier','KNNClassifier','DecisionTree','ForestGridSearch','VotingClassifier','BaggingClassifier','Adaboost','Gboost']

cm = sns.light_palette("lightgreen", as_cmap=True)

ensemble_performance.style.background_gradient(cmap=cm)
y_probas = cross_val_predict(best_model, X_train, 

                                 y_train, cv=10, method='predict_proba')

forest_probas = y_probas[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_train, forest_probas)



fig, ax = plt.subplots(figsize=(10, 6))

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

format_spines(ax, right_border=False)

ax.set_title('Precision/Recall Curve', size=14)

plt.show()



fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(recalls, precisions, 'b-', linewidth=2)

ax.set_xlabel('Recall')

ax.set_ylabel('Precision')

plt.axis([0, 1, 0, 1])

ax.set_title('Precision versus Recall', size=14)

format_spines(ax, right_border=False)

plt.show()