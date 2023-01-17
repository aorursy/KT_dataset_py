# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Import Dependencies

%matplotlib inline



# Data manipulation

import os

import sys

import pandas as pd

import numpy as np

from collections import Counter



# Visualization

import matplotlib.pyplot as plt

import missingno

import seaborn as sns

plt.style.use('seaborn-whitegrid')



# Preprocessing

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import  make_scorer,accuracy_score,precision_score,roc_auc_score,f1_score,recall_score



# Machine learning

from sklearn.model_selection import train_test_split

from sklearn import model_selection, tree, preprocessing, metrics, linear_model

from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import MinMaxScaler

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import GridSearchCV

from sklearn.feature_selection import RFECV

from sklearn.metrics import roc_auc_score

# Let's be rebels and ignore warnings for now

import warnings

warnings.filterwarnings('ignore')
# Import train & test data 

train_data = pd.read_csv('../input/challenge-gh/TrainData.csv', encoding='ISO-8859-1',sep=';')

test_data = pd.read_csv('../input/challenge-gh/TestData.csv', encoding='ISO-8859-1',sep=';')

submission_data = pd.read_csv('../input/challenge-gh/Lsungstemplate.csv', encoding='ISO-8859-1') # example of what a submission should look like


# Exploratory Analysis
# Replace the unknown values with the np.nan values so that we can know the exact percent of missing values.

df_train = train_data.replace('Unbekannt', np.nan, inplace=False)
def visualize_missing_values(df,xlabel,title,fig_name):

    """Function for visualizing the missing values in the dataset.



    This module is used to visualize the missing values in the dataset.



    Attributes:

        df (dataFrame): Data frame of the available data.

        xlabel (string): label for the X- axis in the plot.

        title (string): Title for the plot.

        fig_name (string): Name of the figure to be saved.

    Return:

        None



    """

    fig, ax = plt.subplots(figsize=(15, 10))

    missing_values=df.isnull().sum()

    missing_values.plot.bar(ax=ax)

    for p in ax.patches:

        ax.annotate(p.get_height(), (p.get_x() + 0.10, p.get_height() + 0.15), rotation=0)

    plt.xticks(rotation=90)

    plt.xlabel(xlabel)

    plt.ylabel('Missing value count')

    plt.suptitle(title)

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.3)

    plt.show()

    return
# Function calling to visualize the missing valu counts in the  training data

visualize_missing_values(df_train, 'Features', 'Visualization for Missing value counts', 'Missing_value_visualization ')
def visualize_feature_distribution(df,feature,xlabel,title,fig_name,feature_type):

    """Function for visualizing the feature value distribution.



    This module is used to visualize feature distribution for all available features

    in the data.



    Attributes:

        df (dataFrame): Data frame of the available data.

        feature (string): feature name to be visualized.

        xlabel (string): label for the X- axis in the plot.

        title (string): Title for the plot.

        fig_name (string): Name of the figure to be saved.

        feature_type (string): Indication whether the feature is categorical or continuous.



    Return:

        None



    """

    

    # replacing Nan with the string 'missing_values'.

    df = df.replace(np.nan, 'missing_values', inplace=False)



    fig, ax = plt.subplots(figsize=(15, 10))

    if feature_type=='numerical':

        sns.distplot(df[feature])

    elif feature_type=='categorical':

        sns.countplot(df[feature],linewidth=0.04)

        for p in ax.patches:

            ax.annotate('{:.2%}'.format(p.get_height() / len(df[feature])),(p.get_x() + 0.40, p.get_height() + 0.8), rotation=0)



    plt.xticks(rotation=0)

    plt.xlabel(xlabel)

    plt.suptitle(title)

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.3)

    plt.show()

    return
# droping few columns which are not required for the visualization (not necessary to see the feature distribution)

columns_to_drop_visualization = ['Stammnummer', 'Anruf-ID','Tag','Anzahl Kontakte letzte Kampagne']

# selecting the numerical(continuous value features)

df_numerical = df_train.select_dtypes(include=['int64']).copy()

# fetching the feature names

num_col_names=df_numerical.columns.tolist()

for name in num_col_names:

    if name not in columns_to_drop_visualization:

       visualize_feature_distribution(df_train,name,'Feature values', 'Feature Distribution-'+name, 'Feature Distribution-'+name,'numerical')
# feature distribution of the categorical attributes.

df_categorical = df_train.select_dtypes(include=['object']).copy()

cat_col_names=df_categorical.columns.tolist()

for name in cat_col_names:

    visualize_feature_distribution(df_train,name,'Feature values', 'Feature Distribution-'+name, 'Feature Distribution-'+name,'categorical')

    
def plot_correlation_map(df,fig_name):

    """Function for visualizing thespearman correlation between the features and the target variable in the data set.



     This module is used for visualizing the correlation between the features in the data set.



     Attributes:

         df (dataFrame): Data frame of the available data.

         fig_name (string): Name of the figure to be saved.



     Return:

         None



     """

    # calculate the spearman correlation between the features

    corr = df.corr(method='spearman')

    _ , ax = plt.subplots( figsize =( 22 , 17 ))

    ax.set_title('Correlation matrix')

    # generate the heatmap for the visualization of thr correlation matrix.

    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )

    _ = sns.heatmap(

        corr,

        cmap = cmap,

        square=True,

        cbar_kws={'shrink': .9},

        ax=ax,

        annot = True,

        annot_kws = {'fontsize':10},

    )



    plt.xticks(rotation=90)

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)

    plt.show()

    return
def encode_categorical_features(df):

    """Function for encoding of categorical features.



       This module is used for encoding of categorical features.



       Attributes:

            df (dataFrame): Data frame of the available data.



        Return:

            df (dataFrame): data frame after label encoding.



          """

    # create label encoder object

    lb_make = LabelEncoder()

    # select all the features with type object (categorical)

    obj_df = df.select_dtypes(include=['object']).copy()

    cat_col_names=obj_df.columns.tolist()

    for name in cat_col_names:

        df[name] = lb_make.fit_transform(df[name])

    return df
# Dropping few columns for which correlation finding is not important and they have very high missing values

columns_to_drop_corr=['Stammnummer','Anruf-ID','Tage seit letzter Kampagne','Ergebnis letzte Kampagne']

df_corr=df_train.drop(columns_to_drop_corr,axis=1)

# To fill the missing values in the categorical features looking at the most frequent item in that feature and replacing nan values with that value.

df_corr = df_corr.apply(lambda x: x.fillna(x.value_counts().index[0]))

# convert the categorical features in to the numerical one by performing label encoding.

df_encoded=encode_categorical_features(df_corr)

# visualizing the correlation ma trix wtih all the features

plot_correlation_map(df_encoded,'Feature Correlation matrix')
# Finished Exploratory Analysis.
# Actual classification steps starts from here
# Based on the correlation analysis and by looking at the missing value counts removing few features from the classification task and encoding the categorical features

columns_to_drop=columns_to_drop = ['Stammnummer', 'Anruf-ID','Tage seit letzter Kampagne','Ergebnis letzte Kampagne','Tag','Alter','Kontostand','Anzahl der Ansprachen','Anzahl Kontakte letzte Kampagne']

df_train=df_train.drop(columns_to_drop,axis=1)

df_train = df_train.apply(lambda x: x.fillna(x.value_counts().index[0]))

df_train=encode_categorical_features(df_train)
# Standardizing the duration attribute using min-max scaling to normalize the values

df_train[['Dauer']] = df_train[['Dauer']].apply(np.sqrt)

scaler = MinMaxScaler(feature_range=(0, 1))

df_train[['Dauer']] = scaler.fit_transform(df_train[['Dauer']])

# Extracting all the feature and labels

train_features_all = df_train.drop(['Zielvariable'], axis=1, errors='ignore')

# labels for all feature columns.

train_labels_all =df_train.Zielvariable
def hyper_parameter_tuning(clf_model,Xtrain,ytrain,parameters):

    """Function for hyper parameter tuning.



       This module is used for hyper parameter tuning.



       Attributes:

            clf_model (model): Classification model

            Xtrain (Data Frame): Training samples(features) excluding target variable(label)

            ytrain (Data Frame):  Trining labels .

            parameters(dictionary): parameters that has to be tuned

        Return:

            model (model): Classification model fitted with the best possible hyper parameters.



          """



    # Type of scoring used to compare parameter combinations

    precision_scorer = make_scorer(precision_score)



    # Run the grid search

    grid_obj = GridSearchCV(estimator=clf_model, param_grid=parameters, scoring=precision_scorer,cv = 5, n_jobs = -1, verbose = 2)

    grid_obj = grid_obj.fit(Xtrain, ytrain)



    # Set the model to the best combination of parameters

    model = grid_obj.best_estimator_

    return model

def recursiveFeature_elimination(df,model):

    

    """Function for Recursive feature Elimination to filter the most appropriate festure to use based on the model.



       This module is used for Recursive feature Elimination.



       Attributes:

             model (model): Classification model

             df (Data Frame): Training samples(features) 

        Return:

            features_to_use(list): List of most important and appropriate features to use for the classification.



    """

    features = df.drop('Zielvariable', axis=1)

    label = df['Zielvariable']

    print('original dataset shape %s' % Counter(label))

    # SMOTE is basically sythetic Minority oversampling to balance the target variable distributioon and eliminate skewed class distribution

    sm = SMOTE(random_state=43,sampling_strategy='auto')

    features_resampled, labels_resampled =sm.fit_resample(features, label)

    print('Resampled dataset shape %s' % Counter(labels_resampled))



    # classifier

    clf = model

    RFE = RFECV(clf, step=1, cv=5,scoring='accuracy')

    RFE.fit(features_resampled,labels_resampled)

    support=RFE.support_

    feature_names = np.array(features.columns.tolist())

    features_to_use= feature_names[support].tolist()

    print(len(feature_names))

    print(len(features_to_use))

    print(features_to_use)

    return features_to_use

def run_kfold(model,X_all,y_all):

    """Function for validation of the classification model using  stratified k fold cross validation.



       This module is used for validation of the classification model using stratified k fold cross validation.



       Attributes:

            model (model): Classification model to be validated.

            X_all (Data Frame): Data frame containing only features excluding label

            y_all (Data Frame): Data frame containg only label

        Return:

            None.



          """

    # create K fold object

    kf = StratifiedKFold(n_splits=10, random_state=43, shuffle=True)

    # list to hold precision score.

    Precisionscore_list = []

    # list to hold accuracyscore.

    accuracyscore_list=[]

    # list to hold F1 scores (Harmonic mean of precision and recall)

    F1_score_list= []

    # initialize the fold to 0

    fold = 0

    for train_index, test_index in kf.split(X_all,y_all):

        fold += 1

        X_train, X_test = X_all.values[train_index], X_all.values[test_index]

        y_train, y_test = y_all.values[train_index], y_all.values[test_index]

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        # calculate precision, F1 score and accuracy for each fold.

        precisionscore = precision_score(y_test, predictions)

        accuracy =  accuracy_score(y_test, predictions)

        F1score=  f1_score(y_test,predictions)

        Precisionscore_list.append(precisionscore)

        accuracyscore_list.append(accuracy)

        F1_score_list.append(F1score)

        print("Fold {0} precision_score: {1}".format(fold, precisionscore))

        print("Fold {0} accuracy_score: {1}".format(fold, accuracy))

        print("Fold {0} F1_score: {1}".format(fold, F1score))

    # calculate mean for all the scores.

    mean_precisionscore = np.mean(Precisionscore_list)

    mean_accuracy = np.mean(accuracyscore_list)

    mean_F1= np.mean(F1_score_list)

    # print the mean values of the scores

    print("Mean precision_score: {0}".format(mean_precisionscore))

    print("Mean accuracy_score: {0}".format(mean_accuracy))

    print("Mean F1_score: {0}".format(mean_F1))

    return

# Case-1: Model- Logistic Regressin Classifier
# for logistic regression initialization

classification_model=LogisticRegression(random_state=1)



# Choose  parameter combinations to try to hyper-parameter tuning

parameters = { 'C': [0.8,0.9,1.0,1.2],

                'class_weight': [None, 'balanced'],

                'solver': ['liblinear','sag', 'saga'],

            }

# find the best possible model using hyper-parameter tuning function

best_model=hyper_parameter_tuning(classification_model,train_features_all,train_labels_all,parameters)

# find out the appropriate features to use for logistic regression model using recursive feature elimination

features_to_use=recursiveFeature_elimination(df_train,best_model)



# filter the training dataframe using the best features.

df_filtered_features = df_train[features_to_use]

# labels for all feature columns.

df_filtered_label =df_train.Zielvariable



# validation of the best logistic regression model using kfold cross validation.

run_kfold(best_model,df_filtered_features,df_filtered_label)



# Applying SMOTE Resampling technique to balance the target classes.

sm_obj = SMOTE(random_state=43,sampling_strategy='auto')

features_resampled, labels_resampled =sm_obj.fit_resample(df_filtered_features, df_filtered_label)

best_model.fit(features_resampled,labels_resampled)





# Prediction of the test data

# Make the test data as equivalent to the traning data by removing the non important features and encodeing the categorical features.

#df_test=test_data.drop(columns_to_drop,axis=1)

df_test= test_data[features_to_use]

df_test= encode_categorical_features(df_test)

if 'Dauer' in df_test.columns:

    # Normalize the Dauer feature 

    df_test[['Dauer']] = df_test[['Dauer']].apply(np.sqrt)

    scaler = MinMaxScaler(feature_range=(0, 1))

    df_test[['Dauer']] = scaler.fit_transform(df_test[['Dauer']])



test_features_all = df_test

# labels for all feature columns.

test_labels_all =test_data.Zielvariable





# Prediction Phase

# prediction of the target variables

predictions=best_model.predict(test_features_all)

# Prediction of probabilities for each target classes

prediction_proba =best_model.predict_proba(test_features_all)



# Reading the probabilities of target variable value(0- positive class(Ja))

pos_label_proba= prediction_proba[:,0]







# Finding AUC for ROC curve for positive label 0

y = predictions

pred = pos_label_proba

fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=0)

AUC_score= metrics.auc(fpr, tpr)



# preparing the submission file for logistic regression

df_submission=pd.DataFrame()

df_submission['ID']=submission_data['ID']

df_submission['Expected']=pos_label_proba

df_submission.to_csv('Logistc_Regression_prediction.csv',index=False)

# Case-2: Model- GradientBoosting Classifier
# for  initialization

classification_model=GradientBoostingClassifier(random_state=1)



# Choose  parameter combinations to try to hyper-parameter tuning

parameters = { 'n_estimators': [100, 200, 300],

                'max_features': ['sqrt','auto'],

                'max_depth': [2, 3, 5],

                }



# find the best possible model using hyper-parameter tuning function

best_model=hyper_parameter_tuning(classification_model,train_features_all,train_labels_all,parameters)

# find out the appropriate features to use for gradient boosting model using recursive feature elimination

features_to_use=recursiveFeature_elimination(df_train,best_model)



# filter the training dataframe using the best features.

df_filtered_features = df_train[features_to_use]

# labels for all feature columns.

df_filtered_label =df_train.Zielvariable



# validation of the best gradient boosting model using kfold cross validation.

run_kfold(best_model,df_filtered_features,df_filtered_label)



# Applying SMOTE Resampling technique to balance the target classes.

sm_obj = SMOTE(random_state=43,sampling_strategy='auto')

features_resampled, labels_resampled =sm_obj.fit_resample(df_filtered_features, df_filtered_label)

best_model.fit(features_resampled,labels_resampled)





# Prediction of the test data

# Make the test data as equivalent to the traning data by removing the non important features and encodeing the categorical features.

#df_test=test_data.drop(columns_to_drop,axis=1)

df_test= test_data[features_to_use]

df_test= encode_categorical_features(df_test)

if 'Dauer' in df_test.columns:

    # Normalize the Dauer feature 

    df_test[['Dauer']] = df_test[['Dauer']].apply(np.sqrt)

    scaler = MinMaxScaler(feature_range=(0, 1))

    df_test[['Dauer']] = scaler.fit_transform(df_test[['Dauer']])



test_features_all = df_test

# labels for all feature columns.

test_labels_all =test_data.Zielvariable





# Prediction Phase

# prediction of the target variables

predictions=best_model.predict(test_features_all)

# Prediction of probabilities for each target classes

prediction_proba =best_model.predict_proba(test_features_all)



# Reading the probabilities of target variable value(0- positive class(Ja))

pos_label_proba= prediction_proba[:,0]







# Finding AUC for ROC curve for positive label 0

y = predictions

pred = pos_label_proba

fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=0)

AUC_score= metrics.auc(fpr, tpr)

AUC_score

# preparing the submission file for Gradient boosting 

df_submission=pd.DataFrame()

df_submission['ID']=submission_data['ID']

df_submission['Expected']=pos_label_proba

df_submission.to_csv('Gradient_Boosting_prediction.csv',index=False)

# case-3: Decision Tree classifier
# for Decision Tree classifier initialization

classification_model=DecisionTreeClassifier(random_state=1)



# Choose  parameter combinations to try to hyper-parameter tuning

parameters = { 'class_weight': [None, 'balanced'],

                'max_features': ['sqrt','auto','log2'],

                'max_depth': [None, 2, 3, 5],

                }



# find the best possible model using hyper-parameter tuning function

best_model=hyper_parameter_tuning(classification_model,train_features_all,train_labels_all,parameters)

# find out the appropriate features to use for Decision Tree classifier model using recursive feature elimination

features_to_use=recursiveFeature_elimination(df_train,best_model)



# filter the training dataframe using the best features.

df_filtered_features = df_train[features_to_use]

# labels for all feature columns.

df_filtered_label =df_train.Zielvariable



# validation of the best Decision Tree classifier model using kfold cross validation.

run_kfold(best_model,df_filtered_features,df_filtered_label)



# Applying SMOTE Resampling technique to balance the target classes.

sm_obj = SMOTE(random_state=43,sampling_strategy='auto')

features_resampled, labels_resampled =sm_obj.fit_resample(df_filtered_features, df_filtered_label)

best_model.fit(features_resampled,labels_resampled)





# Prediction of the test data

# Make the test data as equivalent to the traning data by removing the non important features and encodeing the categorical features.

#df_test=test_data.drop(columns_to_drop,axis=1)

df_test= test_data[features_to_use]

df_test= encode_categorical_features(df_test)

if 'Dauer' in df_test.columns:

    # Normalize the Dauer feature 

    df_test[['Dauer']] = df_test[['Dauer']].apply(np.sqrt)

    scaler = MinMaxScaler(feature_range=(0, 1))

    df_test[['Dauer']] = scaler.fit_transform(df_test[['Dauer']])



test_features_all = df_test

# labels for all feature columns.

test_labels_all =test_data.Zielvariable





# Prediction Phase

# prediction of the target variables

predictions=best_model.predict(test_features_all)

# Prediction of probabilities for each target classes

prediction_proba =best_model.predict_proba(test_features_all)



# Reading the probabilities of target variable value(0- positive class(Ja))

pos_label_proba= prediction_proba[:,0]



# Finding AUC for ROC curve for positive label 0

y = predictions

pred = pos_label_proba

fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=0)

AUC_score= metrics.auc(fpr, tpr)

# preparing the submission file for Gradient boosting 

df_submission=pd.DataFrame()

df_submission['ID']=submission_data['ID']

df_submission['Expected']=pos_label_proba

df_submission.to_csv('Decision_tree_prediction.csv',index=False)



# case:4- Random Forest Classifier
# for Decision Tree classifier initialization

classification_model=RandomForestClassifier(random_state=1)



# Choose  parameter combinations to try to hyper-parameter tuning

parameters = { 'class_weight': [None, 'balanced'],

                'n_estimators': [100, 200, 300],

                'max_features': ['sqrt','auto'],

                'max_depth': [None, 2, 3, 5],

                }



# find the best possible model using hyper-parameter tuning function

best_model=hyper_parameter_tuning(classification_model,train_features_all,train_labels_all,parameters)

# find out the appropriate features to use for Random Forest Classifier model using recursive feature elimination

features_to_use=recursiveFeature_elimination(df_train,best_model)



# filter the training dataframe using the best features.

df_filtered_features = df_train[features_to_use]

# labels for all feature columns.

df_filtered_label =df_train.Zielvariable



# validation of the bestRandom Forest Classifier model using kfold cross validation.

run_kfold(best_model,df_filtered_features,df_filtered_label)



# Applying SMOTE Resampling technique to balance the target classes.

sm_obj = SMOTE(random_state=43,sampling_strategy='auto')

features_resampled, labels_resampled =sm_obj.fit_resample(df_filtered_features, df_filtered_label)

best_model.fit(features_resampled,labels_resampled)





# Prediction of the test data

# Make the test data as equivalent to the traning data by removing the non important features and encodeing the categorical features.

#df_test=test_data.drop(columns_to_drop,axis=1)

df_test= test_data[features_to_use]

df_test= encode_categorical_features(df_test)

if 'Dauer' in df_test.columns:

    # Normalize the Dauer feature 

    df_test[['Dauer']] = df_test[['Dauer']].apply(np.sqrt)

    scaler = MinMaxScaler(feature_range=(0, 1))

    df_test[['Dauer']] = scaler.fit_transform(df_test[['Dauer']])



test_features_all = df_test

# labels for all feature columns.

test_labels_all =test_data.Zielvariable





# Prediction Phase

# prediction of the target variables

predictions=best_model.predict(test_features_all)

# Prediction of probabilities for each target classes

prediction_proba =best_model.predict_proba(test_features_all)



# Reading the probabilities of target variable value(0- positive class(Ja))

pos_label_proba= prediction_proba[:,0]



# Finding AUC for ROC curve for positive label 0

y = predictions

pred = pos_label_proba

fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=0)

AUC_score= metrics.auc(fpr, tpr)

# preparing the submission file for Gradient boosting 

df_submission=pd.DataFrame()

df_submission['ID']=submission_data['ID']

df_submission['Expected']=pos_label_proba

df_submission.to_csv('Random_Forest_Classifier_prediction.csv',index=False)



    