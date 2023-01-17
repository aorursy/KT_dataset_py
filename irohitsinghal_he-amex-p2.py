# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
header = pd.read_csv('../input/header.csv')
train = pd.read_csv('../input/train.csv',names=header.columns)
test = pd.read_csv('../input/test.csv', names=header.columns[:-1])
sample = pd.read_csv('../input/sample_submission.csv')
test_key = test['key']
test.drop(columns=['key'], inplace=True)
train.drop(columns=['key'], inplace=True)
train.head()
# Function to calculate missing values by column# Funct 
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns
print(missing_values_table(train), '\n')
missing_values_table(test)
def numRel(feature):
    exp = train[[feature, 'label']]
    exp=exp.groupby(feature).label.agg(['count', 'sum']).reset_index()
    exp['label']=exp['sum']/exp['count']
    exp.plot(x=feature, y='label', marker='.')
def kdeRel(feature):
    plt.figure(figsize = (5, 4))

    # KDE plot for area_assesed_Building removed
    sns.kdeplot(train.loc[train['label'] == 0, feature], label = 'label == 0')
    # KDE plot for area_assesed_Building removed
    sns.kdeplot(train.loc[train['label'] == 1, feature], label = 'label == 1')

    # Labeling of plot
    plt.xlabel(feature); plt.ylabel('label'); plt.title('Matrix');
numRel('V1')
kdeRel('V3')
exp = train[['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'label']]
train.corr()['label'].sort_values(ascending=False).head(10)
train.corr()['label'].sort_values(ascending=True).head(10)
train['v10v26'] = train['V10'] * train['V26']
test['v10v26'] = test['V10'] * test['V26']

train['v1v11'] = train['V1'] * train['V11']
test['v1v11'] = test['V1'] * test['V11']

train['v1v14'] = train['V1'] * train['V14']
test['v1v14'] = test['V1'] * test['V14']

train['v6v11'] = train['V6'] * train['V11']
test['v6v11'] = test['V6'] * test['V11']

train['v10v11'] = train['V10'] * train['V11']
test['v10v11'] = test['V10'] * test['V11']
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

train_labels = train['label'].astype(float)
# Drop the target from the training data
train.drop(columns = ['label'], inplace=True)
    
# Feature names
features = list(train.columns)

# Scale each feature to 0-1
scaler = MinMaxScaler(feature_range = (0, 1))

# with the scaler
scaler.fit(train)
strain = scaler.transform(train)
stest = scaler.transform(test)
train['label'] = train_labels
print('Training data shape: ', strain.shape)
print('Testing data shape: ', stest.shape)
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor

# Make the random forest classifier
# clf = RandomForestClassifier(n_estimators = 100, random_state = 50, verbose = 1, n_jobs = -1)

# Make the model with the specified regularization parameter
# clf = LogisticRegression(C = 0.0001, n_jobs=-1)

#Use XGBooster
clf = XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7)
from sklearn.metrics import roc_auc_score
from sklearn import model_selection

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    strain, train_labels, test_size=0.30)
# Train on the training data
clf.fit(X_train, y_train, early_stopping_rounds=5, 
             eval_set=[(X_test, y_test)], verbose=True)
# clf.fit(X_train, y_train)

# roc_auc_score(y_test, clf.predict(X_test), average='weighted')
# clf.score(X_test, y_test)
def plot_feature_importances(df):
    """
    Plot importances returned by a model. This can work with any measure of
    feature importance provided that higher importance is better. 
    
    Args:
        df (dataframe): feature importances. Must have the features in a column
        called `features` and the importances in a column called `importance
        
    Returns:
        shows a plot of the 15 most importance features
        
        df (dataframe): feature importances sorted by importance (highest to lowest) 
        with a column for normalized importance
        """
    
    # Sort features according to importance
    df = df.sort_values('importance', ascending = False).reset_index()
    
    # Normalize the feature importances to add up to one
    df['importance_normalized'] = df['importance'] / df['importance'].sum()

    # Make a horizontal bar chart of feature importances
    plt.figure(figsize = (10, 6))
    ax = plt.subplot()
    
    # Need to reverse the index to plot most important on top
    ax.barh(list(reversed(list(df.index[:15]))), 
            df['importance_normalized'].head(15), 
            align = 'center', edgecolor = 'k')
    
    # Set the yticks and labels
    ax.set_yticks(list(reversed(list(df.index[:15]))))
    ax.set_yticklabels(df['feature'].head(15))
    
    # Plot labeling
    plt.xlabel('Normalized Importance'); plt.title('Feature Importances')
    plt.show()
    
    return df
# Extract feature importances
feature_importance_values = clf.feature_importances_
feature_importances = pd.DataFrame({'feature': features, 'importance': feature_importance_values})
# Show the feature importances for the default features
feature_importances_sorted = plot_feature_importances(feature_importances)
#Prediction with classifier
y=clf.predict(stest)
prediction=pd.DataFrame({'key': test_key, 'score':y})
prediction.to_csv('submission.csv', index=False)
score = [round(i) for i in sample['score'].tolist()]
pred_score = [round(i) for i in y[:20]]
print("Score: ", roc_auc_score(score, pred_score, average='weighted'))
prediction.head(20)
sample