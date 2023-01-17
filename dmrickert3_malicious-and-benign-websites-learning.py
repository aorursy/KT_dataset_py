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
# Print out a quick overview of the data
dataset=pd.read_csv("../input/dataset.csv")
dataset.head()
# Quick statistical summary of data
dataset.describe(include='all')
# Drop the URL column since that is a unique column for training
dataset.drop('URL', axis=1, inplace=True)
# Take a look at any null values to clean up data
# Likely need to do something with these empty datasets
print(dataset.isnull().sum())
dataset[pd.isnull(dataset).any(axis=1)]
# Interpolate our data to get rid of null values
dataset = dataset.interpolate()
print(dataset.isnull().sum())
# For some reason there's still a isnull in the SERVER column
dataset['SERVER'].fillna('RARE_VALUE', inplace=True)
dataset.describe(include='all')
# Convert categorical columns to numbered categorical columns
dataset_with_dummies = pd.get_dummies(dataset,prefix_sep='--')
print(dataset_with_dummies.head())
# Separate predictors and response
X = dataset_with_dummies.drop('Type',axis=1) #Predictors
y = dataset_with_dummies['Type']

X.head()
# Get a training and test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Train a Random Forest Regressor
from sklearn.ensemble import RandomForestClassifier

# n_estimators is the number of random forests to use
# n_jobs says to use all processors available
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, max_depth=30, criterion = 'entropy')
rf.fit(X_train, y_train)

print('Training Accuracy Score: {}'.format(rf.score(X_train, y_train)))
y_pred = rf.predict(X_test)
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

# Visualize our results
def print_score(classifier,X_train,y_train,X_test,y_test,train=True):
    if train == True:
        print("Training results:\n")
        print('Accuracy Score: {0:.4f}\n'.format(accuracy_score(y_train,classifier.predict(X_train))))
        print('Classification Report:\n{}\n'.format(classification_report(y_train,classifier.predict(X_train))))
        print('Confusion Matrix:\n{}\n'.format(confusion_matrix(y_train,classifier.predict(X_train))))
        res = cross_val_score(classifier, X_train, y_train, cv=10, n_jobs=-1, scoring='accuracy')
        print('Average Accuracy:\t{0:.4f}\n'.format(res.mean()))
        print('Standard Deviation:\t{0:.4f}'.format(res.std()))
    elif train == False:
        print("Test results:\n")
        print('Accuracy Score: {0:.4f}\n'.format(accuracy_score(y_test,classifier.predict(X_test))))
        print('Classification Report:\n{}\n'.format(classification_report(y_test,classifier.predict(X_test))))
        print('Confusion Matrix:\n{}\n'.format(confusion_matrix(y_test,classifier.predict(X_test))))

print_score(rf,X_train,y_train,X_test,y_test,train=False)
from sklearn.tree import export_graphviz

def create_graph(forest, feature_names):
    estimator = forest.estimators_[5]

    export_graphviz(estimator, out_file='tree.dot',
                    feature_names = feature_names,
                    class_names = ['benign', 'malicious'],
                    rounded = True, proportion = False, precision = 2, filled = True)

    # Convert to png using system command (requires Graphviz)
    from subprocess import call
    call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=200'])
    
create_graph(rf, list(X))
# Display in jupyter notebook
from IPython.display import Image
Image(filename = 'tree.png')

# View our feature importances
feature_importance_zip = zip(list(X), rf.feature_importances_)

# Sort the feature_importance_zip
sorted_importance = sorted(feature_importance_zip, key=lambda x: x[1], reverse=True)

for feature in sorted_importance[:15]:
    print(feature)
# Look at our feature importances without the dummy variables
original_feature_dict = {}
for feature, importance in zip(list(X), rf.feature_importances_):
    # Check for our dummy variable delimeter --
    if '--' in feature:
        original_feature_name = feature.split('--')[0]
    else:
        original_feature_name = feature
        
    # Add to our original_feature_dict, incrememnt if it's already there
    if original_feature_name in original_feature_dict:
        original_feature_dict[original_feature_name] += importance
    else:
        original_feature_dict[original_feature_name] = importance
      
# Sort the original_feature_dict
sorted_importance = sorted(original_feature_dict.items(), key=lambda x: x[1], reverse=True)

for feature, importance in sorted_importance:
    print(feature, importance)
    
# Copy over our data so we don't overwrite the original results if we want
#    to tweak above
dataset_preprocessed = dataset

# Converting server types with only 1 unique count to a RARE_VALUE for classification
test = dataset_preprocessed ['SERVER'].value_counts()
col = 'SERVER'
dataset_preprocessed.loc[dataset_preprocessed[col].value_counts()[dataset_preprocessed[col]].values < 2, col] = "RARE_VALUE"
# Function to extract the registration year
def extract_reg_year(x):
    # If no year was reported, leave it as None
    if str(x) == 'None':
        return(x)
    
    # Try different parses for different date formats
    parse_error = False
    try:
        date = x.split(' ')[0]
        year = date.split('/')[2]
    except:
        parse_error = True
    
    # One more date format to try if there's a parse error
    if parse_error:
        try:
            date = x.split('T')[0]
            year = date.split('-')[0]
            parse_error = False
        except:
            parse_error = True
            raise ValueError('Error parsing {}'.format(x))

    return(year)

dataset_preprocessed['WHOIS_REGDATE'] = dataset_preprocessed['WHOIS_REGDATE'].apply(extract_reg_year)
dataset_preprocessed['WHOIS_UPDATED_DATE'] = dataset_preprocessed['WHOIS_UPDATED_DATE'].apply(extract_reg_year)
# State without country doesn't make sense
dataset_preprocessed['WHOIS_STATEPRO'] = dataset_preprocessed[['WHOIS_COUNTRY','WHOIS_STATEPRO']].apply(lambda x : '{}-{}'.format(x[0],x[1]), axis=1)
dataset_preprocessed.describe(include='all')
# As above, create our random forest classifier the same way

# Convert categorical columns to numbered categorical columns
dataset_pp_with_dummies = pd.get_dummies(dataset_preprocessed, prefix_sep='--')

# Separate predictors and response
X_pp = dataset_pp_with_dummies.drop('Type',axis=1) #Predictors
y_pp = dataset_pp_with_dummies['Type']

# Get a training and test dataset
from sklearn.model_selection import train_test_split
X_pp_train, X_pp_test, y_pp_train, y_pp_test = train_test_split(X_pp, y_pp, test_size=0.3, random_state=42)

# Train a Random Forest Regressor
from sklearn.ensemble import RandomForestClassifier

# n_estimators is the number of random forests to use
# n_jobs says to use all processors available
# Properties we can play with for the RandomForestClassifier function:
#    max_depth=int( default None, n_estimators=int(default 10), min_samples_split=int(default 2)
rf_pp = RandomForestClassifier(n_estimators=100, n_jobs=-1, max_depth=30, criterion = 'entropy')
rf_pp.fit(X_pp_train, y_pp_train)

print('Training Accuracy Score: {}'.format(rf_pp.score(X_pp_train, y_pp_train)))
y_pp_pred = rf_pp.predict(X_pp_test)
print_score(rf_pp,X_pp_train,y_pp_train,X_pp_test,y_pp_test,train=False)
# View our feature importances
feature_importance_zip = zip(list(X_pp), rf_pp.feature_importances_)

# Sort the feature_importance_zip
sorted_importance = sorted(feature_importance_zip, key=lambda x: x[1], reverse=True)

for feature in sorted_importance[:15]:
    print(feature)
# Look at our feature importances without the dummy variables
original_feature_dict = {}
for feature, importance in zip(list(X_pp), rf_pp.feature_importances_):
    # Check for our dummy variable delimeter --
    if '--' in feature:
        original_feature_name = feature.split('--')[0]
    else:
        original_feature_name = feature
        
    # Add to our original_feature_dict, incrememnt if it's already there
    if original_feature_name in original_feature_dict:
        original_feature_dict[original_feature_name] += importance
    else:
        original_feature_dict[original_feature_name] = importance
      
# Sort the original_feature_dict
sorted_importance = sorted(original_feature_dict.items(), key=lambda x: x[1], reverse=True)

for feature, importance in sorted_importance:
    print(feature, importance)
create_graph(rf_pp, list(X_pp))
# Display in jupyter notebook
from IPython.display import Image
Image(filename = 'tree.png')