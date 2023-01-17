# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # data visualization library  
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from time import time
from IPython.display import display # Allows the use of display() for DataFrames

# Pretty display for notebooks
%matplotlib inline



import os
print(os.listdir("../input"))

# Load the Census dataset
data = pd.read_csv("../input/ntl-data/NTL_Data.csv")
test_new_data = pd.read_csv("../input/new-test-data/New_data.csv")

# Success - Display the first record
#display(data.head(n=1))

display(data.shape)
# Any results you write to the current directory are saved as output.
# Split the data into features and target label
Graduated_raw = data['Graduated']
features_raw = data.drop('Graduated', axis = 1)


from sklearn.preprocessing import LabelEncoder 
lb_make = LabelEncoder()

graduated = lb_make.fit_transform(Graduated_raw)

from sklearn.model_selection import train_test_split

# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_raw, 
                                                    graduated, 
                                                    test_size = 0.2, 
                                                    random_state = 0)

# Show the results of the split
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))
X_train
test_new_data_X = pd.DataFrame(test_new_data)
type(test_new_data_X)
all_data = pd.concat((X_train,X_test,test_new_data_X))

#for column in all_data.select_dtypes(include=[np.object]).columns:
for column in all_data.columns:
    #print(column, all_data[column].unique())
    X_train[column] = X_train[column].astype('category', categories = all_data[column].unique())
    X_test[column] = X_test[column].astype('category', categories = all_data[column].unique())
    test_new_data_X[column] = test_new_data_X[column].astype('category', categories = all_data[column].unique())
print(X_train.shape)
print(X_test.shape)
print(test_new_data_X.shape)
from sklearn.preprocessing import LabelEncoder
X_train_sklearn = X_train.copy()
lb_make = LabelEncoder()
X_train_sklearn['Governorate'] = lb_make.fit_transform(X_train['Governorate'])
X_train_sklearn['Learner Profile'] = lb_make.fit_transform(X_train['Learner Profile'])
X_train_sklearn['University'] = lb_make.fit_transform(X_train['University'])
X_train_sklearn['Cons.'] = lb_make.fit_transform(X_train['Cons.'])
X_train_sklearn['Track'] = lb_make.fit_transform(X_train['Track'])
X_train_sklearn.head()
from sklearn.preprocessing import LabelEncoder
X_test_sklearn = X_test.copy()
lb_make = LabelEncoder()
X_test_sklearn['Governorate'] = lb_make.fit_transform(X_test['Governorate'])
X_test_sklearn['Learner Profile'] = lb_make.fit_transform(X_test['Learner Profile'])
X_test_sklearn['University'] = lb_make.fit_transform(X_test['University'])
X_test_sklearn['Cons.'] = lb_make.fit_transform(X_test['Cons.'])
X_test_sklearn['Track'] = lb_make.fit_transform(X_test['Track'])
X_test_sklearn.head()
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()
X_train_scaled = np.array(X_train_sklearn)

X_train_scaled = min_max_scaler.fit_transform(X_train_scaled)

#df = pandas.DataFrame(X_train_scaled)

X_train_scaled = pd.DataFrame(X_train_scaled)
X_test_scaled = np.array(X_test_sklearn)
X_test_scaled = min_max_scaler.fit_transform(X_test_scaled)
train_X = pd.get_dummies(X_train)
test_X = pd.get_dummies(X_test)
test_new_X = pd.get_dummies(test_new_data_X)
print(train_X.shape)
print(test_X.shape)
print(test_new_X.shape)
print(y_train.shape)
print(y_test.shape)
from sklearn.metrics import fbeta_score, accuracy_score
def train_predict(learner, sample_size, train_X, y_train, test_X, y_test): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''
    
    results = {}
    
    # TODO: Fit the learner to the training data using slicing with 'sample_size' using .fit(training_features[:], training_labels[:])
    start = time() # Get start time
    learner = learner.fit(train_X[:sample_size], y_train[:sample_size])
    end = time() # Get end time
    
    # TODO: Calculate the training time
    results['train_time'] = end - start
        
    # TODO: Get the predictions on the test set(X_test),
    #       then get predictions on the first 300 training samples(X_train) using .predict()
    start = time() # Get start time
    predictions_test = learner.predict(test_X)
    predictions_train = learner.predict(train_X[:300])
    end = time() # Get end time
    
    # TODO: Calculate the total prediction time
    results['pred_time'] = end - start
            
    # TODO: Compute accuracy on the first 300 training samples which is y_train[:300]
    results['acc_train'] = accuracy_score(y_train[:300], predictions_train)
        
    # TODO: Compute accuracy on test set using accuracy_score()
    results['acc_test'] = accuracy_score(y_test, predictions_test)
    
    # TODO: Compute F-score on the the first 300 training samples using fbeta_score()
    results['f_train'] = fbeta_score(y_train[:300], predictions_train, beta = 0.5)
        
    # TODO: Compute F-score on the test set which is y_test
    results['f_test'] = fbeta_score(y_test, predictions_test, beta = 0.5)
       
    # Success
    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))
        
    # Return the results
    return results
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# TODO: Initialize the three models
clf_A = KNeighborsClassifier()
clf_B = DecisionTreeClassifier(random_state=0)
clf_C = LogisticRegression(random_state=0)

print('The scikit-learn version is {}.'.format(sklearn.__version__))

# TODO: Calculate the number of samples for 1%, 10%, and 100% of the training data
# HINT: samples_100 is the entire training set i.e. len(y_train)
# HINT: samples_10 is 10% of samples_100 (ensure to set the count of the values to be `int` and not `float`)
# HINT: samples_1 is 1% of samples_100 (ensure to set the count of the values to be `int` and not `float`)
samples_100 = len(y_train)
samples_10 = int(0.1 * len(y_train))
samples_1 = int(0.01 * len(y_train))
#print(samples_1)

# Collect results on the learners
results = {}
for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] = \
        train_predict(clf, samples, train_X, y_train, test_X, y_test)

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

# TODO: Initialize the classifier
clf = LogisticRegression(random_state=0)

# TODO: Create the parameters list you wish to tune, using a dictionary if needed.
# HINT: parameters = {'parameter_1': [value1, value2], 'parameter_2': [value1, value2]}
parameters = {'penalty': ['l1', 'l2'], 'max_iter': [100, 200]}

# TODO: Make an fbeta_score scoring object using make_scorer()
scorer = make_scorer(fbeta_score, beta=0.5)

# TODO: Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV()
grid_obj = GridSearchCV(clf,parameters,scorer)

# TODO: Fit the grid search object to the training data and find the optimal parameters using fit()
grid_fit = grid_obj.fit(train_X,y_train)

# Get the estimator
best_clf = grid_fit.best_estimator_

# Make predictions using the unoptimized and model
predictions = (clf.fit(train_X, y_train)).predict(test_X)
best_predictions = best_clf.predict(test_X)

# Report the before-and-afterscores
print("Unoptimized model\n------")
print("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 0.5)))
print("\nOptimized Model\n------")
print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))
# TODO: Import a supervised learning model that has 'feature_importances_'
from sklearn.ensemble import RandomForestClassifier

# TODO: Train the supervised model on the training set using .fit(X_train, y_train)
model = RandomForestClassifier()
model.fit(X_train_scaled, y_train)

# TODO: Extract the feature importances using .feature_importances_ 
#importances = model.feature_importances_

# Plot
#vs.feature_plot(importances, X_train, y_train)
predictions = pd.DataFrame(best_clf.predict(test_new_X))

predictions.to_csv('submission.csv')
from IPython.display import HTML
import base64

def create_download_link( df, title = "Download CSV file", filename = "data.csv"):  
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = f'<a target="_blank">{title}</a>'
    return HTML(html)
create_download_link(predictions)
comat = X_train_scaled.corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(comat, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(comat, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
y_train_scaled = pd.DataFrame(y_train)
sns.scatterplot(X_train_scaled[1], X_train_scaled[3])
