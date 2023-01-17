# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from IPython.display import display # Allows the use of display() for DataFrames
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import fbeta_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.preprocessing import Imputer 
from sklearn.base import TransformerMixin

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Reading data 

# Load the Census dataset
data = pd.read_csv("../input/census.csv")

# Success - Display the first record
display(data.head(n=1))
#Data exploration

#Total number of records
n_records = data.shape [0]

#Number of records where individual's income is more than $50,000
n_greater_50k = data [data ['income']== '>50K'].shape [0]

#Number of records where individual's income is at most $50,000
n_at_most_50k = data [data ['income']== '<=50K'].shape [0]

#Percentage of individuals whose income is more than $50,000
greater_percent = (n_greater_50k/n_records)*100

# Print the results
print("Total number of records: {}".format(n_records))
print("Individuals making more than $50,000: {}".format(n_greater_50k))
print("Individuals making at most $50,000: {}".format(n_at_most_50k))
print("Percentage of individuals making more than $50,000: {}%".format(greater_percent))
#Function for data pre-processing

def data_pros (df):
    
    # Log-transform the skewed features
    skewed = ['capital-gain', 'capital-loss']
    features_log_transformed = pd.DataFrame(data = df)
    features_log_transformed[skewed] = df[skewed].apply(lambda x: np.log(x + 1))
    
    #Normalize numerical features
    # Initialize a scaler, then apply it to the features
    scaler = MinMaxScaler() # default=(0, 1)
    numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

    features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)
    features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])
    
    # One-hot encode the 'features_log_minmax_transform' data using pandas.get_dummies()
    features_final = pd.get_dummies (features_log_minmax_transform)
    return features_final
#Data pre-processing

# Split the data into features and target label
income_raw = data['income']
features_raw = data.drop('income', axis = 1)

#runing a function to receive processed data
features_final = data_pros (features_raw)

# Encode the 'income_raw' data to numerical values
mapping = {'<=50K': 0, '>50K': 1}
income = pd.DataFrame ([mapping [item] for item in income_raw]) 


#Shuffle and split data

# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_final, 
                                                    income, 
                                                    test_size = 0.2, 
                                                    random_state = 0)
clf = GradientBoostingClassifier (learning_rate = 0.09, max_depth = 6, min_samples_split = 100, n_estimators = 300, 
                                  warm_start=True, random_state = 42, subsample = 0.8, max_features = 40, 
                                  min_samples_leaf =40)

predictions = (clf.fit(X_train, y_train)).predict(X_test)

# Report scores from the model on Test data
print("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 0.5)))
#load test data
test = pd.read_csv("../input/test_census.csv")
display(test.head(n=1))
#renaming first column from test dataset
new_columns = test.columns.values
new_columns[0] = 'id'
test.columns = new_columns

display(test.head(n=1))
# Storing id into separate df 
test_id = test['id']
features_test = test.drop('id', axis = 1)
display(features_test.head(n=1))
#Evaluating number of values missing per column in percentage from all dataset
nat_miss = (features_test.isnull().sum () / (features_test.shape [0]))*100
print ('Missing per column')
display (nat_miss)
#Applying imputation for missing values using sklearn. 
#Non-numeric missing values will be replaced by most frequent, while others will be replaced by median 
class DFImputer (TransformerMixin):
    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with median values 
        in column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].median() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)
#Imputation 
test_im = pd.DataFrame (DFImputer ().fit_transform (features_test)) #preparing new datafram with imputed values.
test_im.columns = features_test.columns #restoring column names
test_im.index = features_test.index #restoring indeces
display (test_im.head (n=2))
#Test data pre-processing

X_test_final = data_pros (test_im)
display (X_test_final.head (n=2))

#train model on the whole dataset
clf.fit(features_final, income)

#Make predictions
test['income'] = clf.predict(X_test_final)
display (test.head (n=5))
# generate output file
test[['id', 'income']].to_csv("submission.csv", index=False)

