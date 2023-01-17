# Importing the libraries:



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_recall_fscore_support as error_metric

from sklearn.metrics import confusion_matrix, accuracy_score

import os
# Loading the training and testing set:



print(os.listdir("../input"))

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
# Getting a look at our training data:



train.head()
# let us also check for null values in the training and test sets:



print(train.isnull().values.any())

print(test.isnull().values.any())
# Now let us remove the columns that are not necessary from both the training as well as the test set:



train.drop('subject', axis = 1, inplace = True)

test.drop('subject', axis = 1, inplace = True)
# Let us check the datatype we are dealing with in both the training and testing set:



print(train.dtypes.value_counts())

print(test.dtypes.value_counts())
# Getting a better sense of our training data:



train.describe()
# Now we will try adn find the columns which have data type object because we need to one hot encode them:





object_feature = train.dtypes == np.object

object_feature = train.columns[object_feature]

object_feature



# Thus we observe that only one column that is 'Activity' is of data type object and thus we will one hot encode it.

train['Activity']
# Label Encoding the 'Activity' column:



label_encoder = LabelEncoder()

for x in [train, test]:

    x['Activity'] = label_encoder.fit_transform(x.Activity)
# Checking if our label encoding worked as expected:



train.Activity.sample(5)
# Now, we try finding the correlations between different features using pandas.corr():



feature_cols = train.columns[: -1]   #exclude the Activity column



# Calculate the correlation values:



correlated_values = train[feature_cols].corr()



# Stack the data and convert to a dataframe:



correlated_values = (correlated_values.stack().to_frame().reset_index()

                    .rename(columns={'level_0': 'Feature_1', 'level_1': 'Feature_2', 0:'Correlations'}))

correlated_values.head()
# Creating an abs column for correlation column:



correlated_values['abs_correlation'] = correlated_values.Correlations.abs()

correlated_values.head()
# Now we pick the most correlated features:



train_fields = correlated_values.sort_values('Correlations', ascending = False).query('abs_correlation>0.8')

train_fields.sample(5)
# Now splitting the training and validation sets:



# Getting the split indexes:



split_data = StratifiedShuffleSplit(n_splits = 1, test_size = 0.3, random_state = 42)

train_idx, val_idx = next(split_data.split(train[feature_cols], train.Activity))



# Creating the dataframes:



x_train = train.loc[train_idx, feature_cols]

y_train = train.loc[train_idx, 'Activity']



x_val = train.loc[val_idx, feature_cols]

y_val = train.loc[val_idx, 'Activity']
print(y_train.value_counts(normalize = True))

print(y_val.value_counts(normalize = True))



# Thus, we observe that we have the same ratio of all the classes in both the training and validation or development set.
# Building a model:



lr_l2 = LogisticRegressionCV(cv=4, penalty='l2', max_iter = 1000, n_jobs = -1)

lr_l2 = lr_l2.fit(x_train, y_train)
# Predicitng using the Logistic Regression model:



y_predict = list()

y_proba = list()



labels = ['lr_l2']

models = [lr_l2]



for lab, mod in zip(labels, models):

    y_predict.append(pd.Series(mod.predict(x_val), name = lab))

    y_proba.append(pd.Series(mod.predict_proba(x_val).max(axis=1), name = lab))

    #.max(axis = 1) for a 1 dimensional dataframe



y_predict = pd.concat(y_predict, axis = 1)

y_proba = pd.concat(y_proba, axis = 1)



y_predict.head()
# Calculating the precision, recall and F1 score for our model:



metrics = list()

confusion_m = dict()



for lab in labels:

    precision, recall, f_score, _ = error_metric(y_val, y_predict[lab], average = 'weighted')

    

    accuracy = accuracy_score(y_val, y_predict[lab])

    

    confusion_m[lab] = confusion_matrix(y_val, y_predict[lab])

    

    metrics.append(pd.Series({'Precision': precision, 'Recall': recall,

                            'F_score': f_score, 'Accuracy': accuracy}, name = lab))

    

metrics= pd.concat(metrics, axis =1) 

metrics
#Building the second network:



import sklearn.neural_network as nn

mlpADAM =  nn.MLPClassifier(hidden_layer_sizes=(900,), max_iter=1000 , alpha=1e-4, solver='adam' , verbose=10, tol=1e-19, random_state=1, learning_rate_init=.001)

nnModelADAM = mlpADAM.fit(x_train , y_train)
# Visualising the convergence:



X = np.linspace(1, nnModelADAM.n_iter_ , nnModelADAM.n_iter_)

plt.plot(X , nnModelADAM.loss_curve_, label = 'ADAM Convergence')

plt.title('Error Convergence ')

plt.ylabel('Cost function')

plt.xlabel('Iterations')

plt.legend()

plt.show()
# Generating the training and test scores:



print("Training set score for ADAM: %f" % mlpADAM.score(x_train, y_train))

print("Validation set score for ADAM: %f" % mlpADAM.score(x_val, y_val))
# Importing the test data and dividing into test data and label:



testData  = test.drop('Activity' , axis=1).values

testLabel = test.Activity.values
# Label encoding the Activity:



encoder = LabelEncoder()



# Encoding test labels:



encoder.fit(testLabel)

testLabelE = encoder.transform(testLabel)
# Calculating the test score our Adam Classifier:



print("Test set score for ADAM: %f"     % mlpADAM.score(testData , testLabelE ))