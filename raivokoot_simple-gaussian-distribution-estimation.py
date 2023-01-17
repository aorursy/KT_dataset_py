# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy.stats as stats



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    print(dirname)

    for filename in filenames:

        credit_data = pd.read_csv(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
credit_data.shape
credit_data.describe()
from sklearn.model_selection import train_test_split



def split_data(credit_data):

    """

    Splits the data into sets of size 0.6, 0.2 and 0.2

    where the train set does not contain any outliers and the

    validation and test sets both contain the same amount

    of outliers.

    """

    

    # Separate the outliers because we want

    # both sets to have the same amount of outliers

    non_outliers = credit_data.loc[credit_data['Class'] == 0]

    outliers = credit_data.loc[credit_data['Class'] == 1]

    

    # Split the data

    train, test = train_test_split(non_outliers, test_size=0.4, random_state=42)

    val, test = train_test_split(test, test_size=0.5, random_state=42)

    

    

    val_outliers, test_outliers = train_test_split(outliers, test_size=0.5, random_state=42)

    

    # Add the outliers back to both sets

    val = pd.concat([val, val_outliers])

    test = pd.concat([test, test_outliers])

    

    return train, val, test



train, val, test = split_data(credit_data)



print(train['Class'].value_counts())

print(val['Class'].value_counts())

print(test['Class'].value_counts())
TRIVIAL_COLUMN_NAMES = ['Class', 'Time', 'Amount']

CLASS_COLUMN_NAME = 'Class'



def separate_x_y(data):

    X = data.drop(columns=TRIVIAL_COLUMN_NAMES)

    y = data[CLASS_COLUMN_NAME]

    

    return X,y
X_train, _ = separate_x_y(train)

X_val, y_val = separate_x_y(val)

X_test, y_test = separate_x_y(test)
from sklearn.base import BaseEstimator, ClassifierMixin



class GaussianAnomalyClassifier(BaseEstimator, ClassifierMixin):

    """

    An anomaly detection classifier.

    """

    

    ANOMALY_CLASS_LABEL = 1

    NON_ANOMALY_CLASS_LABEL = 0

    

    def __init__(self, anomaly_threshold=None):

        """

        params:

        anomaly_threshold - The minimum probability a sample

                            can have before being classified

                            as an anomaly.

        """

        

        self.anomaly_threshold = anomaly_threshold

        

        

    def fit(self, X, y=None):

        """

        Estimates the parameters of a multivariate Gaussian

        distribution on X.

        """

        covariance_matrix = np.cov(X, rowvar=0)

        means = np.mean(X,axis=0)

        

        self.distribution = stats.multivariate_normal(mean=means, cov=covariance_matrix)

        

        

        return self

    

    def predict_proba(self, X):

        """

        Calculates the likelihoods of X

        coming from the estimated distribution

        """

        if self.distribution is None:

            raise RuntimeError("You must train the classifier before prediction")

            

        probabilities = self.distribution.pdf(X)

        

        return probabilities

    

    def predict(self, X, y=None):

        """

        Classifies each sample in X

        """

        

        probabilities = self.predict_proba(X)

        

        predictions = np.where(probabilities < self.anomaly_threshold, \

                               self.ANOMALY_CLASS_LABEL, self.NON_ANOMALY_CLASS_LABEL)

        

        return predictions

        
from sklearn.metrics import precision_recall_curve

import matplotlib.pyplot as plt



classifier = GaussianAnomalyClassifier(0.01)

classifier = classifier.fit(X_train)

predictions = classifier.predict_proba(X_val)



precision_recall = precision_recall_curve(y_val, predictions)



plt.figure()

plt.step(precision_recall[1],precision_recall[0], where='post')

plt.xlabel('Recall')

plt.ylabel('Precision')

plt.ylim([0.0, 0.1])

plt.xlim([0.0, 1.0])

from sklearn.metrics import precision_recall_fscore_support



def print_scores(predictions, y_true):

    precision, recall, fscore, _ = precision_recall_fscore_support(y_true, predictions, average='binary', pos_label=1)

    print('Precision: ', precision)

    print('Recall: ', recall)

    print('Fscore: ', fscore)

    print()

    print(pd.Series(predictions).value_counts())

    

    return precision,recall,fscore
print("REAL VALUE COUNTS VALIDATION SET")

print(pd.Series(y_val).value_counts())

print('------------------------------')

print('------------------------------\n')





# create a list of increasingly smaller thresholds to test

thresholds = [0.000000000000000001 * (0.1)**x for x in range(170)]

thresholds.reverse()



counter = 159

fscores = [] # Save fscores to plot them afterwards



for threshold in thresholds:

    classifier = GaussianAnomalyClassifier(threshold)

    classifier = classifier.fit(X_train)

    predictions = classifier.predict(X_val)



    _,_, fscore = print_scores(predictions, y_val)

    

    print('threshold index: ', counter)

    print('------------------------')

    

    

    fscores.append(fscore)

    counter -= 1

    

fscores.reverse()
plt.scatter(range(len(fscores)), fscores, s=4)



plt.title("Fscores of increasingly smaller probability thresholds")

plt.xlabel('Threshold Index')

plt.ylabel('Fscore')
threshold = 0.000000000000000001 * (0.1)**80



classifier = GaussianAnomalyClassifier(threshold)

classifier = classifier.fit(X_train)

predictions = classifier.predict(X_test)



_,_,_ = print_scores(predictions, y_test)