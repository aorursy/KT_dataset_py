# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

# print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



mushrooms = pd.read_csv('../input/mushrooms.csv')

pd.set_option('display.max_columns', None)

pd.set_option('display.max_colwidth', 0)

X = mushrooms.loc[:, 'cap-shape':'habitat']

y = pd.DataFrame(mushrooms['class'])



from sklearn.preprocessing import LabelEncoder

# A multi column label encoder to make it easy to encode all labels at once.

class MultiColumnLabelEncoder:

    def __init__(self,columns = None):

        self.columns = columns # array of column names to encode



    def fit(self,X,y=None):

        return self # not relevant here



    def transform(self,X):

        '''

        Transforms columns of X specified in self.columns using

        LabelEncoder(). If no columns specified, transforms all

        columns in X.

        '''

        output = X.copy()

        if self.columns is not None:

            for col in self.columns:

                output[col] = LabelEncoder().fit_transform(output[col])

        else:

            for colname,col in output.iteritems():

                output[colname] = LabelEncoder().fit_transform(col)

        return output



    def fit_transform(self,X,y=None):

        return self.fit(X,y).transform(X)



from sklearn.preprocessing import OneHotEncoder

onehot_encoder = OneHotEncoder(categorical_features=[0])

X_encoded = MultiColumnLabelEncoder(columns=list(X.columns)).fit_transform(X)

X_1h_encoder = OneHotEncoder(categorical_features='all')

X_processed = X_1h_encoder.fit_transform(X_encoded).toarray()



from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.25, random_state=0)



from sklearn.linear_model import LogisticRegression

logisticRegressor = LogisticRegression()

logisticRegressor.fit(x_train, y_train)



predictions = logisticRegressor.predict(x_test)

# print(predictions)

score = logisticRegressor.score(x_test, y_test)

# print(score)



from sklearn.metrics import classification_report

regression_coefficients = logisticRegressor.coef_[0][0]



# Now a confusion matrix

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import metrics



# Constructing the confusion matrix

cm = metrics.confusion_matrix(y_test, predictions)



# Plotting a confusion matrix

plt.figure(figsize=(2,2))

sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')

plt.ylabel('Actual category')

plt.xlabel('Predicted category')

all_sample_title = 'Accuracy Score: {0}'.format(score)

plt.title(all_sample_title, size = 15)
