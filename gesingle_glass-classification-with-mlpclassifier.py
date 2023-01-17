import pandas as pd

import numpy as np

import matplotlib as mpl

import missingno as mno

import seaborn as sns

import matplotlib.pyplot as plt 

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.preprocessing import normalize

from collections import Counter



data = pd.read_csv('../input/glass.csv')
data.head()
data.info()
data.describe()
data['Type'].value_counts()
features = data.columns[:9]



for feat in features:

    skew = data[feat].skew()

    sns.distplot(data[feat], label='Skew = %.3f' %(skew))

    plt.legend(loc='best')

    plt.show()
def outlier_hunt(df):

    """

    Takes a dataframe df of features and returns a list of the indices

    corresponding to the observations containing more than 1 outlier. 

    """

    outlier_indices = []

    

    # iterate over features(columns)

    for col in df.columns.tolist():

        # 1st quartile (25%)

        Q1 = np.percentile(df[col], 25)

        

        # 3rd quartile (75%)

        Q3 = np.percentile(df[col],75)

        

        # Interquartile rrange (IQR)

        IQR = Q3 - Q1

        

        # outlier step

        outlier_step = 1.5 * IQR

        

        # Determine a list of indices of outliers for feature col

        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index

        

        # append the found outlier indices for col to the list of outlier indices 

        outlier_indices.extend(outlier_list_col)

        

    # select observations containing more than 2 outliers

    outlier_indices = Counter(outlier_indices)        

    multiple_outliers = list( k for k, v in outlier_indices.items() if v > 1 )

    

    return multiple_outliers  



print('The dataset contains %d observations with multiple outliers' %(len(outlier_hunt(data[features]))))
outliers = outlier_hunt(data[features])

data.drop(outliers, inplace=True)



for feat in features:

    skew = data[feat].skew()

    sns.distplot(data[feat], label='Skew = %.3f' %(skew))

    plt.legend(loc='best')

    plt.show()
labels = data['Type'].values

data.drop(['Type'], axis=1, inplace=True)

labels
X = data.values

normalize(X,copy=False)

X
sss = StratifiedShuffleSplit(n_splits=3, test_size=0.3, random_state=0)

for train_index, test_index in sss.split(X, labels):

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = labels[train_index], labels[test_index]
nn = MLPClassifier(alpha=.1)

nn.fit(X_train, y_train)
y_pred = nn.predict(X_test)



print('Accuracy: %2f' % accuracy_score(y_test, y_pred))
ulabels = np.unique(labels)


print(classification_report(y_test, y_pred, ulabels))