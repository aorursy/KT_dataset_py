# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# All Imports

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.linear_model import LogisticRegression



import warnings

warnings.filterwarnings('ignore')

pd.options.display.max_columns = None



from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, confusion_matrix, roc_curve



from sklearn.metrics import precision_recall_curve

pd.set_option('max_colwidth', 200)



from sklearn.preprocessing import MinMaxScaler
#Read data file

df = pd.read_csv('/kaggle/input/glass/glass.csv')
# Check the data

df.info()
sns.pairplot(df)
df.Type.value_counts(normalize=True) * 100
#From the above we can see there is huge class imbalance for Type 3,5,6

#So we may need to use some kind of sampling technique
all_cols = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
# One Vs Rest Classification

from sklearn.model_selection import train_test_split

logreg = LogisticRegression()

for i in df['Type'].unique():

    new_df = df.copy()

    y = new_df['Type']

    index_ref = new_df['Type'] == i

    X = new_df[all_cols]

    y[index_ref] = 1

    y[-index_ref] = 0



    print(" Training for class: ", i)



    X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.7, test_size = 0.3, random_state = 42)



    '''

    Scale the training data

    '''

    scaler = MinMaxScaler()

    #all_cols = list(X_train.columns)

    print(X_train.shape)

    #X_train.reshape(-1,1)

    X_train = scaler.fit_transform(X_train)



    '''

    Scale Test Set and create the X_test and y_test

    '''

    # Scale Test Set

    #X_test.reshape(-1,1)

    X_test = scaler.transform(X_test)    



    logreg.fit(X_train, y_train)



    y_train_pred = logreg.predict(X_train)

    y_test_pred = logreg.predict(X_test)





    print("The Confusion matrix for Training for class :", i) 

    # Confusion matrix 

    confusion = confusion_matrix(y_train,y_train_pred )

    print('Confusion matrix: \n',confusion)

    print('Accuracy_Score:',accuracy_score(y_train,y_train_pred))

    print('Recall_Score:',recall_score(y_train,y_train_pred))

    print('Precision_Score:',precision_score(y_train,y_train_pred))



    TP = confusion[1,1] # true positive 

    TN = confusion[0,0] # true negatives

    FP = confusion[0,1] # false positives

    FN = confusion[1,0] # false negatives



    # Let's see the sensitivity of our logistic regression model

    print('Sensitivity',TP / float(TP+FN))



    # Let us calculate specificity

    print('Specificity', TN / float(TN+FP))



    # Calculate false postive rate - predicting churn when customer does not have churned

    print('false postive rate', FP/ float(TN+FP))



    # positive predictive value 

    print ('positive predictive value',TP / float(TP+FP))



    # Negative predictive value

    print ('Negative predictive value',TN / float(TN+ FN))



    print("The Confusion matrix for Test set for class :", i) 

    # Confusion matrix 

    confusion = confusion_matrix(y_test,y_test_pred )

    print('Confusion matrix: \n',confusion)

    print('Accuracy_Score:',accuracy_score(y_test,y_test_pred))

    print('Recall_Score:',recall_score(y_test,y_test_pred))

    print('Precision_Score:',precision_score(y_test,y_test_pred))



    TP = confusion[1,1] # true positive 

    TN = confusion[0,0] # true negatives

    FP = confusion[0,1] # false positives

    FN = confusion[1,0] # false negatives



    # Let's see the sensitivity of our logistic regression model

    print('Sensitivity',TP / float(TP+FN))



    # Let us calculate specificity

    print('Specificity', TN / float(TN+FP))



    # Calculate false postive rate - predicting churn when customer does not have churned

    print('false postive rate', FP/ float(TN+FP))



    # positive predictive value 

    print ('positive predictive value',TP / float(TP+FP))



    # Negative predictive value

    print ('Negative predictive value',TN / float(TN+ FN))



    print(" ----------Done for Class----------", i)