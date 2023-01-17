# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import math

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression as Logit

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, precision_recall_curve

from sklearn.metrics import confusion_matrix



def all_logit(X_train, y_train, X_test):

    lmodel = Logit().fit(X_train, y_train)

    y_bar = lmodel.predict(X_test)

    return y_bar





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

def main(df):

    #df = pd.read_csv(f)



   

    endog = df['Class']

    exog = df.drop(['Time', 'Class', 'Amount'], axis=1).dropna()

    X_train, X_test, y_train, y_test = train_test_split(exog, endog)

    # Undersampling the non-defaulters due to unbalanced dataset

    # We make it N:1 non-frauds:frauds

    f = df[df['Class'] == 1]    # All fraudsters

    nf = df[df['Class'] == 0]   # Non fraudsters

    if math.ceil(len(nf)/len(f)) > 40:

        nf = nf.sample(len(f)*40) # This is the main trick

    df = nf.append(f)

    endog = df['Class']

    exog = df.drop(['Time', 'Class', 'Amount'], axis=1).dropna()

    X_train, _, y_train, _ = train_test_split(exog, endog)

    y_bar = all_logit(X_train, y_train, X_test)



    # Classification report

    #print(classification_report(y_test, y_bar))

    cm = confusion_matrix(y_test, y_bar)

    

    #print("miss-classification of non-fraud as fraud percentage:", cm[0,1]/(cm[0,1]+cm[0,0])*100)

    #print("Recall of fraudulent percentage:", cm[1,1]/(cm[1,0]+cm[1,1])*100)

    return (cm[0,1]/(cm[0,1]+cm[0,0])), (cm[1,1]/(cm[1,0]+cm[1,1]))





df = pd.read_csv('../input/creditcard.csv')

miss, recall = 0, 0

N = 20

for i in range(N):

    m, r = main(df)

    miss += m

    recall += r



print('miss-classification of non-fraudulent as fraudulent percentage:', miss/N*100)

print("Recall of fraudulent usage percentage:", recall/N*100)
