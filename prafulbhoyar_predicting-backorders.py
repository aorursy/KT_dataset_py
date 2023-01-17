# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from imblearn.under_sampling import NearMiss

from imblearn.under_sampling import RandomUnderSampler

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier



%matplotlib inline





def preprocess(df):

    columns = [ 'national_inv', 

       'forecast_3_month', 'forecast_6_month', 'forecast_9_month',

       'sales_1_month', 'sales_3_month', 'sales_6_month', 'sales_9_month',

        'perf_6_month_avg','perf_12_month_avg',  'went_on_backorder']

    df = df[columns]

    df = df.dropna(how='any')

    df['perf_6_month_avg'] = df['perf_6_month_avg'].apply(lambda x: 0.6 if x == -99 else x)

    df['perf_12_month_avg'] = df['perf_12_month_avg'].apply(lambda x: 0.6 if x == -99 else x)    

    le = preprocessing.LabelEncoder()

    df['went_on_backorder'] = le.fit_transform(df['went_on_backorder'])

    X = df.loc[:,df.columns != 'went_on_backorder']

    y = df['went_on_backorder']

       

    return X,y



def testTrainSplit(X,y):

    return train_test_split(X, y, test_size=0.33, random_state=0)



def drawImportances(importances,columns):

    indices = np.argsort(importances)[::-1]



    # Print the feature ranking

    print("Feature ranking:")



    for f in range(X.shape[1]):

        print("%d. feature %s (%f)" % (f + 1, columns[indices[f]], importances[indices[f]]))



    # Plot the feature importances of the forest

    plt.figure()

    plt.title("Feature importances")

    plt.bar(range(X.shape[1]), importances[indices],

       color="r",  align="center")

    plt.xticks(range(X.shape[1]), indices)

    plt.xlim([-1, X.shape[1]])

    plt.show()



    



    print(importances)

#scale featues

def scale(X_train,y_train):

    rus = RandomUnderSampler()

    return rus.fit_sample(X_train,y_train)



# plot curves

def plotPrecisionRecallCurves(X_train,y_train,X_test,y_test,clfNameArray):

    for clf,name in clfNameArray:

        clf.fit(X_train,y_train)

        y_pred = clf.predict_proba(X_test)

        precision, recall, thresholds = precision_recall_curve(y_test, y_pred[:,1])

        # Plot Precision-Recall curve

        plt.plot(recall, precision,

             label=name)

        plt.xlabel('Recall')

        plt.ylabel('Precision')

        plt.ylim([0.0, 1.05])

        plt.xlim([0.0, 1.0])

        plt.title('Precision-Recall :')

        plt.legend()

    plt.show()

        
#try this on the training data 

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import precision_recall_curve



#read data

df= pd.read_csv("../input/Kaggle_Training_Dataset_v2.csv")

#preproces scale data

X,y = preprocess(df)

X_train,X_test,y_train,y_test =testTrainSplit(X,y)

print(X_train.shape,y_train.shape)

X_train,y_train = scale(X_train,y_train)

print(X_train.shape,y_train.shape)

#create classifiers

dtc = DecisionTreeClassifier(random_state=0,class_weight ='balanced')

rfc = RandomForestClassifier(random_state=0,class_weight ='balanced')

lr = LogisticRegression()

gnb = GaussianNB()

# draw precison recall curves

classifiers = [(dtc,'DecisionTreeClassifier'),

               (rfc,'RandomForestClassifier'),

               (lr,'LogisticRegression'),

               (gnb,'GaussianNB')

                ]

plotPrecisionRecallCurves(X_train,y_train,X_test,y_test,classifiers)

    
#read data

df= pd.read_csv("../input/Kaggle_Test_Dataset_v2.csv")

#preproces scale data

X,y = preprocess(df)

#create classifiers

# draw precison recall curves

classifiers = [(dtc,'DecisionTreeClassifier'),

               (rfc,'RandomForestClassifier'),

               (lr,'LogisticRegression'),

               (gnb,'GaussianNB')

                ]

plotPrecisionRecallCurves(X_train,y_train,X,y,classifiers)