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

train=pd.read_csv("../input/train.csv")
# Defining output directory

output = "../input"
train.columns
var= ['PassengerId','Pclass','Sex', 'Age', 'SibSp','Parch','Fare', 'Cabin', 'Embarked']
def getFeatures(value, featureList, idx):

    features = {}

    if idx%10000==0: print(idx)

    for f in featureList:

        features[f] = value[f][idx]

    return features



X_all = train[var]

    # size_train, size_test = len(train), len(test)

#This step would convert the dataframe into List of feature - value mapping (dict like objects).. type =List

X_all1 = [getFeatures(X_all, var, idx) for idx in X_all.index]



#len(X_all1) : length of dict list



from sklearn.feature_extraction import DictVectorizer

# This would This transformer turns lists of mappings (dict-like objects) of feature 

#names to feature values into Numpy arrays or scipy.sparse matrices for use with scikit-learn estimators

vec = DictVectorizer()

X_all2 = vec.fit_transform(X_all1)



X_all2.shape # It got converted into sparse metrics

from sklearn.preprocessing import Imputer

imp = Imputer(missing_values=np.nan, strategy='mean', axis=0)



X_all3 = imp.fit_transform(X_all2)

print ("size of X_all", X_all3.shape)



from scipy.sparse import csr_matrix

X_all4 = csr_matrix(X_all3)

X_all5 = X_all4.todense()

Y_all5 = np.array(train['Survived'])
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X_all5, Y_all5 , test_size = 0.3, random_state = 398)



from sklearn.linear_model import LogisticRegression

rt_lm = LogisticRegression()

rt_lm.fit(X_train,Y_train)

y_pred_test = rt_lm.predict(X_test)



#rf_lm.predict_proba(rf_enc.transform(rf.apply(X_test)))[:, 1]
# Predicting actual values using "predict" method

train_pred=rt_lm.predict(X_train)

test_pred=rt_lm.predict(X_test)
# Predicting probabilities for Train and test data

train_pred_prob=rt_lm.predict_proba(X_train)[:,1]

#train_pred_prob[:10]

test_pred_prob=rt_lm.predict_proba(X_test)[:,1]
# Predicting the values for complete dataset

all_pred=rt_lm.predict(X_all5)

all_pred_prob = rt_lm.predict_proba(X_all5)



from sklearn import metrics

#sklearn.metrics.roc_curve(y_true, y_score, pos_label=None, sample_weight=None, drop_intermediate=True)



#Calculating false positive rate, true positive rate for different threshold values. All values between 0 to 1



false_pos_rate_tr, true_pos_rate_tr, thresholds_tr = metrics.roc_curve(Y_train, train_pred_prob, pos_label=1)

false_pos_rate, true_pos_rate, thresholds = metrics.roc_curve(Y_test, test_pred_prob, pos_label=1)
# Function to save the ROC curve at "output directory"



import os

def save(path, ext='png', close=True, verbose=True):

    



    directory = os.path.split(path)[0]

    filename = "%s.%s" % (os.path.split(path)[1], ext)

    if directory == '':

        directory = '.'



    if not os.path.exists(directory):

        os.makedirs(directory)



    savepath = os.path.join(directory, filename)



    if verbose:

        print("Saving figure to '%s'..." % savepath),



    plt.savefig(savepath)

    

    if close:

        plt.close()



    if verbose:

        print("Done")



#

# Functioon for plotting ROC Curves



import matplotlib.pyplot as plt



def roc_plot_test(fpr, tpr):

    fig=plt.plot(fpr, tpr)

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.0])

    plt.title('ROC curve using test data for SIU classifier')

    plt.xlabel('False Positive Rate (1 - Specificity)')

    plt.ylabel('True Positive Rate (Sensitivity)')

    plt.grid(True)

    #save(output+"roc_test", ext="png", verbose=True)

	

	

def roc_plot_train(fpr, tpr):

    fig=plt.plot(fpr, tpr)

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.0])

    plt.title('ROC curve using train data for SIU classifier')

    plt.xlabel('False Positive Rate (1 - Specificity)')

    plt.ylabel('True Positive Rate (Sensitivity)')

    plt.grid(True)

    #save(output+"roc_train", ext="png", verbose=True) 

	



#Plotting ROC Curve

#ROC for Training data

roc_plot_train(false_pos_rate_tr, true_pos_rate_tr)
