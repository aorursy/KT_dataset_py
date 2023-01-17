import pandas as pd

import numpy as np  

import matplotlib.pyplot as plt  

#import seaborn as seabornInstance 

from sklearn.model_selection import train_test_split 

from sklearn.ensemble import AdaBoostClassifier

import sklearn.metrics as metrics

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import LabelEncoder
dflabels = pd.read_csv("../input/crmchurn/CRM Churn Labels Shared.tsv", sep="\t", header = None)

df = pd.read_csv("../input/crmchurn/CRM Dataset Shared.tsv", sep="\t")
x = df.dtypes
df.head()
df = df.fillna(0)
dtypes = pd.DataFrame([list(df.columns),list(df.dtypes)]).T

dtypes.columns = ["column","type"]
enccol = []

for index,row in dtypes.iterrows():

    if row["type"] == "object":

        enccol.append(row["column"])
df[enccol] = df[enccol].astype("string")
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
df.dtypes
df = MultiColumnLabelEncoder(columns = enccol).fit_transform(df)
dflabels.head()
df.shape,dflabels.shape
dflabels.columns = ["label"]
dffinal = pd.concat([df, dflabels.reindex(df.index)], axis=1)
dffinal.head()
dffinal.shape
X_train, X_test, y_train, y_test = train_test_split(df, dflabels, test_size=0.3)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
model1 = LogisticRegression(solver="liblinear")

model1.fit(X_train, y_train)
model2 = AdaBoostClassifier()

model2.fit(X_train, y_train)
def evaluate(model,X_test,y_test):

    probs = model.predict_proba(X_test)

    preds = probs[:,1]

    predicted_classes = model1.predict(X_test)

    accuracy = metrics.accuracy_score(y_test,predicted_classes)

    confusion = metrics.confusion_matrix(y_test, model.predict(X_test))

    TP = confusion[1, 1]

    TN = confusion[0, 0]

    FP = confusion[0, 1]

    FN = confusion[1, 0]

    classification_error = (FP + FN) / float(TP + TN + FP + FN)

    sensitivity = TP / float(FN + TP)

    specificity = TN / (TN + FP)

    false_positive_rate = FP / float(TN + FP)

    precision = TP / float(TP + FP)

    print("Accuracy:",accuracy)

    print("\nConfusion Matrix:\n",confusion)    

    print("\nROC_AUC_Score:",(metrics.roc_auc_score(y_test, preds)))

    print("\nClassification Error:", classification_error)

    print("\nSensitivity:", sensitivity)

    print("\nSpecificity:", specificity)

    print("\nFalse Positive Rate:", false_positive_rate)

    print("\nPrecision:", precision)

    # calculate the fpr and tpr for all thresholds of the classification

    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)

    roc_auc = metrics.auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')

    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

    plt.legend(loc = 'lower right')

    plt.plot([0, 1], [0, 1],'r--')

    plt.xlim([0, 1])

    plt.ylim([0, 1])

    plt.ylabel('True Positive Rate')

    plt.xlabel('False Positive Rate')

    plt.show()

    return None
evaluate(model1,X_test,y_test)
evaluate(model2,X_test,y_test)