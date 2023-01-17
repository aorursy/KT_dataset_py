import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
#import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from tpot import TPOTClassifier
import sklearn.metrics as metrics
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("../input/crmchurnautoml/crm-churn.csv")
df.shape
df.head()
df.dtypes
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
df = MultiColumnLabelEncoder(columns = enccol).fit_transform(df)
df["Col1"] = df["Col1"].replace(-1,0)
X_train, X_test, y_train, y_test = train_test_split(df.drop(["Col1"],axis=1), df["Col1"], test_size=0.3)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
pipeline_optimizer = TPOTClassifier(generations=100, population_size=20, cv=5,
                                    random_state=42, verbosity=2, max_time_mins = 60)

pipeline_optimizer.fit(X_train, y_train)
def evaluate(model,X_test,y_test):
    probs = model.predict_proba(X_test)
    preds = probs[:,1]
    predicted_classes = model.predict(X_test)
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
evaluate(pipeline_optimizer,X_test,y_test)