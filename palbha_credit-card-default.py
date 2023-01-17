import numpy as np 

import pandas as pd

import os

print(os.listdir("../input"))



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.metrics import *



from sklearn.ensemble import IsolationForest

from sklearn.neighbors import LocalOutlierFactor



from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import export_graphviz



import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot



init_notebook_mode(connected=True)



import warnings

warnings.filterwarnings('ignore')
df=pd.read_csv("../input/creditcard.csv")
df.head()
df.describe().T
plt.figure(figsize=(15,10))

cor_df=df.corr()

cor_df

sns.heatmap(cor_df, cmap="Blues",

        xticklabels=cor_df.columns,

        yticklabels=cor_df.columns)

plt.title("Correlation Plot")

plt.show()
print(df["Class"].value_counts())

sns.countplot(df["Class"])

plt.title("Countplot of Class: Fraud/Not Fraud")

plt.show()
trace1 = go.Box(

    y=df["Amount"],

    name = 'Log transformed Distribution of Amount',

    marker = dict(

        color = 'rgb(12, 128, 128)',

    )

)



data = [trace0,trace1]

iplot(data, filename = 'basic-line')
trace1 = go.Box(

    y=np.log1p(df["Amount"]),

    name = 'Log transformed Distribution of Amount',

    marker = dict(

        color = 'rgb(12, 128, 128)',

    )

)



data = [trace1]

iplot(data, filename = 'basic-line')
print("Number of values with class 0 and amount< 1",len(df[(df["Class"]==0 )& (df["Amount"]<1)]["Amount"]))
sns.distplot(df[df["Class"]==1]["Amount"])

plt.xlabel("Frauduelent case")

plt.ylabel("Amount variations")

plt.title("Distribution of frauduelet cases with amount")

plt.show()
#Analysing Outliers in data

for i in df.columns:

    print(i)

    q1, q3= np.percentile(df[i],[25,75])

    iqr = q3 - q1

    lower_bound = q1 -(1.5 * iqr) 

    upper_bound = q3 +(1.5 * iqr) 

    print("Lower bound ",df[df[i]<lower_bound][i].count())

    print("Upper bound ",df[df[i]>upper_bound][i].count())
#Since we saw that the Amount values were highly skewed 

#we tried converting to a normalized version using log transformation



df["log_amount"]=np.log1p(df["Amount"])



df.head().T
#Fraud : -1 , Not Fraud : 1

df["unsup_class"]=np.where(df["Class"]==1,-1,1)
X_train, X_test, y_train, y_test = train_test_split(df.drop(["unsup_class","Class","Amount"],axis=1), df.unsup_class, test_size=0.25, random_state=0)
iso_fac=IsolationForest(n_estimators=100, max_samples='auto', 

                   contamination=float(.2),max_features=1.0,

                   bootstrap=False, n_jobs=-1, random_state=42, verbose=0,behaviour="new")

iso_fac.fit(X_train)

print("IsolationForest\n ",iso_fac)

train_pred=iso_fac.predict(X_train)

labels=y_train.unique()

print("Result of Training data")

print("Confusion Matrix")

cm=confusion_matrix(y_train,train_pred)

print(pd.DataFrame(confusion_matrix(y_train,train_pred, labels=labels), index=labels, columns=labels))

#Since through Isolation Forest we get -1 as outliers and 1 as correct values

#Our main concern here is to predict error values lets calculate TNR :equivalent to recall



print("True Neagtive rate",cm[1,1]/(cm[1,1]+cm[1,0]))

print("Accuracy score",accuracy_score(y_train,train_pred))



test_pred=iso_fac.predict(X_test)

labels=y_test.unique()



print("\nResult of Testing data")

print("Confusion Matrix")

cm=confusion_matrix(y_test,test_pred)

print(pd.DataFrame(confusion_matrix(y_test,test_pred, labels=labels), index=labels, columns=labels))

print("True Neagtive rate",cm[1,1]/(cm[1,1]+cm[1,0]))

print("Accuracy score",accuracy_score(y_test,test_pred))
lf= LocalOutlierFactor(n_neighbors=20, contamination=0.2)

pred=lf.fit_predict(X_train)

print("Local Outlier Factor \n",lf)

train_pred=lf.fit_predict(X_train)

labels=y_train.unique()

print("Result of Training data")

print("Confusion Matrix")

cm=confusion_matrix(y_train,train_pred)

print(pd.DataFrame(confusion_matrix(y_train,train_pred, labels=labels), index=labels, columns=labels))

#Since through Isolation Forest we get -1 as outliers and 1 as correct values

#Our main concern here is to predict error values lets calculate TNR :equivalent to recall



print("True Neagtive rate",cm[1,1]/(cm[1,1]+cm[1,0]))

print("Accuracy score",accuracy_score(y_train,train_pred))



test_pred=lf.fit_predict(X_test)

labels=y_test.unique()



print("\nResult of Testing data")

print("Confusion Matrix")

cm=confusion_matrix(y_test,test_pred)

print(pd.DataFrame(confusion_matrix(y_test,test_pred, labels=labels), index=labels, columns=labels))

print("True Neagtive rate",cm[1,1]/(cm[1,1]+cm[1,0]))

print("Accuracy score",accuracy_score(y_test,test_pred))
X_train, X_test, y_train, y_test = train_test_split(df.drop(["unsup_class","Class","Amount"],axis=1), df.Class, test_size=0.25, random_state=0)
def plot_roc():

    plt.plot(fpr, tpr, label = 'ROC curve', linewidth = 2)

    plt.plot([0,1],[0,1], 'k--', linewidth = 2)

    plt.xlim([0.0,0.001])

    plt.ylim([0.0,1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('ROC Curve')

    plt.show();
# precision-recall curve

def plot_precision_recall():

    plt.step(recall, precision, color = 'b', alpha = 0.2,

             where = 'post')

    plt.fill_between(recall, precision, step ='post', alpha = 0.2,

                 color = 'b')



    plt.plot(recall, precision, linewidth=2)

    plt.xlim([0.0,1])

    plt.ylim([0.0,1.05])

    plt.xlabel('Recall')

    plt.ylabel('Precision')

    plt.title('Precision Recall Curve')

    plt.show();
lr= LogisticRegression(C=0.5,class_weight="balanced")

lr.fit(X_train,y_train)
train_pred=lr.predict(X_train)

labels=y_train.unique()

print("Result of Training data")

print("Confusion Matrix")

print(pd.DataFrame(confusion_matrix(y_train,train_pred, labels=labels), index=labels, columns=labels))

print("roc-auc score on train data :",roc_auc_score(y_train,train_pred))

print("precision",precision_score(y_train, train_pred))

print("recall",recall_score(y_train, train_pred))

print("F1 score",f1_score(y_train, train_pred))



test_pred=lr.predict(X_test)

labels=y_test.unique()



print("\n Result of Testing data")

print("Confusion Matrix")

print(pd.DataFrame(confusion_matrix(y_test,test_pred, labels=labels), index=labels, columns=labels))

print("roc-auc score on test data :",roc_auc_score(y_test,test_pred))

print("precision",precision_score(y_test,test_pred))

print("recall",recall_score(y_test,test_pred))

print("F1 score",f1_score(y_test,test_pred))

fpr, tpr, t = roc_curve(y_test, test_pred)

plot_roc()



# Precision-recall curve

precision, recall, thresholds = precision_recall_curve(y_test, test_pred)

plot_precision_recall()
rf=RandomForestClassifier(n_estimators=100, max_depth=12,random_state=0,class_weight="balanced")

rf.fit(X_train, y_train)

print("Random Forest \n",rf)
train_pred=rf.predict(X_train)

labels=y_train.unique()

print("Result of Training data")

print("Confusion Matrix")

print(pd.DataFrame(confusion_matrix(y_train,train_pred, labels=labels), index=labels, columns=labels))

print("roc-auc score on train data :",roc_auc_score(y_train,train_pred))

print("precision",precision_score(y_train, train_pred))

print("recall",recall_score(y_train, train_pred))

print("F1 score",f1_score(y_train, train_pred))



test_pred=rf.predict(X_test)

labels=y_test.unique()



print("\n Result of Testing data")

print("Confusion Matrix")

print(pd.DataFrame(confusion_matrix(y_test,test_pred, labels=labels), index=labels, columns=labels))

print("roc-auc score on test data :",roc_auc_score(y_test,test_pred))

print("precision",precision_score(y_test,test_pred))

print("recall",recall_score(y_test,test_pred))

print("F1 score",f1_score(y_test,test_pred))

fpr, tpr, t = roc_curve(y_test, test_pred)

plot_roc()



# Precision-recall curve

precision, recall, thresholds = precision_recall_curve(y_test, test_pred)

plot_precision_recall()
scale_pos_weight = [0.125,0.5,1,2,3]

for i in scale_pos_weight:

    print('scale_pos_weight = {}: '.format(i))

    xgb = XGBClassifier(scale_pos_weight=i)

    print(xgb)

    xgb.fit(X_train, y_train)

    train_pred=xgb.predict(X_train)

    labels=y_train.unique()

    print("Result of Training data")

    print("Confusion Matrix")

    print(pd.DataFrame(confusion_matrix(y_train,train_pred, labels=labels), index=labels, columns=labels))

    print("roc-auc score on train data :",roc_auc_score(y_train,train_pred))

    print("precision",precision_score(y_train, train_pred))

    print("recall",recall_score(y_train, train_pred))

    print("F1 score",f1_score(y_train, train_pred))



    test_pred=xgb.predict(X_test)

    labels=y_test.unique()



    print("\nResult of Testing data")

    print("Confusion Matrix")

    print(pd.DataFrame(confusion_matrix(y_test,test_pred, labels=labels), index=labels, columns=labels))

    print("roc-auc score on test data :",roc_auc_score(y_test,test_pred))

    print("precision",precision_score(y_test,test_pred))

    print("recall",recall_score(y_test,test_pred))

    print("F1 score",f1_score(y_test,test_pred))

#We have select xgb classifier  with 0.125 scale_pos_weight as the better performing classifier 

xgb = XGBClassifier(scale_pos_weight=0.125)

xgb.fit(X_train, y_train)



test_pred = xgb.predict(X_test)



fpr, tpr, t = roc_curve(y_test, test_pred)

plot_roc()



# Precision-recall curve

precision, recall, thresholds = precision_recall_curve(y_test, test_pred)

plot_precision_recall()


import keras.backend as K

from keras.layers import Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D, BatchNormalization, Input, Conv1D

from keras.callbacks import ModelCheckpoint

from keras import metrics

from keras.optimizers import Adam 

from sklearn.preprocessing import StandardScaler

import keras

from keras.models import Model

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.metrics import *

from keras import regularizers

import tensorflow as tf

from keras.losses import binary_crossentropy

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

BATCH_SIZE = 1024

NUM_FEATURES = 1200

#Creation of our own evaluation metric

def recall(y_true, y_pred):

        """Recall metric.



        Only computes a batch-wise average of recall.



        Computes the recall, a metric for multi-label classification of

        how many relevant items are selected.

        """

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = true_positives / (possible_positives + K.epsilon())

        return recall




def _Model1():

    inp = Input(shape=(30, 1))

    d1 = Dense(16, activation='sigmoid')(inp)

    d2 = Dense(16, activation='relu')(d1)

    d3 = Dense(8, activation='sigmoid')(d2)

    f2 = Flatten()(d3)

    preds = Dense(1, activation='sigmoid')(f2)

    model = Model(inputs=inp, outputs=preds)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy',recall])

    return model



cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)

X_train.columns


scaler=StandardScaler()

df_train=X_train

df_test=X_test


# also add dropout

# different activations as well



preds = []

c = 0

oof_preds = np.zeros((len(df_train), 1))

for train, valid in cv.split(df_train, y_train):

    print("VAL %s" % c)

    X_train = np.reshape(df_train.iloc[train].values, (-1, 30, 1))

    y_train_ = y_train.iloc[train].values

    X_valid = np.reshape(df_train.iloc[valid].values, (-1, 30, 1))

    y_valid = y_train.iloc[valid].values

    model = _Model1()

    history=model.fit(X_train, y_train_, validation_data=(X_valid, y_valid), epochs=20, verbose=2, batch_size=128)



    

    X_test = np.reshape(df_test.values, (-1, 30, 1))

    curr_preds = model.predict(X_test, batch_size=256)

    oof_preds[valid] = model.predict(X_valid)

    preds.append(curr_preds)

    c += 1





print("Training data results:")

auc = roc_auc_score(y_train, oof_preds)

print("CV_AUC: {}".format(auc))





print("\nTest data results: ")

print("Precision score",round(precision_score(y_test, np.around(preds[1])),2))

print("Confusion metrics \n",confusion_matrix(y_test, np.around(preds[1])))
