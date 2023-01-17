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
df = pd.read_csv("/kaggle/input/cardiovascular-disease-dataset/cardio_train.csv", sep=';')
df.shape
df.head(5)
df.isnull().values.any()
del df['id']
df.head(40)
import matplotlib.pyplot as plt

def plot_corr(df, size=12):

    corr = df.corr()

    fig, ax = plt.subplots(figsize=(size, size))

    ax.matshow(corr)

    plt.xticks(range(len(corr.columns)), corr.columns)

    plt.yticks(range(len(corr.columns)), corr.columns)
plot_corr(df)
df.corr()
df.hist(figsize=(11,12))

plt.show()
df['age'] = df['age'].map(lambda x : x // 365)

df.head(5)
# Visualizing the data

dataset_plot = df

dataset_plot[['active','age','alco','ap_hi','ap_lo','cholesterol','gender','gluc','height','smoke','weight']].head(100).plot(style=['o','x','r--','g^'])

plt.legend(bbox_to_anchor=(0.,1.02,1., .102), loc=3,ncol=2, mode="expand", fontsize="x-large", borderaxespad=0.)

plt.show()
from sklearn.model_selection import train_test_split



X = df.drop(['cardio'], axis=1)

y = df['cardio']



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X = scaler.fit_transform(X)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42) 
print("{0:0.2f}% in training set".format((len(X_train)/len(df.index)) * 100))

print("{0:0.2f}% in test set".format((len(X_test)/len(df.index)) * 100))
print("Original True  : {0} ({1:0.2f}%)".format(len(df.loc[df['cardio'] == 1]), (len(df.loc[df['cardio'] == 1])/len(df.index)) * 100.0))

print("Original False : {0} ({1:0.2f}%)".format(len(df.loc[df['cardio'] == 0]), (len(df.loc[df['cardio'] == 0])/len(df.index)) * 100.0))

print("")

print("Training True  : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 1]), (len(y_train[y_train[:] == 1])/len(y_train) * 100.0)))

print("Training False : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 0]), (len(y_train[y_train[:] == 0])/len(y_train) * 100.0)))

print("")

print("Test True      : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 1]), (len(y_test[y_test[:] == 1])/len(y_test) * 100.0)))

print("Test False     : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 0]), (len(y_test[y_test[:] == 0])/len(y_test) * 100.0)))

from sklearn.linear_model import LogisticRegression

from sklearn import metrics



C_start = 0.1

C_end = 5

C_inc = 0.1



C_values, recall_scores = [], []



C_val = C_start

best_recall_score = 0

while (C_val < C_end):

    C_values.append(C_val)

    lr_model_loop = LogisticRegression(C=C_val, class_weight="balanced", random_state=42, solver='liblinear', max_iter=10000)

    lr_model_loop.fit(X_train, y_train.ravel())

    lr_predict_loop_test = lr_model_loop.predict(X_test)

    recall_score = metrics.recall_score(y_test, lr_predict_loop_test)

    recall_scores.append(recall_score)

    if (recall_score > best_recall_score):

        best_recall_score = recall_score

        best_lr_predict_test = lr_predict_loop_test

        

    C_val = C_val + C_inc



best_score_C_val = C_values[recall_scores.index(best_recall_score)]

print("1st max value of {0:.3f} occured at C={1:.3f}".format(best_recall_score, best_score_C_val))



%matplotlib inline 

plt.plot(C_values, recall_scores, "-")

plt.xlabel("C value")

plt.ylabel("recall score")
lr_model =LogisticRegression( class_weight="balanced", C=best_score_C_val, random_state=42, solver='liblinear', max_iter=10000)

lr_model.fit(X_train, y_train.ravel())

lr_predict_test = lr_model.predict(X_test)



# training metrics

print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_test, lr_predict_test)))

print(metrics.confusion_matrix(y_test, lr_predict_test) )

print("")

print("Classification Report")

print(metrics.classification_report(y_test, lr_predict_test))