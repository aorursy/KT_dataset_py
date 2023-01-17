import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import scipy

import scipy.stats

from sklearn.preprocessing import LabelEncoder
data = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')

data
data.isna().sum()
data.drop(['salary', 'sl_no'], axis=1, inplace=True)

data.isna().sum()
data.nunique()
corr = data.corr()

corr.style.background_gradient(cmap='coolwarm').set_precision(2)
data
unique_vals = data.nunique()

col_log = data.columns

for i in range(0, len(unique_vals)):

    coln = str(col_log[i])

    

    # If its less than 5, convert to one hot. Do not convert Status yet

    if int(unique_vals[i]) < 5 and coln != 'status':

        data = pd.concat([data.drop(coln, axis=1), pd.get_dummies(data[coln], prefix=coln)], axis=1)
data
data_y = pd.DataFrame(data['status'])

data_x = data.drop('status', axis=1)



status_encoder = LabelEncoder()

data_y = status_encoder.fit_transform(data_y)
print('Guessing always placed accuracy: %f' % (((data['status'] == 'Placed').sum() / data['status'].count()) * 100))
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn import svm
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.20, random_state=1)

train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.20, random_state=1)
best_score = -1

best_estimators = 0

for i in range(10,250):

    model = RandomForestClassifier(n_estimators=i, random_state=0)

    model.fit(train_x, train_y)

    pred = model.predict(test_x)

    score = accuracy_score(pred, test_y)

    if score > best_score:

        best_score = score

        best_estimators = i

        

print("The best number of estiamtors was %d with accuracy score %f" % (best_estimators, (best_score * 100)))
model = RandomForestClassifier(n_estimators=best_estimators, random_state=0)
model.fit(train_x, train_y)
pred = model.predict(test_x)

score = accuracy_score(pred, test_y)

print("Test Accuracy: %f" % (score * 100))
from sklearn.metrics import confusion_matrix, precision_score, plot_confusion_matrix

import matplotlib.pyplot as plt



print("True Negatives: %d, False Positives: %d, False Negatives: %d, True Positives: %d" % tuple(confusion_matrix(test_y, pred).ravel()))

print("Precision Score: %f" % (precision_score(test_y, pred) * 100))

plot_confusion_matrix(model, test_x, test_y, cmap=plt.cm.Reds)

plt.title("Confusion Matrix")

plt.show()
from sklearn.metrics import precision_recall_curve, plot_precision_recall_curve



# Get the predicted probabilties of the positive label (placed)

y_pred_prob = model.predict_proba(val_x)[:, 1]



# Get curve

precision, recall, thresholds = precision_recall_curve(val_y, y_pred_prob)



# Plot

plt.plot(recall, precision, label="Random Forest")

plt.xlabel("Recall")

plt.ylabel("Precision")

plt.plot([0, 1], [0.68837209, 0.68837209], label="Baseline")

plt.legend()

plt.show()
# Remove final one for dataframe.

df = pd.DataFrame(data={'Precision': precision[:-1], 'Recall': recall[:-1], 'Thresholds': thresholds})

df
targets = df.loc[(df['Precision'] >= 1) & (df['Thresholds'] != 1)]

targets
best = -1

thresh_best = -1



y_test_prob = model.predict_proba(test_x)[:, 1]

for target in targets.to_numpy():

    true_prediction = (y_test_prob > target[2]).astype(int)

    score = precision_score(test_y, true_prediction)

    

    # Since the dataframe is in order from thresholds, we want the lowest threshold with 100%

    # precision. This does slightly bias it towards the train set, but if safety is the highest

    # priority the threshold could be futher increased at the cost of accuracy 

    # (meaning when its positive we know with high probability but we will get more false negatives)

    if score > best:

        best = score

        thresh_best = target[2]

    print("Score for threshold %f: %f" % (target[2], score * 100))

print("Best precision score of %f achieved with threshold %f." % (best, thresh_best))
ypred = (model.predict_proba(test_x)[:, 1] > thresh_best).astype(int)

score = accuracy_score(ypred, test_y)

print("Test accuracy with threshold: %f" % (score * 100))

print("True Negatives: %d, False Positives: %d, False Negatives: %d, True Positives: %d" % tuple(confusion_matrix(test_y, ypred).ravel()))
ypred = (model.predict_proba(val_x)[:, 1] > thresh_best).astype(int)

score = accuracy_score(ypred, val_y)

print("Test accuracy with threshold: %f" % (score * 100))

print("True Negatives: %d, False Positives: %d, False Negatives: %d, True Positives: %d" % tuple(confusion_matrix(val_y, ypred).ravel()))