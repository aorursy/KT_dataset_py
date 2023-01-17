import seaborn as sns

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
path = "/kaggle/input/vertebralcolumndataset/"

df1 = pd.read_csv(path+'column_2C.csv', delimiter=',')

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')
df1.head(5)
sns.pairplot(df1, hue="class", size=3, diag_kind="kde")
df1['class'] = df1['class'].map({'Normal': 0, 'Abnormal': 1})
from sklearn.model_selection import train_test_split

from sklearn import model_selection

from sklearn.metrics import accuracy_score

X = df1[['pelvic_incidence','pelvic_tilt','lumbar_lordosis_angle','sacral_slope', 'pelvic_radius','degree_spondylolisthesis']]

Y = df1['class']

# split data into train and test sets

seed = 2020

test_size = 0.33

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
import xgboost as xgb



# fit model no training data

model = xgb.XGBClassifier()

model.fit(X_train, y_train)

# save model to file

model.save_model("model.bst")
# make predictions for test data

y_pred = model.predict(X_test)

predictions = [round(value) for value in y_pred]

# evaluate predictions

accuracy = accuracy_score(y_test, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
# make predictions proba for test data

y_pred_prob = model.predict_proba(X_test)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc



false_pos_rate, true_pos_rate, proba = roc_curve(y_test, y_pred_prob[:, -1])

plt.figure()

plt.plot([0,1], [0,1], linestyle="--") # plot random curve

plt.plot(false_pos_rate, true_pos_rate, marker=".", label=f"AUC = {roc_auc_score(y_test, predictions)}")

plt.title("ROC Curve")

plt.ylabel("True Positive Rate")

plt.xlabel("False Positive Rate")

plt.legend(loc="lower right")
num_classes = 2



fpr = dict()

tpr = dict()

thresholds = dict()

roc_auc = dict()

# Compute False Positive and True Positive Rates for each class

for i in range(num_classes):

    fpr[i], tpr[i], thresholds[i] = roc_curve(y_test, y_pred_prob[:, -1], drop_intermediate=False)

    roc_auc[i] = auc(fpr[i], tpr[i])
J_stats = [None]*num_classes

opt_thresholds = [None]*num_classes



# Compute Youden's J Statistic for each class

for i in range(num_classes):

    J_stats[i] = tpr[i] - fpr[i]

    opt_thresholds[i] = thresholds[i][np.argmax(J_stats[i])]

    print('Optimum threshold for classe ',i,': '+str(opt_thresholds[i]))

    
optimal_proba_cutoff = sorted(list(zip(np.abs(true_pos_rate - false_pos_rate), y_pred_prob[:, -1])), key=lambda i: i[0], reverse=True)[0][1]

roc_predictions = [1 if i >= optimal_proba_cutoff else 0 for i in y_pred_prob[:,-1]]
optimal_proba_cutoff
print("Accuracy Score Before Thresholding: {}".format(accuracy_score(y_test, predictions)))

print("Precision Score Before Thresholding: {}".format(precision_score(y_test, predictions)))

print("Recall Score Before Thresholding: {}".format(recall_score(y_test, predictions)))

print("F1 Score Before Thresholding: {}".format(f1_score(y_test, predictions)))

print("ROC AUC Score: {}".format(roc_auc_score(y_test, y_pred_prob[:, -1])))
print("Accuracy Score Before and After Thresholding: {}, {}".format(accuracy_score(y_test, predictions), accuracy_score(y_test, roc_predictions)))

print("Precision Score Before and After Thresholding: {}, {}".format(precision_score(y_test, predictions), precision_score(y_test, roc_predictions)))

print("Recall Score Before and After Thresholding: {}, {}".format(recall_score(y_test, predictions), recall_score(y_test, roc_predictions)))

print("F1 Score Before and After Thresholding: {}, {}".format(f1_score(y_test, predictions), f1_score(y_test, roc_predictions)) )
y_actual = pd.Series(y_test, name='Actual')

y_predict_tf = pd.Series(roc_predictions, name='Predicted')

df_confusion = pd.crosstab(y_actual, y_predict_tf, rownames=['Actual'], colnames=['Predicted'], margins=True)

print (df_confusion)