from IPython.display import YouTubeVideo

YouTubeVideo("pN4HqWRybwk")
# show plots inside the notebook  

%matplotlib inline 



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
diabetes_dataset = pd.read_csv("../input/diabetes.csv")
diabetes_dataset.shape
diabetes_dataset.head()
diabetes_dataset.describe()
diabetes_dataset.groupby("Outcome").size()
# This replaces zero/invalid values with the mean in the group.

# But it does not seem to improve the results, that's why it's deactivated.

# dataset_nozeros = diabetes_dataset.copy()



# zero_fields = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'] 

# diabetes_dataset[zero_fields] = diabetes_dataset[zero_fields].replace(0, np.nan)

# diabetes_dataset[zero_fields] = diabetes_dataset[zero_fields].fillna(dataset_nozeros.mean())

# diabetes_dataset.describe()  # check that there are no invalid values left
from sklearn.model_selection import train_test_split 



# divide into training and testing data

train,test = train_test_split(diabetes_dataset, test_size=0.25, random_state=0, stratify=diabetes_dataset['Outcome']) 



# separate the 'Outcome' column from training/testing data

train_X = train[train.columns[:8]]

test_X = test[test.columns[:8]]

train_Y = train['Outcome']

test_Y = test['Outcome']
from sklearn.linear_model import LogisticRegression



model = LogisticRegression()

model.fit(train_X,train_Y)

prediction = model.predict(test_X)
# calculate accuracy

from sklearn import metrics



print(metrics.accuracy_score(test_Y, prediction))
the_most_outcome = diabetes_dataset['Outcome'].median()

prediction2 = [the_most_outcome for i in range(len(test_Y))]

print(metrics.accuracy_score(test_Y, prediction2))
from sklearn import metrics



confusion_matrix = metrics.confusion_matrix(test_Y, prediction)

confusion_matrix
plt.figure()

plt.matshow(confusion_matrix, cmap='Pastel1')



for x in range(0, 2):

    for y in range(0, 2):

        plt.text(x, y, confusion_matrix[x, y])

        

plt.ylabel('expected label')

plt.xlabel('predicted label')

plt.show()
# [row, column]

TP = confusion_matrix[1, 1]

TN = confusion_matrix[0, 0]

FP = confusion_matrix[0, 1]

FN = confusion_matrix[1, 0]



print("Sensitivity: %.4f" % (TP / float(TP + FN)))

print("Specificy  : %.4f" % (TN / float(TN + FP)))
# print the first 10 predicted responses

# 1D array (vector) of binary values (0, 1)

model.predict(test_X)[0:10]
# print the first 10 predicted probabilities of class membership

model.predict_proba(test_X)[0:10]
# histogram of predicted probabilities



save_predictions_proba = model.predict_proba(test_X)[:, 1]  # column 1



plt.hist(save_predictions_proba, bins=10)

plt.xlim(0,1) # x-axis limit from 0 to 1

plt.title('Histogram of predicted probabilities')

plt.xlabel('Predicted probability of diabetes')

plt.ylabel('Frequency')

plt.show()
# predict diabetes if the predicted probability is greater than 0.3

from sklearn.preprocessing import binarize



# it will return 1 for all values above 0.3 and 0 otherwise

# results are 2D so we slice out the first column

prediction2 = binarize(save_predictions_proba.reshape(-1, 1), 0.3)  # [0]
confusion_matrix2 = metrics.confusion_matrix(test_Y, prediction2)

confusion_matrix2



# previous confusion matrix

# array([[110,  15],

#        [ 28,  39]])
TP = confusion_matrix2[1, 1]

TN = confusion_matrix2[0, 0]

FP = confusion_matrix2[0, 1]

FN = confusion_matrix2[1, 0]



print("new Sensitivity: %.4f" % (TP / float(TP + FN)))

print("new Specificy  : %.4f" % (TN / float(TN + FP)))



# old Sensitivity: 0.5821

# old Specificy  : 0.8800
from sklearn.metrics import roc_curve, auc



# function roc_curve

# input: IMPORTANT: first argument is true values, second argument is predicted probabilities

#                   we do not use y_pred_class, because it will give incorrect results without 

#                   generating an error

# output: FPR, TPR, thresholds

# FPR: false positive rate

# TPR: true positive rate

FPR, TPR, thresholds = roc_curve(test_Y, save_predictions_proba)



plt.figure(figsize=(10,5))  # figsize in inches

plt.plot(FPR, TPR)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # 50%  

plt.plot(FPR, TPR, lw=2, label='Logaristic Regression (AUC = %0.2f)' % auc(FPR, TPR))

plt.title('ROC curve for diabetes classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.grid(True)

plt.legend(loc="lower right")
# define a function that accepts a threshold and prints sensitivity and specificity

def evaluate_threshold(threshold):

    print("Sensitivity: %.4f" % (TPR[thresholds > threshold][-1]))

    print("Specificy  : %.4f" % (1 - FPR[thresholds > threshold][-1]))



print ('Threshold = 0.5')

evaluate_threshold(0.5)

print ()

print ('Threshold = 0.35')

evaluate_threshold(0.35)
spec = []

sens = []

thres = []



threshold = 0.00

for x in range(0, 90):

    thres.append(threshold)

    sens.append(TPR[thresholds > threshold][-1])

    spec.append(1 - FPR[thresholds > threshold][-1])

    threshold += 0.01

    

plt.plot(thres, sens, lw=2, label='Sensitivity')

plt.plot(thres, spec, lw=2, label='Specificity')

ax = plt.gca()

ax.set_xticks(np.arange(0, 1, 0.1))

ax.set_yticks(np.arange(0, 1, 0.1))

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.title('Sensitivity vs. Specificity')

plt.xlabel('Threshold')

plt.grid(True)

plt.legend(loc="center right")