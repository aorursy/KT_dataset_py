import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
df = pd.read_csv("../input/heart.csv")
df.isnull().sum()
pie_slices = [round(((df['chd'] == 1).sum() / df.shape[0])*100), round(((df['chd'] == 0).sum() / df.shape[0])*100)]

plt.pie(pie_slices, labels = ['With CHD', 'Without CHD'], colors = ['red', 'lightskyblue'], autopct = '%1.0f%%')

        

# Draw a circle at the center of pie to make it look like a doughnut

ccircle = plt.Circle((0,0), 0.75, color='white', fc='white', linewidth=1.25)

plt.gcf().gca().add_artist(ccircle)





# Set aspect ratio to be equal so that pie is drawn as a circle.

plt.axis('equal')

plt.show()  
# Change present to 1 and absent to 0 in famhist column

df.loc[df.famhist == 'Present', 'famhist'] = 1

df.loc[df.famhist == 'Absent', 'famhist'] = 0



# Plot a pie chart

sizes = [round(((df['famhist'] == 1).sum() / df.shape[0])*100), round(((df['famhist'] == 0).sum() / df.shape[0])*100)]

plt.pie(sizes, labels  = ['CHD present in family', 'CHD absent in family'], colors  = ['red', 'lightskyblue'], autopct = '%1.0f%%')

        

# Draw a circle at the center of pie to make it look like a donut

ccircle = plt.Circle((0,0), 0.75, color='white', fc='white', linewidth=1.25)

plt.gcf().gca().add_artist(ccircle)





# Set aspect ratio to be equal so that pie is drawn as a circle.

plt.axis('equal')

plt.show()  
# Get the age of chd affected people

chd_age = df.loc[df['chd'] == 1, 'age']



# Get the count of different ages

age_count = chd_age.value_counts() 



# Create a bar plot

ax = age_count.sort_index().plot(kind = 'bar', figsize = (15,10), rot = 0)
from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size = 0.2)
from sklearn.linear_model import LogisticRegression

X = df[list(train.keys())[:-1:]]

y = df[list(train.keys())[-1]]



# Build a logistic regression and compute the feature importances

model = LogisticRegression(solver='liblinear')

model.fit(X, y)
from sklearn.feature_selection import RFE

rfe = RFE(model, 9)

rfe = rfe.fit(X, y)



# Summarize the selection of the attributes

# print (rfe.support_)

print('Selected features:', list(X.columns[rfe.support_]))
# RFECV - Feature ranking with recursive feature elimination 

# and cross-validated selection of the best number of features.

from sklearn.feature_selection import RFECV

rfecv = RFECV(estimator=LogisticRegression(solver='liblinear'), step=1, cv=10, scoring='accuracy')

rfecv.fit(X, y)



print('Optimal number of features:', rfecv.n_features_)

print('Selected features:', list(X.columns[rfecv.support_]))



# Plot number of features VS. cross-validation score

plt.figure(figsize=(10,7))

plt.xlabel("Number of features selected")

plt.ylabel("Cross validation score (nb of correct classifications)")

plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)

plt.show()
selected_features = list(X.columns[rfecv.support_])

X = train[selected_features]

plt.subplots(figsize=(10, 7))

sns.heatmap(X.corr(), annot=True)

plt.show()
X_test = df[list(test.keys())[:-1:]]

y_test = df[list(test.keys())[-1]]

y_pred = model.predict(X_test)
# Get test accuracy score

from sklearn.metrics import accuracy_score

print('Test data accuracy:', round(accuracy_score(y_test,y_pred) * 100, 2),'%')
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

conf_matrix = pd.DataFrame(data=cm, columns=['Predicted:0','Predicted:1'], index=['Actual:0','Actual:1'])



plt.figure(figsize = (8,5))

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='flag');

TP = cm[1,1]

TN = cm[0,0]

FP = cm[0,1]

FN = cm[1,0]



precision = TP / (TP + FP) 

recall = TP / (TP + FN)

f1score = 2 * precision * recall / (precision + recall)



print ('Precision = ', round(precision, 2))

print ('Recall = ', round(recall, 2))

print ('F1 score = ', round(f1score, 2))

print()



sensitivity = TP / float(TP + FN)

specificity = TN / float(TN + FP)



print('Correct predictions: ', TP + TN)

print('Accuracy of the model: ',             round((TP + TN) / float(TP + TN + FP + FN), 2))

print('Misclassification: ',                 round(1-((TP + TN) / float(TP + TN + FP + FN)), 2))

print('Sensitivity or True Positive Rate: ', round(TP / float(TP + FN), 2))

print('Specificity or True Negative Rate: ', round(TN / float(TN + FP), 2))

print('Positive predictive value: ',         round(TP / float(TP + FP), 2))

print('Negative predictive Value: ',         round(TN / float(TN + FN), 2))

print('Positive likelihood Ratio: ',         round(sensitivity / (1 - specificity), 2))

print('Negative likelihood Ratio: ',         round((1 - sensitivity) / specificity, 2))
# Compute the probability of the model predicting chd=0 (No Coronary Heart Disease) and chd=1 ( Coronary Heart Disease: Yes)

# on each test data with a default classification threshold of 0.5

y_pred_prob = model.predict_proba(X_test)[:,:]

y_pred_prob_df = pd.DataFrame(data = y_pred_prob, columns = ['P(chd = 0)', 'P(chd = 1)'])

y_pred_prob_df.head()
from sklearn.preprocessing import binarize

for i in range(1,5):

    y_pred_prob_yes = model.predict_proba(X_test)

    y_pred2 = binarize(y_pred_prob_yes,i / 10)[:,1]

    cm2 = confusion_matrix(y_test, y_pred2)

    

    print('With', i/10, 'threshold, the confusion matrix is\n', cm2)

    print('with', cm2[0,0] + cm2[1,1], 'correct predictions and', cm2[1,0], 'Type II errors (False Negatives)')

    print('Accuracy of the model: ', round((cm2[1,1] + cm2[0,0]) / float(cm2[0,0] + cm2[0,1] + cm2[1,0] + cm2[1,1]), 2))

    print('Sensitivity: ', round(cm2[1,1] / (float(cm2[1,1] + cm2[1,0])), 2))

    print('Specificity: ', round(cm2[0,0] / (float(cm2[0,0] + cm2[0,1])), 2))

    print()
# Plot ROC curve

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob_yes[:,1])

plt.plot(fpr,tpr)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.title('ROC curve for Heart disease classifier')

plt.xlabel('False positive rate (1-Specificity)')

plt.ylabel('True positive rate (Sensitivity)')

plt.grid(True)
# Find area under curve

from sklearn.metrics import roc_auc_score

round(roc_auc_score(y_test,y_pred_prob_yes[:,1]), 2)