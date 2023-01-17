# Load dataset

import pandas as pd



data = pd.read_csv("../input/creditcard.csv")
# Scale data in 'Amount' columns

from sklearn.preprocessing import StandardScaler



data['stdAmount'] = StandardScaler().fit_transform(data['Amount'].reshape(-1,1))



# Drop 'Amount'

data = data.drop(['Time', 'Amount'],axis=1)



data.head()
# Create X and y



y = data['Class']

X = data.drop(['Class'], axis=1)
# Import 'train_test_split'

from sklearn.model_selection import train_test_split



# Shuffle and split the data into training and testing subsets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Try RandomForestClassifier

'''Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks, that operate by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees (source: Wikipedia)'''



from sklearn.ensemble import RandomForestClassifier



clf1 = RandomForestClassifier()

clf1 = clf1.fit(X_train, y_train)



y_pred1 = clf1.predict(X_train)



# calculate cross validation score



from sklearn.model_selection import cross_val_score



scores = cross_val_score(clf1, X_train, y_train, cv=5)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# Compute Area Under the Curve (AUC) from prediction scores



from sklearn.metrics import roc_auc_score



AUC1 = roc_auc_score(y_train, y_pred1)

print("Roc AUC score: %0.2f " % (AUC1))
# calculate precision and recall scores



from sklearn.metrics import precision_score

from sklearn.metrics import recall_score



precision = precision_score(y_train, y_pred1, average='binary')

recall = recall_score(y_train, y_pred1, average='binary')



print("Precision score: %0.2f " % (precision))

print("Recall score: %0.2f " % (recall))
# generate confusion matrix



from sklearn.metrics import confusion_matrix

import pandas as pd

import seaborn as sns

%matplotlib inline





cm1 = confusion_matrix(y_train,y_pred1)



df_cm1 = pd.DataFrame(cm1, index = ['True (positive)', 'True (negative)'])

df_cm1.columns = ['Predicted (positive)', 'Predicted (negative)']



sns.heatmap(df_cm1, annot=True, fmt="d")
# Try Gaussian Naive Bayes Classifier

'''Naive Bayes is a simple technique for constructing classifiers: models that assign class labels to problem instances, represented as vectors of feature values, where the class labels are drawn from some finite set. It is not a single algorithm for training such classifiers, but a family of algorithms based on a common principle: all naive Bayes classifiers assume that the value of a particular feature is independent of the value of any other feature, given the class variable.(source: Wikipedia)'''

from sklearn.naive_bayes import GaussianNB 



clf2 = GaussianNB()

clf2.fit(X_train, y_train)

y_pred2 = clf2.predict(X_train)



# calculate cross validation score



scores = cross_val_score(clf2, X_train, y_train, cv=5)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
2# Compute Area Under the Curve (AUC) from prediction scores



AUC2 = roc_auc_score(y_train, y_pred2)

print("Roc AUC score: %0.2f " % (AUC2))
# calculate precision and recall score



precision = precision_score(y_train, y_pred2, average='binary')

recall = recall_score(y_train, y_pred2, average='binary')



print("Precision score: %0.2f " % (precision))

print("Recall score: %0.2f " % (recall))
# generate confusion matrix



cm2 = confusion_matrix(y_train,y_pred2)



df_cm2 = pd.DataFrame(cm2, index = ['True (positive)', 'True (negative)'])

df_cm2.columns = ['Predicted (positive)', 'Predicted (negative)']



sns.heatmap(df_cm2, annot=True, fmt="d")
# Try Logistic Regression Classifier

'''Logistic regression is a regression model where the dependent variable is categorical (source: Wikipedia)'''



from sklearn.linear_model import LogisticRegression



clf3 = LogisticRegression()

clf3.fit(X,y)

y_pred3 = clf3.predict(X_train)



# calculate cross validation score



scores = cross_val_score(clf3, X_train, y_train, cv=5)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# Compute Area Under the Curve (AUC) from prediction scores



AUC3 = roc_auc_score(y_train, y_pred3)

print("Roc AUC score: %0.2f " % (AUC3))
# calculate precision and recall score



precision = precision_score(y_train, y_pred3, average='binary')

recall = recall_score(y_train, y_pred3, average='binary')



print("Precision score: %0.2f " % (precision))

print("Recall score: %0.2f " % (recall))
# generate confusion matrix



cm3 = confusion_matrix(y_train,y_pred3)



df_cm3 = pd.DataFrame(cm3, index = ['True(positive)', 'True(negative)'])

df_cm3.columns = ['Predicted (positive)', 'Predicted (negative)']



sns.heatmap(df_cm3, annot=True, fmt="d")