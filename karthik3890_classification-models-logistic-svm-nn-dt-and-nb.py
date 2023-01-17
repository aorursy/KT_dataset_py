import random
import pandas as pd
import numpy as np
bank=pd.read_csv("../input/bank-training.csv", index_col=0) # index_col=0 will remove the index column from the csv file
bank.head()
bank.shape # which tellls us the 3090 rows with 20 features
# Assign outcome as 0 if y=no and as 1 if y=yes
bank['y'] = [0 if x == 'no' else 1 for x in bank['y']]
X = bank.drop('y', 1) # 1 represents column, we are dropping as we are doing classification
y = bank.y
bank['y'].value_counts()
# 339 people opened term deposit account and 2751 have not opened the term deposit account as per training data
X.head() # dsiplays first 5 tuples of the tarining datapoints
y.head() # dsiplays first 5 tuples of the tarining datapoints labels
# Decide which categorical variables you want to use in model
for col_name in X.columns:
    if X[col_name].dtypes == 'object':# in pandas it is object
        unique_cat = len(X[col_name].unique())
        print("Feature '{col_name}' has {unique_cat} unique categories".format(col_name=col_name, unique_cat=unique_cat))
        print(X[col_name].value_counts())
        print()
# Create a list of features to dummy
todummy_list = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact','month','day_of_week','poutcome']
# Function to dummy all the categorical variables used for modeling
# dummy_na : bool, default False
# Add a column to indicate NaNs, if False NaNs are ignored even though we know thier are no missing values # Just for Reference

def dummy_df(df, todummy_list):
    for x in todummy_list:
        dummies = pd.get_dummies(df[x], prefix=x, dummy_na=False) # prefix give name
        df = df.drop(x, 1)
        df = pd.concat([df, dummies], axis=1)
    return df
X = dummy_df(X, todummy_list)
print(X.shape) # now we have 62 features instead of 20 features
# we have to same pre processing for test data set to match the algorithm to work
import pandas as pd
import numpy as np
banktest=pd.read_csv("../input/bank-test.csv", index_col=0) # index_col will remove the index column from the csv file
banktest.head()
banktest.shape # we have 1029 training datapoints with 20 features
# Assign outcome as 0 if no and as 1 if yes
banktest['y'] = [0 if x == 'no' else 1 for x in banktest['y']]


X_test = banktest.drop('y', 1) # 1 represents column, we are dropping as we are doing classification
y_test = banktest.y
banktest['y'].value_counts()
# Create a list of features to dummy
todummy_list_test = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact','month','day_of_week','poutcome']
# Function to dummy all the categorical variables used for modeling
def dummy_df(df, todummy_list):
    for x in todummy_list:
        dummies = pd.get_dummies(df[x], prefix=x, dummy_na=False) # prefix give name
        df = df.drop(x, 1)
        df = pd.concat([df, dummies], axis=1)
    return df
X_test = dummy_df(X_test, todummy_list)
X_test.shape # features increased to 60 by using dummy fucntion
print(X.shape)
print(X_test.shape)
print(y.shape)
print(y_test.shape)

# we can drop extra columns to match the test and train data set and need to deleted two columns from X
X_test.columns
X.columns
# Get missing columns in the training test
missing_cols = set( X.columns ) - set( X_test.columns )
# Add a missing column in test set with default value equal to 0
for c in missing_cols:
    X_test[c] = 0
# Ensure the order of column in the test set is in the same order than in train set
X_test = X_test[X.columns]
print(X.shape)
print(X_test.shape)
print(y.shape)
print(y_test.shape)
# Use PolynomialFeatures in sklearn.preprocessing to create two-way interactions for all features
from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures

def add_interactions(df):
    # Get feature names
    combos = list(combinations(list(df.columns), 2))
    colnames = list(df.columns) + ['_'.join(x) for x in combos] # new name giving to feature
    
    # Find interactions
    poly = PolynomialFeatures(interaction_only=True, include_bias=False)
    df = poly.fit_transform(df)
    df = pd.DataFrame(df)
    df.columns = colnames
    
    # Remove interaction terms with all 0 values            
    noint_indicies = [i for i, x in enumerate(list((df == 0).all())) if x]
    df = df.drop(df.columns[noint_indicies], axis=1)
    
    return df
X = add_interactions(X)
print(X.head(5)) # no of features increased to 1662 for the training datsaset by using above add interaction function
X_test = add_interactions(X_test)
print(X_test.head(5))# no of features increased to 1540 for the test dataset by using above add interaction function
# Get missing columns in the training test
missing_cols = set( X.columns ) - set( X_test.columns )
# Add a missing column in test set with default value equal to 0
for c in missing_cols:
    X_test[c] = 0
# Ensure the order of column in the test set is in the same order than in train set
X_test = X_test[X.columns]
# lets check all the feaures are equal in both the datasets

print(X.shape)
print(X_test.shape)
print(y.shape)
print(y_test.shape)

# Such a large set of features can cause overfitting and also slow computing
# Use feature selection to select the most important features
import sklearn.feature_selection

select = sklearn.feature_selection.SelectKBest(k=15) #which variables are selected
selected_features = select.fit(X, y)
indices_selected = selected_features.get_support(indices=True)
colnames_selected = [X.columns[i] for i in indices_selected]

X_train_selected = X[colnames_selected]
X_test_selected = X_test[colnames_selected]
print(colnames_selected)
print(X_train_selected.shape)
print(X_test_selected.shape)
print(y.shape)
print(y_test.shape)

#Create an Logistic classifier and train it on 70% of the data set.
from sklearn import svm
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf
clf.fit(X, y)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, X, y, cv=10, scoring='accuracy')
scores.mean() # Cross Validation mean score
#Prediction using test data
y_pred = clf.predict(X_test)
#classification accuracy
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred))
# Imports
from sklearn.metrics import confusion_matrix, classification_report

# Confusion matrix
cm=confusion_matrix(y_test, y_pred)
print(cm)

# New line
print('\n')

# Classification report
print(classification_report(y_test,y_pred))
# As we know beta value should be more as we are looking senistivity or recall
#consider taking beta as 10 0r 100 0r 1000 also used weighted as measure because it considers label imbalance into account
from sklearn.metrics import fbeta_score
fbeta_score(y_test, y_pred, average='binary', beta=10)
# Area under curve
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_test, y_pred)
auc
# plot confusion matrix to describe the performance of classifier.
import seaborn as sns
import matplotlib.pyplot as plt
class_label = ["No", "Yes"]
df_cm = pd.DataFrame(cm, index = class_label, columns = class_label)
sns.heatmap(df_cm, annot = True, fmt = "d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()
from sklearn.tree import DecisionTreeClassifier
class_tree = DecisionTreeClassifier(min_samples_split=30,min_samples_leaf=10,random_state=10)
class_tree.fit(X,y)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(class_tree, X, y, cv=10, scoring='accuracy')
scores.mean()
#Predictions
predictions1 = class_tree.predict(X_test)
#classification accuracy
from sklearn import metrics
print(metrics.accuracy_score(y_test, predictions1))
from sklearn.metrics import roc_auc_score
aucd = roc_auc_score(y_test, predictions1)
aucd
# Imports
from sklearn.metrics import confusion_matrix, classification_report

# Confusion matrix
cm1=confusion_matrix(y_test, predictions1)
print(cm1)

# New line
print('\n')

# Classification report
print(classification_report(y_test,predictions1))
from sklearn.metrics import fbeta_score
fbeta_score(y_test, predictions1, average='binary', beta=10)
# plot confusion matrix to describe the performance of classifier.
import seaborn as sns
import matplotlib.pyplot as plt
class_label = ["No", "Yes"]
df_cm1 = pd.DataFrame(cm1, index = class_label, columns = class_label)
sns.heatmap(df_cm1, annot = True, fmt = "d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()
from sklearn.naive_bayes import GaussianNB
NBC=GaussianNB()
NBC.fit(X,y)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(NBC, X, y, cv=10, scoring='accuracy')
scores.mean()
#Prediction using test data
y_pred2 = NBC.predict(X_test)
#classification accuracy
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred2))
from sklearn.metrics import roc_auc_score
aucnb = roc_auc_score(y_test, y_pred2)
aucnb
#Predictions
predictions2 = NBC.predict(X_test)
# Imports
from sklearn.metrics import confusion_matrix, classification_report

# Confusion matrix
cm2=confusion_matrix(y_test, predictions2)
print(cm2)

# New line
print('\n')

# Classification report
print(classification_report(y_test,predictions2))
from sklearn.metrics import fbeta_score
fbeta_score(y_test, predictions2, average='binary', beta=10)
# plot confusion matrix to describe the performance of classifier.
import seaborn as sns
import matplotlib.pyplot as plt
class_label = ["No", "Yes"]
df_nb = pd.DataFrame(cm2, index = class_label, columns = class_label)
sns.heatmap(df_nb, annot = True, fmt = "d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()
#Import svm model
from sklearn import svm

#Create a svm Classifier
sv = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
sv.fit(X, y)


# Note: Cross validation takes so much time to run for Suppor vector machines 
from sklearn.model_selection import cross_val_score
scores = cross_val_score(sv, X, y, cv=10, scoring='accuracy')
scores.mean()
#Predict the response for test dataset
y_pred3 = sv.predict(X_test)
#classification accuracy
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred3))
# Imports
from sklearn.metrics import confusion_matrix, classification_report

# Confusion matrix
cm3=confusion_matrix(y_test, y_pred3)
print(cm3)

# New line
print('\n')

# Classification report
print(classification_report(y_test,y_pred3))
from sklearn.metrics import fbeta_score
fbeta_score(y_test, y_pred3, average='binary', beta=10)
from sklearn.metrics import roc_auc_score
auc3 = roc_auc_score(y_test, y_pred3)
auc3
# plot confusion matrix to describe the performance of classifier.
import seaborn as sns
import matplotlib.pyplot as plt
class_label = ["No", "Yes"]
df_cm3 = pd.DataFrame(cm3, index = class_label, columns = class_label)
sns.heatmap(df_cm3, annot = True, fmt = "d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

mlp = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500,random_state=10)
mlp.fit(X,y)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(mlp, X, y, cv=10, scoring='accuracy')
scores.mean()
#Predict the response for test dataset
y_pred4 = mlp.predict(X_test)
#classification accuracy
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred4))
from sklearn.metrics import roc_auc_score
auc4 = roc_auc_score(y_test, y_pred4)
auc4
# Imports
from sklearn.metrics import confusion_matrix, classification_report

# Confusion matrix
cm4=confusion_matrix(y_test, y_pred4)
print(cm4)

# New line
print('\n')

# Classification report
print(classification_report(y_test,y_pred4))
from sklearn.metrics import fbeta_score
fbeta_score(y_test, y_pred4, average='binary', beta=10)
# plot confusion matrix to describe the performance of classifier.
import seaborn as sns
import matplotlib.pyplot as plt
class_label = ["No", "Yes"]
df_nn = pd.DataFrame(cm4, index = class_label, columns = class_label)
sns.heatmap(df_nn, annot = True, fmt = "d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()
from IPython.display import HTML, display
import tabulate
table = [["Algorithm Name","Accuracy","Senestivity","Precision","Auc Score","CV score"],
    ["Logistic Regression",0.90,0.17,0.70,0.58,0.89],
         ["Decision Tree",0.895,0.26,0.54,0.615,0.889,],
         ["Naive Bayes",0.8367,0.54,0.34,0.708,0.8217],
         ["Support Vector Machines",0.896,0.10,0.69,0.546,0.896],
        ["MLP Neural Netrworks",0.884,0.33,0.46,0.641,0.824]]
display(HTML(tabulate.tabulate(table, tablefmt='html')))
from IPython.display import HTML, display
import tabulate
table = [["Algorithm Name","Senestivity","Precision","Fbeta score","Auc score","CV score"],
    ["Logistic Regression",0.17,0.70,0.17,0.58,0.89],
         ["Decision Tree",0.26,0.54,0.260,0.615,0.889,],
         ["Naive Bayes",0.54,0.34,0.54,0.708,0.8217],
         ["Support Vector Machines",0.10,0.69,0.09,0.546,0.896],
        ["MLP Neural Netrworks",0.33,0.46,0.33,0.641,0.824]]
display(HTML(tabulate.tabulate(table, tablefmt='html')))
from IPython.display import HTML, display
import tabulate
table = [["Algorithm Name","True Positives","True Negatives","False Positives","False Negatives","Total Cost","Total Sales","Profit"],
    ["Logistic Regression",19,909,8,93,93*10+8*1,19*10+909*1,(19*10+909*1)-(93*10+8*1)],
         ["Decision Tree",29,892,25,83,83*10+25*1,29*10+892*1,(29*10+892*1-(83*10+25*1))],
         ["Naive Bayes",61,800,117,51,51*10+117*1,61*10+800*1,(61*10+800*1)-(51*10+117*1)],
         ["Support Vector Machines",11,912,5,101,101*10+5*1,11*10+912*1,(11*10+912*1)-(101*10+5*1)],
        ["MLP Neural Netrworks",37,873,44,75,75*10+44*1,37*10+873*1,(37*10+873*1)-(75*10+44*1)]]
display(HTML(tabulate.tabulate(table, tablefmt='html')))