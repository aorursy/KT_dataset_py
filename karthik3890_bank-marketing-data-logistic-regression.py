import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import seaborn as sns

from pandas.tools.plotting import scatter_matrix

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.preprocessing import Imputer
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
bank=pd.read_csv("../input/bank-training.csv", index_col=0) # index_col will remove the index column from the csv file
bank.head()
# Assign outcome as 0 if income <=50K and as 1 if income >50K
bank['y'] = [0 if x == 'no' else 1 for x in bank['y']]

# Assign X as a DataFrame of features and y as a Series of the outcome variable
# axis : {0 or ‘index’, 1 or ‘columns’}, default 0
# Whether to drop labels from the index (0 or ‘index’) or columns (1 or ‘columns’).

X = bank.drop('y', 1) # 1 represents column, we are dropping as we are doing classification
y = bank.y
bank['y'].value_counts()
# 339 people opened term deposit account and 2751 have not opened the term deposit account
X.head()
y.head()
bank['y'].value_counts()
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

# prefix : string, list of strings, or dict of strings, default None
# String to append DataFrame column names.

# dummy_na : bool, default False
# Add a column to indicate NaNs, if False NaNs are ignored.

def dummy_df(df, todummy_list):
    for x in todummy_list:
        dummies = pd.get_dummies(df[x], prefix=x, dummy_na=False) # prefix give name
        df = df.drop(x, 1)
        df = pd.concat([df, dummies], axis=1)
    return df
X = dummy_df(X, todummy_list)
print(X.shape)
X.columns
del X['education_illiterate']
del X['default_yes']
import pandas as pd
import numpy as np
banktest=pd.read_csv("../input/bank-test.csv", index_col=0) # index_col will remove the index column from the csv file
banktest.head()
# Assign outcome as 0 if no and as 1 if yes
banktest['y'] = [0 if x == 'no' else 1 for x in banktest['y']]

# Assign X as a DataFrame of features and y as a Series of the outcome variable
# axis : {0 or ‘index’, 1 or ‘columns’}, default 0
# Whether to drop labels from the index (0 or ‘index’) or columns (1 or ‘columns’).

X_test = banktest.drop('y', 1) # 1 represents column, we are dropping as we are doing classification
y_test = banktest.y
banktest['y'].value_counts()
# Create a list of features to dummy
todummy_list_test = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact','month','day_of_week','poutcome']
# Function to dummy all the categorical variables used for modeling

# prefix : string, list of strings, or dict of strings, default None
# String to append DataFrame column names.

# dummy_na : bool, default False
# Add a column to indicate NaNs, if False NaNs are ignored.

def dummy_df(df, todummy_list):
    for x in todummy_list:
        dummies = pd.get_dummies(df[x], prefix=x, dummy_na=False) # prefix give name
        df = df.drop(x, 1)
        df = pd.concat([df, dummies], axis=1)
    return df
X_test = dummy_df(X_test, todummy_list)
X_test.shape
print(X.shape)
print(X_test.shape)
print(y.shape)
print(y_test.shape)
X_test.columns
X.columns
#Create an Logistic classifier and train it on 70% of the data set.
from sklearn import svm

clf = LogisticRegression()
clf
clf.fit(X, y)
#Prediction using test data
y_pred = clf.predict(X_test)
#classification accuracy
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred))
#Predictions
predictions = clf.predict(X_test)
# Imports
from sklearn.metrics import confusion_matrix, classification_report

# Confusion matrix
cm=confusion_matrix(y_test, predictions)
print(cm)

# New line
print('\n')

# Classification report
print(classification_report(y_test,predictions))
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