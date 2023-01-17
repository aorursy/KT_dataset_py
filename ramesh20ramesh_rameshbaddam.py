import pandas as pd
import numpy as np
from pandas.api.types import is_string_dtype, is_numeric_dtype
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, matthews_corrcoef, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
my_df = pd.read_csv('../input/adult-census-income/adult.csv')
my_df.head()
my_df.isnull().sum()
my_df = my_df.replace("?",np.NaN)
#Converting object type data to category
my_df[['workclass', 'education', 'marital.status','occupation','relationship','race','sex','native.country','income']].apply(lambda x: x.astype('category'))
#Category: Missing values imputation using SimpleImputer as Imputer class is deprecated
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy="most_frequent")
my_df_category = my_df[['workclass','occupation','native.country']]
imputer = imputer.fit(my_df_category[['workclass','occupation','native.country']])
my_df_category = imputer.transform(my_df_category[['workclass','occupation','native.country']])
my_df_category = pd.DataFrame(data=my_df_category , columns=[['workclass','occupation','native.country']])
my_df_category.head()
#Numeric: Missing values imputation using SimpleImputer as Imputer class is deprecated
imputer1 = SimpleImputer(missing_values = np.nan, strategy="mean")
my_df_numeric = my_df[['age','fnlwgt','education.num','capital.gain','capital.loss','hours.per.week']]
imputer1 = imputer1.fit(my_df_numeric[['age','fnlwgt','education.num','capital.gain','capital.loss','hours.per.week']])
my_df_numeric = imputer1.transform(my_df_numeric[['age','fnlwgt','education.num','capital.gain','capital.loss','hours.per.week']])
my_df_numeric = pd.DataFrame(data=my_df_numeric , columns=[['age','fnlwgt','education.num','capital.gain','capital.loss','hours.per.week']])
#merging back to original dataframe
my_df[['workclass','occupation','native.country']] = my_df_category[['workclass','occupation','native.country']]
my_df[['age','fnlwgt','education.num','capital.gain','capital.loss','hours.per.week']] = my_df_numeric[['age','fnlwgt','education.num','capital.gain','capital.loss','hours.per.week']]
import seaborn as sns
categorical_attributes = my_df[['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country', 'income']]

for i, attribute in enumerate(categorical_attributes):
    # Set the width and height of the figure
    plt.figure(figsize=(16,6))
    plt.figure(i)
    sns.countplot(categorical_attributes[attribute])
    plt.xticks(rotation=90)
X = my_df.drop(['income'], axis=1)
y = my_df['income']
# here Y variable is binary >=50, <=50
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
from sklearn import preprocessing
categorical_variables = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
for feature in categorical_variables:
        le = preprocessing.LabelEncoder()
        X_train[feature] = le.fit_transform(X_train[feature])
        X_test[feature] = le.transform(X_test[feature])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
cols = X_train.columns
temp_train = X_train.copy()
temp_test = X_test.copy()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = cols)
X_test = pd.DataFrame(scaler.fit_transform(X_test), columns = cols)

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score
#Create a Gaussian Classifier
naive = GaussianNB()
# Train the model using the training sets
naive.fit(X_train, y_train)
y_pred = naive.predict(X_test)
print("Accuracy  :",naive.score(X_test, y_test))
print("Precision :",precision_score(y_test, y_pred, average='weighted'))
# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall    :",metrics.recall_score(y_test, y_pred,average='weighted'))
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=2019, max_depth=6)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print("Accuracy  :",rf.score(X_test, y_test))
print("Precision :",precision_score(y_test, y_pred, average='weighted'))
# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall    :",metrics.recall_score(y_test, y_pred,average='weighted'))
from sklearn.linear_model import LogisticRegression
lg = LogisticRegression()
lg.fit(X_train, y_train)
y_pred = lg.predict(X_test)
print("Accuracy  :",lg.score(X_test, y_test))
print("Precision :",precision_score(y_test, y_pred, average='weighted'))
# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall    :",metrics.recall_score(y_test, y_pred,average='weighted'))
from sklearn.neighbors import KNeighborsClassifier
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Create a k-NN classifier with 7 neighbors: knn
knn = KNeighborsClassifier(n_neighbors=7)

# Fit the classifier to the training data
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
# Print the accuracy
print("Accuracy  :",knn.score(X_test, y_test))
print("Precision :",precision_score(y_test, y_pred, average='weighted'))
# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall    :",metrics.recall_score(y_test, y_pred,average='weighted'))
from sklearn.tree import DecisionTreeClassifier 
tree = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth = 3, min_samples_leaf = 5) 
# Train the model using the training sets
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
print("Accuracy  :",tree.score(X_test, y_test))
print("Precision :",precision_score(y_test, y_pred, average='weighted'))
# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall    :",metrics.recall_score(y_test, y_pred,average='weighted'))
from xgboost import XGBClassifier
xg_reg = XGBClassifier(n_estimators=250, random_state=0)
xg_reg.fit(X_train,y_train)
y_pred = xg_reg.predict(X_test)
print("Accuracy  :",xg_reg.score(X_test, y_test))
print("Precision :",precision_score(y_test, y_pred, average='weighted'))
# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall    :",metrics.recall_score(y_test, y_pred,average='weighted'))
