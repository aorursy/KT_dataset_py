# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv("../input/titanic/train.csv")
numeric_data = train_df.iloc[:, [0,1,2,5,6,7,9]]
numeric_data.head()
category_data = train_df.iloc[:, [3,4,8,10,11]]
category_data.head()
sns.factorplot("Sex", data = train_df, kind = "count")
sns.factorplot("Sex", kind = "count", data = train_df, hue = "Survived")
sns.factorplot("Pclass", data = train_df, kind = "count")
sns.factorplot("Pclass", data = train_df, hue = "Sex", kind = "count")
def titanic_children(passenger):
    age, sex = passenger
    if age < 16:
        return "child"
    else:
        return sex
train_df["person"] = train_df[["Age","Sex"]].apply(titanic_children, axis = 1)
sns.factorplot("Pclass", data = train_df, hue = "person", kind = "count")
train_df["Age"].hist(bins = 70)
as_fig = sns.FacetGrid(train_df, hue="Sex", aspect = 5)
as_fig.map(sns.kdeplot, "Age", shade = True)
oldest = train_df["Age"].max()
as_fig.set(xlim = (0, oldest))
as_fig.add_legend()
as_fig = sns.FacetGrid(train_df, hue="person", aspect = 5)
as_fig.map(sns.kdeplot, "Age", shade = True)
oldest = train_df["Age"].max()
as_fig.set(xlim = (0, oldest))
as_fig.add_legend()
as_fig = sns.FacetGrid(train_df, hue="Pclass", aspect = 5)
as_fig.map(sns.kdeplot, "Age", shade = True)
oldest = train_df["Age"].max()
as_fig.set(xlim = (0, oldest))
as_fig.add_legend()
train_df["Embarked"] = train_df["Embarked"].fillna('S')
sns.factorplot("Embarked", data = train_df, kind = 'count')
sns.factorplot("Embarked", data = train_df, hue = "Pclass", kind = "count")
sns.factorplot("Survived", data = train_df, kind = "count", hue ="Pclass")
sns.factorplot("Pclass","Survived", data = train_df, hue = "person")
sns.lmplot("Age", "Survived", data = train_df)
sns.lmplot("Age","Survived", data = train_df, hue = "Pclass")
sns.lmplot("Age","Survived", data = train_df, hue = "Sex")
sns.lmplot("Age","Survived", data = train_df, hue = "Embarked")
# Import modules
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
import re
from sklearn import tree
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from ycimpute.imputer import knnimput

# Figures inline and set visualization style
%matplotlib inline
sns.set()


# Import data
train_df = pd.read_csv("../input/titanic/train.csv")
test_df = pd.read_csv("../input/titanic/test.csv")
train_df.columns
train_df.info()
train_df.describe().T
train_df["Survived"].value_counts()
train_df["Pclass"].value_counts()
train_df["Age"].value_counts()
train_df["SibSp"].value_counts()
train_df["Parch"].value_counts()
train_df["Fare"].value_counts()
df_fare = train_df["Fare"]
df_fare.head()
sns.boxplot(x = df_fare);
Q1 = df_fare.quantile(0.25)
Q3 = df_fare.quantile(0.75)
IQR = Q3 - Q1
Q1
Q3
IQR
lower_limit = Q1 - 1.5*IQR
upper_limit = Q3 + 1.5*IQR
lower_limit
upper_limit
(df_fare < lower_limit) | (df_fare > upper_limit)
outlier_tf = (df_fare > upper_limit)
outlier_tf.head()
df_fare[outlier_tf]
df_fare[outlier_tf].index
type(df_fare)
df_fare = pd.DataFrame(df_fare)
df_fare.shape
clear_tf = df_fare[~((df_fare < (lower_limit)) | (df_fare > (upper_limit)).any(axis=1))]
clear_tf.shape

# 116 observations flew. When we evaluate our observation in two ways.
df_fare = train_df["Fare"]
lower_limit
df_fare[outlier_tf]
upper_limit
df_fare[outlier_tf] = upper_limit
df_fare[outlier_tf] 

# I suppressed the values above the upper limit value to the upper limit value, that is, we have suppressed the values above our threshold value according to our threshold value.
train_df = train_df.select_dtypes(include = ["float64", "int64"])
df = train_df.copy()
df= df.dropna()
df.head()
clf = LocalOutlierFactor(n_neighbors = 20, contamination = 0.1)

# n_neighbors: refers to the neighborhood number.
#contamination: an argument expressing density
clf.fit_predict(df)

# The clf object contains the formal properties of lOFT.
# We perform algorithm execution.
df_scores = clf.negative_outlier_factor_
df_scores[0:10]

# the score of each observation unit we have, i.e. the density score or LOF score
np.sort(df_scores)[0:20]

# We sorted the df scores with the sort function
threshold_value = np.sort(df_scores)[9]
outlier_tf = df_scores > threshold_value

# I saved those above the threshold as outlier_tf
outlier_tf
new_df = df[df_scores > threshold_value]

# We took all that fulfill this condition that is above the threshold.
# that means accessing non-outlier values and deleting outliers.
new_df
df[df_scores < threshold_value]

# When I reverse it, we see contrary observations.
df[df_scores == threshold_value]
print_worth = df[df_scores == threshold_value]

# I saved it as the hang value;
# Our aim is to perform the filling process according to this observation unit.
outliers = df[~outlier_tf]

# I'm also recording the outliers
outliers
res = outliers.to_records(index = False)
res[:] = print_worth.to_records(index = False)
res
df[~outlier_tf]
df[~outlier_tf] = pd.DataFrame(res, index = df[~outlier_tf].index)
df[~outlier_tf]
from sklearn.impute import KNNImputer
train_df = train_df.select_dtypes(include = ["float64","int64"])
train_df.head(20)
!pip install ycimpute
var_names = list(train_df)

# to keep the names of the variables I keep the names of the dataframe somewhere.
n_df = np.array(train_df)

# I'm creating a new array
n_df[0:10]
n_df.shape

# number of observations and variables
dff = knnimput.KNN(k=4).complete(n_df)

# neighborhood number = 4, complete: means fill
# He filled the observations he saw missing
type(dff)
dff = pd.DataFrame(dff, columns = var_names)

# Convert to pandas dataframe...
type(dff)
train_df = train_df.select_dtypes(include = ["float64","int64"])
train_df.head(20)
train_df.isnull().sum()
var_names = list(train_df)
n_df = np.array(train_df)
from ycimpute.imputer import EM
dff = EM().complete(n_df)
dff = pd.DataFrame(dff, columns = var_names)
dff.isnull().sum()
train_df = pd.read_csv("../input/titanic/train.csv")
test_df = pd.read_csv("../input/titanic/test.csv")
survived_train = train_df.Survived # store target variable of training data in a safe place
data = pd.concat([train_df.drop(["Survived"], axis = 1), test_df]) #concatenate training and test sets
data.head(20)
data.info()
data["Age"] = data.Age.fillna(data.Age.median())
data["Fare"] = data.Fare.fillna(data.Fare.median())
# Impute missing numerical variables
data.info()
# Check out info of data
data = pd.get_dummies(data, columns = ["Sex"], drop_first = True)
data.head(20)
# Select columns and view head
data = data[['Sex_male', 'Fare', 'Age','Pclass', 'SibSp']]
data.head()
data.info()
data_train = data.iloc[:891]
data_test = data.iloc[891:]
X = data_train.values
test = data_test.values
y = survived_train.values
# Instantiate model and fit to data
clf = tree.DecisionTreeClassifier(max_depth=3)
clf.fit(X, y)
# Make predictions and store in 'Survived' column of df_test
Y_pred = clf.predict(test)
test_df['Survived'] = Y_pred
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)
# Setup arrays to store train and test accuracies
dep = np.arange(1, 9)
train_accuracy = np.empty(len(dep))
test_accuracy = np.empty(len(dep))

# Loop over different values of k
for i, k in enumerate(dep):
    # Setup a Decision Tree Classifier
    clf = tree.DecisionTreeClassifier(max_depth=k)

    # Fit the classifier to the training data
    clf.fit(X_train, y_train)

    #Compute accuracy on the training set
    train_accuracy[i] = clf.score(X_train, y_train)

    #Compute accuracy on the testing set
    test_accuracy[i] = clf.score(X_test, y_test)

# Generate plot
plt.title('clf: Varying depth of tree')
plt.plot(dep, test_accuracy, label = 'Testing Accuracy')
plt.plot(dep, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Depth of tree')
plt.ylabel('Accuracy')
plt.show()