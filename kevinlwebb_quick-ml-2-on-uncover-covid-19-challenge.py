# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_excel('/kaggle/input/covid19/dataset.xlsx')
print("# of Rows & Columns = ", df.shape)
df.head()
pd.set_option('display.max_rows', None)

df.describe().T
pd.set_option('display.max_rows', 10)
print(f"Number of columns that have more than 100 missing values: {len(df.isna().sum()[df.isna().sum() > 100])}")
for col in list(df.columns):
    print(col, " :: Unique values = ", len(df[col].unique()))
for col in ["Patient age quantile", "SARS-Cov-2 exam result", "Inf A H1N1 2009"]:
    print(col, " :")
    print(df[col].value_counts(),"\n")


df["SARS-Cov-2 exam result"].value_counts()
df["Patient age quantile"].value_counts()
df["Inf A H1N1 2009"].value_counts()
list(df.columns[~df.isna().any()]) #Provide a list of columns with 0 missing values.
for col in list(df.columns[~df.isna().any()]):
    print("Columns = ", col, " :: Unique values = ", len(df[col].unique()))
# .loc
# 
deduced_df = df.loc[:, df.isin([' ','NULL',np.nan]).mean() < .6]
deduced_df.drop(["Patient ID"], axis=1, inplace = True)
deduced_df
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from joblib import dump, load
# Read the data
df = deduced_df.copy()

# Remove rows with missing target
target = "SARS-Cov-2 exam result"
df.dropna(axis=0, subset=[target], inplace=True)

# Reset index
df.reset_index(drop=True, inplace=True)

# Replace (possible) None types with np.NaN
df.fillna(value=pd.np.nan, inplace=True)

# Separate target from predictors
y = df[target]        
X = df.drop([target], axis=1)

# Break off validation set from training data
X_train_full, X_test_full, y_train, y_test = train_test_split(X, y,
                                                                train_size=0.8,
                                                                test_size=0.2,
                                                                random_state=0)

# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
categorical_cols = [str(cname) for cname in X_train_full.columns if X_train_full[cname].dtype == "object"]

# Select numeric columns
numerical_cols = [str(cname) for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

print(len(categorical_cols),len(numerical_cols))

# Keep selected columns only
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define model
model = RandomForestClassifier(n_estimators=100, random_state=0)

# Bundle preprocessing and modeling code in a pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)
                     ])

# Preprocessing of training data, fit model 
clf.fit(X_train, y_train)
from sklearn.metrics import confusion_matrix

def evaluate_classifier(model, X_test, y_test):
    '''
    Prints labels, the confusion matrix, and accuracy of the given model
    Parameters:
        model (Pipeline):A pipeline for the input data to follow before the classifier.
        X_test (Array):The array to be used for predictions.
        y_test (Array):The array to be used for comparisons.
    '''
   
    y_pred = model.predict(X_test)
    labels = np.unique(y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred, labels=labels)
    accuracy = (y_pred == y_test).mean()

    print("Labels:", labels)
    print("Confusion Matrix:\n", confusion_mat)
    print("Accuracy:", accuracy)
evaluate_classifier(clf, X_test, y_test)
from joblib import dump
dump(clf, "/kaggle/working/classifier.pkl")
loaded_clf = load("/kaggle/working/classifier.pkl")
out_dict = {}

out_df = df.drop("SARS-Cov-2 exam result", axis = 1).copy()

for col in out_df.columns:
    out_dict[col] = [ int(input(f"Enter {col} value: ")) ]
    
predict_dict = pd.DataFrame(out_dict)

predict_dict
model.predict(predict_dict)
out_dict
