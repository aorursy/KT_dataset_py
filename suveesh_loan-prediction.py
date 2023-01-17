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
X_train = pd.read_csv('/kaggle/input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv')

X_test = pd.read_csv('/kaggle/input/loan-prediction-problem-dataset/test_Y3wMUE5_7gLdaTN.csv')
X_train.head()
X_test.head()
X_train.shape
X_train.isnull().sum().sort_values(ascending=False)
X_train = X_train.drop(["Loan_ID"], axis = 1)

X_test = X_test.drop(["Loan_ID"], axis = 1)
r = X_train[X_train['Married'].isnull()].index.tolist()
X_train.drop(X_train.index[r])
X_train.isnull().sum().sort_values(ascending=False)
X_test.isnull().sum().sort_values(ascending=False)
X_train.describe()
import matplotlib.pyplot as plt

import seaborn as sns

X_train.hist(figsize = (10,10))

plt.show()
# Get list of categorical variables

y = X_train.Loan_Status

X_train = X_train.drop(['Loan_Status'], axis=1)

s = (X_train.dtypes == 'object')

object_cols = list(s[s].index)



print("Categorical variables:")

print(object_cols)



from sklearn.preprocessing import OneHotEncoder



# Remove categorical columns (will replace with one-hot encoding)

num_X_train = X_train.drop(object_cols, axis=1)

num_X_test = X_test.drop(object_cols, axis=1)

cat_X_train = X_train[object_cols]

cat_X_test = X_test[object_cols]
cat_X_train = cat_X_train.fillna(method = 'ffill')

cat_X_test = cat_X_test.fillna(method = 'ffill')
from sklearn.impute import SimpleImputer



my_imputer = SimpleImputer()



imputed_num_X_train = pd.DataFrame(my_imputer.fit_transform(num_X_train))

imputed_num_X_test = pd.DataFrame(my_imputer.transform(num_X_test))



# Imputation removed column names; put them back

imputed_num_X_train.columns = num_X_train.columns

imputed_num_X_test.columns = num_X_test.columns



from sklearn.preprocessing import LabelEncoder



# Make copy to avoid changing original data 

label_X_train = X_train.copy()

label_X_test = X_test.copy()



# Apply label encoder to each column with categorical data

label_encoder = LabelEncoder()

for col in object_cols:

    cat_X_train[col] = label_encoder.fit_transform(cat_X_train[col])

    cat_X_test[col] = label_encoder.transform(cat_X_test[col])
cat_X_train.head(3)
cat_X_test.head(3)
concat_X_train = pd.concat([imputed_num_X_train, cat_X_train], axis = 1, sort = False)

concat_X_test = pd.concat([imputed_num_X_test, cat_X_test], axis =1, sort = False)



concat_X_train.head()
y.isnull().any()
target = {'Y': 1, 'N':0}

y_train = y.map(target)

y_train.head()
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC
models = []

models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

models.append(('SVM', SVC(gamma='auto')))

# evaluate each model in turn

results = []

names = []

for name, model in models:

    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)

    cv_results = cross_val_score(model, concat_X_train, y_train, cv=kfold, scoring='accuracy')

    results.append(cv_results)

    names.append(name)

    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
# Compare Algorithms

plt.boxplot(results, labels=names)

plt.title('Algorithm Comparison')

plt.show()
# Make predictions on validation dataset

model = LinearDiscriminantAnalysis()

model.fit(concat_X_train, y_train)

predictions = model.predict(concat_X_test)