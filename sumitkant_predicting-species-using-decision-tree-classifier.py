# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Loading Data

df = pd.read_csv('../input/Iris.csv')

df.head()
df.dtypes
categorical_vars = df.dtypes[df.dtypes == 'object'].index

categorical_vars
# Determine unique entities for each categorical variables. Here we have only one

df[categorical_vars].apply(lambda x: len(x.unique()))
df['Species'].value_counts()
df.describe()
import seaborn as sns
sns.lmplot('SepalLengthCm','SepalWidthCm', data = df, hue = 'Species', fit_reg = False)
sns.lmplot('PetalLengthCm','PetalWidthCm', data = df, hue= 'Species', fit_reg = False )
sns.lmplot('SepalLengthCm','PetalWidthCm', data = df, hue= 'Species', fit_reg = False )
sns.lmplot('PetalLengthCm','SepalWidthCm', data = df, hue= 'Species', fit_reg = False )
df.apply(lambda x: sum(x.isnull()))
X = df

y = df.index

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
dependent_var = 'Species'

independent_vars = [x for x in X_train.columns if not x in ['Id', dependent_var]]

independent_vars
# Initialize the algorithm

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth = 6, min_samples_leaf =30, max_features = 'sqrt')

model.fit(X_train[independent_vars], X_train[dependent_var])
# Making Predictions

predictions_train = model.predict(X_train[independent_vars])

predictions_test = model.predict(X_test[independent_vars])
# Analysing Metrics

from sklearn.metrics import accuracy_score
# For Training Set

acc_train  = accuracy_score(X_train[dependent_var], predictions_train)

# For test set

acc_test = accuracy_score(X_test[dependent_var], predictions_test)
print ('Accuracy Score for training is {} and testing dataset is {}'.format(acc_train, acc_test))