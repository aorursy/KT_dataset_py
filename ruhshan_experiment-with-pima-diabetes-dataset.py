import numpy as np

import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split

from sklearn import svm

from sklearn import neighbors

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

%matplotlib inline

# reading data

df = pd.read_csv('../input/diabetes.csv')





#visualizing correlations

correlations = {}



for col in df.columns:

    correlations[col]=df[col].corr(df['Outcome'])



correlations = pd.DataFrame.from_dict(correlations,orient='index')



correlations.plot.bar()





X = df.iloc[:, 0:8]

y = df.iloc[:, 8]



replace_zero_in = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']



for column in replace_zero_in:

    X[column] = X[column].replace(0, np.NaN)

    mean = int(X[column].mean(skipna=True))

    X[column] = X[column].replace(np.NaN, mean)



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.20)
#trying with cross_validator

clf = svm.SVC(kernel='linear', C=1,degree=3)

scores = cross_val_score(clf, X, y, cv=10)

print(max(scores))

clf = neighbors.KNeighborsClassifier(n_neighbors=30, leaf_size=100, algorithm='kd_tree', p=3)

scores = cross_val_score(clf, X, y, cv=10)

print(max(scores))