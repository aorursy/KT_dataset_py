import pandas as pd
import seaborn as sb
from collections import Counter
import numpy as np
# Reading wine quality dataset
df = pd.read_csv('../input/winequality-red.csv', engine = 'python', error_bad_lines = False)
width = len(df.columns)
print(df.columns)
# Binning values of quality attribute
df['quality'] = pd.cut(df['quality'], (2, 6.5, 8), labels = [0, 1])
# Dividing dataframe to data and target labels

X = np.asarray(df.loc[:, df.columns != 'quality'])
Y = np.asarray(df.loc[:, df.columns == 'quality']).ravel()
print('Bad wine : %d, Good wine : %d'%(Counter(Y)[0], Counter(Y)[1]))
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
np.random.seed(12)
rf = RandomForestClassifier()
ada = AdaBoostClassifier(base_estimator = rf, n_estimators = 50, algorithm='SAMME')
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .5, random_state = 42)
ada.fit(X_train, Y_train)
print(classification_report(Y_test, ada.predict(X_test)))

