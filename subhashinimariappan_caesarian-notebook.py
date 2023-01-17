# import libraries

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_curve

from sklearn.metrics import auc
# data doesn't have headers, so let's create headers

_headers = ['Age', 'Delivery_Nbr', 'Delivery_Time', 'Blood_Pressure', 'Heart_Problem', 'Caesarian']

# read in cars dataset

df = pd.read_csv('https://raw.githubusercontent.com/PacktWorkshops/The-Data-Science-Workshop/master/Chapter06/Dataset/caesarian.csv.arff', names=_headers, index_col=None, skiprows=15)

df.head()

# target column is 'Caesarian'

features = df.drop(['Caesarian'], axis=1).values

labels = df[['Caesarian']].values



# split 80% for training and 20% into an evaluation set

X_train, X_eval, y_train, y_eval = train_test_split(features, labels, test_size=0.2, random_state=0)



# further split the evaluation set into validation and test sets of 10% each

X_val, X_test, y_val, y_test = train_test_split(X_eval, y_eval, test_size=0.5, random_state=0)

model = LogisticRegression()

model.fit(X_train, y_train)

y_proba = model.predict_proba(X_val)
from sklearn.metrics import roc_auc_score

_auc = roc_auc_score(y_val, y_proba[:, 0])

print(_auc)

_false_positive, _true_positive, _thresholds = roc_curve(y_val, y_proba[:, 0])
print(_false_positive)
print(_true_positive)
print(_thresholds)
# Plot the RoC

import matplotlib.pyplot as plt

%matplotlib inline



plt.plot(_false_positive, _true_positive, lw=2, label='Receiver Operating Characteristic')

plt.xlim(0.0, 1.2)

plt.ylim(0.0, 1.2)

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver Operating Characteristic')

plt.show()
