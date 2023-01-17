import pandas as pd 

import seaborn as sns

data = pd.read_csv('../input/data.csv')

data.head()
data.shape

sns.jointplot('radius_mean','texture_mean',data=data)
sns.heatmap(data.corr())
X = data[['radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst']]

y = data['diagnosis']
from sklearn.cross_validation import train_test_split

from sklearn.linear_model import LogisticRegression

lm = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

lm.fit(X_train,y_train)

pred = lm.predict(X_test)
from sklearn.metrics import classification_report

print(classification_report(y_test, pred))