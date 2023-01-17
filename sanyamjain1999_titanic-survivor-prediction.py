import numpy as np

import pandas as pd
d1 = pd.read_csv("../input/sample_submission.csv")

Xt = pd.read_csv("../input/Test.csv")

d2 = pd.read_csv("../input/Train.csv") 
d2.head()
d2.info()
columns_to_drop = ["name","ticket","cabin","embarked", "boat", "body", "home.dest"]

d2 = d2.drop(columns_to_drop,axis=1)
d2.head()
from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()



d2["sex"] = le.fit_transform(d2["sex"])
d2.head()
Xt.head()
Xt = Xt.drop(columns_to_drop,axis=1)
Xt.head()
Xt["sex"] = le.fit_transform(Xt["sex"])
Xt.head()
d2 = d2.fillna(d2["age"].mean())
d2.head()
Xt = Xt.fillna(Xt["age"].mean())

Xt.head()
from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

rf = RandomForestClassifier(criterion='entropy')

skdt = DecisionTreeClassifier(criterion='entropy')
input_cols = ['pclass','sex','age','sibsp', 'parch', 'fare']

output_col = ['survived']
y = np.array(d2[output_col]).reshape((-1,))

print(y.shape)

y
skdt.fit(d2[input_cols], y)
rf.fit(d2[input_cols], y)
pred1 = skdt.predict(Xt)

pred1
pred2 = rf.predict(Xt)

pred2
d1.head()
rs1 = d1
rs1['survived'] = pred1

rs1.head(7)
rs2 = d1

rs2['survived'] = pred2

rs2.head(7)
rs1.to_csv('submission1.csv', index=None)

rs2.to_csv('submission2.csv', index=None)