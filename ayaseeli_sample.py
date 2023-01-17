import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
df = pd.read_csv("../input/Iris.csv")

df.head(5)
x = df.drop(["Id","Species"], axis=1)

y = df["Species"]
y_dummy = pd.get_dummies(y)

y_dummy.head(5)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y_dummy, test_size=0.2, random_state = 43)
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

from sklearn.svm import SVC, LinearSVC, NuSVC

from sklearn.metrics import accuracy_score

clf = [#SVC(),

       #NuSVC(),

       #LinearSVC(),

       ExtraTreeClassifier(),

       DecisionTreeClassifier(),

       RandomForestClassifier(), 

       #AdaBoostClassifier(), 

       #GradientBoostingClassifier()

]

for i in clf:

    name = i.__class__.__name__

    i.fit(x_train, y_train)

    prediction = i.predict(x_test)

    print("="*30)

    print(name)

    print(accuracy_score(y_test, prediction))

    print("="*30)