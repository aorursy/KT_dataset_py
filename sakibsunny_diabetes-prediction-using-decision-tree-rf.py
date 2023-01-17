#Importing modules

import pandas as pd

from sklearn import tree
data = pd.read_csv('../input/diabetes.csv',header = 0)

data.head()
data.info()
data['BMI'] = data['BMI'].astype(int)

data['DiabetesPedigreeFunction'] = data['DiabetesPedigreeFunction'].astype(int)
features = list(data.columns[:8])

features
y = data['Outcome']

x = data[features]

Tree = tree.DecisionTreeClassifier()

Tree = Tree.fit(x,y)



output = Tree.predict([0,135,40,35,164,41,2.2,32])

print (output)
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators = 100)

forest = forest.fit(x,y)



output =  forest.predict([4,85,40,35,94,28.1,.97,33])

print (output)