import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier
data = pd.read_csv(r'../input/iris-flower-dataset/IRIS.csv')

data
data.info()
data.describe(include = 'all')
x = data[['sepal_length','sepal_width','petal_length','petal_width']]

y = data['species']
enc = LabelEncoder()

y_enc = enc.fit_transform(y)
sc = StandardScaler()

x_sc = sc.fit_transform(x)
x_train,x_test,y_train,y_test = train_test_split(x,y_enc, test_size = 0.3, random_state = 4)
logreg = LogisticRegression()

logreg.fit(x_train,y_train)
logreg.predict(x_test)
log_accuracy = logreg.score(x_test,y_test)*100

print('The Accuracy Of The Logistic Regression Model is ', +  log_accuracy,'%' )
dec = DecisionTreeClassifier()

dec.fit(x_train,y_train)
dec_accuracy = dec.score(x_test,y_test)*100

print('The Accuracy Of The Decision Tree Classifier Model is ', +  dec_accuracy,'%' )