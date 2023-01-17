# @author : Gaurav Kabra
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import pandas as pd

suv = pd.read_csv("../input/suv-data/suv_data.csv")
sns.countplot(suv['Gender'])
suv.drop('User ID', inplace=True, axis=1)
gender = pd.get_dummies(suv['Gender'], drop_first=True)
suv = pd.concat([suv,gender], axis=1)

suv.head()
suv.drop('Gender',axis=1,inplace=True)
suv.head()
# scaling is required

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sns.heatmap(suv.isnull())
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
X = suv.drop('Purchased',axis=1)

y = suv['Purchased']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

X_train = sc.fit_transform(X_train)

X_test = sc.fit_transform(X_test)
logmodel = LogisticRegression()

logmodel.fit(X_train, y_train)

pred = logmodel.predict(X_test)
accuracy_score(y_test,pred)