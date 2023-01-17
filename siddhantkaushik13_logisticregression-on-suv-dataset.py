import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df_suv = pd.read_csv('../input/suv-data/suv_data.csv')
df_suv.head()
sns.heatmap(df_suv.isnull(), cmap = 'YlGnBu', cbar=False)
sns.set_style('whitegrid')
sns.countplot(x = 'Purchased', hue = 'Gender', data = df_suv)
sns.distplot(df_suv['Age'], kde = False)
sns.distplot(df_suv['EstimatedSalary'], kde = False)
sns.countplot(df_suv['Purchased'], hue = 'Gender', data= df_suv)
X = df_suv.iloc[:, [2, 3]].values
y = df_suv.iloc[:, 4].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_pred, y_test)
accuracy
from sklearn.metrics import confusion_matrix
confusion_matrix(y_pred, y_test)
