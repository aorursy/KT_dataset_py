import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from plotly.offline import init_notebook_mode, iplot, plot
import seaborn as sns
import plotly as py
init_notebook_mode(connected=True)
import plotly.graph_objs as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv',encoding='ISO-8859-1')
data1 = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv',encoding='ISO-8859-1')

data.head()
feature_cols = ["Pregnancies" , "Glucose" , "BloodPressure" , "Insulin" , "BMI" , "DiabetesPedigreeFunction" , "Age"]

X = data[feature_cols]
y = data.Outcome
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

tree_reg = DecisionTreeRegressor()
tree_reg.fit(X_train , y_train)
y_pred = tree_reg.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

