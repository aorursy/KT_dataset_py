# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
red_wine_quality = pd.read_csv("/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")
# Bibliotheken, die wir nutzen
import numpy as np
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, recall_score, precision_score
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import linear_model
import xgboost as xgb
from math import sqrt
from prettytable import PrettyTable
import plotly.express as px
import seaborn as sns
# Anzeigen des Datensatzes
red_wine_quality
# Bestimmung der Form
red_wine_quality.shape
# Untersuchung der Datentypen
red_wine_quality.info()
# Untersuchung der Missing Values
red_wine_quality.isnull().sum()
# Anzeigen der Header
red_wine_quality.head(10)
red_wine_quality['quality'].plot(kind = 'hist')
plt.xlabel('Qualität')
plt.show()
#Korrelationen berechnen und sortieren 
correlations = red_wine_quality.corr()['quality'].sort_values(ascending=False)
print(correlations)
correlations.plot(kind='bar')
#Erstellung einer Korrelationsmatrix, um die Korrelationen zwischen den Features abzubilden
plt.figure(figsize=(15,6))
sns.heatmap(red_wine_quality.corr(), annot=True, fmt = '.0%')
#Zeige Korrelationen, die größer als 0.2 sind
print(abs(correlations) > 0.2)
#Target Variable 'quality' in eine binäre Variable teilen
red_wine_quality['goodquality'] = [1 if x >= 7 else 0 for x in red_wine_quality['quality']]

#Aufteilen in Feature Variable und Zielvariable
X = red_wine_quality.drop(['quality', 'goodquality'], axis = 1)
y = red_wine_quality['goodquality']
#Verhältnis der Ausprägungen von guten und schlechten Weinen
red_wine_quality['goodquality'].value_counts()
#Standardisieren der Features
X_features = X
X = StandardScaler().fit_transform(X)
#Aufteilen der Daten in Trainings- und Test-Daten
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.25, random_state=0)
model1 = DecisionTreeClassifier(random_state=1)
model1.fit(X_train, y_train)
y_pred1 = model1.predict(X_test)
print(classification_report(y_test, y_pred1))
#Durchführen des RandomForestClassifier
model2 = RandomForestClassifier(random_state=1)
model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)
print(classification_report(y_test, y_pred2))
model4 = GradientBoostingClassifier(random_state=1)
model4.fit(X_train, y_train)
y_pred4 = model4.predict(X_test)
print(classification_report(y_test, y_pred4))

model5 = xgb.XGBClassifier(random_state=1)
model5.fit(X_train, y_train)
y_pred5 = model5.predict(X_test)
print(classification_report(y_test, y_pred5))
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
knn.predict_proba(X_test)[:,1]
from sklearn import metrics
cm = metrics.confusion_matrix(y_test, y_pred)
print(cm)
from sklearn.metrics import ConfusionMatrixDisplay
cmd = ConfusionMatrixDisplay(cm, display_labels=['bad', 'good'])
cmd.plot()
feat_importances = pd.Series(model2.feature_importances_, index=X_features.columns)
feat_importances.nlargest(25).plot(kind='barh',figsize=(10,10))
feat_importances = pd.Series(model5.feature_importances_, index=X_features.columns)
feat_importances.nlargest(25).plot(kind='barh',figsize=(10,10))
# Filtering df for only good quality
red_wine_quality_temp = red_wine_quality[red_wine_quality['goodquality']==1]
red_wine_quality_temp.describe()
# Filtering df for only bad quality
red_wine_quality_temp2 = red_wine_quality[red_wine_quality['goodquality']==0]
red_wine_quality_temp2.describe()