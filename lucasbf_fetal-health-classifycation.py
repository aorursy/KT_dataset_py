
!pip install 'seaborn==0.11'
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree, export_graphviz
from sklearn.metrics import accuracy_score, classification_report, plot_confusion_matrix, confusion_matrix
import sklearn
sklearn.__version__
!ls /kaggle/input/fetal-health-classification/
df_fetal = pd.read_csv('/kaggle/input/fetal-health-classification/fetal_health.csv')
df_fetal.info()
df_fetal.head(10)
df_fetal.groupby('fetal_health').count()
df_fetal.describe()
m = {1: 0., 2: 1., 3: 1.}
df_fetal.loc[:, 'fetal_health'] = df_fetal.loc[:, 'fetal_health'].map(m)
df_fetal.groupby('fetal_health').count()
# Pegando o target y
y = df_fetal.loc[:, 'fetal_health'].values
# Pegando as features 
X = df_fetal.drop('fetal_health', axis=1).values
# Separando os conjuntos em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=58)
clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X_train, y_train)
print(export_text(clf))
import graphviz
data = export_graphviz(clf, out_file=None,
                       feature_names=df_fetal.columns.drop('fetal_health').values,
                       class_names=df_fetal.columns[21],
                       filled=True, rounded=True,
                       special_characters=True)
graph = graphviz.Source(data)
graph
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
sns.heatmap(confusion_matrix(y_pred, y_test))
