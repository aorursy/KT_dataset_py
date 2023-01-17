# Directive pour afficher les graphiques dans Jupyter
%matplotlib inline
# Pandas : librairie de manipulation de données
# NumPy : librairie de calcul scientifique
# MatPlotLib : librairie de visualisation et graphiques
# SeaBorn : librairie de graphiques avancés
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
iris=pd.read_csv('/kaggle/input/iris/Iris.csv')
iris.head(10)

iris['Species']
from IPython.core.display import HTML # permet d'afficher du code html dans jupyter
display(HTML(iris.head(10).to_html()))
iris.shape
iris.describe()
iris.columns
iris.Species.value_counts()
sns.jointplot("PetalLengthCm", "PetalWidthCm", iris, kind='kde');
sns.boxplot(x="Species", y="SepalLengthCm", data=iris)
sns.violinplot(x="Species", y="SepalLengthCm", data=iris)
fig = sns.FacetGrid(iris, hue="Species", aspect=3, palette="Set2") # aspect=3 permet d'allonger le graphique
fig.map(sns.kdeplot, "SepalLengthCm", shade=True)
fig.add_legend()
sns.pairplot(iris, hue="Species")
iris['classe'] = iris.Species.map({"Iris-setosa":0, "Iris-versicolor":1, "Iris-virginica":2})
iris.head()
iris = iris.drop(['Species'], axis=1) 
data_train = iris.sample(frac=0.8)          # 80% des données avec frac=0.8
data_test = iris.drop(data_train.index) 
X_train = data_train.drop(['classe'], axis=1)
y_train = data_train.classe
X_test = data_test.drop(['classe'], axis=1)
y_test = data_test.classe
plt.figure(figsize=(9,9))

logistique = lambda x: np.exp(x)/(1+np.exp(x))   

x_range = np.linspace(-10,10,50)       
y_values = logistique(x_range)

plt.plot(x_range, y_values, color="red")
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)
y_lr = lr.predict(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix
lr_score = accuracy_score(y_test, y_lr)
print(lr_score)
cm = confusion_matrix(y_test, y_lr)
print(cm)
pd.crosstab(y_test, y_lr, rownames=['Reel'], colnames=['Prediction'], margins=True)
from sklearn import tree
dtc = tree.DecisionTreeClassifier()
dtc.fit(X_train,y_train)
y_dtc = dtc.predict(X_test)
print(accuracy_score(y_test, y_dtc))
plt.figure(figsize=(30,30))
tree.plot_tree(dtc, feature_names=X_train.columns, class_names=['0','1','2'], fontsize=14, filled=True)  
from sklearn import ensemble
rf = ensemble.RandomForestClassifier()
rf.fit(X_train, y_train)
y_rf = rf.predict(X_test)
rf_score = accuracy_score(y_test, y_rf)
print(rf_score)
pd.crosstab(y_test, y_rf, rownames=['Reel'], colnames=['Prediction'], margins=True)
importances = rf.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(12,8))
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), iris.columns[indices])
plt.title('Importance des caracteristiques')