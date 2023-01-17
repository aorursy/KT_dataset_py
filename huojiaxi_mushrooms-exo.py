import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
df = pd.read_csv("../input/mushrooms.csv")
df.head(10)
for col in df.columns:

    print('colonne :',col)

    print(df[col].value_counts())
X = df.drop('class', axis=1)

y = df['class']
X1 = pd.get_dummies(data=df)
X1.head().T
X1.head()
label = LabelEncoder()

import sklearn

for col in df.columns:

    df[col]=label.fit_transform(df[col])    
df.head()
# on a 23 columns pour séparer x et y

y=df['class']

x=df.iloc[:,1:23]
x.shape
y.shape
x.head()
y.head()
sns.jointplot("cap-shape", "cap-color", df, kind='kde');
fig = sns.FacetGrid(df, hue="class", aspect=3, palette="Set2") # aspect=3 permet d'allonger le graphique

fig.map(sns.kdeplot, "cap-color", shade=True)

fig.add_legend()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()



X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)



X_train = pca.fit_transform(X_train)

X_test = pca.transform(X_test)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 50, criterion = 'entropy', random_state = 42)

rf.fit(X_train, y_train)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
y_rf = rf.predict(X_test)
rf_score = accuracy_score(y_test, y_rf)

print(rf_score)
cm = confusion_matrix(y_test, y_rf)

print(cm)
print(classification_report(y_test, y_rf))
def visualization_train(model,classifier):

    sns.set_context(context='notebook',font_scale=2)

    plt.figure(figsize=(16,9))

    from matplotlib.colors import ListedColormap

    X_set, y_set = X_train, y_train

    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),

                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),

             alpha = 0.6, cmap = ListedColormap(('red', 'green')))

    plt.xlim(X1.min(), X1.max())

    plt.ylim(X2.min(), X2.max())

    for i, j in enumerate(np.unique(y_set)):

        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],

                    c = ListedColormap(('red', 'green'))(i), label = j)

    plt.title("%s Training Set" %(model))

    plt.xlabel('PC 1')

    plt.ylabel('PC 2')

    plt.legend()

def visualization_test(model,classifier):

    sns.set_context(context='notebook',font_scale=2)

    plt.figure(figsize=(16,9))

    from matplotlib.colors import ListedColormap

    X_set, y_set = X_test, y_test

    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),

                         np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),

                 alpha = 0.6, cmap = ListedColormap(('red', 'green')))

    plt.xlim(X1.min(), X1.max())

    plt.ylim(X2.min(), X2.max())

    for i, j in enumerate(np.unique(y_set)):

        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],

                    c = ListedColormap(('red', 'green'))(i), label = j)

    plt.title("%s Test Set" %(model))

    plt.xlabel('PC 1')

    plt.ylabel('PC 2')

    plt.legend()
visualization_train('Random Forest',rf)
visualization_test('Random Forest',rf)
from sklearn.linear_model import Perceptron
per = Perceptron()

per.fit(X_train,y_train)

y_per = per.predict(X_test)
per_score = accuracy_score(y_test, y_per)

print("Pertinence : ")

print(per_score)

print()

print("Matrice de confusion :")

print(confusion_matrix(y_test, y_per))

print()

print("Rapport de classification :")

print(classification_report(y_test, y_per))
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

lr.fit(X_train,y_train)
y_lr = lr.predict(X_train)
lr_score = accuracy_score(y_train, y_lr)

print(lr_score)
cm = confusion_matrix(y_train, y_lr)

print(cm)
print(classification_report(y_train, y_lr))
y_lr = lr.predict(X_test)

lr_score = accuracy_score(y_test, y_lr)

print(lr_score)
from sklearn.tree import DecisionTreeClassifier as DT



dt = DT(criterion='entropy',random_state=42)

dt.fit(X_train,y_train)
y_dt = dt.predict(X_train)
dt_score = accuracy_score(y_train, y_dt)

print(dt_score)
cm = confusion_matrix(y_train, y_dt)

print(cm)
y_dt = dt.predict(X_test)
dt_score = accuracy_score(y_test, y_dt)

print(dt_score)
visualization_train('Decision tree',dt)
visualization_test('Decision tree',dt)
from xgboost import XGBClassifier

from xgboost import plot_importance
xg = XGBClassifier(learning_rate=0.01,

                      n_estimators=10,         

                      max_depth=4,              

                      min_child_weight = 1,      

                      gamma=0.,                  

                      subsample=1,           

                      colsample_btree=1,      

                      scale_pos_weight=1,     

                      random_state=27,           

                      slient = 0

                      )
xg.fit(X_train,y_train)
y_xg=xg.predict(X_train)
xg_score = accuracy_score(y_train, y_xg)

print(xg_score)
cm = confusion_matrix(y_train, y_xg)

print(cm)
y_xg=xg.predict(X_test)
xg_score = accuracy_score(y_test, y_xg)

print(xg_score)
cm = confusion_matrix(y_test, y_xg)

print(cm)
visualization_train('Xgboost',xg)
visualization_test('Xgboost',xg)