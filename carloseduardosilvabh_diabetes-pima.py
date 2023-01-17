# https://www.linkedin.com/in/carlos-eduardo-silva-80463316/

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
pima = pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")
pima.head()
from pandas_profiling import ProfileReport
#Generating the Visualization
report = ProfileReport(pima, minimal=False, progress_bar=False)
report.to_notebook_iframe()
pima.describe()
pima.info()
pima.shape
pima.hist(figsize=(18,12), bins=15, edgecolor = 'white')
# Diabetes per class
pima1=pima[pima['Outcome']==1]
columns = pima.columns[:8]
length = len(columns)
plt.subplots(figsize=(18,15))
for i,j in zip(columns, range(length)):
    plt.subplot((length/2),3,1+j)
    pima1[i].hist(bins=20, edgecolor = 'black')
    plt.title(i)
plt.show()
sns.countplot(x='Outcome', data=pima)
plt.title('Outcome Values')
pima.plot(kind='box', subplots=True, layout=(3,3), figsize=(14,10) )
sns.boxplot(x='Outcome', y='Glucose', data=pima)
pima['Pregnancies'].value_counts().plot.bar()
sns.pairplot(pima, hue='Outcome')
# Correlation without treating the data
corr = pima.corr()
plt.figure(figsize=(12,10))
#cmap = sns.diverging_palette(180,5, as_cmap=True)
sns.heatmap(corr, square=True,annot=True, cmap='coolwarm')
pima[pima.isnull().any(axis=1)] 
# Values iqual to zero
(pima[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']] == 0).sum()
# making pivot for find values mean by atributes and class
pima.pivot_table(index='Outcome',values=['Glucose','Insulin','BloodPressure','SkinThickness','BMI'])\
.round(decimals=2)
# values equal to zero by class
print("Value Glucose per", pima[pima.Glucose == 0].groupby('Outcome')['Outcome'].count())
print("Value BloodPressure per", pima[pima.BloodPressure == 0].groupby('Outcome')['Outcome'].count())
print("Value SkinThickness per", pima[pima.SkinThickness == 0].groupby('Outcome')['Outcome'].count())
print("Value Insulin per", pima[pima.Insulin == 0].groupby('Outcome')['Outcome'].count())
print("Value BMI per", pima[pima.BMI == 0].groupby('Outcome')['BMI'].count())
pima.loc[(pima['Outcome'] == 1) & (pima['Insulin']==0), 'Insulin'] = 100.34
pima.loc[(pima['Outcome'] == 0) & (pima['Insulin']==0), 'Insulin'] = 68.79
pima.loc[(pima['Outcome']==1) & (pima['Glucose']==0), 'Glucose'] = 141.26
pima.loc[(pima['Outcome']==0) & (pima['Glucose']==0), 'Glucose'] = 109.98
pima.loc[(pima['Outcome']==0) & (pima['BloodPressure']==0), 'BloodPressure'] = 68.18
pima.loc[(pima['Outcome']==1) & (pima['BloodPressure']==0), 'BloodPressure'] = 70.82
pima.loc[(pima['Outcome']==0) & (pima['SkinThickness']==0), 'SkinThickness'] = 19.66
pima.loc[(pima['Outcome']==1) & (pima['SkinThickness']==0), 'SkinThickness'] = 22.16
pima.loc[(pima['Outcome']==0) & (pima['BMI'] == 0), 'BMI'] = 30.30
pima.loc[(pima['Outcome']==1) & (pima['BMI'] == 0), 'BMI'] = 30.30
pima.hist(figsize=(18,12), bins=20, edgecolor = 'black')
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, plot_confusion_matrix
X = pima.iloc[:, 0:8]
y = pima.iloc[:,8]
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
features=["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]
X
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0 )
model = RandomForestClassifier(n_estimators=50, random_state=0,
                              criterion='entropy', max_depth=3,
                              max_features = 'log2', min_samples_split=4,
                              )
model.fit(X_train,y_train)

lista = []
pred = model.predict(X_test)
score = accuracy_score(pred, y_test)
lista.append(score)
print(score)
model.feature_importances_
pd.DataFrame({
    "Variable": features,
    "coef": model.feature_importances_
}) \
.round(decimals=2) \
.sort_values('coef', ascending=False)\
.style.bar(color=['grey', 'lightblue'], align ='zero')
import eli5
from eli5.sklearn import PermutationImportance
permutation = PermutationImportance(model, random_state = 0)
permutation.fit(X_test, y_test)
eli5.show_weights(permutation, feature_names = features)
corr = pima.corr()
plt.figure(figsize=(12,10))
cmap = sns.diverging_palette(180,5, as_cmap=True)
sns.heatmap(corr, square=True,annot=True, cmap=cmap)
from sklearn.neighbors import KNeighborsClassifier
model_knn = KNeighborsClassifier(n_neighbors=24)
model_knn.fit(X_train, y_train)
knn = model_knn.score(X_test, y_test)
lista.append(knn)
print(knn)
rate = []
for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred = knn.predict(X_test)
    rate.append(np.mean(pred != y_test))

plt.figure(figsize=(10,6))
plt.plot(range(1,40),rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate x K Values')
plt.xlabel('K Values')
plt.ylabel('Error Rate')
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
model_tree = DecisionTreeClassifier(random_state=0)
model_tree.fit(X_train, y_train)
pred = model_tree.predict(X_test)  
from sklearn.metrics import confusion_matrix, accuracy_score
prec = accuracy_score(y_test, pred)
matriz = confusion_matrix(y_test, pred)
lista.append(prec)
print(prec)
print(matriz)
resul = (pd.DataFrame(lista, index=['Random Forest','KNeighborsClassifier','Tree Classifier'])*100).round(decimals=2)
resul.columns=['Accuracy']
resul