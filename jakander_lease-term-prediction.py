import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import tree 
import sklearn.linear_model as linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import neighbors
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.grid_search import GridSearchCV
from sklearn import preprocessing
import matplotlib.pyplot as plt
import plotly.offline as py 
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls 
import warnings 
from collections import Counter

%matplotlib inline
path = "../input/german_credit_data.csv"
data = pd.read_csv(path)
data1=data
data.head()
plot_risk_numberinit = data['Risk'].value_counts().plot(title = 'RISK vs APPLICANT', \
                                                                kind = 'barh', color = 'green')
plot_risk_numberinit.set_ylabel(" RISK ")
plot_risk_numberinit.set_xlabel("APPLICANT")
plt.show()
interval = (18, 25, 35, 60, 120)

cats = ['Student', 'Young', 'Adult', 'Senior']
data["Age_cat"] = pd.cut(data.Age, interval, labels=cats)


df_good = data[data["Risk"] == 'good']
df_bad = data[data["Risk"] == 'bad']
#Let's look the Credit Amount column
interval = (18, 25, 35, 60, 120)

cats = ['Student', 'Young', 'Adult', 'Senior']
data["Age_cat"] = pd.cut(data.Age, interval, labels=cats)


df_good = data[data["Risk"] == 'good']
df_bad = data[data["Risk"] == 'bad']

trace0 = go.Box(
    y=df_good["Credit amount"],
    x=df_good["Age_cat"],
    name='Good credit',
    marker=dict(
        color='#3D9970'
    )
)

trace1 = go.Box(
    y=df_bad['Credit amount'],
    x=df_bad['Age_cat'],
    name='Bad credit',
    marker=dict(
        color='#FF4136'
    )
)
    
dat = [trace0, trace1]

layout = go.Layout(
    yaxis=dict(
        title='Credit Amount (US Dollar)',
        zeroline=False
    ),
    xaxis=dict(
        title='Age Categorical'
    ),
    boxmode='group'
)
fig = go.Figure(data=dat, layout=layout)

py.iplot(fig, filename='box-age-cat')

#First plot
trace0 = go.Bar(
    x = data[data["Risk"]== 'good']["Checking account"].value_counts().index.values,
    y = data[data["Risk"]== 'good']["Checking account"].value_counts().values,
    name='Good credit Distribuition' 
    
)

#Second plot
trace1 = go.Bar(
    x = data[data["Risk"]== 'bad']["Checking account"].value_counts().index.values,
    y = data[data["Risk"]== 'bad']["Checking account"].value_counts().values,
    name="Bad Credit Distribuition"
)

dat = [trace0, trace1]

layout = go.Layout(
    title='Checking accounts Distribuition',
    xaxis=dict(title='Checking accounts name'),
    yaxis=dict(title='Count'),
    barmode='group'
)


fig = go.Figure(data=dat, layout=layout)

py.iplot(fig, filename = 'Age-ba', validate = False)

date_int = ["Purpose", 'Sex']
cm = sns.light_palette("red", as_cmap=True)
pd.crosstab(data[date_int[0]], data[date_int[1]]).style.background_gradient(cmap = cm)
print("Purpose : ",data.Purpose.unique())
print("Sex : ",data.Sex.unique())
print("Housing : ",data.Housing.unique())
print("Saving accounts : ",data['Saving accounts'].unique())
print("Risk : ",data['Risk'].unique())
print("Checking account : ",data['Checking account'].unique())
print("Aget_cat : ",data['Age_cat'].unique())
data.isna().sum()

data['Saving accounts'] = data['Saving accounts'].fillna('no_inf')
data['Checking account'] = data['Checking account'].fillna('no_inf')
data.isna().sum()

data = data.merge(pd.get_dummies(data.Purpose, drop_first=True, prefix='Purpose'), left_index=True, right_index=True)

data = data.merge(pd.get_dummies(data.Sex, drop_first=True, prefix='Sex'), left_index=True, right_index=True)

data = data.merge(pd.get_dummies(data.Housing, drop_first=True, prefix='Housing'), left_index=True, right_index=True)

data = data.merge(pd.get_dummies(data["Saving accounts"], drop_first=True, prefix='Savings'), left_index=True, right_index=True)

data = data.merge(pd.get_dummies(data.Risk, prefix='Risk'), left_index=True, right_index=True)

data = data.merge(pd.get_dummies(data["Checking account"], drop_first=True, prefix='Check'), left_index=True, right_index=True)

data = data.merge(pd.get_dummies(data["Age_cat"], drop_first=True, prefix='Age_cat'), left_index=True, right_index=True)

del data["Unnamed: 0"]
del data["Saving accounts"]
del data["Checking account"]
del data["Purpose"]
del data["Sex"]
del data["Housing"]
del data["Age_cat"]
del data["Risk"]
del data['Risk_good']
data.head()
plt.figure(figsize=(14,12))
sns.heatmap(data.corr(),linewidths=0.1,vmax=1.0, 
           square=True,  linecolor='white', annot=True)
plt.show()
X = data.drop('Risk_bad', 1).values
y = data["Risk_bad"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=42)


model =RandomForestClassifier(random_state=2)
param_grid = {"max_depth": [3,5, 7, 10,None],
              "n_estimators":[3,5,10,25,50,150,180,220],
              "max_features": [4,7,15,20]}
grid_search = GridSearchCV(model, param_grid=param_grid, cv=5, scoring='recall', verbose=4)
grid_search.fit(X_train, y_train)

print(grid_search.best_score_)
print(grid_search.best_params_)
forest =RandomForestClassifier(max_depth=None, max_features=20, n_estimators=5, random_state=2)

forest = forest.fit(X_train, y_train)
print("erreur de prévision sur l'apprentissage : ", 1-forest.score(X_train, y_train))

print("erreur de prévision sur le test :  ",1-forest.score(X_test,y_test))

y_pred = forest.predict(X_test)
print(classification_report(y_test, y_pred))

table=pd.crosstab(y_test,y_pred,)
print(table)
plt.matshow(table)
plt.title("Matrice de Confusion")
plt.colorbar()
plt.show()
GNB = GaussianNB()

model = GNB.fit(X_train, y_train)

print("erreur de prévision sur l'apprentissage :",model.score(X_train, y_train))

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

table=pd.crosstab(y_test,y_pred,)
print(table)
plt.matshow(table)
plt.title("Matrice de Confusion")
plt.colorbar()
plt.show()
knn = neighbors.KNeighborsClassifier(n_neighbors=10)
digit_knn=knn.fit(X_train, y_train)
# Estimation de l’erreur de prévision
# sur l’échantillon test
1-digit_knn.score(X_test,y_test)
# grille de valeurs
param=[{"n_neighbors":list(range(1,15))}]
knn= GridSearchCV(neighbors.KNeighborsClassifier(),
param,cv=5,n_jobs=-1)
digit_knn=knn.fit(X_train, y_train)
# paramètre optimal
digit_knn.best_params_["n_neighbors"]

knn = neighbors.KNeighborsClassifier(n_neighbors=10)
digit_knn=knn.fit(X_train, y_train)
# Estimation de l’erreur de prévision
print("l’erreur de prévision",1-digit_knn.score(X_test,y_test))
# Prévision
y_chap = digit_knn.predict(X_test)
print(classification_report(y_test, y_chap))
# matrice de confusion
table=pd.crosstab(y_test,y_chap)
print(table)
plt.matshow(table)
plt.title("Matrice de Confusion")
plt.colorbar()
plt.show()

logit = LogisticRegression()
data_logit=logit.fit(X_train, y_train)
# Erreur
print("l’erreur et :",data_logit.score(X_test, y_test))

# Coefficients
data_logit.coef_

# grille de valeurs
param=[{"C":[0.01,0.05,0.1,0.15,1,10]}]
logit = GridSearchCV(LogisticRegression(penalty="l1"),
param,cv=5,n_jobs=-1)
data_logit=logit.fit(X_train, y_train)
# paramètre optimal
data_logit.best_params_["C"]
logit = LogisticRegression(C=1,penalty="l1")
data_logit=logit.fit(X_train, y_train)
# Erreur
print("Erreur",data_logit.score(X_test, y_test))
# Coefficients
data_logit.coef_
y_chap = data_logit.predict(X_test)
print(classification_report(y_test, y_chap))
tree=DecisionTreeClassifier()
digit_tree=tree.fit(X_train, y_train)
# Estimation de l’erreur de prévision
print("Estimation de l’erreur de prévision : ",1-digit_tree.score(X_test,y_test))

param=[{"max_depth":list(range(2,10))}]
data_tree= GridSearchCV(DecisionTreeClassifier(),
param,cv=5,n_jobs=-1)
data_opt=data_tree.fit(X_train, y_train)
# paramètre optimal
data_opt.best_params_
tree=DecisionTreeClassifier(max_depth=3)
data_tree=tree.fit(X_train, y_train)
# Estimation de l’erreur de prévision
print("Estimation de l’erreur de prévision : ",1-data_tree.score(X_train,y_train))
# sur l’échantillon test
print("Estimation de l’erreur sur l’échantillon test : ",1-data_tree.score(X_test,y_test))

# prévision de l’échantillon test
z_chap = data_tree.predict(X_test)
# matrice de confusion
table=pd.crosstab(y_test,z_chap)
print(table)
print(classification_report(y_test, z_chap))
plt.matshow(table)
plt.title("Matrice de Confusion")
plt.colorbar()
plt.show()
forest = RandomForestClassifier(n_estimators=500,
criterion='gini', max_depth=None,
min_samples_split=2, min_samples_leaf=1,
max_features=8, max_leaf_nodes=None,
bootstrap=True, oob_score=True)
# apprentissage
forest = forest.fit(X_train,y_train)
print('erreur de prévision sur le test :',1-forest.oob_score_)

# prévision
y_chap = forest.predict(X_test)
print(classification_report(y_test, y_chap))
# matrice de confusion
table=pd.crosstab(y_test,y_chap)
print(table)
plt.matshow(table)
plt.title("Matrice de Confusion")
plt.colorbar()
plt.show()