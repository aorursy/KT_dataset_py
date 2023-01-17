import numpy as np 
import pandas as pd 
import pyodbc 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso 
from sklearn.model_selection import train_test_split, cross_val_score 
from statistics import mean 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.model_selection import cross_val_score
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot
import plotly.offline as ply
ply.init_notebook_mode(connected=True)
import plotly.express as px

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv("/kaggle/input/corona-virus-brazil/brazil_covid19_macro.csv")
data
print(data.shape[0])
print(data.shape[1])
print(data.columns.tolist())
print(data.dtypes)
print(data['cases'].describe())
import matplotlib.pyplot as plt
%matplotlib inline
ax = plt.axes()
ax.scatter(data.deaths, data.recovered)

ax.set(xlabel='Ölen Kişi',
       ylabel='Kurtarılan Kişi',
       title='Ölen ve Kurtarılan Kişi');
df = data.dropna(how='any',axis=0)
df=data.filter(['cases','deaths','recovered'])
df.head(75)
df["cases"]=pd.to_numeric(df["cases"])
df["recovered"]=pd.to_numeric(df["recovered"])
df["deaths"]=pd.to_numeric(df["deaths"])
#plot
plt.figure(figsize=(16,8))
plt.plot(df['cases'], label='Confirmed cases');
X_data = df[['cases','recovered']]
X_data.sample(5)
fig, ax = plt.subplots(figsize=(8,8))
sns.heatmap(df.corr(),cmap="YlGnBu");
sns.pairplot(df,kind="reg");
y_data = df['deaths'].values
X_train,X_test,y_train,y_test = train_test_split(X_data,y_data, random_state=42)
lR = LinearRegression().fit(X_train,y_train)
y_predict = lR.predict(X_test)
y_predict
from sklearn.metrics import r2_score
r2_score(y_predict,y_test)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3,weights= 'uniform').fit(X_data, y_data)

y_pred = knn.predict(X_data)
y_pred
import matplotlib.pyplot as plt
plt.figure(figsize=(16,8))
plt.plot(y_pred[:])
plt.show()
GNB = GaussianNB()
nb = {'gaussian': GaussianNB(),
      'bernoulli': BernoulliNB(),
      'multinomial': MultinomialNB()}
scores = {}
X_data = df[['cases','recovered']]
y_data = df['deaths'].values
for key, model in nb.items():
    s = cross_val_score(model, X_data, y_data, cv=10, n_jobs=3, scoring='accuracy')
    scores[key] = np.mean(s)
scores
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=42)
parameters = {'kernel':('linear', 'rbf' , 'poly'), 'C':[.1, 1, 10 , 20], 'gamma':[.5, 1, 2, 10 , 20]}

SVC_Gaussian = svm.SVC(gamma='scale')
gscv = GridSearchCV(SVC_Gaussian, param_grid=parameters, cv=3).fit(X_train, y_train)

print(gscv.best_estimator_)
print(gscv.best_params_)
SVC_Gaussian.fit(X_train, y_train)
y_pred = SVC_Gaussian.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
print(y_pred)
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize = (15,15))
plot = sns.heatmap(cm, annot=True, fmt='d' , cmap="YlGnBu");
from sklearn.linear_model import LinearRegression
df = data.dropna(how='any',axis=0)
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=42)
regressor = LinearRegression()
regressor_fit = regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
print(regressor_fit.score(X_test, y_test)*100)
fig=go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data["cases"],
                    mode='lines+markers',
                    name='Vaka Sayısı'))
fig.add_trace(go.Scatter(x=data.index, y=data["recovered"],
                    mode='lines+markers',
                    name='Kurtarılan Kişiler'))
fig.add_trace(go.Scatter(x=data.index, y=data["deaths"],
                    mode='lines+markers',
                    name='Ölen Kişiler'))
fig.update_layout(xaxis_title="Gün",yaxis_title="Vaka Sayısı",legend=dict(x=0,y=1,traceorder="normal"))
fig.show()
from sklearn.tree import DecisionTreeClassifier
X_train,X_test,y_train,y_test = train_test_split(X_data,y_data, random_state=42)
tree = DecisionTreeClassifier(random_state=42,min_samples_split=2)
tree.fit(X_train, y_train)
print(f'Model Accuracy: {tree.score(X_data, y_data)}')
print(tree.fit)
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
X_train,X_test,y_train,y_test = train_test_split(X_data,y_data, random_state=42)
model = RandomForestClassifier(n_estimators=100, 
                               bootstrap = True,
                               max_features = 'sqrt')
model.fit(X_train, y_train)
y_pred=model.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
