#'''Importing Data Manipulation Modules'''
import numpy as np                 # Linear Algebra
import pandas as pd                # Data Processing, CSV file I/O (e.g. pd.read_csv)

#'''Seaborn and Matplotlib Visualization'''
import matplotlib                  # 2D Plotting Library
import matplotlib.pyplot as plt
import seaborn as sns              # Python Data Visualization Library based on matplotlib
plt.style.use('fivethirtyeight')
%matplotlib inline

#'''Plotly Visualizations'''
import plotly as plotly                # Interactive Graphing Library for Python
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot, plot
import plotly.offline as py
init_notebook_mode(connected=True)
import os
%pylab inline
df = pd.read_csv('../input/palmer-archipelago-antarctica-penguin-data/penguins_size.csv')
df.head()
df.info()
df.describe()
df.isnull().sum()
df['culmen_length_mm'].fillna(df['culmen_length_mm'].mean(), inplace = True)
df['culmen_depth_mm'].fillna(df['culmen_depth_mm'].mean(),inplace = True)
df['flipper_length_mm'].fillna(df['flipper_length_mm'].mean(),inplace = True)
df['body_mass_g'].fillna(df['body_mass_g'].mean(),inplace = True)
df.drop(df[df['sex'] == '.'].index, inplace = True)
df['sex'].fillna(df['sex'].mode()[0],inplace = True)
df.isnull().sum()
df.head()
labels = sorted(df.island.unique())
values = df.island.value_counts().sort_index()
colors = ['BlanchedAlmond', 'GreenYellow', 'PaleTurquoise']


fig = go.Figure(data=[go.Pie(labels=labels,
                             values=values)])
fig.update_traces(hoverinfo='label+percent', textinfo='value',textfont_size=20,
                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))

fig.update_layout(title_text="Distribution of Islands")
fig.show()

labels = sorted(df.species.unique())
values = df.species.value_counts().sort_index()
colors = ['Aquamarine', 'Yellow', 'Coral']


fig = go.Figure(data=[go.Pie(labels=labels,
                             values=values)])
fig.update_traces(hoverinfo='label+percent', textinfo='value',textfont_size=20,
                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))

fig.update_layout(title_text="Distribution of Species")
fig.show()
labels = sorted(df.sex.unique())
values = df.sex.value_counts().sort_index()
colors = ['Aqua', 'Chocolate']


fig = go.Figure(data=[go.Pie(labels=labels,
                             values=values)])
fig.update_traces(hoverinfo='label+percent', textinfo='value',textfont_size=20,
                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))

fig.update_layout(title_text="Distribution of Gender")
fig.show()
fig = px.violin(df, y="culmen_length_mm", x="sex", color="sex", box=True, points="all")
fig.update_layout(title="Interquartile distribution of culmen_length by Gender")
fig.show()
fig = px.violin(df, y="culmen_depth_mm", x="sex", color="sex", box=True, points="all")
fig.update_layout(title="Interquartile distribution of culmen_depth_mm by Gender")
fig.show()
fig = px.violin(df, y="flipper_length_mm", x="sex", color="sex", box=True, points="all")
fig.update_layout(title="Interquartile distribution of flipper_length by Gender")
fig.show()
fig = px.violin(df, y="body_mass_g", x="sex", color="sex", box=True, points="all")
fig.update_layout(title="Interquartile distribution of body_mass by Gender")
fig.show()
y = df['sex']
x = df.drop(['species','island','sex'], axis = 1 )
data_dia = y
data = x
data_n_2 = (data - data.mean()) / (data.std())              # standardization
data = pd.concat([y,data_n_2],axis=1)
data = pd.melt(data,id_vars="sex",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,12))
tic = time.time()
sns.swarmplot(x="features", y="value", hue="sex", data=data,palette=["black", "silver"])

plt.xticks(rotation=90)
fig = px.scatter_matrix(df, dimensions=["culmen_length_mm", "culmen_depth_mm", "flipper_length_mm", "body_mass_g"], color="species",height=600, width=1000)
fig.show()
f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(x.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFECV
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
clf_rf = RandomForestClassifier(random_state=43)      
clr_rf = clf_rf.fit(x_train,y_train)
ac = accuracy_score(y_test,clf_rf.predict(x_test))
print('Accuracy is: ',ac)
cm = confusion_matrix(y_test,clf_rf.predict(x_test))
sns.heatmap(cm,annot=True,fmt="d")
clf_rf_2 = RandomForestClassifier() 
rfecv = RFECV(estimator=clf_rf_2, step=1, cv=4,scoring='accuracy')   #5-fold cross-validation
rfecv = rfecv.fit(x_train, y_train)

print('Optimal number of features :', rfecv.n_features_)
print('Best features :', x_train.columns[rfecv.support_])
import matplotlib.pyplot as plt
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score of number of selected features")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()