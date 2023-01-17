import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import plotly.express as px



plt.rcParams['figure.figsize'] = 8, 5

plt.style.use("fivethirtyeight")

pd.options.plotting.backend = "plotly"



data = pd.read_csv('../input/pokemon/Pokemon.csv')

data.drop('#', axis=1, inplace=True)
sns.heatmap(data.corr())

plt.title('Correlation')

plt.show()
fig = data.isnull().sum().reset_index().plot(kind='bar', x='index', y=0)

fig.update_layout(title='Missing values plot', xaxis_title='Columns', yaxis_title='Missing Count')

fig.show()
fig = data['Type 1'].value_counts().reset_index().plot(kind='bar', y='index', x='Type 1', color='Type 1')

fig.update_layout(title='Abilities of Pokemons', yaxis_title='Ability', xaxis_title='Count')

fig.show()
sns.FacetGrid(data, hue="Legendary", height=6,).map(sns.kdeplot, "Total",shade=True).add_legend()

plt.title('KDE Plot for Total Strength')

plt.show()
sns.pairplot(data=data[['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed','Generation']], hue='Generation')

print('Pairplot segregated on generation')

plt.show()
sns.pairplot(data=data[['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed','Legendary']], hue='Legendary')

print('Pairplot segregated on Legendariness')

plt.show()
sns.boxplot(data=data, x='Generation', y='Total')

plt.title('Total /b Generation')

plt.show()
data[(data['Generation']==1) & (data['Total']>700)]
sns.boxplot(data=data, x='Legendary', y='HP')

plt.title('HP /b Legendary')

plt.show()
data[(data['HP']>160) | (data['HP']==data['HP'].min())]
#dropping 121, 261 and 316 indices in the dataset



data = data.drop([121, 261, 316]).reset_index(drop=True)
fig = plt.figure(figsize=(15,30))

fig.add_subplot(5,2,1)

sns.boxplot(data=data, x='Generation', y='Attack', hue='Legendary')

plt.title('Attack /b Generation')



fig.add_subplot(5,2,2)

sns.boxplot(data=data, x='Generation', y='Defense')

plt.title('Defense /b Generation')



fig.add_subplot(5,2,3)

sns.violinplot(data=data, x='Legendary', y='Speed')

plt.title('Speed /b Legendary')

plt.show()
data['Type 1'] = data['Type 1'].astype('category')
from sklearn import preprocessing



le = preprocessing.LabelEncoder()

data['Type 1'] = le.fit_transform(data['Type 1'])
df = data.drop(['Name','Type 2','Generation','Legendary'], axis=1)



from sklearn.cluster import KMeans



X = np.array(df)



kmeans = KMeans(n_clusters=6, random_state=0)

kmeans.fit(X)



df['cluster_label'] = pd.Series(list(kmeans.labels_))
fig = df['cluster_label'].value_counts().reset_index().plot(kind='bar',x='index',y='cluster_label', color='cluster_label')

fig.update_layout(title='Distribution of the cluster labels', xaxis_title='Cluster', yaxis_title='Count')

fig.show()
fig = data['Generation'].value_counts().reset_index().plot(kind='bar',x='index',y='Generation', color='Generation')

fig.update_layout(title='Distribution of the Generation', xaxis_title='Generation', yaxis_title='Count')

fig.show()
fig = plt.figure(figsize=(15,30))

fig.add_subplot(5,2,1)

sns.kdeplot(data=df['cluster_label'])

plt.title('Cluster Label')





fig.add_subplot(5,2,2)

sns.kdeplot(data=data['Generation'])

plt.title('Generations')



plt.show()
data = pd.read_csv('../input/pokemon/Pokemon.csv')

data.drop(['#','Type 2'], axis=1, inplace=True)

data = data.drop([121, 261, 316]).reset_index(drop=True)
data.head()
#one hot encoding



dummy = pd.get_dummies(data['Type 1'])

dummy.drop('Grass', axis=1, inplace=True) #dropping to avoid dummy variable trap

data = pd.concat([data, dummy], axis=1)

del data['Type 1'] #dropping the variable after encoded
from sklearn.utils import shuffle

data = shuffle(data)

data = data.reset_index(drop=True)
sns.barplot(data=data['Legendary'].value_counts().reset_index(), x='index', y='Legendary')

plt.xlabel('Legendary')

plt.ylabel('')

plt.title('Distribution of target variable')

plt.show()
data.drop('Name', axis=1, inplace=True)
#splitting the data



from sklearn.model_selection import train_test_split



X = data.drop('Legendary', axis=1)

y = data['Legendary']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier



from sklearn.metrics import roc_curve,accuracy_score,plot_confusion_matrix
model = DecisionTreeClassifier(max_depth=5, random_state=13)

model.fit(X_train, y_train)

prediction = model.predict(X_test)

print('The accuracy of the Decision Tree is', accuracy_score(prediction, y_test))
plot_confusion_matrix(model, X_test, y_test)

plt.title('Decision Tree Confusion Matrix')

plt.show()



model = DecisionTreeClassifier(max_depth=5, random_state=13)

model.fit(X_train, y_train)

y_pred_prob = model.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr, tpr, label='Logistic Regression')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Decision Tree ROC Curve')

plt.show()