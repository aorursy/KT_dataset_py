import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.offline as py





import warnings

warnings.filterwarnings('ignore')



py.offline.init_notebook_mode(connected=True)

import plotly.graph_objs as go



%matplotlib inline

# Plot in SVG format since this format is more sharp and legible

%config InlineBackend.figure_format = 'svg'
path = '../input/heart.csv'

df = pd.read_csv(path)

df.head()
num_cont_feat = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

cat_feat = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
df['sex'] = df['sex'].apply(lambda x: 'male' if x == 1 else 'female')

df['exang'] = df['exang'].map({1: 'Yes', 0:'No'})
df.shape
df.columns
df.info()
df.describe()
plt.rcParams['figure.figsize']= (12,8) # figure size

sns.set_style('darkgrid') # Style
df['age'].hist(grid=True, bins=10); 

plt.title('Age distribuition')
sns.distplot(df[df['sex']=='female']['age'], rug=True, hist=True, label='female')

sns.distplot(df[df['sex']=='male']['age'], rug=True, hist=True, label='male')

plt.legend()

plt.title('Density plot of age by sex');
age = df['age']

layout = go.Layout(barmode='overlay')

data = go.Histogram(x=age, opacity=0.6, xbins={'size': 4})

fig = go.Figure(data=[data], layout=layout)

py.offline.iplot(fig)
df['trestbps'].hist()

plt.title('Resting Blood pressure distribuition')
sns.distplot(df['trestbps'], bins=10)

plt.title('Resting Blood pressure desnity plot');
plt.rcParams['figure.figsize']= (15,8) # reajustar o tamanho da figura 



df[[ 'age','trestbps', 'chol', 'thalach', 'oldpeak']].hist();
fig, axes = plt.subplots(nrows = 1, ncols=2)

sns.boxplot(x='chol', data=df, orient='v', ax=axes[0])

sns.boxplot(x='oldpeak', data=df,  orient='v', ax=axes[1]);
df['target'].value_counts()
df['sex'].value_counts()
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(17,10))

cat_feat = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'target']



for idx, feature in enumerate(cat_feat):

    ax = axes[int(idx/4), idx%4]

    if feature != 'target':

        sns.countplot(x=feature, hue='target', data=df, ax=ax)

plt.rcParams['figure.figsize'] = (10,8)

sns.countplot(x='target', hue='sex', data=df);

plt.title('Count of target feature by sex')
pd.crosstab(df['sex'], df['target'], normalize=True)
sex_target = df.groupby(by=['sex', 'target']).size()

sex_target_pcts = sex_target.groupby(level=0).apply(lambda x: 100*x/x.sum())



sex_target_pcts
sns.countplot(x='cp', hue='target', data=df)
df['sex'] = df['sex'].map({'female': 0, 'male': 1})

df['exang'] = df['exang'].map({'No': 0, 'Yes': 1})

plt.figure(figsize=(12,8))

sns.heatmap(df.drop(['target', 'sex', 'cp', 'fbs'], axis=1).corr(), annot=True, cmap='coolwarm');
plt.figure(figsize=(30,30))

sns.set_style('darkgrid')

sns.pairplot(df[num_cont_feat])
plt.rcParams['figure.figsize'] = (8,8)

sns.scatterplot(x='chol', y='trestbps', hue='sex', size=None, data=df)

plt.title(' Cholesterol vs Blood pressure in rest')
sns.jointplot(x='thalach', y='trestbps',  data=df)
sns.jointplot(kind='kde', x='thalach', y='trestbps', data=df)
sns.scatterplot(x='age', y='chol', hue='target', data = df)
plt.figure()

plt.scatter(df[df['target'] == 0]['age'], df[df['target'] == 0]['chol'], marker='o', c='blue', label='healthy')

plt.scatter(df[df['target'] == 1]['age'], df[df['target'] == 1]['chol'], marker='x', c='red', label='sick')



plt.legend()
sns.boxplot(x='sex', y='chol', data=df)
plt.figure(figsize=(15,10))

sns.catplot(x='sex', y='chol', col='target', data=df, kind='box', height=4, aspect=.8)
sns.boxplot(y='age', x='target', data = df)
# First let's create a list to append the data to be plotted

data = []

for pain in df.cp.unique():

    data.append(go.Box(y=df[df.cp == pain].chol, name=str(pain)))



layout = go.Layout(yaxis=dict(title ='Cholesterol', zeroline=False))

                   

fig = go.Figure(data=data, layout=layout)               

py.iplot(fig, show_link=False)
# First let's create a list to append the data to be plotted

data = []

for target in df.target.unique():

    data.append(go.Box(y=df[df.target == target].thalach, name=str(target)))



layout = go.Layout(yaxis=dict(title ='maximum heart rate achieved', zeroline=False))

                   

fig = go.Figure(data=data, layout=layout)               

py.iplot(fig, show_link=False)
# Import train test split



from sklearn.model_selection import train_test_split
df.head()
# Split the DataFrame into a matrix X and vecto Y which form the train set

X, y = df.drop('target', axis=1), df['target']
X.shape, y.shape
X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.3, random_state = 17)
X_train.shape, X_holdout.shape
from sklearn.model_selection import GridSearchCV, StratifiedKFold
# Let's specify 5 kfold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)
# Let's create our hyperparameter grid using a dictionary



params = {'max_depth': np.arange(2,10), 

         'min_samples_leaf': np.arange(2,10),

         }

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
tree = DecisionTreeClassifier(random_state=17)
best_tree = GridSearchCV(estimator = tree, param_grid=params, n_jobs=1, verbose=1)
best_tree.fit(X_train, y_train)
best_tree.best_params_
best_tree.best_estimator_
best_tree.best_score_
pred_holdout_better = best_tree.predict(X_holdout)
accuracy_score(y_holdout, pred_holdout_better)
# First we'll import graphviz from sklearn.tree

from sklearn.tree import export_graphviz
export_graphviz(decision_tree=best_tree.best_estimator_,

               out_file='tree.dot', filled=True, 

                feature_names=df.drop('target', axis=1).columns)



from subprocess import call



call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])



from IPython.display import Image

Image(filename = 'tree.png')
from sklearn.metrics import f1_score
print(f1_score(y_holdout, pred_holdout_better))
importances = best_tree.best_estimator_.feature_importances_
plt.figure(figsize=(10,4))

plt.bar(X_train.columns.values,importances)
from sklearn.base import clone

from sklearn.metrics import f1_score, accuracy_score
X_train_reduced = X_train.drop(['sex', 'fbs', 'thal'], axis=1)

X_holdout_reduced = X_holdout.drop(['sex', 'fbs', 'thal'], axis=1)
X_train_reduced.shape, X_holdout_reduced.shape
# Train on the "best" model found from grid search earlier

tree_reduceded_clone = (clone(best_tree.best_estimator_)).fit(X_train_reduced, y_train)



# Make new predictions

reduced_predictions = tree_reduceded_clone.predict(X_holdout_reduced)



print("\nFinal Model trained on reduced data\n------")

print("Accuracy on holdout data: {:.4f}".format(accuracy_score(y_holdout, reduced_predictions)))

print("F1-score on holdout data: {:.4f}".format(f1_score(y_holdout, reduced_predictions)))