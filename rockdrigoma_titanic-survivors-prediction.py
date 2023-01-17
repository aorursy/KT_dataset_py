# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

from sklearn.metrics import precision_recall_curve

from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve, roc_curve, accuracy_score, confusion_matrix

from sklearn.model_selection import cross_val_predict, cross_val_score, GridSearchCV

from sklearn.linear_model import SGDClassifier, LogisticRegression

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.base import BaseEstimator, TransformerMixin

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from mpl_toolkits.mplot3d import Axes3D

from sklearn.manifold import TSNE

import seaborn as sns

from sklearn.naive_bayes import GaussianNB, MultinomialNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, LabelBinarizer, OneHotEncoder, LabelEncoder





pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/titanic/train.csv')

y = data['Survived']

survived = data['Survived'].values.reshape(-1,1)



data.head(10)
data.info()
data.drop('Cabin', axis=1, inplace=True)

data.drop('PassengerId', axis=1, inplace=True)

data['Embarked'].fillna('Unknown', inplace=True)

data['Title'] = data['Name'].str.extract(pat = '^.+?, (.+?\.) .*$')

data['Lastname'] = data['Name'].str.extract(pat = '(^.+?),')

data['Age']=data.groupby(['Title'])['Age'].transform(lambda x: x.fillna(x.mean())).astype(int)

data['Fare']=data.groupby(['Pclass'])['Fare'].transform(lambda x: x.fillna(x.mean())).astype(int)

data['Family_size'] = data['SibSp']+data['Parch']+1

data['Fare_per_pass'] = data['Fare']/data['Family_size']

replacing = {"Sex":{"male":0, "female":1}}

data.replace(replacing, inplace=True)

replacing = {"Embarked":{"S":0, "C":1, "Q":2, "Unknown":3}}

data.replace(replacing, inplace=True)

data['Ticket'] = data['Ticket'].str.extract(pat = '.*([\d]{6}$)')

data['Ticket'] = data['Ticket'].str.extract(pat = '^([\d]{1}).*')

data['Ticket'].fillna(0, inplace=True)

data['Ticket'] = data['Ticket'].astype(int)
data.head(10)

data.info()
correlation = data.corr()

sns.set(style="white")

plt.figure(figsize=(15,15))

sns.heatmap(correlation, 

        xticklabels=correlation.columns,

        yticklabels=correlation.columns,

        vmin=-1,

        cmap='coolwarm',

        annot=True)
attributes = ['Pclass','Lastname','Title','Embarked','Age','Fare','Fare_per_pass','Survived','Sex']

pd.plotting.scatter_matrix(data[attributes], figsize=(25,15))
data.groupby(['Ticket','Survived']).size().reset_index().sort_values(by=0, ascending=False)
data.drop('Survived', axis=1, inplace=True)

cat_attr = ['Pclass','Lastname','Title','Embarked','Sex']

attr = ['SibSp','Parch','Fare_per_pass','Family_size','Fare','Age','Ticket']

data.count()
data['Age'].hist(color='#A9C5D3', edgecolor='black',grid=False, bins=4)
plot_age = data.groupby('Age').size().reset_index().sort_values(by='Age',ascending=True)

plt.plot(plot_age)

plot_age.sort_values(by='Age',ascending=True)
bin_ranges = [-1, 19, 39, 100]

bin_names = [0, 1, 2]

data['Age_bin'] = pd.cut(np.array(data['Age']), bins=bin_ranges, labels=bin_names)

data[data['Age']==0]
class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self.attribute_names = attribute_names

    def fit(self, X , y=None):

        return self

    def transform(self, X):

        return X[self.attribute_names].values



cat_pipeline = Pipeline([

    ('selector', DataFrameSelector(cat_attr)),

    ('cat', OneHotEncoder(handle_unknown='ignore')),

])



attr_pipeline = Pipeline([

    ('selector', DataFrameSelector(attr)),

    ('scaler', MinMaxScaler(feature_range = (0,1))),

    #('poly', PolynomialFeatures(2)),

])



full_pipeline = FeatureUnion(transformer_list=[

                             ('cat_pipe', cat_pipeline),

                             ('attr_pipe', attr_pipeline),

                             ])
data.head()
x = full_pipeline.fit_transform(data)

x.shape[1]


num = data.shape[0]



labeler = LabelEncoder()

y = labeler.fit_transform(survived.ravel())



shuffle_index = np.random.permutation(num)

x1, y1 = x[shuffle_index], y[shuffle_index]

train_ratio = int(num*0.66)

ax_tr, ax_te, ay_tr, ay_te = x1[:train_ratio], x1[train_ratio:], y1[:train_ratio], y1[train_ratio:]
#amodel = SGDClassifier(random_state=41)

#amodel = GaussianNB()

amodel = RandomForestClassifier(criterion='entropy', max_depth=40, min_samples_leaf=1, min_samples_split=5, n_estimators=40)

#amodel = RandomForestClassifier(max_depth=5, min_samples_leaf=1, min_samples_split=5, n_estimators=20)

#amodel = LogisticRegression(random_state=42, dual=False, multi_class='multinomial', penalty='l2', solver='newton-cg')

amodel.fit(ax_tr, ay_tr)

#cross_val_score(amodel, ax_te.toarray(), ay_te, cv=50, scoring='neg_mean_absolute_error')

cross_val_score(amodel, ax_te.toarray(), ay_te, cv=50)
y_train_pred = cross_val_predict(amodel, ax_te.toarray(), ay_te, cv=50)

confusion_matrix(ay_te, y_train_pred)
accuracy_score(ay_te, y_train_pred)
precision_score(ay_te, y_train_pred)
recall_score(ay_te, y_train_pred)
f1_score(ay_te, y_train_pred)
pca = PCA(n_components=3)

X = pca.fit_transform(ax_tr.toarray())



result=pd.DataFrame(X, columns=['PCA%i' % i for i in range(3)])

color=pd.DataFrame(ay_tr, columns=['Survived'])



color['Survived']=pd.Categorical(color['Survived'])

my_color=color['Survived'].cat.codes

    

# Plot initialisation

fig = plt.figure(figsize=(40,30))

ax = fig.add_subplot(111, projection='3d')

ax.scatter(result['PCA0'], result['PCA1'], result['PCA2'], c=my_color, cmap="spring", s=60)

 

# make simple, bare axis lines through space:

xAxisLine = ((min(result['PCA0']), max(result['PCA0'])), (0, 0), (0,0))

ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'r')

yAxisLine = ((0, 0), (min(result['PCA1']), max(result['PCA1'])), (0,0))

ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'r')

zAxisLine = ((0, 0), (0,0), (min(result['PCA2']), max(result['PCA2'])))

ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'r')

 

# label the axes

ax.set_xlabel("PC1")

ax.set_ylabel("PC2")

ax.set_zlabel("PC3")

plt.show()
pca = PCA(n_components=2)

X = pca.fit_transform(ax_tr.toarray())



result=pd.DataFrame(X, columns=['PCA%i' % i for i in range(2)])

color=pd.DataFrame(ay_tr, columns=['Survived'])



color['Survived']=pd.Categorical(color['Survived'])

my_color=color['Survived'].cat.codes

    

# Plot initialisation

fig = plt.figure(figsize=(25,18))

plt.scatter(result['PCA0'], result['PCA1'], c=my_color, cmap="spring", s=60)

plt.show()
X = TSNE(n_components=2, perplexity=25).fit_transform(ax_tr.toarray())



result=pd.DataFrame(X, columns=['PCA%i' % i for i in range(2)])

color=pd.DataFrame(ay_tr, columns=['Survived'])



color['Survived']=pd.Categorical(color['Survived'])

my_color=color['Survived'].cat.codes

    

# Plot initialisation

fig = plt.figure(figsize=(25,18))

plt.scatter(result['PCA0'], result['PCA1'], c=my_color, cmap="spring", s=60)



plt.show()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data['Embarked'].fillna('Unknown', inplace=True)

test_data.drop('Cabin', axis=1, inplace=True)

test_data['Title'] = test_data['Name'].str.extract(pat = '^.+?, (.+?\.) .*$')

test_data['Lastname'] = test_data['Name'].str.extract(pat = '(^.+?),')

test_data['Age']=test_data.groupby(['Title'])['Age'].transform(lambda x: x.fillna(x.mean()))

test_data['Fare']=test_data.groupby(['Pclass'])['Fare'].transform(lambda x: x.fillna(x.mean()))

test_data['Age'].fillna(28.0, inplace=True)

test_data['Family_size'] = test_data['SibSp']+test_data['Parch']+1

test_data['Fare_per_pass'] = test_data['Fare']/test_data['Family_size']

replacing = {"Sex":{"male":0, "female":1}}

test_data.replace(replacing, inplace=True)

replacing = {"Embarked":{"S":0, "C":1, "Q":2, "Unknown":3}}

test_data.replace(replacing, inplace=True)

test_data['Ticket'] = test_data['Ticket'].str.extract(pat = '.*([\d]{6}$)')

test_data['Ticket'] = test_data['Ticket'].str.extract(pat = '^([\d]{1}).*')

test_data['Ticket'].fillna(0, inplace=True)

test_data['Ticket'] = test_data['Ticket'].astype(int)

test_data.head(5)
test_x = full_pipeline.transform(test_data)
predictions = amodel.predict(test_x.toarray())

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output['Survived'] = output.Survived.astype(int)

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")

y_scores = cross_val_predict(amodel, ax_te.toarray(), ay_te, cv=50)

precisions, recalls, thresholds = precision_recall_curve(ay_te, y_scores)



def plot_precision(pre, rec, thr):

    fig = plt.figure(figsize=(25,18))

    plt.plot(thr, pre[:-1], 'b--', label='Precision')

    plt.plot(thr, rec[:-1],'g-', label='Recall')

    plt.xlabel('Threshold')

    plt.legend(loc='center left')

    plt.ylim([0,1])



plot_precision(precisions, recalls, thresholds)

plt.show()
#y_better = (y_scores<0)

y_better = (y_scores>-0.5)
precision_score(ay_te, y_better)
recall_score(ay_te, y_better)
fpr, tpr, thresholds = roc_curve(ay_te, y_scores)



def plot_roc(fpr, tpr, label=None):

    fig = plt.figure(figsize=(25,18))

    plt.plot(fpr, tpr, linewidth=2, label=label)

    plt.plot([0,1], [0,1], 'k--')

    plt.axis([0,1,0,1])

    plt.xlabel('False positive rate')

    plt.ylabel('True positive rate')



plot_roc(fpr,tpr)

plt.show()
#param_grid = [{'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], 'multi_class':['ovr', 'auto']}]#

#amodel = LogisticRegression()

#grid = GridSearchCV(amodel, param_grid, cv=5, scoring='neg_mean_squared_error')

#grid.fit(ax_tr, ay_tr)

#grid.best_params_
#param_grid = [{'penalty': ['l2', 'none'], 'dual':[False],'solver':['newton-cg', 'lbfgs', 'sag', 'saga'], 'multi_class':['ovr', 'multinomial', 'auto']}]

#amodel = LogisticRegression()

#grid = GridSearchCV(amodel, param_grid, cv=5, scoring='neg_mean_squared_error')

#grid.fit(ax_tr, ay_tr)

#grid.best_params_
#param_grid = [{'criterion':['gini','entropy'], 'n_estimators': [10, 15, 20, 30, 40, 50, 60], 'max_depth':[5,10,20,30,40,50],'min_samples_split':[5,10,20,30,40,50], 'min_samples_leaf':[1,2,3,4,5,6,7,8,9,10]}]

#amodel = RandomForestClassifier()

#grid = GridSearchCV(amodel, param_grid, cv=5, scoring='neg_mean_squared_error')

#grid.fit(ax_tr, ay_tr)

#grid.best_params_