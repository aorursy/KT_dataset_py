# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Load the datasets
train = pd.read_csv('/kaggle/input/dataanalyticscoaching/train.csv')
test = pd.read_csv('/kaggle/input/dataanalyticscoaching/test.csv')
train.head()
test.head()
# Check columns
train.columns
test.columns
# Check the dataset information
train.info()
# take-way: Non of the columns have missing values
test.info()
# Look for duplicates
train.duplicated().sum()
test.duplicated().sum()
# Find unique values count and store them in a dictionary
unique_count = train.nunique()
inx = list(unique_count.index)
res = dict(zip(inx,unique_count)) 
# Find out categorical variables
cat_attr = []
for key, val in res.items(): 
    if val <= 5 and key != 'target':
            print(key, ":", val)
            cat_attr.append(key)
print('Number of Categorical attributes:', len(cat_attr))
cat_feat = train[cat_attr]
# Find out numerical attributes
num_attr = set(train.columns) - set(cat_attr)
num_feat = train[num_attr]
num_attr
train['target'].value_counts()
for attr in cat_attr:
    sns.countplot(x = attr, data = train)
    plt.show()
for attr in num_attr:
    if attr == 'target' or attr == 'id':
        pass
    else:
        sns.distplot(train[attr])
        plt.show()
facet = sns.FacetGrid(train, hue="target",aspect=4) 
facet.map(sns.kdeplot, 'age',shade = True) 
facet.set(xlim=(0, train['age'].max()))
facet.add_legend()
facet = sns.FacetGrid(train, hue="target",aspect=4) 
facet.map(sns.kdeplot, 'age',shade = True) 
facet.set(xlim=(0, train['age'].max()))
facet.add_legend()
plt.xlim(25,55)
facet = sns.FacetGrid(train, hue="target",aspect=4) 
facet.map(sns.kdeplot, 'age',shade = True) 
facet.set(xlim=(0, train['age'].max()))
facet.add_legend()
plt.xlim(50,75)
def age_grouping(dataset):
    dataset.loc[dataset["age"] <= 44, "age"] = 0
    dataset.loc[(dataset["age"] > 44) & (dataset["age"] <= 58), "age"] = 1
    dataset.loc[(dataset["age"] > 58) & (dataset["age"] <= 77), "age"] = 2
age_grouping(train)
zero = train[train['age']==0].count()
one = train[train['age']==1].count()
two = train[train['age']==2].count()
print(zero[0])
print(one[0])
print(two[0])
print(train['age'].unique())
age_grouping(test)
zero = test[test['age']==0].count()
one = test[test['age']==1].count()
two = test[test['age']==2].count()
print(zero[0])
print(one[0])
print(two[0])
print(test['age'].unique())
res = train.groupby(['target', 'sex'])['sex'].count()
res
sns.countplot(x = 'target', hue ='sex', data = train)
for attr in train:
    if attr == 'id' or attr == 'target':
        pass
    else:
        sns.boxplot(x = 'target', y = attr, data = train)
        plt.show()
num_feat.corr()
plt.figure(figsize=(10,10))
sns.heatmap(num_feat.corr(), 
            annot=True,
            fmt='.2f',
            cmap=sns.diverging_palette(240, 10, n=25),
            square=True)
num_attr.difference_update({'id', 'target'})
cat_attr = list(cat_attr)
num_attr= list(num_attr)
X_train = train[cat_attr + num_attr]
y_train = train['target']
X_train
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.attribute_names].values
num_pipeline = Pipeline([
                            ('selector', DataFrameSelector(num_attr)),
                            ('std_scaler', StandardScaler())
                        ])
cat_pipeline = Pipeline([
                            ('selector', DataFrameSelector(cat_attr)),
                        ])
num_pipeline.fit_transform(X_train).shape
cat_pipeline.fit_transform(X_train).shape
full_pipeline = FeatureUnion(transformer_list=[
                                ('num_pipeline', num_pipeline),
                                ('cat_pipeline', cat_pipeline)
                            ])
full_pipeline.fit_transform(X_train).shape
X_prep_train = full_pipeline.fit_transform(X_train)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import GridSearchCV
# Create a Dataframe to store the model accuracy
acc_df = pd.DataFrame([], columns=['Algo', 'Train_acc', 'Val_acc', 'Log_loss'])
# Create a Param and find best param, CV score, accuracy, Log loss
param_grid = { 
                'penalty' : ['l1', 'l2', 'elasticnet', 'none'],
                'C':[0.1, 0.3, 0.6, 1.0, 3.0],
                'solver' : ['lbfgs','newton-cg','liblinear','sag','saga'],
                'max_iter' : [10,20,30,40,50,60]
             }
LR_clf = LogisticRegression()
kfold = KFold(n_splits=5)
grid_search = GridSearchCV(LR_clf, param_grid, cv=kfold, scoring='accuracy', verbose=1)

grid_search.fit(X_prep_train, y_train)

print(grid_search.best_params_)

LR_clf = grid_search.best_estimator_
scores = cross_val_score(LR_clf, X_prep_train, y_train, scoring='accuracy', cv=kfold)

LR_clf.fit(X_prep_train, y_train)
y_pred = LR_clf.predict(X_prep_train)
acc = accuracy_score(y_train, y_pred)
loss = log_loss(y_train, y_pred)

row = {'Algo': 'LR','Train_acc':np.round(acc,2),'Val_acc':np.round(np.mean(scores),3), 'Log_loss':np.round(loss,2)}
acc_df = acc_df.append(row, ignore_index = True)
# Print the accuracy df
acc_df
# Print confusion matrix
confusion_matrix(y_train, y_pred)
print(classification_report(y_train, y_pred))
arr = np.array([1,1,1,0,1,1,0,1,0,1,1,1,1,0,0,1,0,1,1,1,1,0,1,0,0,1,1,1,0,0,0,1,1,1,0,1,1,0,1,0,1,1,0,1,1,0,1,1,0,0,1,1,1,0,0,1,0,1,0,1,1,0,0,1,1,1,1,1,1,0,1,0,0,0,0,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,0])
# test the model
X_test = test.drop(columns=['id'])
X_test = full_pipeline.fit_transform(X_test)
y_tpred = LR_clf.predict(X_test)
print(y_tpred)
print(arr)
# Create submission file
submission = pd.DataFrame({
        "id": test['id'],
        "target": y_tpred
    })

submission.to_csv('_sub.csv',index=False)
submission.head()
