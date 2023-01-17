# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
df = pd.read_csv("../input/Prodigy18 Train Data.csv")
print(df.shape)
df.head()
df['is_promoted'].value_counts()
for key,value in df.iteritems():
    print("{0}: {1}".format(key,len(df[key].unique())))
print(df.isnull().sum())
df['previous_year_rating'].fillna(3.0, inplace=True) # replacing NaN values in 'previous_year_rating' field with numerical 3 (which is neither a good rating nor a bad rating)
sns.FacetGrid(df, hue="is_promoted", height=10) \
   .map(sns.distplot, "no_of_trainings") \
   .add_legend();
plt.show();
print(len(df[df['no_of_trainings']>5]))
print(len(df[ (df['no_of_trainings']>5) & (df['is_promoted']==0) ]))
sns.boxplot(x='is_promoted',y='no_of_trainings', data=df)
plt.show()
print(df['previous_year_rating'].value_counts())
print(df['previous_year_rating'].unique())
for i in df['previous_year_rating'].unique():
    print(len(df[(df['previous_year_rating']==i) & (df['is_promoted']==1)]))
for i in df['age'].unique():
    if True:
        print("age {0}".format(i),end=" ")
        nolens = len(df[(df['age']==i) & (df['is_promoted']==1)])
        novc = len(df[df['age']==i])
        pracc = nolens/novc
        print("Total:{0} and Promoted:{1} and Acc:{2}".format(novc,nolens,pracc))
sns.FacetGrid(df, hue="is_promoted", height=10) \
   .map(sns.distplot, "length_of_service") \
   .add_legend();
plt.show();
for i in df['length_of_service'].unique():
    if i>25:
        print("service {0}".format(i),end=" ")
        nolens = len(df[(df['length_of_service']==i) & (df['is_promoted']==1)])
        novc = len(df[df['length_of_service']==i])
        pracc = nolens/novc
        print("Total:{0} and Promoted:{1} and Acc:{2}".format(novc,nolens,pracc))
sns.boxplot(x='is_promoted',y='length_of_service', data=df)
plt.show()
result = df.sort_values(['length_of_service'], ascending=False)
result
df = df.drop(31071)
sns.FacetGrid(df, hue="is_promoted", height=10) \
   .map(sns.distplot, "avg_training_score") \
   .add_legend();
plt.show();
for i in df['avg_training_score'].unique():
    if i>92:
        print("score {0}".format(i),end=" ")
        nolens = len(df[(df['avg_training_score']==i) & (df['is_promoted']==1)])
        novc = len(df[df['avg_training_score']==i])
        pracc = nolens/novc
        print("Total:{0} and Promoted:{1} and Acc:{2}".format(novc,nolens,pracc))
df[df['education'].isnull()]
df['education'].fillna("Bachelor's", inplace=True) 
# save the labels into a variable label.
label = df['is_promoted']
# Drop the label feature and store the remaining data in data.
data = df.drop(['employee_id','is_promoted','gender'],axis=1)
# Get the shape of data and label
print(data.shape)
print(label.shape)
print(df.isnull().sum())
ohedf = pd.get_dummies(data)
ohedf
#sm = SMOTE(random_state=12, ratio = 1.0)
#x_train_res, y_train_res = sm.fit_sample(ohedf, label)
ohedf2 = ohedf.drop(['region_region_18'],axis=1)
x_train_res = ohedf2
y_train_res = label

df_test = pd.read_csv("../input/Prodigy18 Test Data.csv")
print(df_test.shape)
print(df_test.head())
print(df_test.columns)
df_test['previous_year_rating'].fillna(3.0, inplace=True)
df_test['education'].fillna("Bachelor's", inplace=True)
df_test2 = df_test.drop(['employee_id','gender'],axis=1)
df_test2.shape
ohedf.head()
ohedf.shape
ohetest = pd.get_dummies(df_test2)
ohetest.shape
ohetest.head()
ohetest = ohetest.drop(['region_region_18'],axis=1)
'''# Implementing Randomized Search for hyperparameter tuning
import random
from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import MultinomialNB

mylist = np.random.uniform(10**-3, 10**2, 10**5) # choose 1000 uniform random numbers within the given range.
# specify parameters and distributions to sample from
param_dist = {"alpha": mylist}

# run randomized search
n_iter_search = 500
model = RandomizedSearchCV(MultinomialNB(), param_distributions=param_dist,
                                   n_iter=n_iter_search, scoring = 'accuracy', cv=10)
model.fit(data, label)

print(model.best_estimator_)
'''
from sklearn import tree
'''from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
# Setup the parameters and distributions to sample from: param_dist
param_dist = {"max_depth": [3, None],
              "max_features": randint(1, 9),
              "min_samples_leaf": randint(1, 9),
              "criterion": ["gini", "entropy"]}
# run randomized search
n_iter_search = 10
model = RandomizedSearchCV(tree.DecisionTreeClassifier(), param_distributions=param_dist,
                                   n_iter=n_iter_search, scoring = 'accuracy', cv=10)
clf2 = model.fit(x_train_res, y_train_res)

print(model.best_estimator_)'''
'''from sklearn.ensemble import RandomForestClassifier
clf_rf = RandomForestClassifier()
clf = clf_rf.fit(x_train_res, y_train_res)'''

'''from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier()
clf = neigh.fit(x_train_res, y_train_res)'''
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_train_res, y_train_res)

'''optimal_alpha = model.best_estimator_.alpha
clf = MultinomialNB(alpha = optimal_alpha)
clf.fit(data, label)
print("Optimal value of alpha = {0:0.2f}".format(optimal_alpha))'''
# predict the response
y_pred = clf.predict(ohetest)
c=0
for i in ohetest['avg_training_score']:
    if i > 93:
        y_pred[c]=1
    c = c+1    
c=0
for i in ohetest['length_of_service']:
    if i > 25:
        y_pred[c] = 0
    c = c+1    
employee = df_test['employee_id']
df_submit_file = pd.DataFrame({
    "employee_id" : employee,
    "is_promoted" : y_pred
})
df_submit_file.head()
df_submit_file.to_csv("newsubmission5.csv", index = False)
