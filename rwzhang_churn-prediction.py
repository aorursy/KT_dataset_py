import pandas as pd 
import os
import matplotlib 
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import matplotlib.pyplot as plt 
from sklearn.model_selection import GridSearchCV
df_=pd.read_csv('../input/churn.csv',index_col=0);
g=sns.heatmap(df_.corr())
plt.title('Churn negative correlates to satisfaction');
g=sns.FacetGrid(df_, hue="churn", size=6) \
   .map(plt.hist, "department") \
   .add_legend()
plt.title('Churn in different department')
g.set_xticklabels(rotation=30)
plt.show()
sns.FacetGrid(df_, hue="churn", size=6) \
   .map(sns.kdeplot, "satisfaction") \
   .add_legend()
plt.title('Leaving employees are less satisfied');
sns.FacetGrid(df_, hue="churn", size=6) \
   .map(sns.kdeplot, "evaluation") \
   .add_legend()
plt.title('Evaluation for leaving employees is distributed at two ends');
sns.FacetGrid(df_, hue="churn", size=6) \
   .map(sns.kdeplot, "number_of_projects") \
   .add_legend()
plt.title('Leaving employees have less projects');
sns.FacetGrid(df_, hue="churn", size=6) \
   .map(sns.kdeplot, "time_spend_company") \
   .add_legend()
plt.title('Leaving concentrates at time spent in company of 3 to 5 years');
df_.salary = df_.salary.astype('category')
df_.salary = df_.salary.cat.reorder_categories(['low', 'medium', 'high'])
df_.salary = df_.salary.cat.codes
departments_ = pd.get_dummies(df_.department)
departments = departments_.drop("sales", axis=1)
df = df_.drop("department", axis=1)
df = df.join(departments)

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
target = df.churn
features = df.drop("churn",axis=1)
from sklearn.model_selection import train_test_split
target_train, target_test, features_train, features_test = train_test_split(target,features,test_size=0.25,random_state=42)
model= DecisionTreeClassifier(random_state=24)
depth = [i for i in range(4,31,1)]
samples = [i for i in range(30,600,20)]
parameters = dict(max_depth=depth, min_samples_leaf=samples)
from sklearn.model_selection import GridSearchCV
parameters = dict(max_depth=depth, min_samples_leaf=samples)
param_search = GridSearchCV(model, parameters)
param_search.fit(features_train, target_train)
print('best parameter for decisiontreeclassifier:',param_search.best_params_)

best_model=tree.DecisionTreeClassifier(max_depth=4, min_samples_leaf=50)
best_model.fit(features_train,target_train)
print('accuracy score for training:',best_model.score(features_train,target_train) )
print('accuracy score for test:',best_model.score(features_test,target_test))

import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

feature_importances = best_model.feature_importances_
feature_list = list(features)
relative_importances = pd.DataFrame(index=feature_list, data=feature_importances, columns=["importance"])
objects = relative_importances.sort_values(by="importance", ascending=False).index.values
y_pos = np.arange(len(objects))
performance = relative_importances.sort_values(by="importance", ascending=False).importance.values
relative_importances.head()

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects,rotation=90)
plt.ylabel('Importance')
plt.title('Feature related impotance')
plt.show()
from sklearn import linear_model
num_C = 10
parameterlist = [1.0] * num_C
for i in range(num_C):
    parameterlist[i] = pow(10, i-3)
parameters = dict(C=parameterlist)
regr=linear_model.LogisticRegression(solver = 'newton-cg')
param_search = GridSearchCV(regr, parameters)
param_search.fit(features_train, target_train)
print('best parameter for logisticregression:',param_search.best_params_)
best_regr=linear_model.LogisticRegression(C=1,solver = 'newton-cg')
best_regr.fit(features_train, target_train)
print('accuracy score for training:',best_regr.score(features_train,target_train) )
print('accuracy score for test:',best_regr.score(features_test,target_test))
from sklearn import svm 
from sklearn.model_selection import GridSearchCV     
best_svm=svm.SVC(C=10, gamma=0.5, kernel= 'rbf')
best_svm.fit(features_train, target_train)
print('accuracy score for training:',best_svm.score(features_train,target_train) )
print('accuracy score for test:',best_svm.score(features_test,target_test))