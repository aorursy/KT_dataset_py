#supress warning 

import warnings

warnings.filterwarnings('ignore')



import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt

import seaborn as sns



from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

from sklearn.model_selection import train_test_split



from sklearn.linear_model import LogisticRegression

from sklearn import tree

data = pd.read_csv("/kaggle/input/breast-cancer-wisconsin-data/data.csv")
data.head()
data.info()
#remove last columns, also we don't need id

data.drop(data.columns[len(data.columns)-1], axis=1, inplace=True)

data.drop('id', axis=1, inplace=True)
data.diagnosis.value_counts()
sns.set_style('whitegrid')

data.diagnosis.value_counts().plot(kind='bar',color=["lightblue", "salmon"])
categorical_val=[]

continuous_val=[]

for c in data.columns:

    #print('==================')

    #print(f"{c}:{data[c].unique()}")

    if len(data[c].unique()) <= 10:

        categorical_val.append(c)

    else:

        continuous_val.append(c)
print(categorical_val)

print(continuous_val)
plt.figure(figsize=(20,50))

for i, column in enumerate(continuous_val,1):

    plt.subplot(10,3,i)

    sns.distplot(data[data['diagnosis']=='M'][column],rug=False,label="M")

    sns.distplot(data[data['diagnosis']=='B'][column],rug=False,label='B')

    plt.xlabel(column)

    plt.legend()
df = data.replace({'diagnosis':{"M":1,"B":0}})
df.head()
corr_matrix=df.corr()

fig, ax = plt.subplots(figsize=(15,15))

ax = sns.heatmap(corr_matrix, annot=True, linewidths=0.5, fmt='.2f',cmap='YlGnBu')

bottom, top = ax.get_ylim()

ax.set_ylim(bottom+0.5, top-0.5)
col_drop = ['perimeter_mean','radius_mean','compactness_mean',

            'concave points_mean','radius_se','perimeter_se',

            'radius_worst','perimeter_worst','compactness_worst',

            'concave points_worst','compactness_se','concave points_se',

            'texture_worst','area_worst','concavity_worst']

df2 = df.drop(col_drop,axis=1)
df2.head()
fig, ax = plt.subplots(figsize=(15,15))

ax = sns.heatmap(df2.corr(), annot=True, linewidths=0.5, fmt='.2f',cmap='YlGnBu')

bottom, top = ax.get_ylim()

ax.set_ylim(bottom+0.5, top-0.5)
x = df2.drop('diagnosis',axis=1)
x.shape
#Calculate VIF 

#from statsmodels.stats.outliers_influence import variance_inflation_factor



#vif = pd.DataFrame()

#vif["features"] = x.columns

#vif["VIF Factor"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]

#vif
#while vif[vif['VIF Factor'] > 10]['VIF Factor'].any():    

#    remove = vif.sort_values('VIF Factor',ascending=0)['features'][1] 

    #print(remove)

    #print(continuous_val)

#    x.drop(remove,axis=1,inplace=True)

#    vif = pd.DataFrame()

#    vif["features"] = x.columns

#    vif["VIF Factor"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]

#    print(vif)

#    print('======================')

    
df2.drop('diagnosis',axis=1).corrwith(df.diagnosis).plot(kind='bar',grid=True,figsize=(12,8),

                                                       title='Correlation with diagnosis')
#there is no categorical variable other than our dependent variables, so we don't have to creat dummy variables for our models.

categorical_val
#store variable names 

col_sc = list(df2.columns)

col_sc.remove('diagnosis')
col_sc
#scale our data

#from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

df2[col_sc] = sc.fit_transform(df2[col_sc])
df2.head()
#from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score



def score(m, x_train, y_train, x_test, y_test, train=True):

    if train:

        pred=m.predict(x_train)

        print('Train Result:\n')

        print(f"Accuracy Score: {accuracy_score(y_train, pred)*100:.2f}%")

        print(f"Precision Score: {precision_score(y_train, pred)*100:.2f}%")

        print(f"Recall Score: {recall_score(y_train, pred)*100:.2f}%")

        print(f"F1 score: {f1_score(y_train, pred)*100:.2f}%")

        print(f"Confusion Matrix:\n {confusion_matrix(y_train, pred)}")

    elif train == False:

        pred=m.predict(x_test)

        print('Test Result:\n')

        print(f"Accuracy Score: {accuracy_score(y_test, pred)*100:.2f}%")

        print(f"Precision Score: {precision_score(y_test, pred)*100:.2f}%")

        print(f"Recall Score: {recall_score(y_test, pred)*100:.2f}%")

        print(f"F1 score: {f1_score(y_test, pred)*100:.2f}%")

        print(f"Confusion Matrix:\n {confusion_matrix(y_test, pred)}")

            

    
#from sklearn.model_selection import train_test_split



x = df2.drop('diagnosis',axis=1)

y = df2['diagnosis']



x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=42)
from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression()

logreg = logreg.fit(x_train, y_train)
score(logreg, x_train, y_train, x_test, y_test, train=True)
score(logreg, x_train, y_train, x_test, y_test, train=False)
#C represents the strength of the regularization. higher values of C correspond to less regularization

C = [1, .5, .25, .1, .05, .025, .01, .005, .0025] 

l1_metrics = np.zeros((len(C), 5)) 

l1_metrics[:,0] = C



for index in range(0, len(C)):

    logreg = LogisticRegression(penalty='l1', C=C[index], solver='liblinear') 

    logreg = logreg.fit(x_train, y_train)

    pred_test_Y = logreg.predict(x_test)

    l1_metrics[index,1] = np.count_nonzero(logreg.coef_) 

    l1_metrics[index,2] = accuracy_score(y_test, pred_test_Y) 

    l1_metrics[index,3] = precision_score(y_test, pred_test_Y) 

    l1_metrics[index,4] = recall_score(y_test, pred_test_Y)

    

col_names = ['C','Non-Zero Coeffs','Accuracy','Precision','Recall'] 

print(pd.DataFrame(l1_metrics, columns=col_names))
logreg_t = LogisticRegression(penalty='l1', C=0.25, solver='liblinear')

logreg_t = logreg_t.fit(x_train,y_train)
score(logreg_t, x_train, y_train, x_test, y_test, train=True)
score(logreg_t, x_train, y_train, x_test, y_test, train=False)
from sklearn import tree



tree1 = tree.DecisionTreeClassifier()

tree1 = tree1.fit(x_train, y_train)
score(tree1, x_train, y_train, x_test, y_test, train=True)
score(tree1, x_train, y_train, x_test, y_test, train=False)
#decide the tree depth!

depth_list = list(range(2,15))

depth_tuning = np.zeros((len(depth_list), 4)) 

depth_tuning[:,0] = depth_list



for index in range(len(depth_list)):

    mytree = tree.DecisionTreeClassifier(max_depth=depth_list[index]) 

    mytree = mytree.fit(x_train, y_train)

    pred_test_Y = mytree.predict(x_test)

    depth_tuning[index,1] = accuracy_score(y_test, pred_test_Y) 

    depth_tuning[index,2] = precision_score(y_test, pred_test_Y) 

    depth_tuning[index,3] = recall_score(y_test, pred_test_Y)

    

col_names = ['Max_Depth','Accuracy','Precision','Recall'] 

print(pd.DataFrame(depth_tuning, columns=col_names))
tree2 = tree.DecisionTreeClassifier(max_depth=3)

tree2 = tree2.fit(x_train,y_train)
score(tree2, x_train, y_train, x_test, y_test, train=True)
score(tree2, x_train, y_train, x_test, y_test, train=False)
import graphviz

exported = tree.export_graphviz( decision_tree=tree2,

                                out_file=None,

                                feature_names=x.columns,

                                precision=1,

                                class_names=['B','M'], 

                                filled = True)

graph = graphviz.Source(exported) 

display(graph)
from sklearn.ensemble import RandomForestClassifier



forest = RandomForestClassifier(n_estimators=1000, random_state= 42)

forest = forest.fit(x_train,y_train)
score(forest, x_train, y_train, x_test, y_test, train=True)
score(forest, x_train, y_train, x_test, y_test, train=False)
from sklearn.model_selection import RandomizedSearchCV



n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]

max_features = ['auto', 'sqrt']

max_depth = [int(x) for x in np.linspace(10, 110, num=11)]

max_depth.append(None)

min_samples_split = [2, 5, 10]

min_samples_leaf = [1, 2, 4]

bootstrap = [True, False]



random_grid = {'n_estimators': n_estimators, 'max_features': max_features,

               'max_depth': max_depth, 'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf, 'bootstrap': bootstrap}
random_grid
forest2 = RandomForestClassifier(random_state=42)



#Random search of parameters, using 3 fold cross validation, search across 100 different combinations, and use all available cores

rf_random = RandomizedSearchCV(estimator = forest2, param_distributions=random_grid,

                              n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)



rf_random.fit(x_train,y_train)

rf_random.best_params_
forest3 = RandomForestClassifier(bootstrap=True,

                                 max_depth=20, 

                                 max_features='sqrt', 

                                 min_samples_leaf=2, 

                                 min_samples_split=2,

                                 n_estimators=1200)

forest3 = forest3.fit(x_train, y_train)
score(forest3, x_train, y_train, x_test, y_test, train=True)
score(forest3, x_train, y_train, x_test, y_test, train=False)
from sklearn.svm import SVC



svm = SVC()

svm = svm.fit(x_train,y_train)
score(svm, x_train, y_train, x_test, y_test, train=True)
score(svm, x_train, y_train, x_test, y_test, train=False)
from sklearn.model_selection import GridSearchCV



svm_model = SVC()



params = {"C":(0.1, 0.5, 1, 2, 5, 10, 20), 

          "gamma":(0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1), 

          "kernel":('poly', 'rbf')}



svm_grid = GridSearchCV(svm_model, params, n_jobs=-1, cv=5, verbose=1, scoring="accuracy")

svm_grid.fit(x_train, y_train)
svm_grid.best_params_
svm2 = SVC(C=2, gamma=0.01, kernel='rbf')

svm2 = svm2.fit(x_train, y_train)
score(svm2, x_train, y_train, x_test, y_test, train=True)
score(svm2, x_train, y_train, x_test, y_test, train=False)