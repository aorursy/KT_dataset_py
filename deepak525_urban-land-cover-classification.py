# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

sns.set_style('whitegrid')

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score



import warnings

warnings.filterwarnings('ignore')



import time

from sklearn.model_selection import KFold,StratifiedKFold

from sklearn.metrics import roc_auc_score, roc_curve



from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler,MinMaxScaler



import os

print(os.listdir("../input"))





from sklearn.metrics import make_scorer 

from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

from sklearn.ensemble import RandomForestClassifier



from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier



from sklearn.model_selection import KFold,StratifiedKFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import ShuffleSplit

from sklearn.model_selection import LeaveOneOut as loocv

from plotly import tools

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.offline as py

from plotly.graph_objs import Scatter, Layout

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.figure_factory as ff
%time

train = pd.read_csv('../input/training.csv')

test = pd.read_csv('../input/testing.csv')

print("Rows and Columns(Train): ",train.shape)

print("Rows and Columns(Test) : ",test.shape)
train.info()
train.head()
# we have no missing values

train.isnull().any().any()
#'duplicated()' function in pandas return the duplicate row as True and othter as False

#for counting the duplicate elements we sum all the rows

sum(train.duplicated())
p = train.describe().T

p = p.round(4)

table = go.Table(

    columnwidth=[0.8]+[0.5]*8,

    header=dict(

        values=['Attribute'] + list(p.columns),

        line = dict(color='#506784'),

        fill = dict(color='lightblue'),

    ),

    cells=dict(

        values=[p.index] + [p[k].tolist() for k in p.columns[:]],

        line = dict(color='#506784'),

        fill = dict(color=['rgb(173, 216, 220)', '#f5f5fa'])

    )

)

py.iplot([table], filename='table-of-mining-data')
print(train['class'].value_counts())



f,ax=plt.subplots(1,2,figsize=(20,8))

train['class'].value_counts().plot.pie(autopct='%1.1f%%',ax=ax[0])

ax[0].set_title('Distribution of Different Classes (Pie Chart)')

ax[0].set_ylabel('')

sns.countplot('class',data=train,ax=ax[1])

ax[1].set_title('Distribution of Different Classes (Bar Plot)')

plt.show()
from collections import Counter



def detect_outliers(train_data,n,features):

    outlier_indices = []

    for col in features:

        # 1st quartile (25%)

        Q1 = np.percentile(train_data[col], 25)

        # 3rd quartile (75%)

        Q3 = np.percentile(train_data[col],75)

        # Interquartile range (IQR)

        IQR = Q3 - Q1

        outlier_step = 1.5 * IQR

        outlier_list_col = train_data[(train_data[col] < Q1 - outlier_step) | (train_data[col] > Q3 + outlier_step )].index

        outlier_indices.extend(outlier_list_col)

        

    outlier_indices = Counter(outlier_indices)        

    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )

    

    return multiple_outliers   



# detect outliers

list_atributes = train.drop('class',axis=1).columns

Outliers_to_drop = detect_outliers(train,2,list_atributes)
train.loc[Outliers_to_drop]
group_map = {"grass ":0,"building ":1,'concrete ':2,'tree ':3,'shadow ':4,'pool ':5,'asphalt ':6,'soil ':7,'car ':8}



train['class'] = train['class'].map(group_map)

test['class'] = test['class'].map(group_map)

train['class'].unique()
plt.figure(figsize=(16,6))

features = train.columns.values[1:148]

plt.title("Distribution of Mean Values Per Row in the Train and Test Set",fontsize=15)

sns.distplot(train[features].mean(axis=1),color="green", kde=True,bins=120, label='train')

sns.distplot(test[features].mean(axis=1),color="blue", kde=True,bins=120, label='test')

plt.legend()

plt.show()
plt.figure(figsize=(16,6))

plt.title("Distribution of Mean Values Per Column in the Train and Test Set",fontsize=15)

sns.distplot(train[features].mean(axis=0),color="magenta",kde=True,bins=120, label='train')

sns.distplot(test[features].mean(axis=0),color="darkblue", kde=True,bins=120, label='test')

plt.legend()

plt.show()
sns.pairplot(train, vars=['class', 'BrdIndx','Area','Round','Bright','Compact'], hue='class')

plt.show()
correlations = train.corr().abs().unstack().sort_values(kind="quicksort").reset_index()

correlations = correlations[correlations['level_0'] != correlations['level_1']]

correlations.head()
correlations = correlations.loc[correlations[0] == 1]

removable_features = set(list(correlations['level_1']))

correlations.shape

test.head()
X_train = train.drop(['class'], axis=1)

#X_train = X_train.drop(removable_features, axis=1)

y_train = pd.DataFrame(train['class'].values)

X_test = test.drop(['class'], axis=1)

#X_test = X_test.drop(removable_features, axis=1)

y_test = test['class']



scaler = StandardScaler()

X_train_std = scaler.fit_transform(X_train)

X_test_std = scaler.transform(X_test)
clfs = [KNeighborsClassifier(),DecisionTreeClassifier(),RandomForestClassifier(),AdaBoostClassifier(),GaussianNB(),XGBClassifier()]

total_accuracy = {}

total_accuracy_std = {}

for model in clfs:

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    total_accuracy[str((str(model).split('(')[0]))] = accuracy_score(pred,y_test)

    

for model in clfs:

    model.fit(X_train_std, y_train)

    pred = model.predict(X_test_std)

    total_accuracy_std[str((str(model).split('(')[0]))] = accuracy_score(pred,y_test)
data = total_accuracy.values()

labels = total_accuracy.keys()

data1 = total_accuracy_std.values()

labels1 = total_accuracy_std.keys()



fig = plt.figure(figsize=(20,5))

plt.subplot(121)

plt.plot([i for i, e in enumerate(data)], data); plt.xticks([i for i, e in enumerate(labels)], [l[:] for l in labels])

plt.title("Accuracy Score Without Preprocessing",fontsize = 14)

plt.xlabel('Model',fontsize = 13)

plt.xticks(rotation = 10)

plt.ylabel('Accuracy',fontsize = 13)



plt.subplot(122)

plt.plot([i for i, e in enumerate(data1)], data1); plt.xticks([i for i, e in enumerate(labels1)], [l[:] for l in labels1])

plt.title("Accuracy Score After Preprocessing",fontsize = 14)

plt.xlabel('Model',fontsize = 13)

plt.xticks(rotation = 10)

plt.show()
n_fold = 5

folds = KFold(n_splits=n_fold, shuffle=True, random_state=42)
prediction = np.zeros(len(X_test))

total_acc = []

oof = np.zeros(len(X_train))

for fold_n, (train_index, valid_index) in enumerate(folds.split(X_train,y_train)):

    print('Fold', fold_n, 'started at', time.ctime(),end = "  ")

    X_train_, X_valid = X_train.iloc[train_index], X_train.iloc[valid_index]

    y_train_, y_valid = y_train.iloc[train_index], y_train.iloc[valid_index]

        

    clf_rfc2 = RandomForestClassifier(n_estimators=1000, n_jobs=-1, random_state=0)

    clf_rfc2.fit(X_train_, y_train_)

    oof[valid_index] = clf_rfc2.predict(X_train.iloc[valid_index])

            

    prediction = clf_rfc2.predict(X_test)

    print("Validation Score: ",accuracy_score(y_test,prediction))

    total_acc.append(accuracy_score(y_test,prediction))

print("CV score: {:<8.5f}".format(accuracy_score(y_train, oof)))

print("Mean Testing Score: ",np.mean(total_acc))
clf = RandomForestClassifier(class_weight = 'balanced', random_state=0)

parameters = {'n_estimators':[1000], 

              'max_depth':[4, 5, 6],

              'criterion':['gini', 'entropy'], 

              'max_leaf_nodes':[5,11],

              'min_samples_leaf':[2,3]

              #'max_features': ['auto', 'sqrt', 'log2']

             }



grid_obj = GridSearchCV(clf, parameters, scoring='accuracy', verbose=1, cv=10)



grid_fit = grid_obj.fit(X_train, y_train)

best_clf = grid_fit.best_estimator_

print(best_clf)
best_clf.fit(X_train, y_train)

y_pred = best_clf.predict(X_test)

accuracy_score(y_pred,y_test)
selector = RFE(best_clf, 30, step=1)

selector.fit(X_train,y_train)
y_pred = selector.predict(X_test)

accuracy_score(y_pred,y_test)
import eli5

from eli5.sklearn import PermutationImportance



perm = PermutationImportance(best_clf, random_state=0).fit(X_train, y_train)



eli5.show_weights(perm, top=50,feature_names = X_train.columns.tolist())
clf = DecisionTreeClassifier(class_weight = 'balanced', random_state=0)

parameters = {'max_depth':[ 1, 3, 5, 7, 9, 11 ,13,15,17,19],

              'criterion':['gini', 'entropy'], 

              #'max_leaf_nodes':[2,3,5,7,11,9,13],

              #'min_samples_leaf':[2,3,5,7,11,9,13]

              'min_samples_split':[2],

              'max_leaf_nodes':[11],

              'min_samples_leaf':[2],

              #'max_features': ['auto', 'sqrt', 'log2']

             }



grid_obj = GridSearchCV(clf, parameters, scoring='accuracy', verbose=1, cv=10)



grid_fit = grid_obj.fit(X_train, y_train)

best_clf = grid_fit.best_estimator_

print(best_clf)
best_clf.fit(X_train, y_train)

y_pred = best_clf.predict(X_test)

accuracy_score(y_pred,y_test)
mod = xgb.XGBClassifier(learning_rate=0.02,booster = "gbtree",  objective= "multi:softmax")



params = {

        'min_child_weight': [1, 5, 10],

        'gamma': [0.5, 1, 1.5, 2, 5],

        'subsample': [0.6, 0.8, 1.0],

        'colsample_bytree': [0.6, 0.8, 1.0],

        'max_depth': [3, 4, 5,7,9]

        }



folds = 3

param_comb = 5



skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)



r_search = RandomizedSearchCV(mod, params, scoring='accuracy', n_jobs=4, cv=skf.split(X_train,y_train), verbose=3)

r_search.fit(X_train, y_train)

print('\n Best hyperparameters:')

print(r_search.best_params_)
mod = xgb.XGBClassifier(learning_rate=0.02, n_estimators=350,booster = "gbtree",subsample=0.6,objective="multi:softmax",max_depth=3,

                       gamma=0.5,colsample_bytree=0.6,min_child_weight=1,eval_metric='merror')
mod.fit(X_train,y_train)
y_pred = mod.predict(X_test)

accuracy_score(y_pred,y_test)
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

train_scaled = scaler.fit_transform(X_train)         

PCA_train_x = PCA().fit_transform(train_scaled)

plt.scatter(PCA_train_x[:, 0], PCA_train_x[:, 1], c=train['class'], cmap="copper_r")

plt.axis('off')

plt.colorbar()

plt.show()
train_scaled = scaler.fit_transform(X_train)         

PCA_train_x = PCA(4).fit_transform(train_scaled)
from sklearn.decomposition import KernelPCA



lin_pca = KernelPCA(n_components = 2, kernel="linear", fit_inverse_transform=True)

rbf_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.0433, fit_inverse_transform=True)

sig_pca = KernelPCA(n_components = 2, kernel="sigmoid", gamma=0.001, coef0=1, fit_inverse_transform=True)





plt.figure(figsize=(15, 4))

for subplot, pca, title in ((131, lin_pca, "Linear kernel"), (132, rbf_pca, "RBF kernel, $\gamma=0.04$"), 

                            (133, sig_pca, "Sigmoid kernel, $\gamma=10^{-3}, r=1$")):

       

    PCA_train_x = PCA(2).fit_transform(train_scaled)

    plt.subplot(subplot)

    plt.title(title, fontsize=14)

    plt.scatter(PCA_train_x[:, 0], PCA_train_x[:, 1], c=train['class'], cmap="nipy_spectral_r")

    plt.xlabel("$z_1$", fontsize=18)

    if subplot == 131:

        plt.ylabel("$z_2$", fontsize=18, rotation=0)

    plt.grid(True)



plt.show()
X_train = train.drop('class',axis=  1)

y_train = train['class']
from sklearn.datasets import load_boston

from sklearn.feature_selection import RFE

from sklearn.ensemble import RandomForestClassifier





rfc = RandomForestClassifier(n_estimators=500, class_weight='balanced', max_depth=5, random_state=42)

selector = RFE(rfc, n_features_to_select=50)

selector.fit(X_train, y_train)
selector.get_support()
selected = X_train.columns[selector.get_support()]
clfs = [KNeighborsClassifier(),DecisionTreeClassifier(),RandomForestClassifier(),AdaBoostClassifier(),GaussianNB(),XGBClassifier()]

for model in clfs:

    model.fit(train[selected], y_train)

    pred = model.predict(test[selected])

    print(accuracy_score(pred,y_test))
import shap

from eli5.sklearn import PermutationImportance

import eli5
from sklearn.svm import SVC
rfc = SVC(kernel='linear')

rfc.fit(X_train,y_train)

pred = rfc.predict(test.drop('class',axis=1))

print(accuracy_score(pred,test['class']))
perm = PermutationImportance(rfc, random_state=1).fit(X_train, y_train)

eli5.show_weights(perm, top=50)
explainer = shap.LinearExplainer(rfc, X_train)

shap_values = explainer.shap_values(X_train)



shap.summary_plot(shap_values, X_train)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=1)  

X_train = lda.fit_transform(X_train_std, y_train)  

X_test = lda.transform(X_test_std) 
classifier = RandomForestClassifier(max_depth=5, random_state=0)



classifier.fit(X_train_std, y_train)  

y_pred = classifier.predict(X_test_std) 

accuracy_score(y_test, y_pred)