# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# import package
import matplotlib.pyplot as plt
import seaborn as sns 
import statsmodels as sm 
from pathlib import Path
from sklearn.impute import SimpleImputer
import warnings
%matplotlib inline
pd.options.plotting.backend
pd.plotting.register_matplotlib_converters()
sns.set()
warnings.filterwarnings('ignore')
file = '/kaggle/input/covid19/Kaggle_Sirio_Libanes_ICU_Prediction.xlsx'
data_icu = pd.read_excel(file)
data_icu.head()
data_icu.info()
data_icu.columns
data = data_icu.drop(columns='ICU') # data for modelling
target = data_icu['ICU'] # target
definitions = []
for i in data.columns:
    if type(data[i].iloc[0]) == str:
        factor = pd.factorize(data[i])
        data[i] = factor[0]
        definitions.append([np.unique(factor[0]), factor[1]])
definitions
data.head()
target.value_counts()
data.isnull().sum()[data.isnull().sum()>0]
my_imputer = SimpleImputer(strategy='mean')
imputed_data = pd.DataFrame(my_imputer.fit_transform(data.drop(columns='PATIENT_VISIT_IDENTIFIER')))
imputed_data.columns = data.drop(columns='PATIENT_VISIT_IDENTIFIER').columns
imputed_data.shape
imputed_data.head()
#we find the feature that is most correlated or collinear
cor = pd.DataFrame(np.triu(imputed_data.corr().values), index = imputed_data.corr().index,
                   columns=imputed_data.corr().columns).round(3)
x = cor.unstack()
xo = x.sort_values()
most_correlated = xo[xo>0.95][xo[xo>0.90] !=1] # feature more collinear
pd.DataFrame(most_correlated, columns=['correlation'])
n = len(most_correlated)
print('Number of feature most correlated are: {}'.format(n))
# I plot only 20 features most correlated
fig = plt.figure(dpi=200, figsize=(20,20))
fig.subplots_adjust(hspace=0.5, wspace=0.4)
for i in range(1, 21):
    ax = fig.add_subplot(5,4,i)
    col1 = most_correlated.index[i-1][0]
    col2 = most_correlated.index[i-1][1] 
    sns.regplot(col1, col2, imputed_data)
# we take a columns that we remove
col_selected = []
for row in most_correlated.index:
    col_selected.append(row[0])
np.unique(col_selected) # we see that feature
# we take column for standardscaler and pca
new_data = imputed_data.drop(columns=np.unique(col_selected))
new_data.info()
#we divide data to train set and test set following a WINDOW feature
train = new_data[new_data.WINDOW != 4]
test = new_data[new_data.WINDOW == 4]
train.shape
test.shape
# we take target_train and target_test
target_train = target.iloc[:train.shape[0]]
target_test = target.iloc[train.shape[0]:]
target_train.shape
target_test.shape
# define xtrain, xvalid, ytrain, yvalid
from sklearn.model_selection import train_test_split
xtrain, xvalid, ytrain, yvalid = train_test_split(train, target_train, stratify=target_train, random_state=0,\
                                                  train_size=0.8)
print('xtrain shape: {}'.format(xtrain.shape))
print('xvalid shape: {}'.format(xvalid.shape))
print('ytrain shape: {}'.format(ytrain.shape))
print('yvalid shape: {}'.format(yvalid.shape))
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
xtrain_scaled = scaler.fit_transform(xtrain)
xvalid_scaled = scaler.transform(xvalid)
from sklearn.decomposition import PCA
plt.figure(dpi=250, figsize=(10,5))
pca_curve = PCA().fit(xtrain_scaled)
plt.plot(np.cumsum(pca_curve.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cummulative explained variance')
plt.title('PCA curve')
pca = PCA(n_components=50)
xtrain_pca = pca.fit_transform(xtrain_scaled)
xvalid_pca = pca.transform(xvalid_scaled)
xtrain_pca.shape
m = pca.components_.shape
m # eigenvectors
comp =pd.DataFrame(pca.components_, columns=train.columns, index=['PC'+str(i) for i in range(1,m[0]+1)])
plt.figure(figsize=(20, 20))
sns.heatmap(comp.T)
plt.title('Principal component and primitive feature')
plt.show()
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from xgboost  import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold,cross_val_predict
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, f1_score, precision_score, recall_score, classification_report
# Set random seed
np.random.seed(0)
skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = 0)
def model_selection(listof_model, cv=skf, x=xtrain_pca, y=ytrain):
    
    " listof_model is dictionary type containing different model algorithm "
    
    result = {}
    
    for cm in list(listof_model.items()):
        
        name = cm[0]
        model = cm[1]
        
        cvs = cross_val_score(model, x, y, cv=cv).mean()
        ypred = cross_val_predict(model, x, y, cv=cv)
        auc = roc_auc_score(y, ypred)
        precision = precision_score(y, ypred)
        recall = recall_score(y, ypred)
        acc = accuracy_score(y, ypred)
        fscore = f1_score(y, ypred)
        
        result[name] = {'cross_val_score': cvs, 'auc':auc, 'precision':precision, 'recall':recall,
                        'accuracy': acc, 'f1_score': fscore}
        
        print('{} model done !!!'.format(name))
        
        
    return result
listof_model = {'LogisticRegression': LogisticRegression(), 'LinearSVC': LinearSVC(),
                'KNeighbors':KNeighborsClassifier(), 'RandomForest': RandomForestClassifier(),
               'GradientBoosting': GradientBoostingClassifier(), 'ExtraTree':ExtraTreesClassifier(),
               'XGBoost': XGBClassifier()}
all_model = model_selection(listof_model)
pd.DataFrame(all_model) # we see a model that is match well with our data.
# we choose random forest 
search_space = {'n_estimators': [100, 200, 500],
               'criterion': ['gini', 'entropy'], 'min_samples_split': [1,2,3], 
               'max_samples':[0.5363991145732665, 0.1, 1],
               'max_depth': [2,3,4]}
# Create grid search
gridsearch = GridSearchCV(RandomForestClassifier(), search_space, cv=skf, verbose=1)
# Fit grid search
best_model = gridsearch.fit(xtrain_pca, ytrain)
print("Validation set score: {:.2f}".format(gridsearch.score(xvalid_pca, yvalid)))
print("Best parameters: {}".format(gridsearch.best_params_))
print("Best cross-validation score: {:.2f}".format(gridsearch.best_score_))
opt_model = RandomForestClassifier(min_samples_split=2, n_estimators=100, max_depth=2, max_samples=0.5363991145732665).fit(xtrain_pca, ytrain)
ypred = opt_model.predict(xvalid_pca)
print('Validation accuracy score: {}'.format(accuracy_score(yvalid, ypred)))
print(classification_report(yvalid, ypred, target_names=['NO', 'YES']))
xtest_scaled = scaler.transform(test)
xtest_pca = pca.transform(xtest_scaled)
pred = opt_model.predict(xtest_pca)
print('Test accuracy score: {}'.format(accuracy_score(target_test, pred)))