# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_column',100)
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
original_data = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')
original_data.head()
original_data.isnull().sum()
y = original_data.diagnosis
dataset = original_data.drop(['id','Unnamed: 32','diagnosis'],axis =1)
dataset.head()
print('Observations for Malignant and Benign is \n{}'.format(y.value_counts()))
_ = sns.countplot(y).set(title = 'Number of Labels')
dataset.describe()
data_normalize = (dataset - dataset.mean())/dataset.std()
data = pd.concat([y , data_normalize.iloc[:,0:10]] , axis = 1)
data.head()
data = pd.melt(data, id_vars='diagnosis',var_name='features',value_name='value')
data.head()
plt.figure(figsize = (15,8))
sns.violinplot(x = 'features',y = 'value',hue = 'diagnosis' , split=True ,inner='quart' , data = data)
plt.xticks(rotation = 45)
data = pd.concat([y , data_normalize.iloc[:,10:20]] , axis = 1)
data = pd.melt(data, id_vars='diagnosis',var_name='features',value_name='value')
plt.figure(figsize = (15,8))
sns.violinplot(x = 'features',y = 'value',hue = 'diagnosis' , split=True ,inner='quart' , data = data)
plt.xticks(rotation = 45)
data = pd.concat([y , data_normalize.iloc[:,20:]] , axis = 1)
data = pd.melt(data, id_vars='diagnosis',var_name='features',value_name='value')
plt.figure(figsize = (15,8))
sns.violinplot(x = 'features',y = 'value',hue = 'diagnosis' , split=True , inner = 'quart',data = data)
plt.xticks(rotation = 45)
sns.set(style="whitegrid", palette="muted")
data = pd.concat([y , data_normalize.iloc[:,0:10]] , axis = 1)
data = pd.melt(data, id_vars='diagnosis',var_name='features',value_name='value')
plt.figure(figsize = (15,8))
sns.swarmplot(x = 'features',y = 'value',hue = 'diagnosis', data = data)
plt.xticks(rotation = 45)
data = pd.concat([y , data_normalize.iloc[:,10:20]] , axis = 1)
data = pd.melt(data, id_vars='diagnosis',var_name='features',value_name='value')
plt.figure(figsize = (15,8))
sns.swarmplot(x = 'features',y = 'value',hue = 'diagnosis', data = data)
plt.xticks(rotation = 45)
data = pd.concat([y , data_normalize.iloc[:,20:]] , axis = 1)
data = pd.melt(data, id_vars='diagnosis',var_name='features',value_name='value')
plt.figure(figsize = (15,8))
sns.swarmplot(x = 'features',y = 'value',hue = 'diagnosis', data = data)
plt.xticks(rotation = 45)
data = pd.concat([y , data_normalize.iloc[:,0:10]] , axis = 1)
data = pd.melt(data, id_vars='diagnosis',var_name='features',value_name='value')
plt.figure(figsize = (15,8))
sns.boxplot(x = 'features',y='value',data = data,hue = 'diagnosis')
plt.xticks(rotation = 45)
data = pd.concat([y , data_normalize.iloc[:,10:20]] , axis = 1)
data = pd.melt(data, id_vars='diagnosis',var_name='features',value_name='value')
plt.figure(figsize = (15,8))
sns.boxplot(x = 'features',y='value',data = data,hue = 'diagnosis')
plt.xticks(rotation = 45)
data = pd.concat([y , data_normalize.iloc[:,20:]] , axis = 1)
data = pd.melt(data, id_vars='diagnosis',var_name='features',value_name='value')
plt.figure(figsize = (15,8))
sns.boxplot(x = 'features',y='value',data = data,hue = 'diagnosis')
plt.xticks(rotation = 45)
sns.jointplot(x = data_normalize.concavity_mean , y = data_normalize['concave points_mean'] , kind = 'regg').annotate(pearsonr)
sns.set(style = 'white')
data = data_normalize.loc[:,['radius_worst','perimeter_worst','area_worst']]
g = sns.PairGrid(data,diag_sharey=False)
g.map_lower(sns.kdeplot,cmap='Blues_d')
g.map_upper(plt.scatter)
g.map_diag(sns.kdeplot,lw=3)
f,ax = plt.subplots(figsize = (18,18))
sns.heatmap(data_normalize.corr() , annot=True, linewidths=.5, fmt= '.1f',ax=ax)
drop_list1 = ['perimeter_mean','radius_mean','compactness_mean','concave points_mean',
              'radius_se','perimeter_se','radius_worst','perimeter_worst','compactness_worst',
              'concave points_worst','compactness_se','concave points_se','texture_worst','area_worst']

data = data_normalize.drop(drop_list1 , axis = 1)
data.head()
f,ax = plt.subplots(figsize = (18,18))
sns.heatmap(data.corr() , annot=True, linewidths=.5, fmt= '.1f',ax=ax)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix

x_train,x_test,y_train,y_test = train_test_split(data.values,y.values,test_size = 0.2,random_state = 42)
clf_1 = RandomForestClassifier()
clf_1.fit(x_train,y_train)
y_pred = clf_1.predict(x_test)
accuracy_score(y_pred,y_test)
sns.heatmap(confusion_matrix(y_pred,y_test) , annot = True , fmt = '.1f')
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# We don't know how to select K, the simple way is to try different values of K 

def validation_error(x_train,x_test,y_train,y_test):
    m = 0
    for k in range(3,10):
        select_feature = SelectKBest(chi2 , k=k)
        select_feature.fit(x_train,y_train)
        x_train2 = select_feature.transform(x_train)
        x_test2 = select_feature.transform(x_test)

        clf_2 = RandomForestClassifier()
        clf_2.fit(x_train2,y_train)

        y_pred = clf_2.predict(x_test2)
        acc = accuracy_score(y_pred,y_test)
        if acc > m:
            m = acc
            model = clf_2
            pred = y_pred
            scores = select_feature.scores_
    return (m,model,pred,scores)
x_train,x_test,y_train,y_test = train_test_split(dataset.values,y.values,test_size = 0.2,random_state = 42)
(acc,model,y_pred,scores) = validation_error(x_train,x_test,y_train,y_test)
print(scores,dataset.columns)
cm = confusion_matrix(y_pred,y_test)
print('Accuracy Score is {}'.format(acc))
sns.heatmap(cm,annot=True)
from sklearn.feature_selection import RFE
clf_3 = RandomForestClassifier()
rfe = RFE(estimator = clf_3,n_features_to_select=5,step=1)
rfe = rfe.fit(x_train,y_train)
dataset.columns[rfe.support_]
from sklearn.feature_selection import RFECV
clf_4 = RandomForestClassifier()
rfecv = RFECV(estimator=clf_4,step=1,cv=5,scoring='accuracy')
x_train,x_test,y_train,y_test = train_test_split(data.values,y.values,test_size = 0.2,random_state = 42)
rfecv = rfecv.fit(x_train,y_train)
print('Optimal number of features :', rfecv.n_features_)
print('Best features :', data.columns[rfecv.support_])
rfecv.grid_scores_
# Plot number of features VS. cross-validation scores
import matplotlib.pyplot as plt
plt.figure(figsize = (15,8))
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score of number of selected features")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
clf_rf_5 = RandomForestClassifier()      
clr_rf_5 = clf_rf_5.fit(x_train,y_train)
importances = clr_rf_5.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf_rf_5.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(x_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest

plt.figure(1, figsize=(14, 13))
plt.title("Feature importances")
plt.bar(range(x_train.shape[1]), importances[indices],
       color="g", yerr=std[indices], align="center")
plt.xticks(range(x_train.shape[1]), data.columns[indices],rotation=45)
plt.xlim([-1, x_train.shape[1]])
plt.show()
x_train, x_test, y_train, y_test = train_test_split(dataset,y,test_size=0.3, random_state=42)
#normalization
x_train_N = (x_train-x_train.mean())/(x_train.max()-x_train.min())
x_test_N = (x_test-x_test.mean())/(x_test.max()-x_test.min())

from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)
pca.fit(x_train_N)

plt.figure(1, figsize=(14, 13))
plt.clf()
plt.axes([.2, .2, .7, .7])
plt.plot(np.cumsum(pca.explained_variance_ratio_), linewidth=2)
plt.axis('tight')
plt.xlabel('n_components')
plt.ylabel('explained_variance_ratio_')
X_reduced = pca.fit_transform(data)

clf_rf_5.fit(x_train_N,y_train)
y_pred = clf_rf_5.predict(x_test_N)
accuracy_score(y_pred,y_test)
from boruta import BorutaPy
x_train,x_test,y_train,y_test = train_test_split(data_normalize,y,test_size=0.2,random_state=42)
clf_6 = RandomForestClassifier(class_weight='balanced',max_depth=6)
boruta_selector = BorutaPy(clf_6,n_estimators='auto',verbose=2)
boruta_selector.fit(dataset.values,y.values)
print('Number of Selected Features',boruta_selector.n_features_)
feature_df = pd.DataFrame(dataset.columns.tolist(),columns = ['features'])
feature_df['rank'] = boruta_selector.ranking_
feature_df = feature_df.sort_values('rank').reset_index(drop = True)
feature_df.head(boruta_selector.n_features_)
data_boruta = data_normalize[data_normalize.columns[boruta_selector.support_]]
data_boruta.head()
x_train,x_test,y_train,y_test = train_test_split(data_boruta,y,test_size=0.2,random_state=42)
clf_6.fit(x_train,y_train)
y_pred = clf_6.predict(x_test)
accuracy_score(y_test,y_pred)
