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
import pandas as pd
import sys
import numpy as np
import statsmodels.api as sm
from statsmodels.tools import add_constant
from statsmodels.regression.linear_model import RegressionResults
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
sns.set(style="ticks")
%matplotlib inline
import seaborn as sns
import sklearn as sk
from sklearn.neighbors import KNeighborsRegressor
sns.set(style="ticks")
%matplotlib inline
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report, precision_score
from sklearn import preprocessing
from sklearn.grid_search import GridSearchCV


from sklearn.preprocessing import StandardScaler
import seaborn as sns
pd.set_option('display.width', 1500)
pd.set_option('display.max_columns', 150)
pd.set_option('display.max_rows', 200)
sns.set_context('poster')
data = pd.read_csv("../input/train.csv")

data.shape
data.head()
data.columns[data.isna().sum()!=0]
data['meaneduc'].fillna(data['escolari'], inplace=True)
data[data['SQBmeaned'].isnull()][['Id','meaneduc','idhogar','edjefe','edjefa', 'hogar_adul', 'hogar_mayor', 'hogar_nin', 'age', 'escolari']]
data['SQBmeaned'].fillna(data['escolari']**2, inplace=True)
data['v2a1'].isnull().sum()

norent=data[data['v2a1'].isnull()]
print("Total ", data['v2a1'].isnull().sum())
print("Owns his house:", norent[norent['tipovivi1']==1]['Id'].count())
print("Owns his house paying installments", norent[norent['tipovivi2']==1]['Id'].count())
print("Rented ", norent[norent['tipovivi3']==1]['Id'].count())
print("Precarious ", norent[norent['tipovivi4']==1]['Id'].count())
print("Other ", norent[norent['tipovivi5']==1]['Id'].count())

data['v2a1'].fillna(0, inplace=True)
data['v18q1'].isna().sum()
data[data['v18q']==0]['Id'].count()

data['v18q1'].fillna(0, inplace=True)
data['rez_esc'].isnull().sum()

data[data['rez_esc']>1][['age', 'escolari', 'rez_esc']][:10]

data[data['rez_esc'].isnull()][['age', 'escolari', 'rez_esc']][:10]

data['rez_esc'].fillna(0, inplace=True)
data.columns[data.dtypes==object]
data['dependency'].unique()
data[(data['dependency']=='yes') & (data['SQBdependency']!=0)][['idhogar','dependency','SQBdependency','age']].head()
data[(data['dependency']=='no') ][['idhogar','dependency','SQBdependency','age']].head()
data['dependency'] = data['dependency'].replace(['no'], '0')
data['dependency'] = data['dependency'].replace(['yes'], '1')
data.dependency = data.dependency.astype(float)
data['edjefe'].unique()
data['edjefa'].unique()
data[(data['edjefe']=='no')|(data['edjefe']=='yes')][['edjefe', 'edjefa', 'SQBedjefe','parentesco1','escolari']].head()
data['edjefe'] = data['edjefe'].replace(['no'], '0')
data['edjefe'] = data['edjefe'].replace(['yes'], '1')
data[(data['edjefa']=='yes')][[ 'edjefa','parentesco1','escolari','female','idhogar']].head()
data[(data['parentesco1']==1)&(data['female']==1)&(data['edjefa']=="yes")][[ 'edjefa','parentesco1','escolari','female','idhogar']].head()
data[(data['parentesco1']==1)&(data['female']==1)&(data['edjefa']=="no")][[ 'edjefa','parentesco1','escolari','female','idhogar']].head()
data[(data['parentesco2']==1)&(data['female']==1)&(data['edjefa']=="yes")][[ 'edjefa','parentesco2','escolari','female','idhogar']]
data[(data['parentesco2']==1)&(data['female']==1)&(data['edjefa']=="no")][[ 'edjefa','parentesco2','escolari','female','idhogar']].head()
data[(data['parentesco1']==1)&(data['female']==1)][ 'edjefa'].unique()
data[(data['parentesco1']==1)&(data['female']==1)][ 'escolari'].unique()
data[(data['parentesco2']==1)&(data['female']==1)][ 'edjefa'].unique()
data[(data['parentesco2']==1)&(data['female']==1)&(data['edjefa']=='no')][ 'escolari'].unique()
pivot1 = pd.pivot_table(data[data.parentesco1 == 1],index=["idhogar"],values=["escolari"],fill_value=0)
pivot1.columns = ['edjefeh']
pivot1.head()
pivot2 = pd.pivot_table(data[data.parentesco2 == 1],index=["idhogar"],values=["escolari"],fill_value=0)
pivot2.columns = ['edspouseh']
pivot2.head()
data=data.merge(pivot1, on='idhogar', how='left')
data=data.merge(pivot2, on='idhogar', how='left')
data['edspouseh'].fillna(0, inplace=True)
data['edjefeh'].fillna(0, inplace=True)
data=data.drop(['edjefe', 'edjefa'], axis=1)
household = ['Target','v2a1', 'hacdor', 'rooms', 'hacapo','v14a', 'refrig', 'v18q','v18q1', 'r4h1', 'r4h2','r4h3','r4m1','r4m2','r4m3','r4t1','r4t2','r4t3','tamhog','tamviv','hhsize','paredblolad','paredzocalo','paredpreb','pareddes','paredmad','paredzinc','paredfibras','paredother','pisomoscer','pisocemento','pisoother','pisonatur','pisonotiene','pisomadera','techozinc','techoentrepiso','techocane','techootro','cielorazo','abastaguadentro','abastaguafuera','abastaguano','public','planpri','noelec','coopele','sanitario1','sanitario2','sanitario3','sanitario5','sanitario6','energcocinar1','energcocinar2','energcocinar3','energcocinar4','elimbasu1','elimbasu2','elimbasu3','elimbasu4','elimbasu5','elimbasu6','epared1','epared2','epared3','etecho1','etecho2','etecho3','eviv1','eviv2','eviv3']
d={}
weird2=[]
for row in data.iterrows():
    for i in household:
        idhogar=row[1]['idhogar']
        var=row[1][i]
        if idhogar in d:
            if d[idhogar]!=var:
                weird2.append(i)
        else:
            d[idhogar]=var

weird2 = set(weird2)
weird2
tables={x: pd.pivot_table(data[data.parentesco1 == 1],index=["idhogar"],values=[x]) for x in weird2}
for i in tables: 
    tables[i].columns = [i+'h']
    data = data.merge(tables[i], on='idhogar', how = 'left')
    data[i+'h'].fillna(data[i], inplace=True)
    data=data.drop(columns=[i])

data.columns[data.isna().sum()!=0]
def data_cleaning(data):
    data['meaneduc'].fillna(data['escolari'], inplace=True)
    data['SQBmeaned'].fillna(data['escolari']**2, inplace=True)
    data['rez_esc']=data['rez_esc'].fillna(0)
    data['v18q1']=data['v18q1'].fillna(0)
    data['v2a1']=data['v2a1'].fillna(0)
    data['dependency'] = data['dependency'].replace(['no'], '0')
    data['dependency'] = data['dependency'].replace(['yes'], '1')
    data.dependency = data.dependency.astype(float)
    data['edjefe'] = data['edjefe'].replace(['no'], '0')
    data['edjefe'] = data['edjefe'].replace(['yes'], '1')
    pivot1 = pd.pivot_table(data[data.parentesco1 == 1],index=["idhogar"],values=["escolari"],fill_value=0)
    pivot1.columns = ['edjefeh']
    pivot2 = pd.pivot_table(data[data.parentesco2 == 1],index=["idhogar"],values=["escolari"],fill_value=0)
    pivot2.columns = ['edspouseh']
    data=data.merge(pivot1, on='idhogar', how='left')
    data=data.merge(pivot2, on='idhogar', how='left')
    data['edspouseh'].fillna(0, inplace=True)
    data['edjefeh'].fillna(0, inplace=True)
    data=data.drop(['edjefe', 'edjefa'], axis=1)
    household2 = ['v2a1', 'hacdor', 'rooms', 'hacapo','v14a', 'refrig', 'v18q','v18q1', 'r4h1', 'r4h2','r4h3','r4m1','r4m2','r4m3','r4t1','r4t2','r4t3','tamhog','tamviv','hhsize','paredblolad','paredzocalo','paredpreb','pareddes','paredmad','paredzinc','paredfibras','paredother','pisomoscer','pisocemento','pisoother','pisonatur','pisonotiene','pisomadera','techozinc','techoentrepiso','techocane','techootro','cielorazo','abastaguadentro','abastaguafuera','abastaguano','public','planpri','noelec','coopele','sanitario1','sanitario2','sanitario3','sanitario5','sanitario6','energcocinar1','energcocinar2','energcocinar3','energcocinar4','elimbasu1','elimbasu2','elimbasu3','elimbasu4','elimbasu5','elimbasu6','epared1','epared2','epared3','etecho1','etecho2','etecho3','eviv1','eviv2','eviv3']
    d={}
    weird2=[]
    for row in data.iterrows():
        for i in household2:
            idhogar=row[1]['idhogar']
            var=row[1][i]
            if idhogar in d:
                if d[idhogar]!=var:
                    weird2.append(i)
            else:
                d[idhogar]=var
    weird2 = set(weird2)
    
    tables={x: pd.pivot_table(data[data.parentesco1 == 1],index=["idhogar"],values=[x]) for x in weird2}
    
    for i in tables: 
        tables[i].columns = [i+'h']
        data = data.merge(tables[i], on='idhogar', how = 'left')
        data[i+'h'].fillna(data[i], inplace=True)
        data=data.drop(columns=[i])
    return data
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
def makehistogram(x):

    plt.hist(x, color='dodgerblue')
    plt.ylabel('Number of obs')
    plt.axvline(x=np.mean(x), color='red', label='Mean')
    plt.axvline(x=np.median(x), color='salmon', label='Median')



    plt.legend()
    plt.show()

makehistogram(data.Targeth)
makehistogram(data.escolari)
#And now for a plot of histograms across registration type

def multiplehist(x):
    tar1= (x[data.Targeth == 1])
    tar2= (x[data.Targeth == 2])
    tar3= (x[data.Targeth == 3])
    tar4= (x[data.Targeth == 4])

    plt.hist(tar1, alpha=.25, label='target 1', density=True)
    plt.hist(tar2, alpha=.25, label='target 2', density=True)
    plt.hist(tar3, alpha=.25, label='target 3', density=True)
    plt.hist(tar4, alpha=.25, label='target 4', density=True)
    plt.ylabel('%')
    plt.legend()
    #plt.xscale('log')

    plt.show()

multiplehist(data.escolari)
def multiplehist2(x):
    tar1= (x[data.Targeth == 1])
    tar2= (x[data.Targeth == 2])
    tar3= (x[data.Targeth == 3])
    tar4= (x[data.Targeth == 4])
    plt.hist([tar1, tar2, tar3, tar4], density=True, label=['target 1', 'target 2', 'target 3', 'target 4'], color=['blue','dodgerblue','red','salmon'])
    plt.legend(loc='upper right')
    plt.show()
    #plt.xscale('log')

    plt.show()

multiplehist2(data.escolari)
def multiplehist3(x):
    tar1= (x[data.Targeth == 1])
    tar2= (x[data.Targeth == 2])
    tar3= (x[data.Targeth == 3])
    tar4= (x[data.Targeth == 4])
    datal = [tar1,tar2,tar3,tar4]
    titles = ['target 1','target 2','target 3','target 4'] 

    f,a = plt.subplots(2,2, figsize=(12, 6))
    a = a.ravel()
    for idx,ax in enumerate(a):
        ax.hist(datal[idx],density=True, color='dodgerblue')
        ax.set_title(titles[idx])
        ax.axvline(x=np.mean(datal[idx]), color='red', label='Mean')
        ax.axvline(x=np.median(datal[idx]), color='salmon', label='Median')
        
        
    plt.tight_layout()
    
multiplehist3(data.edjefeh)
multiplehist3(data.escolari)
def targetboxplot(x):
    tar1= (x[data.Targeth == 1])
    tar2= (x[data.Targeth == 2])
    tar3= (x[data.Targeth == 3])
    tar4= (x[data.Targeth == 4])
    data1 = [tar1,tar2,tar3,tar4]
    fig = plt.figure(1, figsize=(9, 6))

    # Create an axes instance
    ax = fig.add_subplot(111)

    # Create the boxplot
    bp = ax.boxplot(data1, patch_artist=True)

    ## change outline color, fill color and linewidth of the boxes
    for box in bp['boxes']:
        # change outline color
        box.set( color='#7570b3', linewidth=2)
        # change fill color
        box.set( facecolor = '#1b9e77' )
    ## change whisker color
    for whisker in bp['whiskers']:
        whisker.set(color='#7570b3', linewidth=2)

    ## change color and linewidth of the caps
    for cap in bp['caps']:
        cap.set(color='dodgerblue', linewidth=2)

    ## change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='#b2df8a', linewidth=2)

    ## change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(marker='o', markerfacecolor='dodgerblue', alpha=0.5, markersize=10)
    ax.set_xticklabels(['Target = 1', 'Target = 2', 'Target = 3', 'Target = 4'])
    ## Remove top axes and right axes ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

targetboxplot(data.escolari)
targetboxplot(data.edjefeh)
targetboxplot(data.v2a1h)
targetboxplot(data.edspouseh)
targetboxplot(data.r4t3h)
targetboxplot(data.hhsizeh)
targetboxplot(data.dependency)
def makebar(x):
    data_bar=data.groupby(['Targeth'])[x].mean()*100

    data_bar.plot(kind='bar', color = 'dodgerblue', tick_label = data_bar, title = x)

#data_bar.values
makebar('paredzinch')
makebar('paredbloladh')
makebar('pisocementoh')
makebar('pisonaturh')
makebar('pisonotieneh')
makebar('energcocinar1h')
data.shape
data_work = data.drop(['idhogar', 'Id'], axis=1)
X_data = data_work.drop(['Targeth'],axis=1)
Y_data = data_work[['Targeth']]

def input_confusion(n_neighbors, weights, data, y, test_size):
    X_train, X_test, y_train, y_test = train_test_split(data.drop([y],axis=1), data[y], test_size=test_size)
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    score = clf.score(X_test, y_test)
    return(pred, score, clf.classes_, y_test)

def plot_confusion_matrix(cm, classes, n_neighbors, title='Confusion matrix (Normalized)',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Normalized confusion matrix: KNN-{}'.format(n_neighbors))
    plt.colorbar()
    plt.xticks(np.arange(4), classes)
    plt.yticks(np.arange(4), classes)
    plt.tight_layout()
    plt.xlabel('True label',rotation='horizontal', ha='right')
    plt.ylabel('Predicted label')
    plt.show()
def confusion(n_neighbors, weights, data, y, test_size):    
    pred, score, classes, y_test = input_confusion(n_neighbors, weights, data, y, test_size)
    cm = confusion_matrix(y_test, pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plot_confusion_matrix(cm_normalized.T, classes, n_neighbors=n_neighbors)
    cm_df = pd.DataFrame(cm.T, index=classes, columns=classes)
    cm_df.index.name = 'Predicted'
    cm_df.columns.name = 'True'
    print(cm_df)    
    print(pd.DataFrame(precision_score(y_test, pred, average=None),
                       index=classes, columns=['Precision'])) 

#import pydot
from IPython.display import Image
from sklearn.externals.six import StringIO  
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error,confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier 
from sklearn.metrics import log_loss
from sklearn.calibration import CalibratedClassifierCV
#Split Train / Test
%matplotlib inline
def input_confusion_tree( X_data, data, y, estimators):
    X_train, X_test, y_train, y_test = train_test_split(X_data, data[y], test_size=0.3)
    clf = RandomForestClassifier(n_estimators=estimators, n_jobs=-1)
    calibrated_clf = CalibratedClassifierCV(clf, method='isotonic', cv=5)
    calibrated_clf.fit(X_train, y_train)
    pred = calibrated_clf.predict(X_test)
    score = calibrated_clf.score(X_test, y_test)
    return(pred, score, calibrated_clf.classes_, y_test)

def plot_confusion_matrix(cm, classes, title='Confusion matrix (Normalized)',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Normalized confusion matrix')
    plt.colorbar()
    plt.xticks(np.arange(4), classes)
    plt.yticks(np.arange(4), classes)
    plt.tight_layout()
    plt.xlabel('True label',rotation='horizontal', ha='right')
    plt.ylabel('Predicted label')
    plt.show()
def confusion_tree(X_data,data, y, estimators):    
    pred, score, classes, y_test = input_confusion_tree(X_data, data, y, estimators)
    cm = confusion_matrix(y_test, pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plot_confusion_matrix(cm_normalized.T, classes)
    cm_df = pd.DataFrame(cm.T, index=classes, columns=classes)
    cm_df.index.name = 'Predicted'
    cm_df.columns.name = 'True'
    print('Number of estimators '+str(i))
    print(cm_df) 
    print(score)
    print(pd.DataFrame(precision_score(y_test, pred, average=None),
                       index=classes, columns=['Precision'])) 


for i in np.arange(1, 150, 10):
    confusion_tree(X_data, data_work, 'Targeth', i)
for i in range(1, 10):
    confusion_tree(X_data, data_work, 'Targeth', i)

def finding_random_forest(X_data, data, y, estimators, max_depth):
    scores=[]
    estimators_used = []
    depth_used = []
    X_train, X_test, y_train, y_test = train_test_split(X_data, data[y], test_size=0.5)
    param_grid ={
        'n_estimators': estimators,
        'max_depth': max_depth
        }
    clf = RandomForestClassifier( n_jobs=-1)
    grid_clf = GridSearchCV(clf, param_grid, cv=10)
    grid_clf.fit(X_train, y_train)
    for i in range (len(estimators)*len(max_depth)):
        scores.append(grid_clf.grid_scores_[i][1])
        estimators_used.append(grid_clf.grid_scores_[i][0]['n_estimators'])
        depth_used.append(grid_clf.grid_scores_[i][0]['max_depth'])
    random_forest_scores = pd.DataFrame(
        {'score': scores,
         'depth': depth_used,
         'estimator': estimators_used
        })
    best_score = random_forest_scores['score'].max()
    best_estimator=random_forest_scores[random_forest_scores['score']==random_forest_scores['score'].max()]['estimator'].values
    best_depth=random_forest_scores[random_forest_scores['score']==random_forest_scores['score'].max()]['depth'].values
    print('best score is '+str(best_score) +' with depth '+ str(best_depth)+' and number of estimators '+str(best_estimator))
    return grid_clf , random_forest_scores

    
estimators_1 = [i for i in range(5,20)]
depth_1 = [i for i in range(5,20)]
finding_random_forest(X_data, data_work, 'Targeth', estimators_1, depth_1)
test_data=pd.read_csv("../input/test.csv")


test_data=data_cleaning(test_data)
test_data.columns[test_data.isna().sum()!=0]
ids=test_data['Id']
test_data=test_data.drop(['Id', 'idhogar'], axis=1)
test_data.shape
clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
             max_depth=19, max_features=17, max_leaf_nodes=None,
             min_impurity_decrease=0.0, min_impurity_split=None,
             min_samples_leaf=1, min_samples_split=2,
             min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=-1,
             oob_score=False, random_state=None, verbose=0,
             warm_start=False)

clf.fit(X_data, Y_data)
prediction_1=clf.predict(test_data)
submit_1=pd.DataFrame({'Id': ids, 'Target': prediction_1})


submit_1['Target']=pd.to_numeric(submit_1['Target'], downcast='signed')
submit_1.to_csv('submit_1.csv', index=False)












