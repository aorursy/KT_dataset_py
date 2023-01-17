# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import gc

import seaborn as sns

from time import time



#models

from sklearn.model_selection import train_test_split

from sklearn.metrics import make_scorer, accuracy_score, mean_absolute_error

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, RandomForestRegressor, AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

import xgboost as xgb

from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler

from lightgbm import LGBMClassifier, plot_importance

from catboost import CatBoostClassifier, Pool

from sklearn.mixture import GaussianMixture



#stacking stuff

from mlxtend.classifier import StackingCVClassifier

from sklearn.linear_model import LogisticRegression



#PCA

from sklearn.decomposition import PCA





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
global train

global test

train = pd.read_csv('../input/learn-together/train.csv', index_col='Id')

test = pd.read_csv('../input/learn-together/test.csv', index_col='Id')

train.head()
train.describe()
train.dtypes.unique()
def type_otimizator(df):

    for col in df.columns:

        if df[col].dtype=='float64': 

            df[col] = df[col].astype('float32')

        if df[col].dtype=='int64': 

            if df[col].max()<1: df[col] = df[col].astype(bool)

            elif df[col].max()<128: df[col] = df[col].astype('int8')

            elif df[col].max()<32768: df[col] = df[col].astype('int16')

            else: df[col] = df[col].astype('int32')

    return df



train = type_otimizator(train)

test = type_otimizator(test)

gc.collect()
train.columns
train[train.isna()].count()
[col + ' - ' + str(train[col].unique()) for col in train.filter(like='Soil', axis=1).columns]
# let just atrib

id_test = test.index



y = train.Cover_Type

x = train.drop(['Cover_Type'], axis = 1)



train = train.drop(['Cover_Type'], axis = 1)
def getCorrelation():

    fig, ax = plt.subplots(figsize=(15,13))

    im = ax.imshow(train.astype(float).corr(), cmap=plt.cm.RdBu)



    ax.set_xticks(np.arange(len(x.columns)))

    ax.set_yticks(np.arange(len(x.columns)))

    ax.set_xticklabels(x.columns)

    ax.set_yticklabels(x.columns)



    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",

             rotation_mode="anchor")



    ax.set_title("Correlation")

    fig.tight_layout()

    plt.colorbar(im);

    plt.show()



getCorrelation()
# LightGBM 

def getLGBC():

    return LGBMClassifier(n_estimators=500,  

                     learning_rate= 0.001,

                     objective= 'multiclass', 

                     num_class=7,

                     random_state= 1,

                     n_jobs=-1)
def getImportance():

    clas_lgbc= getLGBC()

    clas_lgbc.fit(x, y)

    plot_importance(clas_lgbc, ignore_zero=False, figsize=(8,40))

    

getImportance()

gc.collect()





#let see some histogram

plt.figure(figsize=(15,5))

sns.distplot(x.Hillshade_3pm)

plt.show()
drop_columns = ['Soil_Type7', 'Soil_Type15', 'Soil_Type36', 'Soil_Type28', 'Soil_Type27', 'Soil_Type25', 'Soil_Type21', 'Soil_Type9', 'Soil_Type8']
x = x.drop(drop_columns, axis =1)

test = test.drop(drop_columns, axis =1)



x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=1)
#see the best correlation for Hillshade_3pm

corr = x[x.Hillshade_3pm!=0].corr()

plt.figure(figsize=(20,16))

sns.heatmap(corr,annot=True)
    

def predictHillshade_3pm():

    t = time()

    all_data = x.append(test)

    num_train = len(x)



    cols_for_HS = ['Aspect','Slope', 'Hillshade_9am','Hillshade_Noon']

    HS_zero = all_data[all_data.Hillshade_3pm==0]

    HS_zero.shape



    HS_train = all_data[all_data.Hillshade_3pm!=0]



    rf_hs = RandomForestRegressor(n_estimators=100).fit(HS_train[cols_for_HS], HS_train.Hillshade_3pm)

    out = rf_hs.predict(HS_zero[cols_for_HS]).astype(int)

    all_data.loc[HS_zero.index,'Hillshade_3pm'] = out



    x['Hillshade_3pm']= all_data.loc[:num_train,'Hillshade_3pm']

    test['Hillshade_3pm']= all_data.loc[num_train:,'Hillshade_3pm']

    print('duration: ' + str(time() - t))

    



predictHillshade_3pm()

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=1)

getCorrelation()
ti = time()

pca = PCA(n_components=0.99)

dfPCA = x.append(test)

num_pca = len(x)

trans = pca.fit_transform(dfPCA)

for i in range(trans.shape[1]):

    col = 'pca_' + str(i)

    x[col] = trans[:num_pca, i]

    test[col] = trans[num_pca:, i]

print('duration: ' + str(time() - ti))
def makePCA(df, columnsOrigin, numElement):

    pca = PCA(n_components=numElement)

    dfPCA = df[columnsOrigin].copy()

    elements = pca.fit_transform(dfPCA)

    columnPCA = ['Column_'+str(c) for c in range(numElement)]

    return pd.DataFrame(data = elements, columns = columnPCA)



#pca_columns = ['Soil_Type11', 'Soil_Type12', 'Soil_Type13', 'Soil_Type18', 'Soil_Type22', 'Soil_Type22', 'Soil_Type26', 'Soil_Type30', 'Soil_Type35', 'Soil_Type37', 'Soil_Type38']

pca_columns = ['Soil_Type10', 'Soil_Type11', 'Soil_Type12', 'Soil_Type13', 'Soil_Type18', 'Soil_Type19', 'Soil_Type22', 'Soil_Type22', 'Soil_Type26', 'Soil_Type30', 'Soil_Type35', 'Soil_Type37', 'Soil_Type38']



#x = x.drop(pca_columns, axis =1)

#test = test.drop(pca_columns, axis =1)

x.join(makePCA(x, pca_columns, 7))

x = x.drop(pca_columns, axis =1)



test.join(makePCA(test, pca_columns, 7))

test = test.drop(pca_columns, axis =1)



x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=1)
def getGausianMixtureClassifier(components):

    return GaussianMixture(n_components=components)



def applyGausianMixture():

    t = time()

    num_train = len(x)

    all_data = x.append(test)

    

    components = 10

    gm = getGausianMixtureClassifier(components)

    

    pred = gm.fit_predict(StandardScaler().fit_transform(all_data))

    x['GM'] = pred[:num_train]

    test['GM'] = pred[num_train:]

    

    for i in range(components):

        x['GM'+str(i)] = pred[:num_train]==i  

        test['GM'+str(i)] = pred[num_train:]==i

    print('duration: '+ str(time()-t))

    

#this is used in final predict

def getGausianMixture(x_gm):

    t = time()

    components = 10

    gm = getGausianMixtureClassifier(components)

    pred = gm.fit_predict(StandardScaler().fit_transform(x_gm))

    

    x_gm['GM'] = pred

    for i in range(components):

        x_gm['GM'+str(i)] = pred[:] == i

    print('duration: '+ str(time()-t))

    return x_gm

        

    

applyGausianMixture()

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=1)

gc.collect()
x.head()
# Fit the classifier

def fitClassifier(classifier, classifier_Name, xx_train, yy_train, xx_test, yy_test):

    classifier.fit(xx_train, yy_train)

    pred_train = classifier.predict(xx_test)

    print(classifier_Name + ': ' + str(round(accuracy_score(yy_test, pred_train),5)))

    

# Random Forest.

def getRandomForest():

    return RandomForestClassifier(n_jobs =  -1, n_estimators = 500, max_features = 15, max_depth = 40, random_state = 1)    



# XGBoost

def getXGB():

    return xgb.XGBClassifier(n_estimators = 800, max_depth = 40, random_state = 1 )

    

    



#Neural network

def getNN():

    return MLPClassifier(verbose = False,

                         max_iter=1000,

                         tol = 0.0000000010,

                         solver = 'adam',

                          hidden_layer_sizes=(100),

                          activation='relu')



#Neural network (Dense)

def getDenseNN():

    return MLPClassifier(verbose = True,

                        max_iter=1000,

                        tol = 0.00000000010,

                        solver = 'adam',

                        hidden_layer_sizes=(120,60),

                        activation='relu')



#extra tree

def getExtraTree():

    return ExtraTreesClassifier(n_estimators = 1000, max_features = 15, max_depth = 40, random_state = 1)    





# logistic regretion from stacking

def getLogisticRegression():

    return LogisticRegression(max_iter=1500,

                              n_jobs=-1,

                              solver= 'lbfgs',

                              multi_class = 'multinomial')



# catBooster

def getCatBooster():

    return CatBoostClassifier(iterations=3000, depth=10, eval_metric = 'Accuracy')

    



# adaBoost

def getAdaBoostClassifier():

    return AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=15), n_estimators = 1500, random_state = 1, learning_rate =0.00001)



#LigthGBMClassifier

def getLGBClassifier():

    return LGBMClassifier(n_estimators=500,  

                     learning_rate= 0.0001,

                     objective= 'multiclass', 

                     num_class=7,

                     random_state= 1,

                     n_jobs=-1)

# 2Layer classifier

def get2LayerClassifier():

    return ExtraTreesClassifier(n_estimators = 500, max_features = 4, max_depth = 20, random_state = 1)    

    
def score_LGBC():

    ti = time()

    clas_lgbc = getLGBC()

    fitClassifier(clas_lgbc, 'LGBC', x_train, y_train, x_test, y_test)

    print('duration of LGBC: ' + str(time() - ti))

    

def score_XGB():

    ti = time()

    clas_xgb = getXGB()

    fitClassifier(clas_xgb, 'XGB', x_train, y_train, x_test, y_test)

    print('duration of XGB: ' + str(time() - ti))

    

def score_ExtraTree():

    ti = time()

    clas_extraTree = getExtraTree()

    fitClassifier(clas_extraTree, 'ExtraTree', x_train, y_train, x_test, y_test)

    print('duration of ExtraTree: ' + str(time() - ti))

    

def score_RandomForest():

    ti = time()

    clas_RandomForest = getRandomForest()

    fitClassifier(clas_RandomForest, 'RandomForest', x_train, y_train, x_test, y_test)

    print('duration of RandomForest: ' + str(time() - ti))



def score_CatBooser():

    ti = time()

    clas_Cat = getCatBooster()

    fitClassifier(clas_Cat, 'CatBooster', x_train, y_train, x_test, y_test)

    #clas_Cat.fit(x_train, y_train, eval_set=Pool(x_test, y_test))

    pred_train = clas_Cat.predict(x_test)

    print('CatBooster' + ': ' + str(round(accuracy_score(y_test, pred_train),5)))    

    print('duration of CatBooster: ' + str(time() - ti))

    

def score_AdaBoost():

    ti = time()

    clas_Ada = getAdaBoostClassifier()

    fitClassifier(clas_Ada, 'AdaBooster', x_train, y_train, x_test, y_test)

    print('duration of AdaBooster: ' + str(time() - ti))





 

#score_CatBooser() # ** -> 0.87103 -> 0.87676

#score_ExtraTree() #0.87235 -> 0.875 -> 0.88536

#score_LGBC() #0.76698 -> 0.76786 -> 0.77888

#score_XGB() #0.87037 -> 0.86728 -> 0.87324

#score_RandomForest() #0.86023 -> 0.85935 -> 0.86905

#score_AdaBoost() # ** -> ** -> 0.79586



gc.collect()
# Stacking

clas_rf = getRandomForest()

clas_xgb = getXGB()

clas_extree = getExtraTree()

clas_cat = getCatBooster()





clas_rf.fit(x, y)

clas_xgb.fit(x, y)

clas_extree.fit(x, y)

clas_cat.fit(x, y)



pred_rf = clas_rf.predict(x)

pred_xgb = clas_xgb.predict(x)

pred_extree = clas_extree.predict(x)

pred_cat = clas_cat.predict(x)



# test

x_stacked = pd.DataFrame({'pred_rf' : pred_rf,

                          'pred_xgb' : pred_xgb,

                          'pred_extree' : pred_extree,

                          'pred_cat' : pred_cat[:,0].astype(int)})



x_stacked.append(getGausianMixture(x_stacked))



clas_final = get2LayerClassifier()

clas_final.fit(x_stacked, y)



pred_rf = clas_rf.predict(test)

pred_xgb = clas_xgb.predict(test)

pred_extree = clas_extree.predict(test)

pred_cat = clas_cat.predict(test)



# test

test_stacked = pd.DataFrame({'pred_rf' : pred_rf,

                             'pred_xgb' : pred_xgb,

                             'pred_extree' : pred_extree,

                             'pred_cat' : pred_cat[:,0].astype(int)})



test_stacked.append(getGausianMixture(test_stacked))



prediction = clas_final.predict(test_stacked)
submission = pd.DataFrame({ 'Id': id_test,

                            'Cover_Type': prediction })

submission.to_csv("submission_example.csv", index=False)