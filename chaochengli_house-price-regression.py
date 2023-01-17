from sklearn.metrics import classification_report

from sklearn.ensemble import AdaBoostRegressor,BaggingRegressor,ExtraTreesRegressor,GradientBoostingRegressor,RandomForestRegressor

from sklearn.model_selection import GridSearchCV,cross_val_predict,KFold,cross_val_score,ShuffleSplit

from sklearn.feature_selection import f_regression,RFE,RFECV,SelectKBest,SelectFromModel

from sklearn.tree import DecisionTreeRegressor,ExtraTreeRegressor

from sklearn.svm import SVR

from sklearn.linear_model import LinearRegression,LogisticRegression,RandomizedLasso

from sklearn.pipeline import Pipeline 

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.metrics import fbeta_score, make_scorer

from sklearn import metrics

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import re

import pdb

from math import sqrt
#－－－－－－－－－－－－define 2-D dict－－－－－－－－－－－－－－#

def addtwodimdict(dic, key_a, key_b, val): 

    if key_a in dic:

        dic[key_a].update({key_b: val})

    else:

        dic.update({key_a:{key_b: val}})

        

"""

map the features according to the data_description,find the pattern [featurename]: 

and map the feature with numbers

""" 

def mapping():

    pattern = re.compile(r'[a-zA-Z0-9]{2,15}:')

    f_file = open('../input/data_description.txt','r')

    mapping={}

    for line in f_file.readlines():

        split_line = line.split()

        if split_line:

            if pattern.match(split_line[0]):



                match_feat = pattern.match(split_line[0]).group()

#                 print ('found the feature:',match_feat)

                i=0

                continue

            else:

                pattern1 = re.compile(r'[0-9]{1,5}')

                if pattern1.match(split_line[0]):

                    continue

                else:

                    addtwodimdict(mapping, match_feat, split_line[0], i)

                    i+=1

        else:

            continue

    f_file.close()

    return mapping

maps = mapping()
#－－－－－－－－－－－－data preprocess－－－－－－－－－－－－－－#

x_train = pd.read_csv('../input/train.csv',header=0)

x_test = pd.read_csv('../input/test.csv',header=0)

y_train = x_train['SalePrice']

test_Id = x_test['Id']



"""data preprocess:feature mapping,fillna,and drop some features"""

def data_preprocess(raw_data,maps):

    feat_to_map = ['MSZoning','Street','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope',                   'Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl',                   'Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual',                   'BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir',                   'Electrical', 'KitchenQual','Functional','FireplaceQu','GarageType','GarageFinish','GarageQual',                   'GarageCond','PavedDrive','PoolQC','Fence','MiscFeature','SaleType','SaleCondition']

    for feat in feat_to_map:

        feat_map = feat+':'

        raw_data[feat] = raw_data[feat].map(maps[feat_map])

    feat_to_drop = []

    

#drop features with too few data

    for feat in raw_data.columns:

        if (raw_data[feat].count()<1000):

            feat_to_drop.append(feat)

    feat_to_drop.append('Id')

    raw_data_new = raw_data.drop(feat_to_drop,axis = 1)



#fill nan of features with mean

    for feat in raw_data_new.columns:

        index_num = raw_data_new.shape[0]

        if (raw_data_new[feat].count()<index_num):

            raw_data_new[feat] = raw_data_new[feat].fillna(raw_data_new[feat].mean())

    return raw_data_new

x_new_train = data_preprocess(x_train,maps)

x_new_train = x_new_train.drop(['SalePrice'],axis = 1)

x_new_test = data_preprocess(x_test,maps)

features = x_new_train.columns

# print (x_new_train.describe())

# print (x_new_test.describe())

# print (y_train.describe())
#－－－－－－－－－－－－plot feature importance－－－－－－－－－－－－－－#

def plot_feature_importance(ax,title,feature_importance,legend_flag = True):

    for i in range(len(feature_importance)):

        label_name = feature_importance[i][0]

        ax.bar(i,feature_importance[i][1],label = label_name)

        ax.hold(True)

    if legend_flag:

        ax.legend(loc = 'best')

    ax.set_title(title)

    plt.show()

    

feat_num = 20#decompose to 20 features

feat_selector = {'SelectKBest':SelectKBest(f_regression,k=feat_num),'Tree_based':ExtraTreeRegressor()                 ,'random_forest':RandomForestRegressor(n_estimators=20, max_depth=4)                 ,'RFE':RFE(estimator=LinearRegression(),n_features_to_select=feat_num,step=3)                 ,'RFECV':RFECV(estimator=LinearRegression(),step=3)                 ,'RandomizedLasso':RandomizedLasso(alpha=0.025)

                }

fig = plt.figure(figsize = (16,9))

i=0

for key in feat_selector:

    clf = feat_selector[key]

    clf = clf.fit(x_new_train,y_train)

    i+=1

    if key=='Tree_based':

        title = 'Tree_based feature importances'

#         print (title,clf.feature_importances_)      

        ax = fig.add_subplot(2,3,i)

        feature_importances = zip(features,clf.feature_importances_)

        plot_feature_importance(ax,title,feature_importances,False)

    elif key=='RFE': 

        title = 'RFE feature importance'

        ax = fig.add_subplot(2,3,i)

        feature_importances = zip(features,1.0/clf.ranking_)

        plot_feature_importance(ax,title,feature_importances,False)

    elif key=='RFECV':

        title = 'RFECV feature importance'

#         print (title,clf.ranking_)

        ax = fig.add_subplot(2,3,i)

        feature_importances = zip(features,1.0/clf.ranking_)

        plot_feature_importance(ax,title,feature_importances,False)

    elif key=='SelectKBest':    

        title = 'SelectKBest scores'

        print (title,clf.scores_)

        ax = fig.add_subplot(2,3,i)

        KBest_score = clf.scores_

        feature_importances = zip(features,KBest_score)

#         print (feature_importances)

        plot_feature_importance(ax,title,feature_importances,False)

    elif key=='random_forest':

        title = 'random forest scores'

#         print (title,clf.feature_importances_)

        feature_importances = zip(features,clf.feature_importances_)          

        ax = fig.add_subplot(2,3,i)

        plot_feature_importance(ax,title,feature_importances,False)

    elif key=='RandomizedLasso':

        title = 'RandomizedLasso feat importance'#1 for top,the higher the more important

        feature_importances =  sorted(zip(features,map(lambda x: round(x, 4), clf.scores_)), reverse=True)      

        ax = fig.add_subplot(2,3,i)

        plot_feature_importance(ax,title,feature_importances,False)

fig.savefig('feature_importance.png')
#－－－－－－－－－－－feature selection－－－－－－－－#

KBest_score_dict = {}

for i in range(len(KBest_score)):

    KBest_score_dict[i] = KBest_score[i]

KBest_score_sorted = sorted(KBest_score_dict.iteritems(),key = lambda x:x[1],reverse = True)

Kbest_feat_index = [x[0] for x in KBest_score_sorted[:feat_num]]

Kbest_feat_name = features[Kbest_feat_index]

x_new_train = x_new_train[Kbest_feat_name]

x_new_test = x_new_test[Kbest_feat_name]

features = x_new_train.columns

print (x_new_train.head(3))

print (x_new_test.head(3))
#－－－－－－－－－－－－Output－－－－－－－－－－－－－－#

filepath = './Result_details.txt'

randomforest__mse = mse_all['random_forest']

DTree_mse = mse_all['Decision_Tree']

Adboost_mse = mse_all['AdaBoostRegressor']

Bagging_mse = mse_all['BaggingRegressor']

Extra_mse = mse_all['ExtraTreesRegressor']

text = 'Features are %s\nRandom forest mse is: %.5f\nDecision Tree mse is: %.5f\nAdaBoostRegressor mse is: %.5f\nBaggingRegressor mse is: %.5f\nExtraTreesRegressor mse is: %.5f\n'%(features,randomforest__mse,DTree_mse,Adboost_mse,                                                                    Bagging_mse,Extra_mse)

f_file = open(filepath,'a+')

f_file.write(text)

f_file.close()



result = pd.DataFrame({'Id':test_Id,'SalePrice':pd.Series(y_test_pred)})

result.to_csv('Submission.csv',index = False)