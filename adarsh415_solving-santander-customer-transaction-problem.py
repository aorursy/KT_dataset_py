# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')
print('Train shape',train_data.shape)

print('Test shape', test_data.shape)

#train_data.head(10).T
#train_data.info()
#train_data.describe().T
import seaborn as sns

# Target distribution in training data

sns.barplot(x= train_data['target'].value_counts().index,y=train_data['target'].value_counts().values)
Target = train_data['target']

features = train_data.drop(columns=['target','ID_code']).columns

test_ids = test_data["ID_code"].values

train_ids = train_data['ID_code'].values
#Number of missing in each colums

def missing_value(data, head=False):

    missing = pd.DataFrame(data.isnull().sum()).rename(columns={0:'total'})

    if head:

        return missing.sort_values('total', ascending=False).head(10)

    else:

        return missing.sort_values('total', ascending=False)
# missing value in train data

missing_value(train_data, head=True)
# missing value in test data

missing_value(test_data, head=True)
# pearson correlation

# Create correlation matrix

#corr_matrix = train_data.corr().round(2)

#sns.heatmap(corr_matrix,annot=True)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split,cross_val_score,StratifiedKFold,GridSearchCV

from sklearn.feature_selection import RFECV

from sklearn.preprocessing import StandardScaler, normalize, MinMaxScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.pipeline import Pipeline

import imblearn as iml

from sklearn.metrics import f1_score,make_scorer, roc_auc_score,roc_curve, confusion_matrix,auc,make_scorer

from sklearn.decomposition import PCA
"""

pca_df = normalize(train_data.drop(columns=['ID_code','target']),axis=1)

pca_test_df = normalize(test_data.drop(columns=['ID_code']),axis=1)



def _get_number_components(model, threshold):

    component_variance = model.explained_variance_ratio_

    explained_variance = 0.0

    component = 0

    for var in component_variance:

        explained_variance += var

        component += 1

        if (explained_variance >= threshold):

            break

    return component



### Get the optimal number of components

pca = PCA()

train_pca = pca.fit_transform(pca_df)

test_pca = pca.fit_transform(pca_test_df)

component = _get_number_components(pca, threshold=0.9)

component



"""
# Implement PCA 

#obj_pca = PCA(n_components=component)

#X_pca = obj_pca.fit_transform(pca_df)

#X_t_pca = obj_pca.fit_transform(pca_test_df)
# add the decomposed features in the train dataset



def _add_decomposition(df, decomp, ncomp, flag):

    for i in range(1, ncomp+1):

        df[flag+"_"+str(i)] = decomp[:,i-1]

#_add_decomposition(train_data, X_pca, 90, 'pca')

#_add_decomposition(test_data, X_t_pca, 90, 'pca')
train_data.shape
#from imblearn.over_sampling import SMOTE
#smote = SMOTE(ratio='minority')

#train_sampled_x,train_sampled_y = smote.fit_sample(train_data.drop(columns=['ID_code','target']),train_data['target'])
#features = train_data.drop(columns=['ID_code','target']).columns

#train_data = pd.DataFrame(train_sampled_x)

#train_data.columns = features

#y = pd.Series(train_sampled_y)
train_data.shape


idx = features = train_data.columns.values[2:]

for df in [train_data, test_data]:

    df['sum'] = df[idx].sum(axis=1)

    df['min'] = df[idx].min(axis=1)

    df['max'] = df[idx].max(axis=1)

    df['mean'] = df[idx].mean(axis=1)

    df['std'] = df[idx].std(axis=1)

    df['skew'] = df[idx].skew(axis=1)

    df['kurt'] = df[idx].kurt(axis=1)

    df['med'] = df[idx].median(axis=1)

    df['var'] = df[idx].var(axis=1)

features = train_data.drop(columns=['target','ID_code']).columns

for feature in features:

    train_data['r2_'+feature] = np.round(train_data[feature], 2)

    test_data['r2_'+feature] = np.round(test_data[feature], 2)

    train_data['r1_'+feature] = np.round(train_data[feature], 1)

    test_data['r1_'+feature] = np.round(test_data[feature], 1)
print('train shape',train_data.shape, 'test shape', test_data.shape)

features = train_data.drop(columns=['target','ID_code']).columns

#features = train_data.columns
train_data=train_data.iloc[:,202:]
test_data=test_data.iloc[:,201:]
pipeline = Pipeline([('StanderScaler', StandardScaler())])

train_data = pipeline.fit_transform(train_data)

test_data = pipeline.transform(test_data)
features[200:]
#pipeline = Pipeline([('StanderScaler', StandardScaler())])

#train_data = pipeline.fit_transform(train_data.drop(columns=['target','ID_code']))

#test_data = pipeline.transform(test_data.drop(columns=['ID_code']))
train_data = pd.DataFrame(data=train_data,columns=features[200:])

test_data = pd.DataFrame(data=test_data,columns=features[200:])
train_data.shape
test_data.shape
#xtrain,xtest,ytrain,ytest = train_test_split(train_data,Target,test_size=0.95,random_state=42)
#type(xtrain)

test_data.shape

#type(Target)

#xtrain.shape
#rndF = RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=-1, class_weight='balanced')

#rndF.fit(xtrain,ytrain)

#rndF_prediction = rndF.predict(xtest)

#roc_auc_score for RandomForest model

#print(roc_auc_score(ytest, rndF_prediction))
#scorer = make_scorer(roc_auc_score)

#estimator = RandomForestClassifier(n_estimators=100, random_state=2020, n_jobs=-1,class_weight='balanced')

#selector = RFECV(estimator=estimator,step=1,cv=5,scoring=scorer)
#selector.fit(xtrain,ytrain)
#selector.ranking_
#rankings = pd.DataFrame({'feature':list(features), 'rank': list(selector.ranking_)}).sort_values('rank')
#rankings.head()
#train_data = selector.transform(train_data)

#test_data = selector.transform(test_data)


#feature_importance = pd.DataFrame({'feature':features,'importance':rndF.feature_importances_})

#feature_importance.sort_values(by='importance', ascending=False).head(10)
#plt.figure(figsize=(24,16))

#top 30 impotant features as per randomforest

#sns.barplot(x= 'feature',y='importance',data=feature_importance.sort_values(by='importance', ascending=False)[:30])

#plt.show()
"""

def performance_mes(prediction, target):

    print('Confusion Matrix:')

    print(confusion_matrix(target,prediction))

    fpr, tpr, _ = roc_curve(target,prediction,pos_label=1)

    print('AUC')

    print( auc(fpr,tpr))

"""
#performance_mes(rndF_prediction,ytest)
import lightgbm as lgbm
def lightGBM(train,target,test, n_folds):

    params = {

        'boosting_type':'gbdt',

        'boost': 'gbdt',

        'objective':'binary',

        'learning_rate':0.008,

        'metric':'auc',

        'max_depth':2,

        'num_leaves':13,

        "bagging_fraction" : 0.4,

        "feature_fraction" : 1.0,

        "min_child_samples":80,

        "bagging_freq" : 5,

        #'min_data':150,

        "bagging_seed" : 2020,

        "verbosity" : -1,

        "random_state" : 2020,

        'early_stopping_round':5500,

        'min_data_in_leaf':80,

        'boost_from_average':False,

        #'reg_lambda': 0.33,

        #'reg_alpha': 0.60,

        'seed':2020,

        'tree_learner': 'serial'

    }

    skfold = StratifiedKFold(n_splits=n_folds, random_state=42)

    train_pred = np.zeros(len(train))

    predictions = np.zeros(len(test))

    for fold_, (xidx, validx) in enumerate(skfold.split(train_data,target)):

        print("Fold idx:{}".format(fold_ + 1))

        lgbmTrain = lgbm.Dataset(train.iloc[xidx],label=target.iloc[xidx])

        lgbmVal = lgbm.Dataset(train.iloc[validx],label=target.iloc[validx])

        evals_result = {}

        model = lgbm.train(params,lgbmTrain,120000,valid_sets=[lgbmTrain,lgbmVal], evals_result=evals_result)

        train_pred[validx] = model.predict(train.iloc[validx], num_iteration=model.best_iteration)

        predictions += model.predict(test, num_iteration=model.best_iteration)/skfold.n_splits

    print("CV score: {:<8.5f}".format(roc_auc_score(target, train_pred)))

    return predictions, model, evals_result
prediction, model, eval_result = lightGBM(train_data,Target, test_data,5)
model.best_score
train_data.shape
#eval_result

#skfold = StratifiedKFold(n_splits=11, random_state=42)

#for i, n in enumerate(skfold.split(train_data.values,Target.values)):

    #print('i',i)

    #print('n',n[0])

    #print('m',n[1])
# Submission dataframe

#prediction[prediction>1] = 1

#prediction[prediction<0] = 0



#submitLGB['ID_code'] = test_ids

#submitLGB["target"] = prediction

submitLGB = pd.DataFrame({'ID_code':test_ids, 'target':prediction})



# Create the Submission File using Light GBM

submitLGB.to_csv('LightGBM.csv', index = False)



submitLGB.head()