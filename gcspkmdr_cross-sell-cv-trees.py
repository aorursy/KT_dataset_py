import pandas as pd

import matplotlib.pyplot as plt

import numpy as np



from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 15,15

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import xgboost as xgb

import lightgbm as lgb

from catboost import CatBoostClassifier



from sklearn.model_selection import train_test_split,StratifiedShuffleSplit,StratifiedKFold

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import roc_auc_score,accuracy_score ,confusion_matrix

from sklearn.preprocessing import KBinsDiscretizer

from sklearn.preprocessing import RobustScaler



import warnings

warnings.filterwarnings("ignore")



from tqdm.notebook import tqdm ,tnrange
train_data = pd.read_csv('/kaggle/input/avcrosssell/train.csv')



print(train_data.shape)



train_data.head()
test_data = pd.read_csv('/kaggle/input/avcrosssell/test.csv')



print(test_data.shape)



test_data.head()
x = train_data[~train_data.iloc[:,1:].duplicated(keep = 'first')]



#reemove confusing ids



train_data = train_data[~train_data.id.isin(x[x.iloc[:,1:-1].duplicated(keep = False)].id)]
def nullColumns(train_data):

    

    list_of_nullcolumns =[]

    

    for column in train_data.columns:

        

        total= train_data[column].isna().sum()

        

        try:

            

            if total !=0:

                

                print('Total Na values is {0} for column {1}' .format(total, column))

                

                list_of_nullcolumns.append(column)

        

        except:

            

            print(column,"-----",total)

    

    print('\n')

    

    return list_of_nullcolumns





def percentMissingFeature(data):

    

    data_na = (data.isnull().sum() / len(data)) * 100

    

    data_na = data_na.drop(data_na[data_na == 0].index).sort_values(ascending=False)[:30]

    

    missing_data = pd.DataFrame({'Missing Ratio' :data_na})

    

    return data_na





def plotMissingFeature(data_na):

    

    f, ax = plt.subplots(figsize=(15, 12))

    

    plt.xticks(rotation='90')

    

    if(data_na.empty ==False):

        

        sns.barplot(x=data_na.index, y=data_na)

        

        plt.xlabel('Features', fontsize=15)

        

        plt.ylabel('Percent of missing values', fontsize=15)

        

        plt.title('Percent missing data by feature', fontsize=15)
print('train data')



print(nullColumns(train_data))



print(percentMissingFeature(train_data))



print('\n')



print('test_data')



print(nullColumns(test_data))



print(percentMissingFeature(test_data))
train_data.describe()
response = train_data.loc[:,"Response"].value_counts().rename('Count')

plt.xlabel("Response")

plt.ylabel('Count')

sns.barplot(response.index , response.values).set_title('Response')
response
sns.distplot(train_data['Vintage'])
sns.distplot(train_data['Annual_Premium'])
sns.distplot(test_data['Annual_Premium'])
sns.distplot(train_data['Age'])
sns.distplot(test_data['Age'])
sns.set(style="white", palette="muted", color_codes=True)



f, axes = plt.subplots(2, 1, figsize=(15, 15))



male = train_data[train_data['Gender'] =='Male']["Response"].value_counts().rename('Count')



female = train_data[train_data['Gender'] =='Female']["Response"].value_counts().rename('Count')



sns.barplot(male.index,male,  color="b", ax=axes[0]).set_title('Gender : Male')



sns.barplot(female.index,female,   color="r", ax=axes[1]).set_title('Gender : Female')



plt.setp(axes, yticks = np.arange(0,50000,10000))



for ax in f.axes:

    

    plt.sca(ax)

    

    plt.xticks(rotation=0)



plt.tight_layout()
sns.set(style="white", palette="muted", color_codes=True)



f, axes = plt.subplots(2, 1, figsize=(15, 15))



dl0 = train_data[train_data['Driving_License'] ==0]["Response"].value_counts().rename('Count')



dl1 = train_data[train_data['Driving_License'] ==1]["Response"].value_counts().rename('Count')



sns.barplot(dl0.index,dl0,  color="b", ax=axes[0]).set_title('Driving_License : No')



sns.barplot(dl1.index,dl1,   color="r", ax=axes[1]).set_title('Driving_License : Yes')



plt.setp(axes, yticks = np.arange(0,200000,1000))



for ax in f.axes:

    

    plt.sca(ax)

    

    plt.xticks(rotation=0)



plt.tight_layout()
sns.set(style="white", palette="muted", color_codes=True)



f, axes = plt.subplots(2, 1, figsize=(15, 15))



pi0 = train_data[train_data['Previously_Insured'] ==0]["Response"].value_counts().rename('Count')



pi1 = train_data[train_data['Previously_Insured'] ==1]["Response"].value_counts().rename('Count')



sns.barplot(dl0.index,dl0,  color="b", ax=axes[0]).set_title('Previously_Insured : No')



sns.barplot(dl1.index,dl1,   color="r", ax=axes[1]).set_title('Previously_Insured : Yes')



plt.setp(axes, yticks = np.arange(0,50000,5000))



for ax in f.axes:

    

    plt.sca(ax)

    

    plt.xticks(rotation=0)



plt.tight_layout()
train_data['Policy_Region'] = train_data['Policy_Sales_Channel'].astype(str)+'_'+train_data['Region_Code'].astype(str)



test_data['Policy_Region'] = test_data['Policy_Sales_Channel'].astype(str)+'_'+test_data['Region_Code'].astype(str)



train_data['Vehicle_Age_License'] = train_data['Vehicle_Age'].astype(str)+'_'+train_data['Driving_License'].astype(str)



test_data['Vehicle_Age_License'] = test_data['Vehicle_Age'].astype(str)+'_'+test_data['Driving_License'].astype(str)

cat_features = ['Gender','Driving_License','Region_Code','Previously_Insured',

                'Vehicle_Damage','Policy_Sales_Channel','Policy_Region',

                'Vehicle_Age','Vintage','Annual_Premium','Vehicle_Age_License']



cont_features = ['Age']



label = 'Response'
def encode_cat_cols(train, test, cat_cols): #target



    train_df = train_data.copy()

    

    test_df = test_data.copy()

    

    # Making a dictionary to store all the labelencoders for categroical columns to transform them later.

    

    le_dict = {}



    for col in cat_cols:

        

        if col!= 'Vehicle_Age':

        

            le = LabelEncoder()



            le.fit(train_df[col].unique().tolist() + test_df[col].unique().tolist())



            train_df[col] = le.transform(train_df[[col]])



            test_df[col] = le.transform(test_df[[col]])



            le_dict[col] = le

        

    train_df['Vehicle_Age'] = train_df['Vehicle_Age'].map({'< 1 Year':1,'1-2 Year':2,'> 2 Years':3})

    

    test_df['Vehicle_Age'] = test_df['Vehicle_Age'].map({'< 1 Year':1,'1-2 Year':2,'> 2 Years':3})



    le = LabelEncoder()

    

    train_df[label] = le.fit_transform(train_df[[label]])

    

    le_dict[label] = le

    

    

    return train_df, test_df, le_dict
train_df, test_df, le_dict = encode_cat_cols(train_data,test_data,cat_features)
train_df = train_df[~train_df.Policy_Sales_Channel.isin(list(set(train_df.Policy_Sales_Channel)-set(test_df.Policy_Sales_Channel)))]



#test_df.loc[(test_df.Policy_Sales_Channel.isin(list(set(test_df.Policy_Sales_Channel) - set(train_df.Policy_Sales_Channel)))),'Policy_Sales_Channel'] = -1



test_df.loc[(test_df.Policy_Sales_Channel==137),'Policy_Sales_Channel'] = -1



test_df.loc[(test_df.Policy_Sales_Channel==136),'Policy_Sales_Channel'] = -1
#Used only for XgBoost and LightGBM

#test_df.loc[(test_df.Annual_Premium.isin(list(set(test_df.Annual_Premium) - set(train_df.Annual_Premium)))),'Annual_Premium'] = -1
train_df['train'] = 1



test_df['train'] = 0



combined_data = pd.concat([train_df,test_df],axis =0).reset_index(drop = True).copy()
premium_discretizer = KBinsDiscretizer(n_bins=8, encode='ordinal', strategy='quantile')



combined_data['Premium_Bins'] =premium_discretizer.fit_transform(combined_data['Annual_Premium'].values.reshape(-1,1)).astype(int)



age_discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')



combined_data['Age_Bins'] =age_discretizer.fit_transform(combined_data['Age'].values.reshape(-1,1)).astype(int)
sns.boxplot(combined_data[combined_data['train']==1]['Response'],combined_data[combined_data['train']==1]['Age'])
# Age Bin demarcates two classses better(compare the follwing graph with the above one)

sns.boxplot(combined_data[combined_data['train']==1]['Response'],combined_data[combined_data['train']==1]['Age_Bins'])
sns.boxplot(combined_data[combined_data['train']==1]['Response'],combined_data[combined_data['train']==1]['Annual_Premium'])
# the same can be seen after binning annual premium

sns.boxplot(combined_data[combined_data['train']==1]['Response'],combined_data[combined_data['train']==1]['Premium_Bins'])
gender_counts = combined_data['Gender'].value_counts().to_dict()



combined_data['Gender_Counts'] = combined_data['Gender'].map(gender_counts)



region_counts = combined_data['Region_Code'].value_counts().to_dict()



combined_data['Region_counts'] = combined_data['Region_Code'].map(region_counts)



vehicle_age_counts = combined_data['Vehicle_Age'].value_counts().to_dict()



combined_data['Vehicle_Age_Counts'] = combined_data['Vehicle_Age'].map(vehicle_age_counts)
combined_data['Nunq_Policy_Per_Region'] = combined_data.groupby('Region_Code')['Policy_Sales_Channel'].transform('nunique') 



combined_data['SDev_Annual_Premium_Per_Region_Code_int'] = combined_data.groupby('Region_Code')['Annual_Premium'].transform('std').fillna(-1) 



combined_data['Nunq_Region_Per_Premium'] = combined_data.groupby('Annual_Premium')['Region_Code'].transform('nunique')



# 1230.45 can be split into “1230” and “45”. LGBM cannot see these pieces on its own, you need to split them.

combined_data['SDev_Annual_Premium_Per_Region_Code_dec'] = combined_data['SDev_Annual_Premium_Per_Region_Code_int'] %1



combined_data['SDev_Annual_Premium_Per_Region_Code_int'] =combined_data['SDev_Annual_Premium_Per_Region_Code_int'].astype(int)





combined_data['Avg_Policy_Region_Age'] = combined_data.groupby(['Policy_Region'])['Age'].transform('mean')



combined_data['Avg_Policy_Region_Premium'] = combined_data.groupby(['Policy_Region'])['Annual_Premium'].transform('mean') 



combined_data['Avg_Region_Premium'] = combined_data.groupby(['Region_Code'])['Annual_Premium'].transform('mean')



combined_data['Nunq_Premium_Region'] = combined_data.groupby(['Annual_Premium'])['Region_Code'].transform('nunique')
#combined_data['f4'] = combined_data.groupby(['Vehicle_Age'])['Annual_Premium'].transform('max')-combined_data.groupby(['Vehicle_Age'])['Annual_Premium'].transform('min')

#combined_data['f1'] =combined_data['Annual_Premium'] /(combined_data['Vintage']/365)





#combined_data['f2'] = combined_data.groupby(['Premium_Bins'])['Annual_Premium'].transform('max')-combined_data.groupby(['Premium_Bins'])['Annual_Premium'].transform('min')

#combined_data['f2'] = combined_data.groupby(['Annual_Premium'])['Vintage'].transform('nunique')
train_df = combined_data[combined_data['train']==1]



test_df = combined_data[combined_data['train']==0]
# Remove duplicate rows---> More trustworthy CV

cols = ['Gender', 'Age', 'Driving_License', 'Region_Code',

       'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage',

        'Annual_Premium','Policy_Sales_Channel', 'Vintage']



train_df = train_df[~train_df.loc[:,cols].duplicated(keep = 'first')].reset_index(drop=True)
target = train_df['Response']



train_df = train_df.drop(columns =['train','id','Response'])



test_df = test_df.drop(columns=['train','id','Response'])
test_size = 0.34



train_df
def feature_importance(model, X_train):



    fI = model.feature_importances_

    

    print(fI)

    

    names = X_train.columns.values

    

    ticks = [i for i in range(len(names))]

    

    plt.bar(ticks, fI)

    

    plt.xticks(ticks, names,rotation = 90)

    

    plt.show()
%%time

##LightGBM



cat_features = ['Driving_License','Gender','Region_Code','Previously_Insured','Vehicle_Damage',

                'Policy_Sales_Channel','Policy_Region','Vehicle_Age','Vintage',

                'Annual_Premium','Vehicle_Age_License','Premium_Bins']



cont_features = ['Age','Age_Bins']



probs_lgb = np.zeros(shape=(len(test_df),))



scores = []



avg_loss = []



seeds = [1]



for seed in tnrange(len(seeds)):

    

    print(' ')

    

    print('#'*100)

    

    print('Seed',seeds[seed])



    X_train_cv,y_train_cv = train_df.copy(), target.copy()



    sssf = StratifiedShuffleSplit(n_splits=5, test_size = test_size ,random_state=seed)

    

    for i, (idxT, idxV) in enumerate(sssf.split(X_train_cv, y_train_cv)):



        print('Fold',i)



        print(' rows of train =',len(idxT),'rows of holdout =',len(idxV))



        clf = lgb.LGBMClassifier(boosting_type='gbdt',

                                 n_estimators=10000,

                                 max_depth=10,

                                 learning_rate=0.02,

                                 subsample=0.9,

                                 colsample_bytree=0.4,

                                 objective ='binary',

                                 random_state = 1,

                                 importance_type='gain',

                                 reg_alpha=2,

                                 reg_lambda=2

                                 #cat_features=cat_features

                                )        

        

        h = clf.fit(X_train_cv.iloc[idxT], y_train_cv.iloc[idxT], 

                    eval_set=[(X_train_cv.iloc[idxV],y_train_cv.iloc[idxV])],

                    verbose=100,eval_metric=['binary_logloss','auc'],

                    early_stopping_rounds=100)

        probs_oof = clf.predict_proba(X_train_cv.iloc[idxV])[:,1]

        

        probs_lgb +=clf.predict_proba(test_df)[:,1]

        

        roc = roc_auc_score(y_train_cv.iloc[idxV],probs_oof)



        scores.append(roc)



        avg_loss.append(clf.best_score_['valid_0']['binary_logloss'])



        print ('LGB Val OOF AUC=',roc)



        print('#'*100)



        if i==0:

            feature_importance(clf,X_train_cv)



print("Log Loss Stats {0:.8f},{1:.8f}".format(np.array(avg_loss).mean(), np.array(avg_loss).std()))



print('%.8f (%.8f)' % (np.array(scores).mean(), np.array(scores).std()))
%%time



##XGBM



probs_xgb = np.zeros(shape=(len(test_df),))



scores = []



avg_loss = []



X_train_cv,y_train_cv = train_df.copy(), target.copy()



seeds = [1]



for seed in tnrange(len(seeds)):

    

    print(' ')

    

    print('#'*100)

    

    print('Seed',seeds[seed])

    

    sssf = StratifiedShuffleSplit(n_splits=5, test_size = test_size ,random_state=seed)

    

    for i, (idxT, idxV) in enumerate(sssf.split(X_train_cv, y_train_cv)):



        print('Fold',i)



        print(' rows of train =',len(idxT),'rows of holdout =',len(idxV))



        clf = xgb.XGBClassifier(n_estimators=1000,

                                max_depth=6,

                                learning_rate=0.04,

                                subsample=0.9,

                                colsample_bytree=0.35,

                                objective = 'binary:logistic',

                                random_state = 1

                               )        





        h = clf.fit(X_train_cv.iloc[idxT], y_train_cv.iloc[idxT], 

                    eval_set=[(X_train_cv.iloc[idxV],y_train_cv.iloc[idxV])],

                    verbose=100,eval_metric=['auc','logloss'],

                    early_stopping_rounds=50)

        

        probs_oof = clf.predict_proba(X_train_cv.iloc[idxV])[:,1]

        

        probs_xgb +=clf.predict_proba(test_df)[:,1]



        roc = roc_auc_score(y_train_cv.iloc[idxV],probs_oof)



        scores.append(roc)

        

        avg_loss.append(clf.best_score)



        print ('XGB Val OOF AUC=',roc)



        print('#'*100)



        if i==0:

            

            feature_importance(clf,X_train_cv)

            

print("Log Loss Stats {0:.5f},{1:.5f}".format(np.array(avg_loss).mean(), np.array(avg_loss).std()))



print('%.6f (%.6f)' % (np.array(scores).mean(), np.array(scores).std()))
%%time



##CatBoost



cat_features = ['Driving_License','Gender','Region_Code','Previously_Insured','Vehicle_Damage',

                'Policy_Sales_Channel','Policy_Region','Vehicle_Age','Vintage','Annual_Premium',

                'Vehicle_Age_License','Premium_Bins']



cont_features = ['Age','Age_Bins']



probs_cb = np.zeros(shape=(len(test_df),))



scores = []



avg_loss = []



X_train_cv,y_train_cv = train_df.copy(), target.copy()



seeds = [1]



for seed in tnrange(len(seeds)):

    

    print(' ')

    

    print('#'*100)

    

    print('Seed',seeds[seed])

    

    sssf = StratifiedShuffleSplit(n_splits=5, test_size = test_size ,random_state=seed)

    

    for i, (idxT, idxV) in enumerate(sssf.split(X_train_cv, y_train_cv)):



        print('Fold',i)



        print(' rows of train =',len(idxT),'rows of holdout =',len(idxV))



        clf = CatBoostClassifier(iterations=10000,

                                learning_rate=0.02,

                                random_strength=0.1,

                                depth=8,

                                loss_function='Logloss',

                                eval_metric='Logloss',

                                leaf_estimation_method='Newton',

                                random_state = 1,

                                cat_features =cat_features,

                                subsample = 0.9,

                                rsm = 0.8

                                )    



        h = clf.fit(X_train_cv.iloc[idxT], y_train_cv.iloc[idxT],

                    eval_set=[(X_train_cv.iloc[idxV],y_train_cv.iloc[idxV])],

                   early_stopping_rounds=50,verbose = 100)



        probs_oof = clf.predict_proba(X_train_cv.iloc[idxV])[:,1]

        

        probs_cb +=clf.predict_proba(test_df)[:,1]

        

        roc = roc_auc_score(y_train_cv.iloc[idxV],probs_oof)



        scores.append(roc)



        print ('CatBoost Val OOF AUC=',roc)



        avg_loss.append(clf.best_score_['validation']['Logloss'])



        if i==0:

            

            feature_importance(clf,X_train_cv)



        print('#'*100)



print("Log Loss Stats {0:.8f},{1:.8f}".format(np.array(avg_loss).mean(), np.array(avg_loss).std()))



print('%.8f (%.8f)' % (np.array(scores).mean(), np.array(scores).std()))
#!pip uninstall -q fastai -y



#!pip install -q /kaggle/input/fast-v2-offline/dataclasses-0.6-py3-none-any.whl



#!pip install -q /kaggle/input/fast-v2-offline/torch-1.6.0-cp37-cp37m-manylinux1_x86_64.whl



#!pip install -q /kaggle/input/fast-v2-offline/torchvision-0.7.0-cp37-cp37m-manylinux1_x86_64.whl



#!pip install -q /kaggle/input/fast-v2-offline/fastcore-1.0.1-py3-none-any.whl



#!pip install -q /kaggle/input/fast-v2-offline/fastai-2.0.8-py3-none-any.whl



#import fastai



#print('fastai version :', fastai.__version__)



#from fastai import *



#from fastai.layers import *



#from fastai.tabular.all import *



#import torch.nn.functional as F



#import torch



#import torch.nn as nn



#import torch.optim as optim



#from torch.utils.data import Dataset, DataLoader
"""

def categorical_features(df,cat_features,uuid_col):

    

    cat_features_subdf = []

    

    for col in df.columns:

        

        if col in cat_features:

            

            cat_features_subdf.append(col)

            

    return cat_features_subdf





def continuous_features(df,uuid_col,cat_features_subdf):

    

    cont_features_subdf = []

    

    for col in df.columns:

        

        if col not in cat_features_subdf:

            

            cont_features_subdf.append(col)

            

    return cont_features_subdf

    

"""
""""

class FocalLoss(nn.Module):

    

    def __init__(self, alpha=4, gamma=2, logits=False, reduction = 'mean'):

        

        super(FocalLoss, self).__init__()

       

        self.alpha = alpha

        

        self.gamma = gamma

        

        self.logits = logits

        

        self.reduction = reduction



    def forward(self, inputs, targets):



        targets =targets.type_as(inputs)

        

        if self.logits:

            

            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        

        else:

            

            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')

        

        pt = torch.exp(-BCE_loss)

        

        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss



        if self.reduction is None:

            

            return F_loss

        

        else:

            

            return torch.mean(F_loss)

            

"""
"""

%%time



scores = []



avg_loss = []



cat_features = ['Gender','Driving_License','Region_Code','Previously_Insured','Vehicle_Damage','Policy_Sales_Channel','Policy_Region','Vehicle_Age','Vintage']



cont_features = ['Age','Annual_Premium',]



#custom sklearn metric

roc =  skm_to_fastai(roc_auc_score,axis=0)



#np.random.seed(10)



probs = np.zeros(shape=(len(test_df)))



#seeds = list(np.random.randint(5,500,1))



seeds = [1]



for seed in seeds:

    

    print(' ')



    print('#'*100)



    print('Seed',seed)



    df = train_df.copy().sample(frac = 1.0,axis =1,random_state = seed)



    cat_features_subdf= categorical_features(df,cat_features,'')



    cont_features_subdf = continuous_features(df,'',cat_features_subdf)



    df = pd.concat([df, target], axis=1)



    sssf = StratifiedShuffleSplit(n_splits=1, test_size = 0.35 ,random_state=seed)



    procs = [FillMissing,Categorify, Normalize]



    for i, (idxT, idxV) in enumerate(sssf.split(train_df, target)):





        data = TabularDataLoaders.from_df(df, procs=procs,

                                              cont_names=cont_features_subdf, cat_names=cat_features_subdf,

                                              y_names='Response', valid_idx=idxV, bs=256, val_bs=256)





        learn = tabular_learner(data, y_range=(0,1), layers=[256,128,64],loss_func=BCELossFlat(),metrics= [roc])



        #learn.lr_find()

        

        learn.fit_one_cycle(5, 1e-3)

            

        val_loss, val_roc = learn.validate()

            

        scores.append(val_roc)



        avg_loss.append(val_loss)

        

        test_dl = learn.dls.test_dl(test_df)

        

        sub = learn.get_preds(dl=test_dl)

        

        probs += sub[0].numpy().ravel()

            

        print('#'*100)



print("Log Loss Stats {0:.5f},{1:.5f}".format(np.array(avg_loss).mean(), np.array(avg_loss).std()))



print('%.6f (%.6f)' % (np.array(scores).mean(), np.array(scores).std()))



"""
p1 =probs_lgb/5



p2 = probs_cb/5



p3 = probs_xgb/5
submission = pd.read_csv('../input/avcrosssell/sample_submission.csv')



submission['Response'] =  0.7*p2+0.3*p3
submission.to_csv('submission.csv',index =False)



submission.head()
np.save('lgb.npy',p1)

np.save('cb.npy',p2)

np.save('xgb.npy',p3)