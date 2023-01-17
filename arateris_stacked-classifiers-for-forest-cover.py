# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from time import time



from collections import Counter

from itertools import combinations



from sklearn.model_selection import cross_val_score,cross_validate, train_test_split, GridSearchCV, StratifiedKFold

from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.metrics import accuracy_score, confusion_matrix, balanced_accuracy_score, make_scorer

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, RandomForestRegressor

from sklearn.linear_model import LogisticRegression



from mlxtend.classifier import StackingCVClassifier, StackingClassifier

from lightgbm import LGBMClassifier

from xgboost import XGBClassifier, plot_importance

from catboost import CatBoostClassifier



from tqdm import tqdm





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/learn-together/train.csv", index_col='Id')

test = pd.read_csv("../input/learn-together/test.csv", index_col='Id')



y = train['Cover_Type'] # this is the target

X = train.drop('Cover_Type', axis = 1)

X_test = test.copy()



print('Train set shape : ', X.shape)

print('Test set shape : ', X_test.shape)



X.rename({'Horizontal_Distance_To_Roadways':'HDR',

              'Horizontal_Distance_To_Hydrology':'HDH',

              'Horizontal_Distance_To_Fire_Points':'HDF',

              'Vertical_Distance_To_Hydrology':'VDH'}, axis="columns", inplace=True)

X_test.rename({'Horizontal_Distance_To_Roadways':'HDR',

              'Horizontal_Distance_To_Hydrology':'HDH',

              'Horizontal_Distance_To_Fire_Points':'HDF',

              'Vertical_Distance_To_Hydrology':'VDH'}, axis="columns", inplace=True)





columns = X.columns



X.head()



# Save test predictions to file

def to_submission(preds, file_name):

    output = pd.DataFrame({'Id': X_test.index,

                           'Cover_Type': preds})

    output.to_csv(file_name+'.csv', index=False)

count = { 1: 0.37062,

 2: 0.49657,

 3: 0.05947,

 4: 0.00106,

 5: 0.01287, 

 6: 0.02698, 

 7: 0.03238} 

weight = [count[x]/(sum(count.values())) for x in range(1,7+1)]

class_weight_lgbm = {i: v for i, v in enumerate(weight)}

def imbalanced_accuracy_score(y_true, y_pred):

    return accuracy_score(y_true, y_pred, sample_weight=[weight[x] for x in y_true-1])



imbalanced_accuracy_scorer = make_scorer(imbalanced_accuracy_score, greater_is_better=True)



def imbalanced_cross_validate(clf, X, y, cfg_args={}, fit_params={}, cv=5):

    return cross_validate(clf, X, y, scoring= imbalanced_accuracy_scorer, cv=cv, n_jobs=-1, fit_params=fit_params )
X.head()
X_test.head()
print('Missing Label? ', y.isnull().any())

print('Missing train data? ', X.isnull().any().any())

print('Missing test data? ', X_test.isnull().any().any())
print (X.dtypes.value_counts())

print (X_test.dtypes.value_counts())
#transform Soil_Type into categorial

def categorify(df, col_string_search, remove_original=False):

    for key_str in col_string_search:

        new_col_name = key_str+'_cat'

        df[new_col_name]=0

        for col in columns:

            if ~str(col).find(key_str):

                df[new_col_name]= df[new_col_name]+int(str(col).lstrip(key_str))*df[col]

                if remove_original:

                    df.drop(col, axis=1, inplace=True)

#         df[new_col_name] = df[new_col_name].astype('category')

    return df
cols_to_categorify = ['Soil_Type', 'Wilderness_Area']

X = categorify(X, cols_to_categorify, remove_original=True)

X_test = categorify(X_test, cols_to_categorify, remove_original=True)

X_test.head()
X.describe()
for col in X.columns:

    plt.figure(figsize=(15,5))

    sns.distplot(X[col])

    plt.show()
print(X.Hillshade_3pm[(X.Hillshade_3pm<130).to_numpy() &  (X.Hillshade_3pm>120).to_numpy()].value_counts())

print((X.Hillshade_3pm==0).sum())

print((X_test.Hillshade_3pm==0).sum())

corr = X[X.Hillshade_3pm!=0].corr()

plt.figure(figsize=(12,12))

sns.heatmap(corr,annot=True)
#replacing the zeros for better guess, mainly to avoid zeros in the feature engineering and fake outliers. 

num_train=len(X)

cols_for_HS = ['Aspect','Slope', 'Hillshade_9am','Hillshade_Noon']

all_data = X.append(X_test)

HS_zero = all_data[all_data.Hillshade_3pm==0]

HS_zero.shape



HS_train = all_data[all_data.Hillshade_3pm!=0]

# res = cross_val_score(RandomForestRegressor(n_estimators=100), HS_train.drop('Hillshade_3pm',axis=1), HS_train.Hillshade_3pm, n_jobs=-1, verbose=True)

# print(res) #[0.9996774  0.99989463 0.9999186 ]

##actually, the CV is so close to zero there is actually no new information here..keeping it for simplicity

rf_hs = RandomForestRegressor(n_estimators=100).fit(HS_train[cols_for_HS], HS_train.Hillshade_3pm)

out = rf_hs.predict(HS_zero[cols_for_HS]).astype(int)

all_data.loc[HS_zero.index,'Hillshade_3pm'] = out

X= all_data[:num_train]

X_test= all_data[num_train:]
#X.nunique()


def quick_fe(df, cols, operations, max_combination=2):

    

    if max_combination>=2:

        for col1, col2 in combinations(cols, 2):

            for ope in operations:

                if ope=='add': df[col1 + "_add_" + col2] = df[col1]+df[col2]

                elif ope=='minus': df[col1 + "_minus_" + col2] = df[col1]-df[col2]

                elif ope=='time': df[col1 + "_time_" + col2] = df[col1]*df[col2]

    if max_combination>=3:

        for col1, col2, col3 in combinations(cols, 3):

            for ope in operations:

                if ope=='add': df[col1 + "_add_" + col2 + "_add_" + col3] = df[col1]+df[col2]+df[col3]

                elif ope=='time': df[col1 + "_time_" + col2+ "_time_" + col3] = df[col1]*df[col2]*df[col3]

    return df







X.head()




def feature_eng(dataset):

    # https://www.kaggle.com/nadare/eda-feature-engineering-and-modeling-4th-359#nadare's-kernel

    #https://www.kaggle.com/lukeimurfather/adversarial-validation-train-vs-test-distribution

    #https://www.kaggle.com/evimarp/top-6-roosevelt-national-forest-competition

    

    cols_to_combine = ['HDH', 'HDF', 'HDR']

    dataset = quick_fe(dataset, cols_to_combine, ['add','time','minus'], max_combination=3)

    

    dataset['Ele_vert'] = dataset.Elevation-dataset.VDH



    dataset['Distance_hyd'] = (dataset['HDH']**2+dataset['VDH']**2)**0.5

    

    dataset['Slope_hyd'] = np.arctan(dataset['VDH']/(dataset['HDH']+0.001))

    dataset.Slope_hyd=dataset.Slope_hyd.map(lambda x: 0 if np.isinf(x) else x) # remove infinite value if any

    

    dataset['Sin_Slope_hyd'] = np.sin(np.radians(dataset['Slope_hyd']))

    dataset['Cos_Slope_hyd'] = np.cos(np.radians(dataset['Slope_hyd']))



    dataset['Mean_Distance']=(dataset.HDF + 

                               dataset.Distance_hyd + 

                               dataset.HDR) / 3 



    # Hillshade

    hillshade_col = ["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"]

    dataset = quick_fe(dataset,hillshade_col, ['add','minus'], max_combination=3)



    dataset["Hillshade_std"] = dataset[hillshade_col].std(axis=1)

    dataset["Hillshade_max"] = dataset[hillshade_col].max(axis=1)

    dataset["Hillshade_min"] = dataset[hillshade_col].min(axis=1)



#     dataset['Is_Overwater'] = dataset['VDH'] > 0   #This shouldn't be very useful for Trees?

            

#     dataset['Elevation_log'] = np.log1p(dataset['Elevation']) #This shouldn't be very useful for Trees?

    

    dataset['Aspect'] = dataset['Aspect'].astype(int) % 360

    

    dataset['Sin_Aspect'] = np.sin(np.radians(dataset['Aspect'])) # not important feature at all

    dataset['Cos_Aspect'] = np.cos(np.radians(dataset['Aspect']))

    

    dataset['Sin_Slope'] = np.sin(np.radians(dataset['Slope'])) # not important feature at all

    dataset['Cos_Slope'] = np.cos(np.radians(dataset['Slope']))

    

      

    dataset['Elevation_Adj_distanceH'] = dataset['Elevation'] - 0.25*dataset['Distance_hyd']

    dataset['Elevation_Adj_distanceV'] = dataset['Elevation'] - 0.19*dataset['HDH']



    from bisect import bisect

    cardinals = [i for i in range(45, 361, 90)]

    points = ['N', 'E', 'S', 'W']

    dataset['Cardinal'] = dataset.Aspect.apply(lambda x: points[bisect(cardinals, x) % 4])

    dataset.loc[:,'North']= dataset['Cardinal']=='N'

    dataset.loc[:,'East']= dataset['Cardinal']=='E'

    dataset.loc[:,'West']= dataset['Cardinal']=='W'

    dataset.loc[:,'South']= dataset['Cardinal']=='S'

#     dataset.drop('Cardinal', axis=1, inplace=True)

    

    # extremely stony = 4, very stony = 3, stony = 2, rubbly = 1, None = 0

    Soil_to_stony = [4, 3, 1, 1, 1, 2, 0, 0, 3, 1,

                1, 2, 1, 0, 0, 0, 0, 0, 0, 0,

                0, 4, 4, 4, 4, 4, 3, 4, 4, 4, 

                4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]

    dataset['Stonyness'] = [Soil_to_stony[x] for x in (dataset['Soil_Type_cat'].astype(int)-1)]

    dataset.loc[:,'Extremely_Stony']= dataset['Stonyness']==4

    dataset.loc[:,'Very_Stony']= dataset['Stonyness']==3

    dataset.loc[:,'Stony']= dataset['Stonyness']==2

    dataset.loc[:,'Rubbly']= dataset['Stonyness']==1

    dataset.loc[:,'Stony_NA']= dataset['Stonyness']==0

#     dataset.drop('stonyness',axis=1,inplace=True)



    

    return dataset



X = feature_eng(X)

X_test = feature_eng(X_test)

columns = X.columns
# Frequency encoding

#https://www.kaggle.com/c/ieee-fraud-detection/discussion/108575#latest-628340

def freq_encoding(df_train, df_test, cols_to_encode):

    df = pd.concat([df_train[cols_to_encode], df_test[cols_to_encode]],axis=0)

    for col in cols_to_encode:

        new_name = col+'_counts'

        temp = df[col].value_counts().to_dict()

        df[new_name] = df[col].map(temp)

        df[new_name] = df[new_name].astype('int32')

        df_train[new_name] = df.loc[:len(df_train),new_name]

        df_test[new_name] = df.loc[len(df_train):,new_name]

    return df_train, df_test
selected_cols = ['Soil_Type_cat', 'Wilderness_Area_cat', 'Stonyness', 'Cardinal']

X, X_test = freq_encoding(X, X_test, selected_cols)



X.drop('Stonyness',axis=1,inplace=True)

X.drop('Cardinal', axis=1, inplace=True)

X_test.drop('Stonyness',axis=1,inplace=True)

X_test.drop('Cardinal', axis=1, inplace=True)

droping_list = []# [col for col in X.columns if ~str(col).find('Soil_Type')]



X.drop(droping_list, axis=1, inplace = True)

X_test.drop(droping_list, axis=1, inplace = True)



columns = X.columns

X_test.describe()
num_train = X.shape[0]

all_data = pd.concat([X, X_test])
# Add PCA features

from sklearn.decomposition import PCA



t = time()



pca = PCA(n_components=0.95).fit(StandardScaler().fit_transform(all_data))

trans = pca.transform(all_data)

print('duration: '+ str(time()-t))

print(trans.shape)
# # https://www.kaggle.com/stevegreenau/stacking-multiple-classifiers-clustering

from sklearn.mixture import GaussianMixture



t = time()

gmix = GaussianMixture(n_components=10) 

gaussian = gmix.fit_predict(StandardScaler().fit_transform(all_data))



print('duration: '+ str(time()-t))
X['Test_Cluster'] = gaussian[:num_train]  #  Do we need to OHE this ? 

X_test['Test_Cluster'] = gaussian[num_train:]#  Do we need to OHE this ? 

for i in range(trans.shape[1]):

    col_name= 'pca'+str(i+1)

    X[col_name] = trans[:num_train, i]

    X_test[col_name] = trans[num_train:, i]

def mem_reduce(df):

    for col in df.columns:

        if df[col].dtype=='float64': 

            df[col] = df[col].astype('float32')

        if df[col].dtype=='int64': 

            if df[col].max()<128: df[col] = df[col].astype('int8')

            elif df[col].max()<32768: df[col] = df[col].astype('int16')

            else: df[col].astype('int32')

    return df



# X= mem_reduce(X)

# X_test=mem_reduce(X_test)
X.describe()
#prepare df to store pred proba

Id_train=train.index

Id_test=test.index



x_train_L2=pd.DataFrame(Id_train)

x_test_L2=pd.DataFrame(Id_test)

def L1_Training(clf, clf_name, cv=5, early_stop=False):

    scores = []

    clf_cul=[str(clf_name)+str(i+1) for i in range(7)]

    for i in clf_cul:

        x_train_L2.loc[:, i]=0

        x_test_L2.loc[:, i]=0



    clf_proba = np.zeros((X_test.shape[0], 7))

    for train, val in tqdm(StratifiedKFold(n_splits=cv, shuffle=True, random_state=9999).split(X, y)): 

        X_train_loc = X.iloc[train,:]

        X_val_loc = X.iloc[val,:]

        y_train_loc = y.iloc[train]

        y_val_loc = y.iloc[val]

        if early_stop:

            # fit the model  ##Do we need to reset the model in between loops??

            clf.fit(X_train_loc, y_train_loc, 

                verbose=False,

                eval_set=[(X_train_loc, y_train_loc), (X_val_loc, y_val_loc)], 

                early_stopping_rounds=50)

            # use this fitted model to predict Test set.

            clf_pred_proba_test = clf.predict_proba(X_test)

            x_test_L2.loc[:, clf_cul] +=  clf_pred_proba_test/ cv  #average over the CV rounds

        else :

            # when no early stoping the prediction of the Test set will be done once for all after (better use the full training set)

            clf.fit(X_train_loc, y_train_loc)

            

        #checking validation

        clf_pred_proba_val = clf.predict_proba(X_val_loc)

        x_train_L2.loc[val, clf_cul]= clf_pred_proba_val

        y_pred = clf.predict(X_val_loc)

        scores.append(imbalanced_accuracy_score(y_pred,y_val_loc))

        

    if ~early_stop:

        #retrain on full data

        clf.fit(X,y)

        clf_pred_proba_test = clf.predict_proba(X_test)

        x_test_L2.loc[:, clf_cul] = clf_pred_proba_test

        

    clf_pred_test = x_test_L2.loc[:,clf_cul].to_numpy().argmax(axis=1)+1

    return scores, clf_pred_test

xgb= XGBClassifier( n_estimator= 500, 

                    learning_rate= 0.1, 

                    max_depth= 50,  

                    objective= 'binary:logistic', #this outputs probability,not one/zero. should we use binary:hinge? is it better for the learning phase?

                    random_state= 2019,

                    sample_weight=count,

                    n_jobs=-1)

lgbc= LGBMClassifier(n_estimators=500,  

                     learning_rate= 0.1,

                     objective= 'multiclass', 

                     num_class=7,

                     random_state= 2019,

                     n_jobs=-1,

                     class_weight=class_weight_lgbm)

rf = RandomForestClassifier(n_estimators = 1000, 

                            max_features = 0.3, 

                            max_depth = 100, 

                            min_samples_split = 2, 

                            min_samples_leaf = 1,

                            bootstrap = False,

                            random_state=2019,

                            class_weight=count)

xtc= ExtraTreesClassifier(n_estimators = 750, 

                            max_features = 0.3, 

                            max_depth = None, 

                            min_samples_split = 2, 

                            min_samples_leaf = 1,

                            bootstrap = False,

                            random_state=2019, 

                            class_weight=count)

# cat= CatBoostClassifier(n_estimators = 1000,

#                         cat_features= ['Soil_Type_cat', 'Wilderness_Area_cat', 'Stonyness', 'Cardinal'],

#                         learning_rate= 0.1, 

#                         random_state=2019,

# #                         class_weights=weight,# ?? 

#                         verbose = 50) 

lr= LogisticRegression(max_iter=1000,

                       n_jobs=-1,

                       solver= 'lbfgs',

                       multi_class = 'multinomial',

                       class_weight=count)
# cat_score, cat_preds =  L1_Training(cat, 'cat', cv=8, early_stop=True) 

# to_submission(cat_preds, 'cat_Kfold_sub')

# print(np.mean(cat_score))


# xgb_score, xgb_preds = L1_Training(xgb, 'xgb', cv=8, early_stop=True) 

# to_submission(xgb_preds, 'xgb_Kfold_sub')

# print(np.mean(xgb_score))

# #v11  LB:0.76478
# xgb.fit(X,y)

# xgb_test_pred_full = xgb.predict(X_test)

# to_submission(xgb_test_pred_full, 'xgb_full_sub')



# lgbc_score, lgb_preds = L1_Training(lgbc, 'lgbc', cv=8, early_stop=True) 

# to_submission(lgb_preds, 'lgb_Kfold_sub')

# print(np.mean(lgbc_score))

# # v11 LB : 0.79057 (with GaussianMixture 10, without PCA)
# lgbc.fit(X,y)

# lgbc_test_pred_full = lgbc.predict(X_test)

# to_submission(lgbc_test_pred_full, 'lgbc_full_sub')

# # v11 LB : 0.79057 (with GaussianMixture 10, without PCA)


# rf_score, rf_preds = L1_Training(rf, 'rf', cv=8, early_stop=False) 

# to_submission(rf_preds, 'rf_sub')

# print(np.mean(rf_score))

## v12 LB:0.76981 (with GaussianMixture 10, without PCA)


# xtc_score, xtc_preds = L1_Training(xtc, 'xtc', cv=8, early_stop=False)

# to_submission(xtc_preds, 'xtc_sub')

# print(np.mean(xtc_score))

# #v12 LB:0.78579 (with GaussianMixture 10, without PCA)


# lr_score, lr_preds = L1_Training(lr, 'lr', cv=8, early_stop=False) 

# to_submission(lr_preds, 'lr_sub')

# print(np.mean(lr_score))
# preds_test = l2_lgbc.predict(x_test_L2).argmax(axis=1)+1

# preds_sum = np.concatenate([xtc_preds, lgb_preds, xgb_preds, rf_preds])

# to_submission(preds_sum, 'sum_preds_sub')
#redefine clf because the weight format doesnt match the StackingClassifier..

xgb= XGBClassifier( n_estimator= 500, 

                    learning_rate= 0.1, 

                    max_depth= 50,  

                    objective= 'binary:logistic', #this outputs probability,not one/zero. should we use binary:hinge? is it better for the learning phase?

                    random_state= 2019,

                    n_jobs=-1)

lgbc= LGBMClassifier(n_estimators=500,  

                     learning_rate= 0.1,

                     objective= 'multiclass', 

                     num_class=7,

                     random_state= 2019,

                     n_jobs=-1)

lgbc2= LGBMClassifier(n_estimators=500,  

                     learning_rate= 0.1,

                     objective= 'multiclass', 

                     num_class=7,

                     random_state= 888,

                     n_jobs=-1)

lgbc3= LGBMClassifier(n_estimators=750,  

                     learning_rate= 0.1,

                     objective= 'multiclass', 

                     num_class=7,

                     random_state= 666,

                     n_jobs=-1)

rf = RandomForestClassifier(n_estimators = 1000, 

                            max_features = 0.3, 

                            max_depth = 100, 

                            min_samples_split = 2, 

                            min_samples_leaf = 1,

                            bootstrap = False,

                            random_state=2019)

xtc= ExtraTreesClassifier(n_estimators = 750, 

                            max_features = 0.3, 

                            max_depth = None, 

                            min_samples_split = 2, 

                            min_samples_leaf = 1,

                            bootstrap = False,

                            random_state=2019)

xtc2= ExtraTreesClassifier(n_estimators = 550, 

                            max_features = 0.3, 

                            max_depth = None, 

                            min_samples_split = 2, 

                            min_samples_leaf = 1,

                            bootstrap = False,

                            random_state=888)

xtc3= ExtraTreesClassifier(n_estimators = 750, 

                            max_features = 0.3, 

                            max_depth = None, 

                            min_samples_split = 2, 

                            min_samples_leaf = 1,

                            bootstrap = False,

                            random_state=777)

# lr= LogisticRegression(max_iter=1000,

#                        n_jobs=-1)





### ugly double stack.. just to check improvements of the hillshade fix.. to be removed...

stack = StackingCVClassifier(classifiers=[lgbc,xtc,rf, xgb],#[lgbc, lgbc2, lgbc3, xtc, xtc2, xtc3, rf, xgb],

                             meta_classifier=lgbc,

                             use_probas=True,

                             use_features_in_secondary=True,

                             verbose=1,

                             random_state=888,

                             n_jobs=-1,

                             cv =8)



# stack_score, stack_preds = L1_Training(stack, 'stack', cv=8, early_stop=False) 

# # to_submission(lr_preds, 'lr_sub')

# print(np.mean(stack_score))



stack.fit(X,y)





stack2 = StackingCVClassifier(classifiers=[lgbc, lgbc2, lgbc3, xtc, xtc2, xtc3, rf, xgb],

                             meta_classifier=lgbc,

                             use_probas=True,

                             use_features_in_secondary=True,

                             verbose=1,

                             random_state=888,

                             n_jobs=-1,

                             cv =8)



# stack_score, stack_preds = L1_Training(stack, 'stack', cv=8, early_stop=False) 

# # to_submission(lr_preds, 'lr_sub')

# print(np.mean(stack_score))



stack2.fit(X,y)

stack_preds = stack.predict(X_test.to_numpy())

to_submission(stack_preds, 'stack_preds_sub')
stack2_preds = stack2.predict(X_test.to_numpy())

to_submission(stack2_preds, 'stack2_preds_sub')