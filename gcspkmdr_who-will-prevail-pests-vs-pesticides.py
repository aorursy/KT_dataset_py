import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt



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



from sklearn.model_selection import train_test_split, GridSearchCV ,RepeatedStratifiedKFold, cross_val_score , StratifiedKFold ,GroupKFold

from sklearn.preprocessing import StandardScaler,MinMaxScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier , ExtraTreesClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score,accuracy_score ,confusion_matrix

from sklearn.preprocessing import PolynomialFeatures

import warnings

warnings.filterwarnings("ignore")



from imblearn.under_sampling import TomekLinks

from imblearn.over_sampling import SMOTE

from imblearn.under_sampling import RandomUnderSampler

from imblearn.over_sampling import RandomOverSampler

from imblearn.combine import SMOTETomek

from imblearn.under_sampling import ClusterCentroids , NearMiss
train_data = pd.read_csv('/kaggle/input/pesticidesagriculturedataset/Agricultre/train.csv')

test_data = pd.read_csv('/kaggle/input/pesticidesagriculturedataset/Agricultre/test.csv')
print(train_data.shape)

train_data.head(20)
train_data_ids = train_data['ID']
print(test_data.shape)

test_data.head()
test_data_ids = test_data['ID']
train_data.Number_Weeks_Used.fillna(-1,inplace = True)

test_data.Number_Weeks_Used.fillna(-1,inplace = True)
train_data = train_data.loc[(train_data.Number_Doses_Week>0)]
sns.boxplot(train_data['Crop_Damage'],train_data['Estimated_Insects_Count'])
sns.boxplot(train_data['Crop_Damage'],train_data['Number_Doses_Week'])
sns.boxplot(train_data['Crop_Damage'],train_data['Number_Weeks_Used'])
sns.boxplot(train_data['Crop_Damage'],train_data['Number_Weeks_Quit'])
sns.boxplot(train_data['Crop_Damage'],np.floor(train_data['Number_Weeks_Quit']-train_data['Number_Weeks_Used']))
sns.boxplot(train_data['Crop_Damage'],np.floor(train_data['Number_Weeks_Used']*train_data['Number_Doses_Week']))
combined_data = pd.concat([train_data,test_data],axis = 0)

combined_data.head()
combined_data['ID'] = combined_data['ID'].str[1:].astype('int')
combined_data = combined_data.sort_values(by ='ID', ascending = 1) 
combined_data = combined_data.reset_index(drop =  True)
sns.barplot(combined_data.loc[30:40,'Estimated_Insects_Count'].index,combined_data.loc[30:40,'Estimated_Insects_Count'])
combined_data['Group'] = 0

combined_data.loc[0,'Group'] = 1







combined_data['Group_Change'] = 0

combined_data['Group_First'] = 0

combined_data['Group_Last'] = 0

combined_data['Batch_Last'] = 0





combined_data.loc[0,'Group_First'] = 1



combined_data['Soil_Change'] = 0
%%time

for idx, row in combined_data.iterrows():

    

    if idx != 0:

        

        if (np.abs(combined_data.loc[idx,'Estimated_Insects_Count'] - combined_data.loc[idx-1,'Estimated_Insects_Count']) > 1):

            

            combined_data.loc[idx,'Group'] = combined_data.loc[idx-1,'Group'] + 1

            combined_data.loc[idx,'Group_First'] = 1

            combined_data.loc[idx-1,'Group_Last'] = 1

            combined_data.loc[idx-1,'Batch_Last'] = 1

        

        else:

            

            if(combined_data.loc[idx,'Number_Doses_Week'] >= combined_data.loc[idx-1,'Number_Doses_Week']):

                

                if(combined_data.loc[idx,'Soil_Type'] == combined_data.loc[idx-1,'Soil_Type']):

                    

                    combined_data.loc[idx,'Group'] = combined_data.loc[idx-1,'Group']

                    

                else:

                    

                    combined_data.loc[idx,'Group'] = combined_data.loc[idx-1,'Group'] + 1

                    

            else:

                

                combined_data.loc[idx,'Group'] = combined_data.loc[idx-1,'Group'] + 1

                combined_data.loc[idx,'Group_Change'] = 1

                combined_data.loc[idx,'Group_First'] = 1

                combined_data.loc[idx-1,'Group_Last'] = 1

                   

   
%%time 

for idx in range(0,len(combined_data)-1):

        if combined_data.loc[idx,'Group_Last'] == 1 & combined_data.loc[idx,'Soil_Type'] != combined_data.loc[idx+1,'Soil_Type']:

            combined_data.loc[idx,'Soil_Change'] == 1
df_group = combined_data.loc[(combined_data.Crop_Damage.isna() == 0),['Group']]

df_group_count = pd.DataFrame({'Group':df_group.Group.value_counts().index ,'Count' :df_group.Group.value_counts()})

df_group_count = df_group_count[df_group_count['Count']>1]
%%time

subgroup_avg = pd.DataFrame()



for idx, row in df_group_count.iterrows():

    #print(row.Group)

    subgroup = combined_data.loc[(combined_data.Group == row.Group) & (combined_data.Crop_Damage.isna() == 0) ,['ID','Crop_Damage']]

    

    for i in subgroup.Crop_Damage.unique():

        id_min = subgroup[subgroup.Crop_Damage == i].ID.min() + 1   

        id_max = subgroup[subgroup.Crop_Damage == i].ID.max() - 1   

        

        if id_min <= id_max:

            subgroup_avg = pd.concat([subgroup_avg,pd.DataFrame({'ID': range(id_min, id_max + 1),'Group_Avg' : i})],axis = 0)

      
%%time

subgroup_one = pd.DataFrame()

subgroup_two = pd.DataFrame()



for grp in combined_data.Group.unique():

    #print(grp)

    subgroup = combined_data.loc[(combined_data.Group == grp ),['ID','Crop_Damage']]

    

    if 1 in subgroup.Crop_Damage.unique():

        

        id_min_one = subgroup.loc[(subgroup.Crop_Damage == 1) & (subgroup.Crop_Damage.isna() == 0),:].ID.min() + 1   

        id_max = subgroup.ID.max()    

        

        if id_min_one <= id_max:

            subgroup_one = pd.concat([subgroup_one,pd.DataFrame({'ID': range(id_min_one, id_max+1),'Group_One' : 1})],axis = 0)

    

    if 2 in subgroup.Crop_Damage.unique():

        

        id_min_two = subgroup.loc[(subgroup.Crop_Damage == 2) & (subgroup.Crop_Damage.isna() == 0),:].ID.min() + 1   

        id_max = subgroup.ID.max()    

        

        if id_min_two <= id_max:

            subgroup_two = pd.concat([subgroup_two,pd.DataFrame({'ID': range(id_min_two, id_max+1),'Group_Two' : 2})],axis = 0)

                                
combined_data = pd.merge(combined_data,subgroup_avg,on='ID',how='left')

combined_data['Group_Avg'].fillna(-1,inplace =True)
combined_data = pd.merge(combined_data,subgroup_one,on='ID',how='left')

combined_data['Group_One'].fillna(0,inplace =True)
combined_data = pd.merge(combined_data,subgroup_two,on='ID',how='left')

combined_data['Group_Two'].fillna(0,inplace =True)
#temp = combined_data.copy()
#temp = temp.drop(columns = ['Group__Number_Doses_Week_count', 'Group__Number_Doses_Week_std','Total_Doses','Insect_Freq','Week_Since_Pesticide_Used'

#                           ,'Group__Number_Doses_Week_count','Group__Number_Doses_Week_std'])

#temp.head()
#combined_data = temp.copy()
#combined_data['Crop_Soil_Type'] = combined_data['Crop_Type'].astype(str) + '_' + combined_data['Soil_Type'].astype(str)

#['Crop_Pesticide_Type'] =  combined_data['Crop_Type'].astype(str) + '_' + combined_data['Pesticide_Use_Category'].astype(str)

#combined_data['Crop_Season_Type'] = combined_data['Crop_Type'].astype(str) + '_' + combined_data['Season'].astype(str)

#combined_data['Soil_Pesticide_Type'] = combined_data['Soil_Type'].astype(str) +'_' +combined_data['Pesticide_Use_Category'].astype(str)

#combined_data['Pesticide_Season_Type'] = combined_data['Pesticide_Use_Category'].astype(str) +'_'+combined_data['Season'].astype(str)

#combined_data['Soil_Season_Type'] = combined_data['Soil_Type'].astype(str) +'_'+combined_data['Season'].astype(str)

#combined_data['Total_Doses']= combined_data['Number_Doses_Week']* combined_data['Number_Weeks_Used']

#combined_data['Insect_Freq']= combined_data['Estimated_Insects_Count']/combined_data['Number_Doses_Week']

#combined_data['Week_Since_Pesticide_Used'] = combined_data['Number_Weeks_Used'] +combined_data['Number_Weeks_Quit']

#combined_data['Week_Used_Quit'] = combined_data['Number_Weeks_Used']/combined_data['Number_Weeks_Quit']

#combined_data['Side_Effect_days'] = combined_data['Number_Weeks_Quit']-combined_data['Number_Weeks_Used']

#combined_data["Unique_Pesticides_Used_Per_Crop_Soil"] = combined_data.groupby(['Crop_Type','Soil_Type'])['Pesticide_Use_Category'].transform('nunique')
#cols =combined_data.columns

#le = LabelEncoder()

#for col in cols:

#    if col != 'Crop_Damage':

#        combined_data[col] = le.fit_transform(combined_data[col])

#combined_data = pd.DataFrame(combined_data,columns = cols)
#combined_data =combined_data.drop(columns = ['Total_Doses', 'Side_Effect_days'])
def agg_numeric(df, parent_var, df_name):



            

    # Only want the numeric variables

    parent_ids = df[parent_var].copy()

    numeric_df = df[['Number_Doses_Week','Number_Weeks_Used','Number_Weeks_Quit']].copy()

    numeric_df[parent_var] = parent_ids



    # Group by the specified variable and calculate the statistics

    agg = numeric_df.groupby(parent_var).agg(['count'

                                              #, 'mean'

                                              #, 'max'

                                              #, 'min'

                                              #, 'sum'

                                              ,'std'  

                                             ])



    # Need to create new column names

    columns = []



    # Iterate through the variables names

    for var in agg.columns.levels[0]:

        if var != parent_var:

            # Iterate through the stat names

            for stat in agg.columns.levels[1]:

                # Make a new column name for the variable and stat

                columns.append('%s_%s_%s' % (df_name, var, stat))

    

    agg.columns = columns

    

    # Remove the columns with all redundant values

    _, idx = np.unique(agg, axis = 1, return_index=True)

    agg = agg.iloc[:, idx]

    

    return agg
df_group_agg = agg_numeric(combined_data,'Group','Group_').reset_index()

df_group_agg = df_group_agg.apply(lambda x : x.fillna(-1))

df_group_agg.head()
combined_data = pd.merge(combined_data,df_group_agg,on='Group',how='left')
combined_data.columns
features = ['Estimated_Insects_Count', 'Crop_Type', 'Soil_Type',

       'Pesticide_Use_Category', 'Number_Doses_Week', 'Number_Weeks_Used',

       'Number_Weeks_Quit', 'Season', 'Group', 'Group_Change',

       'Group_First', 'Group_Last', 'Batch_Last', 'Group_Avg',

       'Group_One', 'Group_Two', 'Group__Number_Doses_Week_count',

       'Group__Number_Weeks_Quit_std', 'Group__Number_Weeks_Used_std',

       'Group__Number_Doses_Week_std']
train_features = combined_data.loc[(combined_data.Crop_Damage.isna() == 0 ),features]

target = combined_data.loc[(combined_data.Crop_Damage.isna() == 0 ),'Crop_Damage']



X_test = combined_data.loc[(combined_data.Crop_Damage.isna() == 1 ),features]
train_features.head()
def resampling_strategy(method,X,y):

    

    if method == 'ROS':

        sm = RandomOverSampler()

        X, y = sm.fit_sample(X, y)

    

    if method == 'RUS':

        sm = RandomUnderSampler()

        X, y = sm.fit_sample(X, y)  

        

    if method == 'Tomek':

        sm = TomekLinks(n_jobs = -1)

        X, y = sm.fit_sample(X, y)

        

    if method == 'SMOTE':

        sm = SMOTE(n_jobs = -1,sampling_strategy = 'all')

        X, y = sm.fit_sample(X, y)

    

    if method == 'SMOTETomek':

        sm = SMOTETomek(n_jobs = -1)

        X, y = sm.fit_sample(X, y)

    

    if method == 'Cluster':

        sm = ClusterCentroids(n_jobs = -1)

        X, y = sm.fit_sample(X, y)

    

    if method == 'NearMiss':

        sm = NearMiss(n_jobs = -1)

        X, y = sm.fit_sample(X, y)

    

    print("Resampling Startegy-------",method)

    print('#'*20)

    counts = y.value_counts()

    print(counts)

    print('#'*20)

    plt.xlabel("Crop_Damage")

    plt.ylabel('Count')

    sns.barplot(counts.index , counts.values)

    

    return X,y
def feature_importance(model, X_train):



    print(model.feature_importances_)

    names = X_train.columns.values

    ticks = [i for i in range(len(names))]

    plt.bar(ticks, model.feature_importances_)

    plt.xticks(ticks, names,rotation =90)

    plt.show()
def create_submission_file(model_list,df):

    preds = 0

    submission = pd.read_csv('/kaggle/input/pesticidesagriculturedataset/Agricultre/sample_submission.csv')

    for model in model_list:

        preds = preds + (model.predict_proba(df))

    submission.loc[:,'Crop_Damage'] = np.argmax(preds/len(model_list),axis =1)

    submission.loc[:,'ID'] = test_data_ids

    submission.loc[(submission.ID.isin(test_data[test_data.Number_Doses_Week == 0].ID.values)),'Crop_Damage'] = 1

    !rm './submission.csv'

    submission.to_csv('submission.csv', index = False, header = True)

    print(submission.head())
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)

scores = []

X_train_cv,y_train_cv = resampling_strategy('',train_features.copy(), target.copy())

for i, (idxT, idxV) in enumerate(rskf.split(X_train_cv, y_train_cv)):

    print('Fold',i)

    print(' rows of train =',len(idxT),'rows of holdout =',len(idxV))

    clf = lgb.LGBMClassifier(

            n_estimators=1000,

            max_depth=6,

            learning_rate=0.1,

            subsample=0.8,

            colsample_bytree=0.4,

            objective = 'multiclass'

        )        

    

    h = clf.fit(X_train_cv.iloc[idxT], y_train_cv.iloc[idxT], 

                eval_set=[(X_train_cv.iloc[idxV],y_train_cv.iloc[idxV])],

                verbose=100,eval_metric='multi_logloss',

                early_stopping_rounds=50)

    acc = accuracy_score(y_train_cv.iloc[idxV],np.argmax(clf.predict_proba(X_train_cv.iloc[idxV]),axis =1))

    scores.append(acc)

    print ('LGB Val CV=',acc)

    print('#'*20)





print('%.3f (%.3f)' % (np.array(scores).mean(), np.array(scores).std()))

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)

scores = []

X_train_cv,y_train_cv = resampling_strategy('',train_features.copy(), target.copy())

for i, (idxT, idxV) in enumerate(rskf.split(X_train_cv, y_train_cv)):

    print('Fold',i)

    print(' rows of train =',len(idxT),'rows of holdout =',len(idxV))

    clf = xgb.XGBClassifier(

            n_estimators=1000,

            max_depth=6,

            learning_rate=0.1,

            subsample=0.8,

            colsample_bytree=0.4,

            objective = 'multi:softprob'

        )        

    

    h = clf.fit(X_train_cv.iloc[idxT], y_train_cv.iloc[idxT], 

                eval_set=[(X_train_cv.iloc[idxV],y_train_cv.iloc[idxV])],

                verbose=100,eval_metric='merror',

                early_stopping_rounds=50)

    acc = accuracy_score(y_train_cv.iloc[idxV],np.argmax(clf.predict_proba(X_train_cv.iloc[idxV]),axis =1))

    scores.append(acc)

    print ('LGB Val CV=',acc)

    print('#'*20)





print('%.3f (%.3f)' % (np.array(scores).mean(), np.array(scores).std()))
X_train, X_val, y_train, y_val = train_test_split(train_features,target , test_size=0.2, random_state=1,stratify = target)
#X_train,y_train = resampling_strategy('',train_features.copy(), target.copy())



model_lgb = lgb.LGBMClassifier(boosting_type='gbdt',

                               n_estimators= 1000,

                               max_depth=6,

                               learning_rate=0.1,

                               subsample=0.8,

                               colsample_bytree=0.4,

                               objective = 'multiclass'

                              )





model_lgb.fit(X_train, y_train,

              eval_set=[(X_train, y_train),(X_val, y_val)],

              eval_metric=['multi_logloss'],

              early_stopping_rounds = 100,

              verbose=2)
print(model_lgb.best_score_['valid_1'])

feature_importance(model_lgb,X_train)
#create_submission_file([model_lgb],X_test)
#X_train,y_train = resampling_strategy('',train_features.copy(), target.copy())



model_xgb = xgb.XGBClassifier(objective = 'multi:softprob' ,max_depth =6 , n_estimators=1000 ,

                              subsample =0.9,colsample_bytree=0.9,eval_metric = 'merror',seed=42)



model_xgb.fit(X_train,y_train,

              eval_set=[(X_train,y_train),(X_val, y_val)],

              early_stopping_rounds = 50,

              verbose=2)
feature_importance(model_xgb,X_train)
#create_submission_file([model_xgb],X_test)
rfc = RandomForestClassifier(n_estimators=500 ,

                             max_depth=6, min_samples_split=2, 

                             min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', 

                             n_jobs=-1, random_state=123, verbose=3)

rfc.fit(train_features.copy(),target.copy())
feature_importance(rfc,X_train)
create_submission_file([rfc,model_lgb,model_xgb],X_test)