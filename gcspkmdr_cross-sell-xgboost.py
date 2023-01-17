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



#confusing ids



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
response = train_data.loc[:,"Response"].value_counts().rename('Count')

plt.xlabel("Response")

plt.ylabel('Count')

sns.barplot(response.index , response.values).set_title('Response')
response
sns.distplot(train_data['Annual_Premium'])
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
test_df.loc[(test_df.Annual_Premium.isin(list(set(test_df.Annual_Premium) - set(train_df.Annual_Premium)))),'Annual_Premium'] = -1
train_df['train'] = 1



test_df['train'] = 0



combined_data = pd.concat([train_df,test_df],axis =0).reset_index(drop = True).copy()
premium_discretizer = KBinsDiscretizer(n_bins=8, encode='ordinal', strategy='quantile')



combined_data['Premium_Bins'] =premium_discretizer.fit_transform(combined_data['Annual_Premium'].values.reshape(-1,1)).astype(int)



age_discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')



combined_data['Age_Bins'] =age_discretizer.fit_transform(combined_data['Age'].values.reshape(-1,1)).astype(int)
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
train_df = combined_data[combined_data['train']==1]



test_df = combined_data[combined_data['train']==0]
cols = ['Gender', 'Age', 'Driving_License', 'Region_Code',

       'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage',

        'Annual_Premium','Policy_Sales_Channel', 'Vintage']



train_df = train_df[~train_df.loc[:,cols].duplicated(keep = 'first')].reset_index(drop=True)
target = train_df['Response']



train_df = train_df.drop(columns =['train','id','Response'])



test_df = test_df.drop(columns=['train','id','Response'])
train_df.head()
trees = 5



n_splits = 5



seeds = [32,432,73,5,2]



submission = pd.read_csv('../input/avcrosssell/sample_submission.csv')



probs = np.zeros(shape=(len(test_df),))



# probablity file per seed per split per tree

submission_probs = pd.DataFrame(columns = ['id','Response'])



submission_probs.iloc[:,0] = submission.iloc[:,0]



submission_probs.iloc[:,1:] = 0
%%time



##XGBM



scores = []



avg_loss = []



submission_name = []



seed_no = []



fold_no = []



X_train_cv,y_train_cv = train_df.copy(), target.copy()



cat_features = ['Driving_License','Gender','Region_Code','Previously_Insured','Vehicle_Damage',

                'Policy_Sales_Channel','Policy_Region','Vehicle_Age','Vintage',

                'Annual_Premium','Vehicle_Age_License','Premium_Bins']



cont_features = ['Age','Age_Bins']



for seed in tnrange(len(seeds)):

    

    sssf = StratifiedShuffleSplit(n_splits=n_splits, test_size = 0.3 ,random_state=seeds[seed])



    for j, (idxT, idxV) in tqdm(enumerate(sssf.split(X_train_cv, y_train_cv))):

        

        print('Fold',j)



        print(' rows of train =',len(idxT),'rows of holdout =',len(idxV))



        model_xgb = [0] *trees



        for i in tnrange(trees):



            print('Tree',i)

            

            seed_no.append(seeds[seed])

            

            fold_no.append(j)

            

            model_xgb[i] = xgb.XGBClassifier(n_estimators=1000,

                                max_depth=6,

                                learning_rate=0.04,

                                subsample=0.9,

                                colsample_bytree=0.35,

                                objective = 'binary:logistic',

                                random_state = i*27

                               )        

  

            

            model_xgb[i].fit(X_train_cv.iloc[idxT], y_train_cv.iloc[idxT], 

                    eval_set=[(X_train_cv.iloc[idxV],y_train_cv.iloc[idxV])],

                    verbose=100,eval_metric=['auc','logloss'],

                    early_stopping_rounds=50)





            probs_file_name = 'probs_'+str(seeds[seed])+'_'+str(j)+'_'+str(i)+".csv"

            

            model_xgb_probs = model_xgb[i].predict_proba(test_df)[:,1]

            

            submission_probs.iloc[:,1:] = model_xgb_probs

            

            # probablity file per seed per split per tree

            submission_probs.to_csv(probs_file_name,index = False)

            

            probs += model_xgb_probs

            

            probs_oof = model_xgb[i].predict_proba(X_train_cv.iloc[idxV])[:,1]

            

            roc = roc_auc_score(y_train_cv.iloc[idxV],probs_oof)



            scores.append(roc)



            avg_loss.append(model_xgb[i].best_score)

            

            submission_name.append(probs_file_name)

            

            print ('XGB ROC OOF =',roc)

            

            print('#'*100)

    



print("Average Log Loss Stats {0:.5f},{1:.5f}".format(np.array(avg_loss).mean(), np.array(avg_loss).std()))

submission_probs.iloc[:,1:] = probs/(len(seeds)*trees*n_splits)



# probablity combined

submission_probs.to_csv('probs.csv',index =False)
model_stats = pd.DataFrame({'submission':submission_name,'seed': seed_no,'fold':fold_no,'oof_roc':scores,'validation_loss':avg_loss})



model_stats.to_csv('model_stats.csv',index =False)



model_stats.head()