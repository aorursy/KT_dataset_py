#for data processing

import numpy as np 

import pandas as pd



#for visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import warnings

warnings.filterwarnings("ignore")
import lightgbm as lgb

from sklearn import preprocessing

from sklearn.metrics import mean_squared_log_error, mean_squared_error

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import log_loss

from sklearn.metrics import classification_report

import seaborn as sns

from collections import Counter

sns.set_style('whitegrid')
train= pd.read_csv("../input/av-janatahack-machine-learning-in-agriculture/train_yaOffsB.csv")

test= pd.read_csv("../input/av-janatahack-machine-learning-in-agriculture/test_pFkWwen.csv")
train.head(2)
test.head(2)
train['train_or_test']='train'

test['train_or_test']='test'

all=pd.concat([train,test])
all.head(2)
#Visualization to check for missing values

sns.heatmap(all.isna())
all.info()
#ID

sum(all['ID'].value_counts()>1)
#Estimated_Insects_Count

sns.set_style('whitegrid')

sns.distplot(all[all['Crop_Damage']==0]['Estimated_Insects_Count'],bins=30,color='blue',kde=False)

sns.distplot(all[all['Crop_Damage']==1]['Estimated_Insects_Count'],bins=30,color='red',kde=False)

sns.distplot(all[all['Crop_Damage']==2]['Estimated_Insects_Count'],bins=30,color='green',kde=False)

plt.legend(labels=['Crop Damage=0', 'Crop Damage=1', 'Crop Damage=2'])
#Crop_Type

sns.set_style('whitegrid')

sns.countplot(x='Crop_Type',data=all,hue='Crop_Damage')

plt.legend(labels=['Crop Damage=0', 'Crop Damage=1', 'Crop Damage=2'])
groupby_df = all[all['train_or_test']=='train'].groupby(['Crop_Type', 'Crop_Damage']).agg({'Crop_Damage': 'count'})

groupby_pcts = groupby_df.groupby(level=0).apply(lambda x:round(100 * x / x.sum(),2))

groupby_df,groupby_pcts
#Soil_Type

sns.set_style('whitegrid')

sns.countplot(x='Soil_Type',data=all,hue='Crop_Damage')

plt.legend(labels=['Crop Damage=0', 'Crop Damage=1', 'Crop Damage=2'])
groupby_df = all[all['train_or_test']=='train'].groupby(['Soil_Type', 'Crop_Damage']).agg({'Crop_Damage': 'count'})

groupby_pcts = groupby_df.groupby(level=0).apply(lambda x:round(100 * x / x.sum(),2))

groupby_df,groupby_pcts
#Pesticide_Use_Category

sns.set_style('whitegrid')

sns.set_style('whitegrid')

sns.countplot(x='Pesticide_Use_Category',data=all,hue='Crop_Damage')

plt.legend(labels=['Crop Damage=0', 'Crop Damage=1', 'Crop Damage=2'])
groupby_df = all[all['train_or_test']=='train'].groupby(['Pesticide_Use_Category', 'Crop_Damage']).agg({'Crop_Damage': 'count'})

groupby_pcts = groupby_df.groupby(level=0).apply(lambda x:round(100 * x / x.sum(),2))

groupby_df,groupby_pcts
#Number_Doses_Week

sns.set_style('whitegrid')

sns.distplot(all[all['Crop_Damage']==0]['Number_Doses_Week'],bins=30,color='blue',kde=False)

sns.distplot(all[all['Crop_Damage']==1]['Number_Doses_Week'],bins=30,color='red',kde=False)

sns.distplot(all[all['Crop_Damage']==2]['Number_Doses_Week'],bins=30,color='green',kde=False)

plt.legend(labels=['Crop Damage=0', 'Crop Damage=1', 'Crop Damage=2'])
#Number_Weeks_Used

sns.set_style('whitegrid')

sns.distplot(all[all['Crop_Damage']==0]['Number_Weeks_Used'],bins=30,color='blue',kde=False)

sns.distplot(all[all['Crop_Damage']==1]['Number_Weeks_Used'],bins=30,color='red',kde=False)

sns.distplot(all[all['Crop_Damage']==2]['Number_Weeks_Used'],bins=30,color='green',kde=False)

plt.legend(labels=['Crop Damage=0', 'Crop Damage=1', 'Crop Damage=2'])
#Number_Weeks_Quit

sns.set_style('whitegrid')

sns.distplot(all[all['Crop_Damage']==0]['Number_Weeks_Quit'],bins=30,color='blue',kde=False)

sns.distplot(all[all['Crop_Damage']==1]['Number_Weeks_Quit'],bins=30,color='red',kde=False)

sns.distplot(all[all['Crop_Damage']==2]['Number_Weeks_Quit'],bins=30,color='green',kde=False)

plt.legend(labels=['Crop Damage=0', 'Crop Damage=1', 'Crop Damage=2'])
#Pesticide_Use_Category

sns.set_style('whitegrid')

sns.set_style('whitegrid')

sns.countplot(x='Season',data=all,hue='Crop_Damage')

plt.legend(labels=['Crop Damage=0', 'Crop Damage=1', 'Crop Damage=2'])
groupby_df = all[all['train_or_test']=='train'].groupby(['Season', 'Crop_Damage']).agg({'Crop_Damage': 'count'})

groupby_pcts = groupby_df.groupby(level=0).apply(lambda x:round(100 * x / x.sum(),2))

groupby_df,groupby_pcts
#Crop_Damage

sns.set_style('whitegrid')

sns.countplot('Crop_Damage',hue='train_or_test',data=all)
all.dtypes
sns.heatmap(all.corr(),annot=True)
# Estimated_Insects_Count & Number_Doses_Week  

# sns.jointplot(x='Estimated_Insects_Count',y='Number_Doses_Week',data=all,kind='kde')
# Estimated_Insects_Count & Number_Weeks_Used

# sns.jointplot(x='Estimated_Insects_Count',y='Number_Weeks_Used',data=all,kind='kde')
# Estimated_Insects_Count & Number_Weeks_Quit

# sns.jointplot(x='Estimated_Insects_Count',y='Number_Weeks_Quit',data=all,kind='kde')
# Number_Doses_Week & Number_Weeks_Used

# sns.jointplot(x='Number_Doses_Week',y='Number_Weeks_Used',data=all,kind='kde')
# Number_Doses_Week & Number_Weeks_Quit

# sns.jointplot(x='Number_Doses_Week',y='Number_Weeks_Quit',data=all,kind='kde')
# Number_Weeks_Used & Number_Weeks_Quit

# sns.jointplot(x='Number_Weeks_Used',y='Number_Weeks_Quit',data=all,kind='kde')
from scipy.stats import chi2
def chi_test(df,col1,col2):

    

    #Contingency Table

    contingency_table=pd.crosstab(df[col1],df[col2])

    #print('contingency_table :-\n',contingency_table)



    #Observed Values

    Observed_Values = contingency_table.values 

    #print("\nObserved Values :-\n",Observed_Values)



    #Expected Values

    import scipy.stats

    b=scipy.stats.chi2_contingency(contingency_table)

    Expected_Values = b[3]

    #print("\nExpected Values :-\n",Expected_Values)



    #Degree of Freedom

    no_of_rows=len(contingency_table.iloc[0:2,0])

    no_of_columns=len(contingency_table.iloc[0,0:2])

    df=(no_of_rows-1)*(no_of_columns-1)

    #print("\nDegree of Freedom:-",df)



    #Significance Level 5%

    alpha=0.05

    #print('\nSignificance level: ',alpha)



    #chi-square statistic - χ2

    chi_square=sum([(o-e)**2./e for o,e in zip(Observed_Values,Expected_Values)])

    chi_square_statistic=chi_square[0]+chi_square[1]

    #print("\nchi-square statistic:-",chi_square_statistic)



    #critical_value

    critical_value=chi2.ppf(q=1-alpha,df=df)

    #print('\ncritical_value:',critical_value)



    #p-value

    p_value=1-chi2.cdf(x=chi_square_statistic,df=df)

    #print('\np-value:',p_value)



    #compare chi_square_statistic with critical_value and p-value which is the probability of getting chi-square>0.09 (chi_square_statistic)

    if chi_square_statistic>=critical_value:

        print("\nchi_square_statistic & critical_value - significant result, reject null hypothesis (H0), dependent.")

    else:

        print("\nchi_square_statistic & critical_value - not significant result, fail to reject null hypothesis (H0), independent.")



    if p_value<=alpha:

        print("\np_value & alpha - significant result, reject null hypothesis (H0), dependent.")

    else:

        print("\np_value & alpha - not significant result, fail to reject null hypothesis (H0), independent.")
#Soil_Type & Crop_Type

chi_test(all,'Soil_Type','Crop_Type')
#Soil_Type & Pesticide_Use_Category

chi_test(all,'Soil_Type','Pesticide_Use_Category')
#Soil_Type & Season

chi_test(all,'Soil_Type','Season')
#Crop_Type & Pesticide_Use_Category

chi_test(all,'Crop_Type','Pesticide_Use_Category')
#Crop_Type & Season

chi_test(all,'Crop_Type','Season')
#Pesticide_Use_Category & Season

chi_test(all,'Pesticide_Use_Category','Season')
#Soil_Type & Estimated_Insects_Count

sns.boxplot(x='Soil_Type',y='Estimated_Insects_Count',data=all)
#Soil_Type & Number_Weeks_Used

sns.boxplot(x='Soil_Type',y='Number_Weeks_Used',data=all)
#Soil_Type & Number_Doses_Week

sns.boxplot(x='Soil_Type',y='Number_Doses_Week',data=all)
#Soil_Type & Number_Weeks_Quit

sns.boxplot(x='Soil_Type',y='Number_Weeks_Quit',data=all)
#Crop_Type & Estimated_Insects_Count

sns.boxplot(x='Crop_Type',y='Estimated_Insects_Count',data=all)
#Crop_Type & Number_Weeks_Used

sns.boxplot(x='Crop_Type',y='Number_Weeks_Used',data=all)
#Crop_Type & Number_Doses_Week

sns.boxplot(x='Crop_Type',y='Number_Doses_Week',data=all)
#Crop_Type & Number_Weeks_Quit

sns.boxplot(x='Crop_Type',y='Number_Weeks_Quit',data=all)
#Pesticide_Use_Category & Estimated_Insects_Count

sns.boxplot(x='Pesticide_Use_Category',y='Estimated_Insects_Count',data=all)
#Pesticide_Use_Category & Number_Weeks_Used

sns.boxplot(x='Pesticide_Use_Category',y='Number_Weeks_Used',data=all)
#Pesticide_Use_Category & Number_Doses_Week

sns.boxplot(x='Pesticide_Use_Category',y='Number_Doses_Week',data=all)
#Pesticide_Use_Category & Number_Weeks_Quit

sns.boxplot(x='Pesticide_Use_Category',y='Number_Weeks_Quit',data=all)
#Season & Estimated_Insects_Count

sns.boxplot(x='Season',y='Estimated_Insects_Count',data=all)
#Season & Number_Weeks_Used

sns.boxplot(x='Season',y='Number_Weeks_Used',data=all)
#Season & Number_Doses_Week

sns.boxplot(x='Season',y='Number_Doses_Week',data=all)
#Season & Number_Weeks_Quit

sns.boxplot(x='Season',y='Number_Doses_Week',data=all)
feature_cols = all.columns.tolist()

feature_cols.remove('ID')

feature_cols.remove('Crop_Damage')

feature_cols.remove('train_or_test')

label_col = 'Crop_Damage'

print(feature_cols)
all['ID_value'] = all['ID'].apply(lambda x: x.strip('F')).astype('int')
all=all.sort_values(['ID_value'])
all.shape
#Performing this operation as datetime has an upper and lower limit

date = np.array('2020-07-24', dtype=np.datetime64)

date=date-74084

date
#Creating a date array 

date_arr=date+np.arange(148168)

date_arr
all['date']=date_arr#pd.to_datetime(date_arr, errors='coerce')
#Estimated_Insects_Count

sns.lineplot(x=all[(all['date'].dt.year>2000) & (all['date'].dt.year<2025)]['date'],y=all[(all['date'].dt.year>2000) & (all['date'].dt.year<2025)]['Estimated_Insects_Count'])
#Number_Doses_Week

sns.lineplot(x=all[(all['date'].dt.year>2020) & (all['date'].dt.year<2025)]['date'],y=all[(all['date'].dt.year>2020) & (all['date'].dt.year<2025)]['Number_Doses_Week'])
#Number_Weeks_Used

sns.lineplot(x=all[(all['date'].dt.year>2020) & (all['date'].dt.year<2025)]['date'],y=all[(all['date'].dt.year>2020) & (all['date'].dt.year<2025)]['Number_Weeks_Used'])
#SeasonNumber_Weeks_Quit

sns.lineplot(x=all[(all['date'].dt.year>2020) & (all['date'].dt.year<2025)]['date'],y=all[(all['date'].dt.year>2020) & (all['date'].dt.year<2025)]['Number_Weeks_Quit'])
#Crop_Damage

sns.lineplot(x=all[(all['date'].dt.year>2000) & (all['date'].dt.year<2025)]['date'],y=all[(all['date'].dt.year>2000) & (all['date'].dt.year<2025)]['Crop_Damage'])
#Resetting Index

all=all.reset_index(drop=True)
all['Crop_Type_Damage_lag1']=all.sort_values(['ID_value']).groupby(['Crop_Type'])['Crop_Damage'].apply(lambda x:x.shift().rolling(5,min_periods=1).mean()).fillna(-999).values

all['Soil_Type_Damage_lag1']=all.sort_values(['ID_value']).groupby(['Soil_Type'])['Crop_Damage'].apply(lambda x:x.shift().rolling(5,min_periods=1).mean()).fillna(-999).values

all['Pesticide_Use_Category_lag1']=all.sort_values(['ID_value']).groupby(['Pesticide_Use_Category'])['Crop_Damage'].apply(lambda x:x.shift().rolling(5,min_periods=1).mean()).fillna(-999).values

all['Season_lag1']=all.sort_values(['ID_value']).groupby(['Season'])['Crop_Damage'].apply(lambda x:x.shift().rolling(5,min_periods=1).mean()).fillna(-999).values



all['Crop_Type_Damage_lag2']=all.sort_values(['ID_value']).groupby(['Crop_Type'])['Crop_Damage'].apply(lambda x:x.shift(periods=2).rolling(5,min_periods=1).mean()).fillna(-999).values

all['Soil_Type_Damage_lag2']=all.sort_values(['ID_value']).groupby(['Soil_Type'])['Crop_Damage'].apply(lambda x:x.shift(periods=2).rolling(5,min_periods=1).mean()).fillna(-999).values

all['Pesticide_Use_Category_lag2']=all.sort_values(['ID_value']).groupby(['Pesticide_Use_Category'])['Crop_Damage'].apply(lambda x:x.shift(periods=2).rolling(5,min_periods=1).mean()).fillna(-999).values

all['Season_lag2']=all.sort_values(['ID_value']).groupby(['Season'])['Crop_Damage'].apply(lambda x:x.shift(periods=2).rolling(5,min_periods=1).mean()).fillna(-999).values

#Setting Crop_Damage=-999 for missing values

all.loc[all['train_or_test'] == 'test', 'Crop_Damage'] = -999
#Creating Other Lag Variables

all['Crop_Damage_lag1'] = all['Crop_Damage'].shift(fill_value=-999)

all['Estimated_Insects_Count_lag1'] = all['Estimated_Insects_Count'].shift(fill_value=-999)

all['Crop_Type_lag1'] = all['Crop_Type'].shift(fill_value=-999)

all['Soil_Type_lag1'] = all['Soil_Type'].shift(fill_value=-999)

all['Pesticide_Use_Category_lag1'] = all['Pesticide_Use_Category'].shift(fill_value=-999)

all['Number_Doses_Week_lag1'] = all['Number_Doses_Week'].shift(fill_value=-999)

all['Number_Weeks_Used_lag1'] = all['Number_Weeks_Used'].shift(fill_value=-999)

all['Number_Weeks_Quit_lag1'] = all['Number_Weeks_Quit'].shift(fill_value=-999)

all['Season_lag1'] = all['Season'].shift(fill_value=-999)



all['Crop_Damage_lag2'] = all['Crop_Damage'].shift(periods=2,fill_value=-999)

all['Estimated_Insects_Count_lag2'] = all['Estimated_Insects_Count'].shift(periods=2,fill_value=-999)

all['Crop_Type_lag2'] = all['Crop_Type'].shift(fill_value=-999)

all['Soil_Type_lag2'] = all['Soil_Type'].shift(fill_value=-999)

all['Pesticide_Use_Category_lag2'] = all['Pesticide_Use_Category'].shift(periods=2,fill_value=-999)

all['Number_Doses_Week_lag2'] = all['Number_Doses_Week'].shift(periods=2,fill_value=-999)

all['Number_Weeks_Used_lag2'] = all['Number_Weeks_Used'].shift(periods=2,fill_value=-999)

all['Number_Weeks_Quit_lag2'] = all['Number_Weeks_Quit'].shift(periods=2,fill_value=-999)

all['Season_lag2'] = all['Season'].shift(periods=2,fill_value=-999)

#train & test split

train, test = all[all.train_or_test == 'train'], all[all.train_or_test == 'test']
train.drop(['train_or_test'], inplace=True, axis=1)

test.drop(['train_or_test'], inplace=True, axis=1)

test.drop([label_col], inplace=True, axis=1)
print(train.shape, test.shape)
del all
missing_impute = -999
train['Number_Weeks_Used'] = train['Number_Weeks_Used'].apply(lambda x: missing_impute if pd.isna(x) else x)

test['Number_Weeks_Used'] = test['Number_Weeks_Used'].apply(lambda x: missing_impute if pd.isna(x) else x)



train['Number_Weeks_Used_lag1'] = train['Number_Weeks_Used_lag1'].apply(lambda x: missing_impute if pd.isna(x) else x)

test['Number_Weeks_Used_lag1'] = test['Number_Weeks_Used_lag1'].apply(lambda x: missing_impute if pd.isna(x) else x)



train['Number_Weeks_Used_lag2'] = train['Number_Weeks_Used_lag2'].apply(lambda x: missing_impute if pd.isna(x) else x)

test['Number_Weeks_Used_lag2'] = test['Number_Weeks_Used_lag2'].apply(lambda x: missing_impute if pd.isna(x) else x)
df_train, df_eval = train_test_split(train, test_size=0.40, random_state=101, shuffle=True, stratify=train[label_col])
feature_cols = train.columns.tolist()

feature_cols.remove('ID')

feature_cols.remove('Crop_Damage')

feature_cols.remove('ID_value')

feature_cols.remove('date')

print(feature_cols)
cat_cols = ['Crop_Type', 'Soil_Type', 'Pesticide_Use_Category', 'Season', 'Crop_Type_lag1', 'Soil_Type_lag1', 'Pesticide_Use_Category_lag1', 'Season_lag1']
params = {}

params['learning_rate'] = 0.04

params['max_depth'] = 18

params['n_estimators'] = 3000

params['objective'] = 'multiclass'

params['boosting_type'] = 'gbdt'

params['subsample'] = 0.7

params['random_state'] = 42

params['colsample_bytree']=0.7

params['min_data_in_leaf'] = 55

params['reg_alpha'] = 1.7

params['reg_lambda'] = 1.11

params['class_weight']: {0: 0.44, 1: 0.4, 2: 0.37}
clf = lgb.LGBMClassifier(**params)

    

clf.fit(df_train[feature_cols], df_train[label_col], early_stopping_rounds=100, eval_set=[(df_train[feature_cols], df_train[label_col]), (df_eval[feature_cols], df_eval[label_col])], eval_metric='multi_error', verbose=True, categorical_feature=cat_cols)



eval_score = accuracy_score(df_eval[label_col], clf.predict(df_eval[feature_cols]))



print('Eval ACC: {}'.format(eval_score))
#Getting best iteration 

best_iter = clf.best_iteration_

params['n_estimators'] = best_iter

print(params)
clf = lgb.LGBMClassifier(**params)



clf.fit(train[feature_cols], train[label_col], eval_metric='multi_error', verbose=False, categorical_feature=cat_cols)



# eval_score_auc = roc_auc_score(df_train[label_col], clf.predict(df_train[feature_cols]))

eval_score_acc = accuracy_score(train[label_col], clf.predict(train[feature_cols]))



print('ACC: {}'.format(eval_score_acc))
preds = clf.predict(test[feature_cols])
Counter(train['Crop_Damage'])
Counter(preds)
submission = pd.DataFrame({'ID':test['ID'], 'Crop_Damage':preds})
plt.rcParams['figure.figsize'] = (12, 6)

lgb.plot_importance(clf)

plt.show()
submission.to_csv('lgbm.csv',index=False)