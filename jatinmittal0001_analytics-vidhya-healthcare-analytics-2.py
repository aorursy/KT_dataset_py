# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from catboost import CatBoostClassifier

import catboost

from catboost import *

import numpy as np

from sklearn.metrics import roc_auc_score

import seaborn as sns

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from lightgbm import LGBMModel,LGBMClassifier

from sklearn.model_selection import train_test_split,StratifiedShuffleSplit,StratifiedKFold



from sklearn.preprocessing import StandardScaler, RobustScaler



'''

https://datahack.analyticsvidhya.com/contest/janatahack-healthcare-analytics-ii/#ProblemStatement

'''



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("../input/janatahack-healthcare-analytics-part-2/train.csv")

test = pd.read_csv("../input/janatahack-healthcare-analytics-part-2/test.csv")

train_data_dict = pd.read_csv("../input/janatahack-healthcare-analytics-part-2/train_data_dict.csv")

sample_sub = pd.read_csv("../input/janatahack-healthcare-analytics-part-2/sample_submission.csv")

print('Shape of raw train data: ',train.shape)

print('Shape of raw test data: ',test.shape)

print(train.columns)

train.head(5)
check = train.drop(['case_id','patientid'],axis=1)



dup_index_list = check[check.duplicated()].index   # duplicaet rows after removing case ID, patient ID, 152 such observations
train.drop(train.index[dup_index_list],inplace=True)

train.reset_index(drop=True, inplace=True)

train_data_dict
print('Num unique in Hospital_code: ',train.Hospital_code.nunique())

print('Num unique in Hospital_type_code: ',train.Hospital_type_code.nunique())

print('Num unique in City_Code_Hospital: ',train.City_Code_Hospital.nunique())

print('Num unique in Hospital_region_code: ',train.Hospital_region_code.nunique())

print('Num unique in Department: ',train.Department.nunique())

print('Num unique in Ward_Type: ',train.Ward_Type.nunique())



print('Num unique in Ward_Facility_Code: ',train.Ward_Facility_Code.nunique())

print('Num unique in Bed Grade: ',train['Bed Grade'].nunique())

print('Num unique in City_Code_Patient: ',train.City_Code_Patient.nunique())

print('Num unique in patientid: ',train.patientid.nunique())



print('Num unique in Type of Admission: ',train['Type of Admission'].nunique())

print('Num unique in Available Extra Rooms in Hospital: ',train['Available Extra Rooms in Hospital'].nunique())

print('Num unique in Severity of Illness: ',train['Severity of Illness'].nunique())



print('Num unique in Age: ',train['Age'].nunique())

print('Num unique in Visitors with Patient: ',train['Visitors with Patient'].nunique())
test['Stay'] = 'test_data'

total = train.append(test)

total.reset_index(inplace=True,drop=True)
test['patientid'].nunique()
total.columns
total['Bed Grade'] = total.groupby(['Department','Hospital_code','Ward_Type'])["Bed Grade"].apply(lambda x: x.fillna(x.value_counts().index[0]))



idx = total['City_Code_Patient'].isnull( )

total['City_Code_Patient'][ idx ] = 17 #Taking distinct values of this variable you will see all values from 1 to 37 but 17 would be missing. SO I am imputing it with 17
total['Bed Grade']=total['Bed Grade'].astype(int)

total['City_Code_Patient']=total['City_Code_Patient'].astype(int)

pd.crosstab(train['Type of Admission'],train['Severity of Illness'])
sns.scatterplot(y='Ward_Facility_Code', x='Admission_Deposit', data=total)    # to plot scatter graph between two continuous variables
sns.boxplot(x = total.Admission_Deposit)
sns.countplot(x="Stay", data=train)
def outlier_treatment(data,p1,p99):

    data_X = data.copy()

    col = "Admission_Deposit"

    data_X[col][data_X[col] <= p1] = p1

    data_X[col][data_X[col] >= p99] = p99

    

  

    return data_X



a = train["Admission_Deposit"].quantile([0.25,0.75]).values

p_cap = a[1] + 1.5*(a[1]-a[0])

p_clip = a[0] - 1.5*(a[1]-a[0])



total = outlier_treatment(total,p_clip,p_cap)
total.columns
# varibale to check if a patient is from same city as that of hospital

def row_same_city(a,b):

    if a ==b:

        return 1

    

    return 0



total['in_same_city'] = total.apply(lambda row: row_same_city(row['City_Code_Hospital'],row['City_Code_Patient']),axis=1)
total['Avg_deposit_1'] = total.groupby(['Hospital_code','Department', 'Ward_Type','Ward_Facility_Code','Bed Grade'])['Admission_Deposit'].transform('mean')

total['Avg_deposit_2'] = total.groupby(['Hospital_code'])['Admission_Deposit'].transform('mean')

total['Avg_deposit_3'] = total.groupby(['Department', 'Ward_Type','Ward_Facility_Code','Bed Grade'])['Admission_Deposit'].transform('mean')

total['Avg_deposit_4'] = total.groupby(['Ward_Type'])['Admission_Deposit'].transform('mean')

total['Avg_deposit_5'] = total.groupby(['Ward_Type','Bed Grade'])['Admission_Deposit'].transform('mean')

total['Avg_deposit_8'] = total.groupby(['Age'])['Admission_Deposit'].transform('mean')

total['Avg_deposit_9'] = total.groupby(['Ward_Type','Ward_Facility_Code'])['Admission_Deposit'].transform('mean')



'''

total['Median_deposit_1'] = total.groupby(['Hospital_code','Department', 'Ward_Type','Ward_Facility_Code','Bed Grade'])['Admission_Deposit'].transform('median')

total['Median_deposit_2'] = total.groupby(['Hospital_code'])['Admission_Deposit'].transform('median')

total['Median_deposit_3'] = total.groupby(['Department', 'Ward_Type','Ward_Facility_Code','Bed Grade'])['Admission_Deposit'].transform('median')

total['Median_deposit_4'] = total.groupby(['Ward_Type'])['Admission_Deposit'].transform('median')

total['Median_deposit_5'] = total.groupby(['Ward_Type','Bed Grade'])['Admission_Deposit'].transform('median')



total['Median_deposit_6'] = total.groupby(['Hospital_code','Department', 'Ward_Type','Ward_Facility_Code','Bed Grade'])['Admission_Deposit'].transform('median')

total['Median_deposit_7'] = total.groupby(['Hospital_code'])['Admission_Deposit'].transform('median')

total['Median_deposit_8'] = total.groupby(['Department', 'Ward_Type','Ward_Facility_Code','Bed Grade'])['Admission_Deposit'].transform('median')

total['Median_deposit_9'] = total.groupby(['Ward_Type'])['Admission_Deposit'].transform('median')

total['Median_deposit_10'] = total.groupby(['Ward_Type','Bed Grade'])['Admission_Deposit'].transform('median')



total['max_deposit_1'] = total.groupby(['Hospital_code','Department', 'Ward_Type','Ward_Facility_Code','Bed Grade'])['Admission_Deposit'].transform('max')

total['max_deposit_2'] = total.groupby(['Hospital_code'])['Admission_Deposit'].transform('max')

total['max_deposit_3'] = total.groupby(['Department', 'Ward_Type','Ward_Facility_Code','Bed Grade'])['Admission_Deposit'].transform('max')

total['max_deposit_4'] = total.groupby(['Ward_Type'])['Admission_Deposit'].transform('max')

total['max_deposit_5'] = total.groupby(['Ward_Type','Bed Grade'])['Admission_Deposit'].transform('max')



total['max_deposit_6'] = total.groupby(['Hospital_code','Department', 'Ward_Type','Ward_Facility_Code','Bed Grade'])['Admission_Deposit'].transform('max')

total['max_deposit_7'] = total.groupby(['Hospital_code'])['Admission_Deposit'].transform('max')

total['max_deposit_8'] = total.groupby(['Department', 'Ward_Type','Ward_Facility_Code','Bed Grade'])['Admission_Deposit'].transform('max')

total['max_deposit_9'] = total.groupby(['Ward_Type'])['Admission_Deposit'].transform('max')

total['max_deposit_10'] = total.groupby(['Ward_Type','Bed Grade'])['Admission_Deposit'].transform('max')



'''



total['min_deposit_1'] = total.groupby(['Hospital_code','Department', 'Ward_Type','Ward_Facility_Code','Bed Grade'])['Admission_Deposit'].transform('min')

total['min_deposit_2'] = total.groupby(['Hospital_code'])['Admission_Deposit'].transform('min')

total['min_deposit_3'] = total.groupby(['Department', 'Ward_Type','Ward_Facility_Code','Bed Grade'])['Admission_Deposit'].transform('min')

total['min_deposit_4'] = total.groupby(['Ward_Type'])['Admission_Deposit'].transform('min')

total['min_deposit_5'] = total.groupby(['Ward_Type','Bed Grade'])['Admission_Deposit'].transform('min')



total['min_deposit_6'] = total.groupby(['Hospital_code','Department', 'Ward_Type','Ward_Facility_Code','Bed Grade'])['Admission_Deposit'].transform('min')



'''

total['min_deposit_7'] = total.groupby(['Hospital_code'])['Admission_Deposit'].transform('min')

'''



total['min_deposit_8'] = total.groupby(['Department', 'Ward_Type','Ward_Facility_Code','Bed Grade'])['Admission_Deposit'].transform('min')

total['min_deposit_9'] = total.groupby(['Ward_Type'])['Admission_Deposit'].transform('min')

total['min_deposit_10'] = total.groupby(['Ward_Type','Bed Grade'])['Admission_Deposit'].transform('min')







total['num_visit_emergency_gp_PDA'] = (total[total['Type of Admission']=='Emergency'].groupby(['patientid','Department','Type of Admission'])['patientid'].transform('count'))

total['num_visit_emergency_gp_PDA'].fillna(0,inplace=True)



total['num_visit_Trauma_gp_PDA'] = (total[total['Type of Admission']=='Trauma'].groupby(['patientid','Department','Type of Admission'])['patientid'].transform('count'))

total['num_visit_Trauma_gp_PDA'].fillna(0,inplace=True)



total['num_visit_Urgent_gp_PDA'] = (total[total['Type of Admission']=='Urgent'].groupby(['patientid','Department','Type of Admission'])['patientid'].transform('count'))

total['num_visit_Urgent_gp_PDA'].fillna(0,inplace=True)



total['num_visit_emergency_gp_H'] = (total[total['Type of Admission']=='Emergency'].groupby(['Hospital_code'])['patientid'].transform('count'))

total['num_visit_emergency_gp_H'].fillna(0,inplace=True)



total['num_visit_Trauma_gp_H'] = (total[total['Type of Admission']=='Trauma'].groupby(['Hospital_code'])['patientid'].transform('count'))

total['num_visit_Trauma_gp_H'].fillna(0,inplace=True)



total['num_visit_Urgent_gp_H'] = (total[total['Type of Admission']=='Urgent'].groupby(['Hospital_code'])['patientid'].transform('count'))

total['num_visit_Urgent_gp_H'].fillna(0,inplace=True)



total['Avg_visitors_1'] = total.groupby(['Hospital_code','Department', 'Ward_Type','Ward_Facility_Code','Bed Grade'])['Visitors with Patient'].transform('mean')

total['Avg_visitors_2'] = total.groupby(['Hospital_code'])['Visitors with Patient'].transform('mean')

total['Avg_visitors_3'] = total.groupby(['Department', 'Ward_Type','Ward_Facility_Code','Bed Grade'])['Visitors with Patient'].transform('mean')

total['Avg_visitors_4'] = total.groupby(['Ward_Type'])['Visitors with Patient'].transform('mean')

total['Avg_visitors_5'] = total.groupby(['Ward_Type','Bed Grade'])['Visitors with Patient'].transform('mean')



total['Avg_visitors_6'] = total.groupby(['Hospital_code','Department', 'Ward_Type','Ward_Facility_Code','Bed Grade'])['Visitors with Patient'].transform('mean')

total['Avg_visitors_8'] = total.groupby(['Department', 'Ward_Type','Ward_Facility_Code','Bed Grade'])['Visitors with Patient'].transform('mean')



total['Median_visitors_1'] = total.groupby(['Hospital_code','Department', 'Ward_Type','Ward_Facility_Code','Bed Grade'])['Visitors with Patient'].transform('median')

total['Median_visitors_2'] = total.groupby(['Hospital_code'])['Visitors with Patient'].transform('median')

total['Median_visitors_3'] = total.groupby(['Department', 'Ward_Type','Ward_Facility_Code','Bed Grade'])['Visitors with Patient'].transform('median')

total['Median_visitors_4'] = total.groupby(['Ward_Type'])['Visitors with Patient'].transform('median')

total['Median_visitors_5'] = total.groupby(['Ward_Type','Bed Grade'])['Visitors with Patient'].transform('median')



total['Median_visitors_6'] = total.groupby(['Hospital_code','Department', 'Ward_Type','Ward_Facility_Code','Bed Grade'])['Visitors with Patient'].transform('median')

total['Median_visitors_7'] = total.groupby(['Hospital_code'])['Visitors with Patient'].transform('median')

total['Median_visitors_8'] = total.groupby(['Department', 'Ward_Type','Ward_Facility_Code','Bed Grade'])['Visitors with Patient'].transform('median')

total['Median_visitors_9'] = total.groupby(['Ward_Type'])['Visitors with Patient'].transform('median')

total['Median_visitors_10'] = total.groupby(['Ward_Type','Bed Grade'])['Visitors with Patient'].transform('median')



total['max_visitors_1'] = total.groupby(['Hospital_code','Department', 'Ward_Type','Ward_Facility_Code','Bed Grade'])['Visitors with Patient'].transform('max')

total['max_visitors_2'] = total.groupby(['Hospital_code'])['Visitors with Patient'].transform('max')

total['max_visitors_3'] = total.groupby(['Department', 'Ward_Type','Ward_Facility_Code','Bed Grade'])['Visitors with Patient'].transform('max')

total['max_visitors_4'] = total.groupby(['Ward_Type'])['Visitors with Patient'].transform('max')

total['max_visitors_5'] = total.groupby(['Ward_Type','Bed Grade'])['Visitors with Patient'].transform('max')



total['max_visitors_6'] = total.groupby(['Hospital_code','Department', 'Ward_Type','Ward_Facility_Code','Bed Grade'])['Visitors with Patient'].transform('max')

total['max_visitors_7'] = total.groupby(['Hospital_code'])['Visitors with Patient'].transform('max')

total['max_visitors_8'] = total.groupby(['Department', 'Ward_Type','Ward_Facility_Code','Bed Grade'])['Visitors with Patient'].transform('max')

total['max_visitors_9'] = total.groupby(['Ward_Type'])['Visitors with Patient'].transform('max')

total['max_visitors_10'] = total.groupby(['Ward_Type','Bed Grade'])['Visitors with Patient'].transform('max')



total['min_visitors_1'] = total.groupby(['Hospital_code','Department', 'Ward_Type','Ward_Facility_Code','Bed Grade'])['Visitors with Patient'].transform('min')

total['min_visitors_2'] = total.groupby(['Hospital_code'])['Visitors with Patient'].transform('min')

total['min_visitors_3'] = total.groupby(['Department', 'Ward_Type','Ward_Facility_Code','Bed Grade'])['Visitors with Patient'].transform('min')

total['min_visitors_4'] = total.groupby(['Ward_Type'])['Visitors with Patient'].transform('min')

total['min_visitors_5'] = total.groupby(['Ward_Type','Bed Grade'])['Visitors with Patient'].transform('min')



total['min_visitors_6'] = total.groupby(['Hospital_code','Department', 'Ward_Type','Ward_Facility_Code','Bed Grade'])['Visitors with Patient'].transform('min')

total['min_visitors_7'] = total.groupby(['Hospital_code'])['Visitors with Patient'].transform('min')

total['min_visitors_8'] = total.groupby(['Department', 'Ward_Type','Ward_Facility_Code','Bed Grade'])['Visitors with Patient'].transform('min')





total['type_of_ad_cnt_1'] = total.groupby(['Ward_Type','Ward_Facility_Code'])['Type of Admission'].transform('count')

total['type_of_ad_cnt_2'] = total.groupby(['Ward_Type'])['Type of Admission'].transform('count')

total['type_of_ad_cnt_3'] = total.groupby(['Ward_Facility_Code'])['Type of Admission'].transform('count')

total['type_of_ad_cnt_4'] = total.groupby(['Age'])['Type of Admission'].transform('count')

total['type_of_ad_cnt_5'] = total.groupby(['Age'])['Ward_Facility_Code'].transform('count')

total['type_of_ad_cnt_6'] = total.groupby(['Hospital_code'])['Age'].transform('count')



m1 = {'Minor':1, 'Moderate':2, 'Extreme':3}

m2 = {'Trauma':1, 'Urgent':2, 'Emergency':3}

m3 = {'0-10':1, '11-20':2, '21-30':3, '31-40':4, '41-50':5, '51-60':6, '61-70':7,

       '71-80':8, '81-90':9, '91-100':10}







total['Type of Admission'] = total['Type of Admission'].map(m2)

total['Severity of Illness'] = total['Severity of Illness'].map(m1)

total['Age'] = total['Age'].map(m3)
abc = pd.crosstab(total.in_same_city,total.Stay).drop(['test_data'],axis=1)

prob_of_same_city_to_stay = abc.iloc[:, :].apply(lambda x: x / x.sum(),axis=1)

prob_of_same_city_to_stay.columns = [str(col) + '_same_city_prob' for col in prob_of_same_city_to_stay.columns]

prob_of_same_city_to_stay.reset_index(inplace=True)

prob_of_same_city_to_stay.head(3)
abc = pd.crosstab(total.Ward_Facility_Code,total.Stay).drop(['test_data'],axis=1) #dropping test_data column will get us values only for train set

prob_of_facility_to_stay = abc.iloc[:, :].apply(lambda x: x / x.sum(),axis=1)

prob_of_facility_to_stay.columns = [str(col) + '_facility_prob' for col in prob_of_facility_to_stay.columns]

prob_of_facility_to_stay.reset_index(inplace=True)

prob_of_facility_to_stay.head(3)
abc = pd.crosstab(total.Age,total.Stay).drop(['test_data'],axis=1) #dropping test_data column will get us values only for train set

prob_of_age_to_stay = abc.iloc[:, :].apply(lambda x: x / x.sum(),axis=1)

prob_of_age_to_stay.columns = [str(col) + '_age_prob' for col in prob_of_age_to_stay.columns]

prob_of_age_to_stay.reset_index(inplace=True)

prob_of_age_to_stay.head(3)

abc = pd.crosstab(total['Severity of Illness'],total.Stay).drop(['test_data'],axis=1) #dropping test_data column will get us values only for train set*/



prob_of_sev_ill_to_stay = abc.iloc[:, :].apply(lambda x: x / x.sum(),axis=1)

prob_of_sev_ill_to_stay.columns = [str(col) + '_sever_prob' for col in prob_of_sev_ill_to_stay.columns]

prob_of_sev_ill_to_stay.reset_index(inplace=True)

prob_of_sev_ill_to_stay.head(3)
abc = pd.crosstab(total['Type of Admission'],total.Stay).drop(['test_data'],axis=1) #dropping test_data column will get us values only for train set*/



prob_of_type_of_ad_to_stay = abc.iloc[:, :].apply(lambda x: x / x.sum(),axis=1)

prob_of_type_of_ad_to_stay.columns = [str(col) + '_type_ad_prob' for col in prob_of_type_of_ad_to_stay.columns]

prob_of_type_of_ad_to_stay.reset_index(inplace=True)

prob_of_type_of_ad_to_stay.head(3)
abc2 = total.groupby(['Age','Severity of Illness']).agg({'Age': 'count'})



abc2 = abc2.groupby(level=0).apply(lambda x:100 * x / float(x.sum()))

abc2.rename(columns={'Age':'perc'},inplace=True)

abc2.reset_index(inplace=True)
total = total.merge(prob_of_age_to_stay, on='Age',how='left')



#total = total.merge(prob_of_sev_ill_to_stay, on='Severity of Illness',how='left')  #decreasing perf



#total = total.merge(prob_of_type_of_ad_to_stay, on='Type of Admission',how='left')  #decreasing perf



total = total.merge(prob_of_same_city_to_stay, on='in_same_city',how='left')



total = total.merge(prob_of_facility_to_stay, on='Ward_Facility_Code',how='left')



total = total.merge(abc2, on=['Age','Severity of Illness'],how='left') 
col_hot = ['Ward_Type', 'Ward_Facility_Code']



total = pd.get_dummies(total,columns=col_hot)  # creating one hot encoding varibales for only these two.
feat_to_drop = ['City_Code_Hospital',

 'is_outlier',

 'Avg_deposit_1',

 'Avg_deposit_3',

 'Avg_deposit_4',

 'Avg_deposit_10',

 'min_deposit_1',

 'min_deposit_2',

 'min_deposit_4',

 'min_deposit_5',

 'min_deposit_8',

 'min_deposit_9',

 'min_deposit_10',

 'num_visit_emergency_gp_H',

 'num_visit_Urgent_gp_H',

 'Avg_visitors_1',

 'Avg_visitors_3',

 'Avg_visitors_8',

 'Median_visitors_1',

 'Median_visitors_6',

 'Median_visitors_7',

 'Median_visitors_9',

 'Median_visitors_10',

 'max_visitors_3',

 'max_visitors_4',

 'max_visitors_6',

 'max_visitors_8',

 'max_visitors_9',

 'min_visitors_1',

 'min_visitors_2',

 'min_visitors_4',

 'min_visitors_5',

 'min_visitors_6',

 'min_visitors_7',

 'min_visitors_8','Median_visitors_8', 'max_visitors_7', 'max_visitors_10','Ward_Type_T', 'Ward_Type_U', 'Median_visitors_2','61-70_age_prob', 

                '71-80_age_prob', '91-100_age_prob',

       'More than 100 Days_age_prob', '11-20_sever_prob', '31-40_sever_prob',

       '41-50_sever_prob', '51-60_sever_prob', '61-70_sever_prob',

       '71-80_sever_prob', '81-90_sever_prob', '91-100_sever_prob',

       'More than 100 Days_sever_prob', 'Ward_Type_P', 'Ward_Type_R',

       'Ward_Type_U', 'Ward_Facility_Code_F',

       'Ward_Facility_Code_A','bin_separater', '11-20_same_city_prob', '21-30_same_city_prob',

       '31-40_same_city_prob', '41-50_same_city_prob', '51-60_same_city_prob',

       '61-70_same_city_prob', '71-80_same_city_prob', '81-90_same_city_prob',

       '91-100_same_city_prob', 'More than 100 Days_same_city_prob',

       '61-70_facility_prob', '71-80_facility_prob', '91-100_facility_prob',

       'More than 100 Days_facility_prob', 'Ward_Facility_Code_B',

       'Ward_Facility_Code_C', 'Ward_Facility_Code_D', 'Ward_Facility_Code_E']

'''

for i in feat_to_drop:

    if i in total.columns:

        total.drop(i,axis=1,inplace=True)

        

'''
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

obj = (total.dtypes=='object') + (total.dtypes=='category')

col = list(total.dtypes[obj].index)



col.remove('Stay')

for i in range(0,len(col)):

    l = col[i]

    total[l] = le.fit_transform(total[l])
corrmat = total[~(total['Stay']=='test_data')].drop(['Stay'],axis=1).corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);    #without exact values



# Threshold for removing correlated variables

threshold = 0.8



# Absolute value correlation matrix

corr_matrix = corrmat.abs()

corr_matrix.head()



# Upper triangle of correlations

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

upper.head()



# Select columns with correlations above threshold

to_drop = [column for column in upper.columns if any(upper[column] > threshold)]



print('There are %d columns to remove.' % (len(to_drop)))



# Drop correlated features

#total.drop(columns = to_drop,axis=1,inplace=True)
# percentage of target counts in the train set



(train['Stay'].value_counts())*100/train.shape[0]
X = total[~(total['Stay']=='test_data')].drop(['Stay'],axis=1)

X.drop(['case_id','patientid'],axis=1,inplace=True)

y = total[~(total['Stay']=='test_data')]['Stay']



final_test = total[(total['Stay']=='test_data')].drop(['Stay'],axis=1)

final_test.drop(['patientid'],axis=1,inplace=True)
final_test.head(3)
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=.30, random_state=150,stratify=y,shuffle=True)
X_train.columns
# categorical column 

cat_col=['Hospital_code','Hospital_type_code', 'Hospital_region_code', 'Department'

            ,'Bed Grade','City_Code_Patient','Type of Admission','Severity of Illness','Age']
from catboost import CatBoostClassifier

catb = CatBoostClassifier(

    iterations=10000,

   cat_features=cat_col,

    random_seed=4,

    learning_rate=0.05,

    #subsample=0.7,

    early_stopping_rounds=150,

    eval_metric='Accuracy',

    task_type='GPU'

)

catb.fit(

    X_train, y_train,

    eval_set=(X_validation, y_validation),verbose=200

)

print('Model is fitted: ' + str(catb.is_fitted()))

print('Model params:')

print(catb.get_params())
feat_importances = pd.Series(catb.feature_importances_, index=X_train.columns)

feat_importances.nlargest(20).plot(kind='barh')





p_train_catb_val = catb.predict(X_validation)

print(accuracy_score(y_validation, p_train_catb_val))



p_train_catb = catb.predict(X_train)

print(accuracy_score(y_train, p_train_catb))
               

case_ids=final_test['case_id'].reset_index(drop=True)



final_test_pred = catb.predict(final_test.drop(['case_id'],axis=1))



ak = pd.DataFrame(case_ids).reset_index(drop=True).merge(pd.DataFrame(final_test_pred,columns=["Stay"]).reset_index(drop=True), left_index=True, right_index=True)



ak = ak[['case_id','Stay']]

ak.reset_index(drop=True,inplace=True)

ak.to_csv('test_pred_catboost_wo_CV.csv', index=False)

ak.head()
from IPython.display import FileLink

FileLink('test_pred_catboost_wo_CV.csv')
%%time

##LightGBM



probs_final_test = np.zeros(shape=(len(final_test),11))



n_fold = 6

scores_cv_train = []

scores_cv_test = []



avg_loss = []



class_list = []





sssf = StratifiedShuffleSplit(n_splits=n_fold, test_size = 0.25 ,random_state=14)



for i, (idxT, idxV) in enumerate(sssf.split(X, y)):



    print('Fold',i)



    print(' rows of train =',len(idxT),'rows of holdout =',len(idxV))



    clf = CatBoostClassifier(

    iterations=10000,

   cat_features=cat_col,

    random_seed=4,

    learning_rate=0.05,

    #subsample=0.7,

    early_stopping_rounds=150,

    eval_metric='Accuracy',

    task_type='GPU'

)    



    clf.fit(X.iloc[idxT], y.iloc[idxT], 

                eval_set=[(X.iloc[idxV],y.iloc[idxV])],

                verbose=100)

    

    class_list = list(clf.classes_)





    probs_cv_train = clf.predict(X.iloc[idxT])

    score_train = accuracy_score( y.iloc[idxT], probs_cv_train)

    scores_cv_train.append(score_train)

    

    

    probs_cv_test = clf.predict(X.iloc[idxV])

    score_test = accuracy_score( y.iloc[idxV], probs_cv_test)

    scores_cv_test.append(score_test)

    



    probs_final_test  += clf.predict_proba(final_test.drop(['case_id'],axis=1))



    print('#'*100)



    if i==0:

        feat_importances = pd.Series(clf.feature_importances_, index=X_train.columns)

        feat_importances.nlargest(20).plot(kind='barh')

        

print(scores_cv_train)

print(scores_cv_test)


case_ids=final_test['case_id'].reset_index(drop=True)

maxInrows= np.argmax(probs_final_test/n_fold, axis=1)

a = class_list

final_test_pred = [a[idx] for idx in maxInrows]



ak = pd.DataFrame(case_ids).reset_index(drop=True).merge(pd.DataFrame(final_test_pred,columns=["Stay"]).reset_index(drop=True), left_index=True, right_index=True)



ak = ak[['case_id','Stay']]

ak.reset_index(drop=True,inplace=True)

ak.to_csv('test_pred_catboost_with_CV.csv', index=False)

ak.head()
from IPython.display import FileLink

FileLink('test_pred_catboost_with_CV.csv')
clf = LGBMClassifier( n_estimators=1000,

                         #objective ='multiclass',

                         feature_name =cat_col,max_depth=20,

                     eval_metric='multiclass',

                     

            )        



clf.fit(X_train, y_train, 

            eval_set=[(X_validation,y_validation)],

            verbose=100,early_stopping_rounds=50

           )

p_train_lgbm = clf.predict(X_train)

print("train score :",accuracy_score(y_train, p_train_lgbm))



p_train_lgbm_val = clf.predict(X_validation)

print("Validation score :",accuracy_score(y_validation, p_train_lgbm_val))
feat_importances = pd.Series(clf.feature_importances_, index=X_train.columns)

feat_importances.nlargest(20).plot(kind='barh')
               

case_ids=final_test['case_id'].reset_index(drop=True)



final_test_pred = clf.predict(final_test.drop(['case_id'],axis=1))



ak = pd.DataFrame(case_ids).reset_index(drop=True).merge(pd.DataFrame(final_test_pred,columns=["Stay"]).reset_index(drop=True), left_index=True, right_index=True)



ak = ak[['case_id','Stay']]

ak.reset_index(drop=True,inplace=True)

ak.to_csv('test_pred_lgb_wo_CV.csv', index=False)

ak.head()
%%time

##LightGBM



probs_final_test = np.zeros(shape=(len(final_test),11))



n_fold = 8

scores_cv_train = []

scores_cv_test = []



avg_loss = []



class_list = []





sssf = StratifiedShuffleSplit(n_splits=n_fold, test_size = 0.25 ,random_state=14)



for i, (idxT, idxV) in enumerate(sssf.split(X, y)):



    print('Fold',i)



    print(' rows of train =',len(idxT),'rows of holdout =',len(idxV))



    clf = LGBMClassifier(colsample_bytree=0.7, n_estimators=700,

                             objective ='multiclass',

                             feature_name =cat_col,

                max_depth=15,learning_rate=0.05

                )        



    clf.fit(X.iloc[idxT], y.iloc[idxT], 

                eval_set=[(X.iloc[idxV],y.iloc[idxV])],

                verbose=100,eval_metric=['multiclass'],

                early_stopping_rounds=30)

    

    class_list = list(clf.classes_)





    probs_cv_train = clf.predict(X.iloc[idxT])

    score_train = accuracy_score( y.iloc[idxT], probs_cv_train)

    scores_cv_train.append(score_train)

    

    

    probs_cv_test = clf.predict(X.iloc[idxV])

    score_test = accuracy_score( y.iloc[idxV], probs_cv_test)

    scores_cv_test.append(score_test)

    



    probs_final_test  += clf.predict_proba(final_test.drop(['case_id'],axis=1))



    print('#'*100)



    if i==0:

        feat_importances = pd.Series(clf.feature_importances_, index=X_train.columns)

        feat_importances.nlargest(20).plot(kind='barh')

        

print(scores_cv_train)

print(scores_cv_test)


case_ids=final_test['case_id'].reset_index(drop=True)

maxInrows= np.argmax(probs_final_test/n_fold, axis=1)

a = class_list

final_test_pred = [a[idx] for idx in maxInrows]



ak = pd.DataFrame(case_ids).reset_index(drop=True).merge(pd.DataFrame(final_test_pred,columns=["Stay"]).reset_index(drop=True), left_index=True, right_index=True)



ak = ak[['case_id','Stay']]

ak.reset_index(drop=True,inplace=True)

ak.to_csv('test_pred_lgb_with_CV.csv', index=False)

ak.head()