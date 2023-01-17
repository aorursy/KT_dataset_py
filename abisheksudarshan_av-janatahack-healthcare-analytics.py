#importing useful libraries

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from datetime import datetime

%matplotlib inline

from statsmodels.stats.outliers_influence import variance_inflation_factor
#reading csv

health_camp_detail=pd.read_csv('../input/janatahack-healthcare-analytics/Train/Health_Camp_Detail.csv')

patient_profile=pd.read_csv('../input/janatahack-healthcare-analytics/Train/Patient_Profile.csv')

fhc=pd.read_csv('../input/janatahack-healthcare-analytics/Train/First_Health_Camp_Attended.csv')

shc=pd.read_csv('../input/janatahack-healthcare-analytics/Train/Second_Health_Camp_Attended.csv')

thc=pd.read_csv('../input/janatahack-healthcare-analytics/Train/Third_Health_Camp_Attended.csv')

train=pd.read_csv('../input/janatahack-healthcare-analytics/Train/Train.csv')

test=pd.read_csv('../input/janatahack-healthcare-analytics/Test.csv')
health_camp_detail.info()
health_camp_detail.head(2)
#converting string date to datetime object

health_camp_detail['Camp_Start_Date']=health_camp_detail['Camp_Start_Date'].apply(lambda x:datetime.strptime(x,'%d-%b-%y'))

health_camp_detail['Camp_End_Date']=health_camp_detail['Camp_End_Date'].apply(lambda x:datetime.strptime(x,'%d-%b-%y'))



#adding suffix for easy identification during one-hot encoding

health_camp_detail['Category1']=health_camp_detail['Category1']+'_cat1'

health_camp_detail['Category2']=health_camp_detail['Category2']+'_cat2'

health_camp_detail['Category3']=health_camp_detail['Category3'].apply(lambda x:str(x)+'_cat3')
health_camp_detail.info()
health_camp_detail.head(2)
health_camp_detail['Health_Camp_ID'].value_counts()
sum(health_camp_detail['Health_Camp_ID'].value_counts()>1) 

# No duplicate health camps, should not cause any issue during join
health_camp_detail[health_camp_detail['Camp_End_Date']<health_camp_detail['Camp_Start_Date']]

# No instances of campaign end date < camp start date
#Camp_Start_Date

health_camp_detail['Camp_Start_Month']=health_camp_detail['Camp_Start_Date'].apply(lambda x:x.month)

health_camp_detail['Camp_Start_Day']=health_camp_detail['Camp_Start_Date'].apply(lambda x:x.day)

health_camp_detail['Camp_Start_Quarter']=health_camp_detail['Camp_Start_Date'].apply(lambda x:x.quarter)



#Camp_End_Date

health_camp_detail['Camp_End_Month']=health_camp_detail['Camp_End_Date'].apply(lambda x:x.month)

health_camp_detail['Camp_End_Day']=health_camp_detail['Camp_End_Date'].apply(lambda x:x.day)

health_camp_detail['Camp_End_Quarter']=health_camp_detail['Camp_End_Date'].apply(lambda x:x.quarter)



#Camp_Duration

health_camp_detail['Camp_Duration']=(health_camp_detail['Camp_End_Date']-health_camp_detail['Camp_Start_Date']).astype('timedelta64[D]')
health_camp_detail.head(2)
#Creating Dummies

#Category1

category1_dummies = pd.get_dummies(health_camp_detail['Category1'],drop_first=True)

health_camp_detail = pd.concat([health_camp_detail.drop('Category1',axis=1),category1_dummies],axis=1)

    

#Category2

category2_dummies = pd.get_dummies(health_camp_detail['Category2'],drop_first=True)

health_camp_detail = pd.concat([health_camp_detail.drop('Category2',axis=1),category2_dummies],axis=1)



#Category3

category3_dummies = pd.get_dummies(health_camp_detail['Category3'],drop_first=True)

health_camp_detail = pd.concat([health_camp_detail.drop('Category3',axis=1),category3_dummies],axis=1)



#Weekends

health_camp_detail['weekends_during_campaign']=[pd.date_range(x,y).weekday.isin([5,6]).sum() for x , y in zip(health_camp_detail['Camp_Start_Date'],health_camp_detail['Camp_End_Date'])]

    
health_camp_detail.head(2)
#camps by quarter

sns.countplot('Camp_Start_Quarter',data=health_camp_detail)
#camps by month

sns.countplot('Camp_Start_Month',data=health_camp_detail)
#camp duration distribution

sns.boxplot(health_camp_detail['Camp_Duration'])
#camp weekends distribution

sns.boxplot(health_camp_detail['weekends_during_campaign'])
patient_profile.info()
patient_profile.head(2)
patient_profile['First_Interaction']=patient_profile['First_Interaction'].apply(lambda x:datetime.strptime(x,'%d-%b-%y'))
#First_Interaction_Date

patient_profile['First_Interaction_Month']=patient_profile['First_Interaction'].apply(lambda x:x.month)

patient_profile['First_Interaction_Day']=patient_profile['First_Interaction'].apply(lambda x:x.day)

patient_profile['First_Interaction_Quarter']=patient_profile['First_Interaction'].apply(lambda x:x.quarter)



#Making Education Score & Age as np.nan

patient_profile['Education_Score']=patient_profile['Education_Score'].apply(lambda x: np.nan if x=='None' else x)

patient_profile['Age']=patient_profile['Age'].apply(lambda x: np.nan if x=='None' else x)

patient_profile['Education_Score']=pd.to_numeric(patient_profile['Education_Score'], downcast="float")

patient_profile['Age']=pd.to_numeric(patient_profile['Age'], downcast="float")



#Consolidating Online Interactions

patient_profile['Online_Interactions']=patient_profile['Facebook_Shared']+patient_profile['Twitter_Shared']+patient_profile['LinkedIn_Shared']+patient_profile['Online_Follower']
patient_profile['Facebook_Shared'].value_counts()
patient_profile['Online_Follower'].value_counts()
patient_profile['LinkedIn_Shared'].value_counts()
patient_profile['Twitter_Shared'].value_counts()
patient_profile['Online_Interactions'].value_counts()
patient_profile.drop(['Facebook_Shared','Twitter_Shared','LinkedIn_Shared','Online_Follower'],axis=1,inplace=True)
#Income

patient_profile['Income']=patient_profile['Income'].apply(lambda x:str(x)+'_inc')

patient_profile['Income'].value_counts()
#City_Type

patient_profile['City_Type'].value_counts()
patient_profile['City_Type']=patient_profile['City_Type'].fillna('None')
patient_profile['City_Type']=patient_profile['City_Type']+'_city'
#Creating Dummies

#Income

income_dummies = pd.get_dummies(patient_profile['Income'],drop_first=True)

patient_profile = pd.concat([patient_profile.drop('Income',axis=1),income_dummies],axis=1)



#City_Type

city_type_dummies = pd.get_dummies(patient_profile['City_Type'],drop_first=True)

patient_profile = pd.concat([patient_profile.drop('City_Type',axis=1),city_type_dummies],axis=1)
patient_profile.head(2)
sum(patient_profile['Patient_ID'].value_counts()>1)

#No duplicates for patients
#Age

patient_profile['Age'].value_counts()
sns.distplot(patient_profile[(patient_profile['Age']!='None')]['Age'])
patient_profile['Age'].mean(),patient_profile['Age'].median()
#Education

patient_profile['Education_Score'].value_counts()
sns.distplot(patient_profile[(patient_profile['Education_Score']!='None')]['Education_Score'])
patient_profile['Education_Score'].mean(),patient_profile['Education_Score'].median()
patient_profile['Employer_Category'].value_counts()
fhc.head(2)
fhc.info()
fhc.drop('Unnamed: 4',inplace=True,axis=1)
shc.head(2)
shc.info()
thc.head(2)
thc.info()
train.head()
train.info()
#dropping 334 records with not registration date

train=train.dropna()
train.isnull().sum()
test.head(2)
test.info()
#converting string date to datetime object

train['Registration_Date']=train['Registration_Date'].apply(lambda x:datetime.strptime(x,'%d-%b-%y'))

test['Registration_Date']=test['Registration_Date'].apply(lambda x:datetime.strptime(x,'%d-%b-%y'))



train['Registration_Quarter']=train['Registration_Date'].apply(lambda x:x.quarter)

test['Registration_Quarter']=test['Registration_Date'].apply(lambda x:x.quarter)



train['Registration_Month']=train['Registration_Date'].apply(lambda x:x.month)

test['Registration_Month']=test['Registration_Date'].apply(lambda x:x.month)



train['Registration_Day']=train['Registration_Date'].apply(lambda x:x.day)

test['Registration_Day']=test['Registration_Date'].apply(lambda x:x.day)
train.head(2)
test.head(2)
#merging health camp details

train=pd.merge(train,health_camp_detail,how='left')

test=pd.merge(test,health_camp_detail,how='left')



#merging patient details

train=pd.merge(train,patient_profile,how='left')

test=pd.merge(test,patient_profile,how='left')



#merging fhc details

train = pd.merge(train, fhc,  how='left', left_on=['Patient_ID','Health_Camp_ID'], right_on = ['Patient_ID','Health_Camp_ID'])



#merging shc details

train = pd.merge(train, shc,  how='left', left_on=['Patient_ID','Health_Camp_ID'], right_on = ['Patient_ID','Health_Camp_ID'])



#merging thc details

train = pd.merge(train, thc,  how='left', left_on=['Patient_ID','Health_Camp_ID'], right_on = ['Patient_ID','Health_Camp_ID'])



#creating outcome value

#creating binary column for first health camp

train['fhc_outcome']=train['Health_Score'].apply(lambda x: 1 if x>0 else 0 )



#creating binary column for second health camp

train['shc_outcome']=train['Health Score'].apply(lambda x: 1 if x>0 else 0 )



#creating binary column for third health camp

train['thc_outcome']=train['Number_of_stall_visited'].apply(lambda x: 1 if x>0 else 0 )



#overall outcome

train['overall_outcome']=train['fhc_outcome']+train['shc_outcome']+train['thc_outcome']



train.drop(['fhc_outcome','shc_outcome','thc_outcome','Health_Score','Health Score','Donation','Number_of_stall_visited','Last_Stall_Visited_Number'],axis=1,inplace=True)

train['overall_outcome'].value_counts()
train.head(2)
train.info()
#checking missing values

train.isnull().sum()
test.isnull().sum()
train_avg_edu=train['Education_Score'].mean()

train_avg_age=train['Age'].mean()

train['Education_Score']=train['Education_Score'].fillna(train_avg_edu)

train['Age']=train['Age'].fillna(train_avg_age)



train.isnull().sum()
test_avg_edu=test['Education_Score'].mean()

test_avg_age=test['Age'].mean()

test['Education_Score']=test['Education_Score'].fillna(test_avg_edu)

test['Age']=test['Age'].fillna(test_avg_age)



test.isnull().sum()
#Replace column as needed to visualize

plt.figure(figsize=(10,6))

train[train['overall_outcome']==1]['Second_cat1'].hist(alpha=0.5,color='blue',

                                              bins=15,label='Outcome=1')

train[train['overall_outcome']==0]['Second_cat1'].hist(alpha=0.5,color='red',

                                              bins=15,label='Outcome=0')

plt.legend()
sns.scatterplot(x='Age',y='Education_Score',data=train,hue='overall_outcome')
#Camp_Start_Date, Camp_End_Date, Registration_Date, First_Interaction

train['regis_cs']=(train['Camp_Start_Date']-train['Registration_Date']).astype('timedelta64[D]')

train['regis_ce']=(train['Camp_End_Date']-train['Registration_Date']).astype('timedelta64[D]')

train['regis_fi']=(train['Registration_Date']-train['First_Interaction']).astype('timedelta64[D]')



test['regis_cs']=(test['Camp_Start_Date']-test['Registration_Date']).astype('timedelta64[D]')

test['regis_ce']=(test['Camp_End_Date']-test['Registration_Date']).astype('timedelta64[D]')

test['regis_fi']=(test['Registration_Date']-test['First_Interaction']).astype('timedelta64[D]')



train['fi_cs']=(train['Camp_Start_Date']-train['First_Interaction']).astype('timedelta64[D]')

train['fi_ce']=(train['Camp_End_Date']-train['First_Interaction']).astype('timedelta64[D]')



test['fi_cs']=(test['Camp_Start_Date']-test['First_Interaction']).astype('timedelta64[D]')

test['fi_ce']=(test['Camp_End_Date']-test['First_Interaction']).astype('timedelta64[D]')
#regis_cs

plt.figure(figsize=(20,6))

train[train['overall_outcome']==1]['regis_cs'].hist(alpha=0.5,color='blue',

                                              bins=15,label='Outcome=1')

train[train['overall_outcome']==0]['regis_cs'].hist(alpha=0.5,color='red',

                                              bins=15,label='Outcome=0')

plt.legend()
#regis_ce

plt.figure(figsize=(20,6))

train[train['overall_outcome']==1]['regis_ce'].hist(alpha=0.5,color='blue',

                                              bins=15,label='Outcome=1')

train[train['overall_outcome']==0]['regis_ce'].hist(alpha=0.5,color='red',

                                              bins=15,label='Outcome=0')

plt.legend()
#regis_fi

plt.figure(figsize=(20,6))

train[train['overall_outcome']==1]['regis_fi'].hist(alpha=0.5,color='blue',

                                              bins=15,label='Outcome=1')

train[train['overall_outcome']==0]['regis_fi'].hist(alpha=0.5,color='red',

                                              bins=15,label='Outcome=0')

plt.legend()
#fi_cs

plt.figure(figsize=(20,6))

train[train['overall_outcome']==1]['fi_cs'].hist(alpha=0.5,color='blue',

                                              bins=15,label='Outcome=1')

train[train['overall_outcome']==0]['fi_cs'].hist(alpha=0.5,color='red',

                                              bins=15,label='Outcome=0')

plt.legend()
#fi_ce

plt.figure(figsize=(20,6))

train[train['overall_outcome']==1]['fi_ce'].hist(alpha=0.5,color='blue',

                                              bins=15,label='Outcome=1')

train[train['overall_outcome']==0]['fi_ce'].hist(alpha=0.5,color='red',

                                              bins=15,label='Outcome=0')

plt.legend()
train['nos_hc_per_patient']=train.groupby('Patient_ID')['Health_Camp_ID'].transform('nunique')

train['nos_pat_per_health']=train.groupby('Health_Camp_ID')['Patient_ID'].transform('nunique')



test['nos_hc_per_patient']=test.groupby('Patient_ID')['Health_Camp_ID'].transform('nunique')

test['nos_pat_per_health']=test.groupby('Health_Camp_ID')['Patient_ID'].transform('nunique')
#nos_hc_per_patient

plt.figure(figsize=(20,6))

train[train['overall_outcome']==1]['nos_hc_per_patient'].hist(alpha=0.5,color='blue',

                                              bins=15,label='Outcome=1')

train[train['overall_outcome']==0]['nos_hc_per_patient'].hist(alpha=0.5,color='red',

                                              bins=15,label='Outcome=0')

plt.legend()
#nos_pat_per_health

plt.figure(figsize=(20,6))

train[train['overall_outcome']==1]['nos_pat_per_health'].hist(alpha=0.5,color='blue',

                                              bins=15,label='Outcome=1')

train[train['overall_outcome']==0]['nos_pat_per_health'].hist(alpha=0.5,color='red',

                                              bins=15,label='Outcome=0')

plt.legend()
corr_train=train.drop(['Registration_Date','Camp_Start_Date','Camp_End_Date','Patient_ID','Health_Camp_ID','First_Interaction','Employer_Category'],axis=1).corr()

corr_train.reset_index(inplace=True)

corr_train=corr_train[['index','overall_outcome']]

corr_train['overall_outcome']=corr_train['overall_outcome'].apply(lambda x:abs(x))

corr_train.sort_values(by='overall_outcome',ascending=False,inplace=True)
independent_variables=corr_train.iloc[1:len(corr_train)//2]['index'] #taking 50% of variables

independent_variables
def calc_vif(X):



    # Calculating VIF

    vif = pd.DataFrame()

    vif["variables"] = [independent_variables.iloc[i] for i in range(len(independent_variables))]

    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]



    return(vif)
calc_vif(train[independent_variables].dropna())
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,Dropout

from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import roc_auc_score
X=train[independent_variables]

y=train['overall_outcome']
X.info()
model = Sequential()



# https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw



# input layer

model.add(Dense(26,  activation='relu'))

model.add(Dropout(0.5))



# hidden layer

model.add(Dense(13,activation='relu'))

model.add(Dropout(0.5))



# output layer

model.add(Dense(units=1,activation='sigmoid'))



# Compile model

model.compile(loss='binary_crossentropy', optimizer='adam')
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

model.fit(x=X_train, 

          y=y_train, 

          epochs=500,

          batch_size=256,

          validation_data=(X_test, y_test),

          callbacks=[early_stop],

          verbose=1

          )
model_loss = pd.DataFrame(model.history.history)

model_loss.plot()
predictions = model.predict_proba(X_test)
roc_auc_score(y_test, predictions)
sns.distplot(predictions,kde=False)
test_scaled=scaler.transform(test[independent_variables])
predictions = model.predict_proba(test_scaled)
test['Outcome']=predictions
output=test[['Patient_ID','Health_Camp_ID','Outcome']]
output.to_csv('nn.csv',index=False)
preds = 0

for seed_val in [1,3,10,15,20,33,333,1997,2020,2021]:

    print (seed_val)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=seed_val)

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)

    X_test = scaler.transform(X_test)

    model.fit(x=X_train, 

              y=y_train, 

              epochs=500,

              batch_size=256,

              validation_data=(X_test, y_test),

              callbacks=[early_stop],

              verbose=1

              )

    test_scaled=scaler.transform(test[independent_variables])

    predictions = model.predict_proba(test_scaled)

    preds += predictions

preds = preds/10
sub = pd.DataFrame({"Patient_ID":test.Patient_ID.values})

sub["Health_Camp_ID"] = test.Health_Camp_ID.values

sub["Outcome"] =  preds

sub.to_csv("nn_blended.csv", index=False)
from lightgbm import LGBMClassifier
tr=train[train['Camp_Start_Date'] <'2005-11-01']

val=train[train['Camp_Start_Date'] >'2005-10-30']
clf = LGBMClassifier(n_estimators=450,

                     learning_rate=0.03,

                     random_state=1,

                     colsample_bytree=0.5,

                     reg_alpha=2,

                     reg_lambda=2)



clf.fit(tr[independent_variables], tr['overall_outcome'], eval_set=[(val[independent_variables], val['overall_outcome'])], verbose=50,

        eval_metric = 'auc', early_stopping_rounds = 100)
preds = 0

for seed_val in [1,3,10,15,20,33,333,1997,2020,2021]:

    print (seed_val)

    m=LGBMClassifier(n_estimators=450,learning_rate=0.03,random_state=seed_val,colsample_bytree=0.5,reg_alpha=2,reg_lambda=2)

    m.fit(train[independent_variables],train['overall_outcome'])

    predict=m.predict_proba(test[independent_variables])[:,1]

    preds += predict

preds = preds/10
sub = pd.DataFrame({"Patient_ID":test.Patient_ID.values})

sub["Health_Camp_ID"] = test.Health_Camp_ID.values

sub["Outcome"] =  preds

sub.to_csv("lgbm_blending1.csv", index=False)
train.sort_values(by=['Patient_ID','Registration_Date'],inplace=True)

train['days_since_last_registration'] = train.groupby('Patient_ID')['Registration_Date'].diff().apply(lambda x: x.days)

train['days_since_next_registration'] = train.groupby('Patient_ID')['Registration_Date'].diff(-1) * (-1) / np.timedelta64(1, 'D')       



test.sort_values(by=['Patient_ID','Registration_Date'],inplace=True)

test['days_since_last_registration'] = test.groupby('Patient_ID')['Registration_Date'].diff().apply(lambda x: x.days)

test['days_since_next_registration'] = test.groupby('Patient_ID')['Registration_Date'].diff(-1) * (-1) / np.timedelta64(1, 'D')
def agg_numeric(df, parent_var, df_name):

    """

    Groups and aggregates the numeric values in a child dataframe

    by the parent variable.

    

    Parameters

    --------

        df (dataframe): 

            the child dataframe to calculate the statistics on

        parent_var (string): 

            the parent variable used for grouping and aggregating

        df_name (string): 

            the variable used to rename the columns

        

    Return

    --------

        agg (dataframe): 

            a dataframe with the statistics aggregated by the `parent_var` for 

            all numeric columns. Each observation of the parent variable will have 

            one row in the dataframe with the parent variable as the index. 

            The columns are also renamed using the `df_name`. Columns with all duplicate

            values are removed. 

    

    """

    

            

    # Only want the numeric variables

    parent_ids = df[parent_var].copy()

    numeric_df = df.select_dtypes('number').drop(columns={'Patient_ID', 'Health_Camp_ID','Second_cat1', 'Third_cat1', 'B_cat2', 'C_cat2', 'D_cat2', 'E_cat2',

       'F_cat2', 'G_cat2', '2_cat3','1_inc', '2_inc', '3_inc', '4_inc', '5_inc',

       '6_inc', 'None_inc', 'B_city', 'C_city', 'D_city', 'E_city', 'F_city',

       'G_city', 'H_city', 'I_city', 'None_city'}).copy()

    numeric_df[parent_var] = parent_ids



    # Group by the specified variable and calculate the statistics

    agg = numeric_df.groupby(parent_var).agg(['count', 'mean', 'max', 'min', 'sum'])



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
PID_aggregate = agg_numeric(train.drop('overall_outcome',axis=1), 'Patient_ID', 'agg')

print('PID aggregate shape: ', PID_aggregate.shape)

train=train.merge(PID_aggregate, on ='Patient_ID', how = 'left')



PID_aggregate = agg_numeric(test, 'Patient_ID', 'agg')

print('PID aggregate shape: ', PID_aggregate.shape)

test=test.merge(PID_aggregate, on ='Patient_ID', how = 'left')
preds = 0

for seed_val in [1,3,10,15,20,33,333,1997,2020,2021]:

    print (seed_val)

    m=LGBMClassifier(n_estimators=450,learning_rate=0.03,random_state=seed_val,colsample_bytree=0.5,reg_alpha=2,reg_lambda=2)

    m.fit(train[independent_variables],train['overall_outcome'])

    predict=m.predict_proba(test[independent_variables])[:,1]

    preds += predict

preds = preds/10
sub = pd.DataFrame({"Patient_ID":test.Patient_ID.values})

sub["Health_Camp_ID"] = test.Health_Camp_ID.values

sub["Outcome"] =  preds

sub.to_csv("lgbm_blending2.csv", index=False)