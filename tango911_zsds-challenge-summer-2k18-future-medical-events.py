import numpy as np 
import pandas as pd 
train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
train.head()
train=train.drop(['Age','Gender'],axis=1)
# Getting Year from Date
def get_year(x):
    date = str(x)
    year = date[:4]
    return year
# Adding the Year column to Train Data
train['Year'] = train.apply(lambda x : get_year(x['Date']), axis=1)
# Getting Month from Date
def get_month(x):
    date = str(x)
    month = date[4:]
    return month
# Adding the Month column to Train Data
train['Month'] = train.apply(lambda x : get_month(x['Date']), axis=1)
train.head()
pid_list = set(train['UID'])
test_df = pd.DataFrame()
for item in pid_list:
    df = train[train['Year']=='2013']
    df1 = df[df['UID']==item]
    if df1.shape[0]<10:
        df1_month = set(df1['Month'])
        df2 = train[train['Year']=='2012']
        df3 = df2[df2['UID']==item]
        df3_month = set(df3['Month'])
        difference_month = df3_month.difference(df1_month)
        for m in difference_month:
            df4 = df3[df3['Month']==m]
            df1 = df1.append(df4)
    if df1.shape[0]<10:
        df1_month = set(df1['Month'])
        df2 = train[train['Year']=='2011']
        df3 = df2[df2['UID']==item]
        df3_month = set(df3['Month'])
        difference_month = df3_month.difference(df1_month)
        for m in difference_month:
            df4 = df3[df3['Month']==m]
            df1 = df1.append(df4)
        
    test_df = test_df.append(df1)  
    
train = test_df
train.head(5)
# Probability for Event P(E)

new_train_df = train.groupby(["UID","Event_Code"]).size().reset_index(name="count_of_event_for_patient")
trial_train = train.merge(new_train_df, on = ['UID','Event_Code'])

new_train_df1 = train.groupby(["UID"]).size().reset_index(name="total_events_for_patient") 
trial_train = trial_train.merge(new_train_df1, on = ['UID'])

trial_train['prob_of_event'] = trial_train['count_of_event_for_patient'] / trial_train['total_events_for_patient']

train = trial_train
train.head()
# Probability for Month P(M)

new_train_df = train.groupby(["UID","Month"]).size().reset_index(name="count_of_month_for_patient")
trial_train = train.merge(new_train_df, on = ['UID','Month'])

new_train_df1 = train.groupby(["UID"]).size().reset_index(name="total_months_for_patient") 
trial_train = trial_train.merge(new_train_df1, on = ['UID'])

trial_train['prob_of_month'] = trial_train['count_of_month_for_patient'] / trial_train['total_months_for_patient']
train = trial_train
# Probability of Month Given the Event P(M/E)

new_train_df = train.groupby(["UID","Event_Code","Month"]).size().reset_index(name="count_of_month_and_event_for_patient")
trial_train = train.merge(new_train_df, on = ['UID','Event_Code','Month'])

new_train_df1 = train.groupby(["UID"]).size().reset_index(name="total_event_for_patient_when_month") 
trial_train = trial_train.merge(new_train_df1, on = ['UID'])

trial_train['prob_of_month_when_event'] = trial_train['count_of_month_and_event_for_patient'] / trial_train['total_event_for_patient_when_month']
train = trial_train
# Computing the Probability
train['prob_of_occurrence'] = (train['prob_of_month_when_event']*train['prob_of_event']) / train['prob_of_month']
new_sort = train
# Extracting the Top 10 probability Events
freq_events1 = pd.crosstab(index=[new_sort['UID']],columns=new_sort['Event_Code'], values = new_sort['prob_of_occurrence'], aggfunc=np.mean)
freq_events1.fillna(0)
freq_events1.reset_index(drop=False, inplace=True)

submit = freq_events1.loc[:,freq_events1.columns != 'UID'].apply(lambda x: pd.Series(x.sort_values(ascending=False).iloc[:10].index, index=['Event'+str(x) for x in range(1,11)]),axis=1).reset_index()
submit.drop('index',inplace=True, axis=1)

submit['UID'] = freq_events1['UID']

cols = submit.columns.tolist()
cols = cols[-1:] + cols[:-1]
submit = submit[cols]
submit
# final submission
submit.to_csv("submission.csv", index=False)