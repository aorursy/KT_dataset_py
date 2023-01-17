from glob import glob

import pandas as pd

import numpy as np

from sklearn import model_selection, preprocessing
# concat quartre files into one

original_csv_path = '../input/'

all_files = glob(original_csv_path + "/*.csv")

look_back_days = 5




df=pd.DataFrame()

for filename in all_files:

    print(filename)

    df_temp = pd.read_csv(filename, header=0,encoding = "ISO-8859-1", error_bad_lines=False)

    df_temp.dropna(axis=1, thresh=999999, inplace=True)

    df = pd.concat([df,df_temp], axis=0, ignore_index=True)

    

df_temp=True

hdd_df = df

hdd_df.drop(labels=['capacity_bytes', 'model'],axis=1,inplace=True)
display(hdd_df.head())

display(hdd_df.shape)



# remove columns that have NA (the new SMART readings that are not collected in previous quarters)

hdd_df.dropna(axis=1, thresh=999999, inplace=True)

display(hdd_df.shape)



hdd_df.dropna(axis=0, inplace=True)

display(hdd_df.shape)
# convert failure column to categorical

hdd_df['failure'] = hdd_df['failure'].astype(float).astype(int).astype(str)

hdd_df['failure'] = pd.Categorical(hdd_df['failure'])

hdd_df['date'] = pd.to_datetime(hdd_df['date'], format="%Y-%m-%d")
# sort by serial number and date

hdd_df.sort_values(by=['serial_number','date'], ascending=[True,False],inplace=True)
# take only the top <look_back_days> for each serial number

time_limited_df = hdd_df.groupby('serial_number').head(look_back_days)

time_limited_df.loc[:,'time_lag'] = time_limited_df.groupby('serial_number').cumcount()
# apply standard scaling on train and fit to test on each individual column

for column in list(time_limited_df.filter(regex='smart')):

    scaler = preprocessing.StandardScaler().fit(time_limited_df[[column]])

    time_limited_df.loc[:,column] = scaler.transform(time_limited_df[[column]])
#get the target feature

hdd_failure = time_limited_df.groupby('serial_number').head(1)[["serial_number" ,"failure"]]

time_roled_df = hdd_failure

for smart_metric in list(time_limited_df.filter(regex='smart')):

    smart_metric_subset= pd.pivot_table(time_limited_df, values=smart_metric, 

                                 index='serial_number', columns='time_lag', aggfunc=np.sum)

    

    new_columns=['serial_number']

    for i in range(look_back_days):

        new_columns.append(smart_metric+"-"+str(i))

        

    smart_metric_subset.reset_index(inplace=True)

    smart_metric_subset.columns=new_columns

    

    #concat all metrics together

    time_roled_df = pd.merge(time_roled_df, smart_metric_subset, on='serial_number')
for smart_metric in list(time_roled_df.filter(regex='smart')):

    time_roled_df[smart_metric].fillna(-10, inplace=True)

y = time_roled_df.pop('failure').to_frame()

X = time_roled_df
# stratified split - train and test

X_train, X_test, y_train, y_test = model_selection.train_test_split(

        X, y,stratify=y, test_size=0.5)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
X_train.loc[:,'failure'] = y_train

X_test.loc[:,'failure'] = y_test
X_train.to_csv('train.csv', index=False)

X_test.to_csv('test.csv', index=False)
#df = df[df.columns.drop(list(df.filter(regex='Test')))]