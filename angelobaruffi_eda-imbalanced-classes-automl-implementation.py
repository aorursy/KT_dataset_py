import pandas as pd

import numpy as np

from IPython.display import HTML, display

import tabulate

import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline  
#UTILS



def missing_data(df):

    total = df.isnull().sum().sort_values(ascending=False)

    percent = (df.isnull().sum())/df.isnull().count().sort_values(ascending=False)

    return pd.concat([total, percent], axis=1, keys=['Total','Percent'], sort=False).sort_values('Total', ascending=False)



def zeros_data(df):

    #Processes zeros values

    total = (df == 0).sum().sort_values(ascending=False)

    percent = ((df == 0).sum())/(df == 0).count().sort_values(ascending=False)

    return pd.concat([total, percent], axis=1, keys=['Total','Percent'], sort=False).sort_values('Total', ascending=False)



def print_table(df):

    display(HTML(tabulate.tabulate(df, tablefmt='html')))

    

def predict_autoH2o(df_predict,  y_column, exclude_algos = [], max_runtime_secs=60*60, nfold=0, sort_metric='aucpr'):

    

    df_h2o = h2o.H2OFrame(df_predict)

    df_h2o = df_h2o[1:,:]

    train,test = df_h2o.split_frame(ratios=[.7], seed = 1)

    

    x = train.columns

    y = y_column

    x.remove(y)

    

    plt.title('Train')

    train.as_data_frame()[y].value_counts().plot(kind='bar', legend = True)

    plt.figure()

    plt.title('Test')

    test.as_data_frame()[y].value_counts().plot(kind='bar', legend = True)



    aml, leaders = predict_autoH2oTrainTest(train, train,  y, x,exclude_algos = exclude_algos, max_runtime_secs=max_runtime_secs, nfold=nfold, sort_metric=sort_metric)

    

    return aml, leaders



def predict_autoH2oTrainTest(train, test,  y, x, exclude_algos = [], max_runtime_secs=60*60, nfold=0, sort_metric='aucpr'):

    # For binary classification, response should be a factor

    train[y] = train[y].asfactor()

    test[y] = test[y].asfactor()



    # Run AutoML for 20 base models (limited to 1 hour max runtime by default)

    exclude_algos = exclude_algos



    aml = H2OAutoML(max_runtime_secs=max_runtime_secs, seed=1, exclude_algos = exclude_algos, nfolds = nfold, sort_metric=sort_metric)

    aml.train(x=x, y=y, training_frame=train, validation_frame=test)



    # AutoML Leaderboard

    lb = aml.leaderboard



    # Optionally edd extra model information to the leaderboard

    lb = get_leaderboard(aml, extra_columns='ALL')



    # Print all rows (instead of default 10 rows)

    return aml, lb.head(rows=lb.nrows)
df = pd.read_excel('../input/covid19/dataset.xlsx')
df.columns
thr = 0.8



fig, ax = plt.subplots(figsize=(20, 5))

plt.xlabel('Fields')

plt.ylabel('% Missing values')

miss_df = missing_data(df)

miss_df = miss_df[miss_df['Percent'] != 0]

miss_df.drop('Total', axis=1).plot(kind='bar', ax=ax)

print('Fields with null {}'.format(miss_df.shape[0]))

print('Number of field in df {}'.format(df.shape[1]))

ax.axhline(y=thr, color='r', linestyle='--', lw=2)

miss_df_remove = miss_df[miss_df['Percent'] >= thr]
column_id = 'Patient ID'



#df_clean = df.drop(miss_df_remove.index, axis=1)

df_clean = df.drop(column_id, axis=1)



df_clean.columns
label_fields = ['SARS-Cov-2 exam result','Patient addmited to regular ward (1=yes, 0=no)',

       'Patient addmited to semi-intensive unit (1=yes, 0=no)',

       'Patient addmited to intensive care unit (1=yes, 0=no)']



fig, ax = plt.subplots(2,2,figsize=(10, 5))



for i,label_field in enumerate(label_fields):

    df[label_field].value_counts().plot(kind='bar', ax=ax.flatten()[i], legend = True)



fig.tight_layout(pad=2.0)
df[label_fields[0]].value_counts()
df_clean.select_dtypes('O')
df_clean = df_clean.astype('float', errors='ignore')

for column in df_clean.select_dtypes('O').columns:

    

    print('Field: {}'.format(column))

    print('Values: {}'.format(set(df_clean[column])))

    print('---')

    
df_clean = df_clean.replace('negative', 0).replace('positive', 1)

df_clean = df_clean.replace('detected',1).replace('not_detected',0)

df_clean = df_clean.replace('present',1).replace('absent',0)

df_clean = df_clean.replace('not_done',np.nan).replace('Não Realizado',np.nan)

df_clean = df_clean.replace('normal',1)

df_clean = df_clean.replace('<1000',1000)

df_clean['Urine - pH'] = df_clean['Urine - pH'].astype('float')

df_clean['Urine - Leukocytes'] = df_clean['Urine - pH'].astype('float')