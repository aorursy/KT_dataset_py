# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
def cluster(Account1, Monthly_Period, History,Sponsors,InstallmentCredit,Plotsize,Housing):

    if Plotsize <= 2.5:

        if History <= 3.5:

            if Sponsors <= 2.5:

                return 1

            else:

                return 2

        else:

            if Account1 <= 0.5:

                return 2

            else:

                return 0

    else:

        if Monthly_Period <= 11.0:

            if Housing <= 1.5:

                return 1

            else:

                return 0

        else:

            if InstallmentCredit <= 4.0913920402526855:

                return 1

            else:

                return 2



import pandas as pd

import numpy as np



df = pd.read_csv('../input/dataset.csv')



test_df = df[df.isnull().any(axis=1)]



def preprocess_rf(df):

    train_df = df.copy(deep=True)

    train_df = train_df.drop('Class',axis=1)

    train_df = train_df.fillna(0)

    train_df['Phone'] = train_df['Phone'].apply(lambda x : 1 if x=='yes' else 0)

    train_df['Post'] = train_df['Post'].apply(lambda x : int(x.split('b')[1]))

    train_df['Housing'] = train_df['Housing'].apply(lambda x :int(x.split('H')[1]))

    train_df['Plan'] = train_df['Plan'].apply(lambda x : int(x.split('L')[1]))

    sizes = {'sm':0,'SM':0,'me':1,'ME':1,'M.E.':1,'LA':2,'la':2,'xl':3,'XL':3}

    train_df['Plotsize'] = train_df['Plotsize'].apply(lambda x : sizes[x])

    train_df['Sponsors'] = train_df['Sponsors'].apply(lambda x : int(x.upper().split('G')[-1]))

    train_df['Employment Period'] = train_df['Employment Period'].apply(lambda x : int(x.split('e')[1]))

    gat = {'M0':0,'M1':1,'F0':2,'F1':3}

    train_df['Gender&Type'] = train_df['Gender&Type'].apply(lambda x : gat[x])

    train_df['Account2'] = train_df['Account2'].apply(lambda x : int(x.split('c')[-1]))

    train_df['Motive'] = train_df['Motive'].apply(lambda x : int(x.split('p')[-1]) if x!='?' else 0)

    train_df['History'] = train_df['History'].apply(lambda x : int(x.split('c')[-1] if x!='?' else 0))

    acct_m = {'aa':0,'ab':1,'ac':2,'ad':3}

    train_df['Account1'] = train_df['Account1'].apply(lambda x : acct_m[x] if x in acct_m else 0)

#     train_df = train_df.drop('id',axis=1)

    for col in train_df.columns:

        train_df[col] = train_df[col].apply(lambda x : x if x!='?' else 0)

    return train_df



test_df_rf = preprocess_rf(test_df)



classes = []

for index,row in test_df_rf.iterrows():

    classes.append(cluster(row['Account1'],float(row['Monthly Period']),row['History'],row['Sponsors'],float(row['InstallmentCredit']),row['Plotsize'],row['Housing']))



test_df_rf = test_df_rf.reset_index()



test_df_rf['Class'] = pd.Series(classes)



test_df_rf_fin = test_df_rf[['id','Class']]



test_df_rf_fin.to_csv('submit.csv',index=False)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



create_download_link(test_df_rf_fin)


