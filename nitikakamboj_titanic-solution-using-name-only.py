# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
df_train=pd.read_csv('../input/train.csv',index_col='PassengerId')
df_test = pd.read_csv('../input/test.csv',index_col='PassengerId')
df_train.head
boy=(df_train.Name.str.contains('Master'))|((df_train.Sex=='male')&(df_train.Age<13))
female=df_train.Sex=='female'
boy_or_female=boy|female
#df_train['Surname']=df_train.Name.apply(lambda x:re.search('^([^,]*)',x).group(1))
n_ticket=df_train[boy_or_female].groupby('Ticket').Survived.count()
#n_surname=df_train[boy_or_female].groupby('Surname').Survived.count()
ticket_surv=df_train[boy_or_female].groupby('Ticket').Survived.mean()
#surname_surv=df_train[boy_or_female].groupby('Surname').Survived.mean()
def create_feature(frame):
    frame['boy']=(frame.Name.str.contains('Master'))|((frame.Sex=='male')&(frame.Age<13))
    frame['female']=(frame.Sex=='female').astype(int)
    #frame['Surname']=frame.Name.apply(lambda x:re.search('^([^,]*)',x).group(0))
    frame['n_ticket']=frame.Ticket.replace(n_ticket)
    #frame.n_surame=frame.n_surname.astype(float)
    frame.loc[~frame.Ticket.isin(n_ticket.index),'n_ticket']=0
    frame['TicketSurv'] = frame.Ticket.replace(ticket_surv)
    frame.loc[~frame.Ticket.isin(ticket_surv.index),'TicketSurv']=0
    return frame[['female','boy','n_ticket','TicketSurv']]
    

def did_survive(row):
    if row.female:
        if(row.n_ticket>0) and (row.TicketSurv==0):
            return 0
        else:
            return 1
    elif row.boy:
        if(row.n_ticket>0) and (row.TicketSurv==1):
            return 1
        else:
            return 0
    else:
        return 0

X=create_feature(df_test)
pred=X.apply(did_survive,axis=1)
pred=pd.DataFrame(pred)
pred.rename(columns={0:'Survived'},inplace=True)
pred.to_csv('submission_file.csv')
print(pred.Survived.value_counts())
