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
df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
yes_mo_map = {'yes':1, 'no':0}
df.loc[:,['dependency','edjefe','edjefa']]=df.loc[:,['dependency','edjefe','edjefa']].applymap(lambda s: yes_mo_map.get(s) if s in yes_mo_map else s)
test_df.loc[:,['dependency','edjefe','edjefa']]=test_df.loc[:,['dependency','edjefe','edjefa']].applymap(lambda s: yes_mo_map.get(s) if s in yes_mo_map else s)

temp_df = df.fillna(-10)
household_feat = [] #household features
personal_feat = []  #personal features
for i in temp_df.columns[1:] :
    if i != 'idhogar':
        if sum(temp_df.groupby('idhogar')[i].nunique()) == 2988:# number of unique households 
            household_feat.append(i)
        else:
            personal_feat.append(i)
df.columns[df.isnull().any()].tolist()
df.loc[:,'v18q1'].fillna(0, inplace=True) 
df.columns[df.isnull().any()].tolist()
df_processed = df.loc[df['parentesco1']==1, household_feat+['Target', 'idhogar']]
df_processed = df_processed.set_index('idhogar')
inst_level=['dis','male','female','estadocivil1','estadocivil2','estadocivil3','estadocivil4','estadocivil5','estadocivil6','estadocivil7','parentesco1','parentesco2','parentesco3','parentesco4','parentesco5','parentesco6','parentesco7','parentesco8','parentesco9','parentesco10','parentesco11','parentesco12','instlevel1','instlevel2','instlevel3','instlevel4','instlevel5','instlevel6','instlevel7','instlevel8','instlevel9']
df_processed2=df.groupby('idhogar')[inst_level].sum()
df_processed2 = df_processed2.loc[df_processed2['parentesco1']==1, :]#.reset_index(drop=True)#converting personal features to household features!
df_tot = pd.concat([df_processed, df_processed2], axis = 1)
df_tot=df_tot.reset_index()
left_out= ['escolari','rez_esc','age','SQBescolari','SQBage','agesq']
df_tot=df_tot.drop(['v2a1', 'meaneduc', 'SQBmeaned'],axis=1)
test_df.loc[:,'v18q1'].fillna(0, inplace=True)
test_df_processed = test_df.loc[test_df['parentesco1']==1, household_feat]
test_df_processed = test_df_processed.reset_index(drop=True)
test_df_processed2=test_df.groupby('idhogar')[inst_level].sum().reset_index()
test_df_tot = pd.concat([test_df_processed, test_df_processed2], axis = 1).iloc[:7334] #Number of household heads in the dataset
test_df_tot=test_df_tot.drop(['v2a1', 'meaneduc', 'SQBmeaned'],axis=1)
feats= list(df_tot.columns)
feats.remove('index')
feats.remove('Target')
target = 'Target'
from sklearn.ensemble import RandomForestClassifier
rd= RandomForestClassifier()
rd.fit(df_tot.loc[:,feats],df_tot.loc[:,target])
predicted_class = rd.predict(test_df.loc[:,feats])
my_submission = pd.DataFrame({'Id': test_df.Id, 'Target': predicted_class})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)
