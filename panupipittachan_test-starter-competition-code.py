import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import os
from tqdm.notebook import tqdm
df_keyword = pd.read_csv('../input/undrg-rd1-listings/Extra Material 2 - keyword list_with substring.csv')
df_testdata = pd.read_csv('../input/undrg-rd1-listings/Keyword_spam_question.csv')
df_keyword.head(10)
df_keyword.info()
df_split_keyword = df_keyword.iloc[0:0]
for idx,ky in enumerate(df_keyword['Keywords']) :
    if len(ky.split(',')) >= 2 :
        para = ky.split(',')   
        for i in range(len(para)) :
            df_split_keyword = df_split_keyword.append(pd.Series([idx,para[i]], index= df_split_keyword.columns ), ignore_index=True)
    else :
        df_split_keyword = df_split_keyword.append(pd.Series([idx,ky], index= df_split_keyword.columns ), ignore_index=True)
check_group = df_split_keyword.groupby('Keywords').count().reset_index()
check_group = check_group[check_group['Group']==2]
check_group
## If 1 keyword has 2 group take less one
check_group = df_split_keyword.groupby('Keywords').count().reset_index()
check_group = check_group[check_group['Group']==2]
del_lessgroup = df_split_keyword[df_split_keyword['Keywords'].isin(check_group['Keywords'])]
del_lessgroup = del_lessgroup.sort_values('Keywords')
del_less =list(zip(del_lessgroup['Keywords'].index,del_lessgroup['Keywords'].values))
g1 = 0
g2 = 0
drop_axis0 = []
for idx,ky in del_less :
    if g1 == g2 :
        g1 = df_split_keyword.iloc[idx,0]
        idx2 = idx
        continue
    g2 = df_split_keyword.iloc[idx,0]
    if g1 > g2 :
        g1 = df_split_keyword.iloc[idx,0]
        df_split_keyword = df_split_keyword.drop(idx,axis=0)
        idx2 = idx
    else :
        g1 = df_split_keyword.iloc[idx,0]
        df_split_keyword = df_split_keyword.drop(idx2,axis=0)
        idx2 = idx
check_group = df_split_keyword.groupby('Keywords').count().reset_index()
check_group[check_group['Group']==2]
df_testdata.sample(5)
##Normalize name
import re
new_name = []
for name in df_testdata['name'] :
    new_name.append(re.sub("[^a-zA-Z1-9\s]", "", name).lower())
df_testdata['Name']= new_name
df_testdata
import re
check = df_testdata['Name']
order= []
for name in tqdm(check) :
    ft= []
    p_ky = []
    p_number = 0
    for idx,ky in enumerate(df_split_keyword['Keywords']) :
        if ky in name :
            if df_split_keyword.iloc[idx,0] == p_number : ##Condition if difference keyword have same group chose one
                continue
            ft.append(df_split_keyword.iloc[idx,0])
            p_ky.append(ky)
            p_number = df_split_keyword.iloc[idx,0]
        else :
            continue
    ind = []
    if len(p_ky) >= 2 :
        for z in range(len(p_ky)) :
            p_i = []
            for i in p_ky[z:] :
                if p_i == [] :
                    p_i = i
                    continue
                else :
                    if i in p_i:
                        ind.append(ft[p_ky.index(i)])
                    elif p_i in i:
                        ind.append(ft[p_ky.index(p_i)])
                    else :
                        continue
        ind_dropduplicates = list(dict.fromkeys(ind))
        for k in ind_dropduplicates :
            ft.remove(k)       
    order.append(ft)
df_testdata['groups_found'] = order
for_submission = df_testdata.drop(['name','Name'],axis=1)
for_submission