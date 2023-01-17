# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df= pd.read_csv('../input/the-human-freedom-index/hfi_cc_2019.csv',na_values= ['-'])
df.info()
df.dtypes
df1= df.drop(columns=['year','ISO_code', 'countries', 'region','hf_rank','pf_rank', 'pf_score','ef_score','ef_rank', 'hf_quartile'])

corr_matrix=df1.corr()
top_15_corr_positive = corr_matrix['hf_score'].sort_values(ascending=False)[:15]
top_15_corr_positive
top_15_corr_negative= corr_matrix['hf_score'].sort_values(ascending=True)[:15]
top_15_corr_negative
df_hfi= df[['year','countries','hf_score','pf_rol_procedural', 'pf_rol','ef_legal','ef_trade','pf_expression_control','pf_expression_influence',    
'pf_expression',             
'pf_ss',                      
'pf_rol_criminal',            
'pf_rol_civil',               
'ef_trade_regulatory',        
'pf_movement',               
'ef_legal_military',         
'ef_money',
'ef_government_transfers',        
'ef_government_consumption',      
'ef_government_tax_payroll',      
'ef_government_tax',              
'ef_government_tax_income']]

df_hfi
plt.figure(figsize=(14,10))

plt.title("CORRELATION HEATMAP",fontsize=20)
sns.heatmap(data=df_hfi.drop(['year'], axis=1).corr(),cmap="PRGn_r",annot=True, fmt='.2f', linewidths=1)
plt.show()
