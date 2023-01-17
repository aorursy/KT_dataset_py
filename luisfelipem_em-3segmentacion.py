# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np 
import pandas as pd
import calendar
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
import gc
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline

sns.set_style('white')

pd.options.display.float_format = '{:,.2f}'.format
df_sorted = pd.read_pickle('/kaggle/input/em-1inicio/EasyMoney_Nuevo.pkl',compression='zip')
df_sorted.info()
productos_easymoney=['loans',
 'mortgage',
 'funds',
 'securities',
 'long_term_deposit',
 'em_account_pp',
 'credit_card',
 'payroll',
 'pension_plan',
 'payroll_account',
 'emc_account',
 'debit_card',
 'em_account_p',
 'em_acount']
dif_productos_easymoney=['dif_loans',
 'dif_mortgage',
 'dif_funds',
 'dif_securities',
 'dif_long_term_deposit',
 'dif_em_account_pp',
 'dif_credit_card',
 'dif_payroll',
 'dif_pension_plan',
 'dif_payroll_account',
 'dif_emc_account',
 'dif_debit_card',
 'dif_em_account_p',
 'dif_em_acount']
df_sorted['debit_card_E']=df_sorted['dif_debit_card']*10
df_sorted['em_account_p_E']=df_sorted['dif_em_account_p']*10
df_sorted['em_account_pp_E']=df_sorted['dif_em_account_pp']*10
df_sorted['em_acount_E']=df_sorted['dif_em_acount']*10
df_sorted['emc_account_E']=df_sorted['dif_emc_account']*10
df_sorted['payroll_E']=df_sorted['dif_payroll']*10
df_sorted['payroll_account_E']=df_sorted['dif_payroll_account']*10
productos_easymoney_cuenta_simple = ['debit_card_E','em_account_p_E','em_account_pp_E','em_acount_E','emc_account_E','payroll_E','payroll_account_E']
df_sorted['funds_E']=df_sorted['dif_funds']*40
df_sorted['long_term_deposit_E']=df_sorted['dif_long_term_deposit']*40
df_sorted['mortgage_E']=df_sorted['dif_mortgage']*40
df_sorted['pension_plan_E']=df_sorted['dif_pension_plan']*40
df_sorted['securities_E']=df_sorted['dif_securities']*40
productos_easymoney_cuenta_ahorro = ['funds_E','long_term_deposit_E','mortgage_E','pension_plan_E',
                                     'securities_E']
df_sorted['loans_E']=df_sorted['dif_loans']*60
df_sorted['credit_card_E']=df_sorted['dif_credit_card']*60
productos_easymoney_cuenta_financiamiento = ['loans_E','credit_card_E']
df_sorted.head().T
df_sorted['IngPCS']= df_sorted['debit_card_E']+df_sorted['em_account_p_E']+df_sorted['em_account_pp_E']+df_sorted['em_acount_E']+df_sorted['emc_account_E']+df_sorted['payroll_E']+df_sorted['payroll_account_E']
df_sorted['IngPCA']= df_sorted['funds_E']+df_sorted['long_term_deposit_E']+df_sorted['mortgage_E']+df_sorted['pension_plan_E']+df_sorted['securities_E']
df_sorted['IngPCF']= df_sorted['loans_E']+df_sorted['credit_card_E']
df_sorted['IngTotal'] = df_sorted['IngPCS']+df_sorted['IngPCA']+df_sorted['IngPCF']
df_sorted.head().T
df_sorted.drop(productos_easymoney,axis=1, inplace=True)
df_sorted.drop(dif_productos_easymoney,axis=1, inplace=True)
df_sorted
df_sorted.drop(productos_easymoney_cuenta_simple,axis=1, inplace=True)
df_sorted.drop(productos_easymoney_cuenta_ahorro,axis=1, inplace=True)
df_sorted.drop(productos_easymoney_cuenta_financiamiento,axis=1, inplace=True)
df_sorted
columns_to_drop = ['short_term_deposit','country_id','mesesAlta']
df_sorted.drop(columns_to_drop,axis=1, inplace=True)
df_sorted
Series = ['region_code', 'age','salary','IngPCS','IngPCA','IngPCF','IngTotal']
df_sorted.groupby(["pk_cid","pk_partition"]).agg({
                                                'region_code':np.mean,
                                                'age':np.mean,
                                                'salary':np.mean,
                                                'IngPCS':np.sum,
                                                'IngPCA':np.sum,
                                                'IngPCF':np.sum,
                                                'IngTotal':np.sum})
df_sorted
df_sorted.sort_values(by="IngTotal", ascending=False)
df_sorted1=df_sorted[df_sorted['IngTotal']>0]
df_sorted1
df_sorted1 = df_sorted1[df_sorted['deceased']=='N']
df_sorted1
Variables = ['age','salary','IngTotal']
sns.pairplot(df_sorted1[Variables].sample(10000))
pipe = Pipeline(
    steps=[
        ('StandardScaler', StandardScaler()),
        ('KMeans', KMeans(n_clusters=6))
    ]
)
pipe.fit(df_sorted1[Variables])
df_sorted1['Label'] = pipe.predict(df_sorted1[Variables])
df_sorted1.head()
df_sorted1['Label'].value_counts()
df_sorted1.pivot_table(index='Label', values=Variables, aggfunc=np.mean)
sse = {}
for k in range(1, 20):
    pipe = Pipeline(
    steps=[
        ('StandardScaler', StandardScaler()),
        ('KMeans', KMeans(n_clusters=k))
        ]
    ).fit(df_sorted1[Variables])
    sse[k] = pipe['KMeans'].inertia_

plt.figure(figsize=(10, 5))
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()
