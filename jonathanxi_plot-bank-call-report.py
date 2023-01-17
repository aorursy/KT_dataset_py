# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

 #   for filename in filenames:

  #      print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
pd.set_option('display.max_columns',1000)

path = '../input/summer-research'

df_rc=pd.read_csv(f'{path}/FFIEC CDR Call Schedule RC 03312020.txt',delimiter="\t")
df_rc=df_rc.dropna(how='all',axis=1)

df_rc.head()
df_rc_totalasset=df_rc[['IDRSSD','RCFD2170','RCFD2948']]

df_rc_totalasset=df_rc_totalasset.dropna(how='all',thresh=3)

df_rc_totalasset=df_rc_totalasset.astype(float)

df_rc_totalasset.tail()
import matplotlib.pyplot as plt

import seaborn as sns



sns.set(rc={"figure.figsize": (8, 6)}); 

sns.set(style="white",palette='deep',color_codes=False)
df_rc_plot=df_rc_totalasset

df_rc_plot_log=df_rc_plot['RCFD2170'].astype(float).apply(np.log)  # Log asset

#sns.kdeplot(data=df_rc_plot_log,label="Bank Total Asset" ,shade=True)





fig = plt.figure(figsize=(16,6))

ax1= fig.add_subplot(1,2,1)

ax2= fig.add_subplot(1,2,2)

sns.kdeplot(df_rc_plot_log,shade=True,label='Log Bank Total Asset',ax=ax1)

sns.kdeplot(df_rc_plot['RCFD2170'],label="Bank Total Asset" ,shade=True,ax=ax2)

sns.despine()
import plotly.express as px

#fig = px.histogram(cp, x="degree_p", y="salary", color="gender")

ax1=px.histogram(df_rc_plot_log,width=600,height=450,title="Log Bank Total Asset",histnorm='density')

#fig.update_layout(

 #   autosize=False)#,paper_bgcolor="LightSteelBlue")

#fig.update_yaxes(automargin=True)

ax2=px.histogram(df_rc_plot['RCFD2170'],width=600,height=450,title="Bank Total Asset",histnorm='density')

ax1.show()

ax2.show()

df_rc_plot_log=pd.DataFrame(df_rc_plot_log)

df_rc_plot_log['IDRSSD']=df_rc_plot['IDRSSD']

fig=px.scatter_polar(df_rc_plot_log, r="RCFD2170", theta="IDRSSD",color="IDRSSD")

#fig.update_layout(font_size=6)

fig.show()
df_rc_plot['RCFD2170'].astype(float).describe()

log_sd_plus=1.851376e+08+4.525690e+08

log_sd_minus=1.851376e+08-4.525690e+08

right=sum(np.where(df_rc_plot['RCFD2170'].astype(float)>log_sd_plus,1,0))

left=sum(np.where(df_rc_plot['RCFD2170'].astype(float)<log_sd_minus,1,0))

(80-right-left)/80
df_rc_plot_log=df_rc_plot['RCFD2170'].astype(float).apply(np.log)

df_rc_plot_log.describe()

log_sd_plus=16.887587+2.541675

log_sd_minus=16.887587-2.541675

right=sum(np.where(df_rc_plot_log>log_sd_plus,1,0))

left=sum(np.where(df_rc_plot_log<log_sd_minus,1,0))

(80-right-left)/80
df_rc_plot_log_sort=sorted(df_rc_plot_log,reverse=True)

Top_n_shares=np.cumsum(df_rc_plot_log_sort)/sum(df_rc_plot_log_sort)

fig = px.bar(Top_n_shares[0:10],title='Top 10 Share for Log Total Asset',width=600,height=400)

fig.show()
df_rc_plot_sort=sorted(df_rc_plot['RCFD2170'].astype(float),reverse=True)

Top_n_shares=np.cumsum(df_rc_plot_sort)/sum(df_rc_plot_sort)

fig = px.bar(Top_n_shares[0:10],title='Top 10 Share for Total Asset',width=600,height=400)

fig.show()
import warnings

warnings.filterwarnings('ignore')
df_rce_totaldeposit=df_rc[['IDRSSD','RCON2200']]

df_rce_totaldeposit=df_rce_totaldeposit.drop(index=0).astype(float)

#df_rce_totaldeposit=np.where(df_rce_totaldeposit==0,np.nan,df_rce_totaldeposit)

#df_rce_totaldeposit=pd.DataFrame(df_rce_totaldeposit)

df_rce_totaldeposit_new=df_rce_totaldeposit.dropna(how='all',thresh=2)

df_rce_totaldeposit_new.columns=['IDRSSD','RCON2200']

df_asset_deposit=pd.merge(df_rc_totalasset,df_rce_totaldeposit_new,on='IDRSSD')

df_asset_deposit.tail()
fig_1=px.histogram(df_asset_deposit['RCON2200'].apply(np.log),width=600,height=450,title="Log Bank Total Deposit",histnorm='density')

fig_2=px.histogram(df_asset_deposit['RCON2200'],width=600,height=450,title="Bank Total Deposit",histnorm='density')

fig_1.show()

fig_2.show()
df_loans=pd.read_csv(f'{path}/FFIEC CDR Call Schedule RCH 03312020.txt',delimiter="\t")

df_totalloans=df_loans[['IDRSSD','RCONHT71']].dropna(how='all',thresh=2)

df_totalloans=df_totalloans.astype(float)

df_asset_deposit_loan=pd.merge(df_asset_deposit,df_totalloans,on='IDRSSD')

df_asset_deposit_loan.columns=['IDRSSD','Total Asset','Total Liabilities','Total Deposit','Total loans held for trading']

df_asset_deposit_loan.tail()
df_asset_deposit_loan['Total loans held for trading']=np.where(df_asset_deposit_loan['Total loans held for trading']>0,1,0)

df_asset_deposit_loan.tail()
px.scatter(df_asset_deposit_loan,x='Total Asset',y='Total Deposit',size='Total Liabilities',color='Total loans held for trading',width=800,trendline="ols")
df_asset_deposit_loan_log=df_asset_deposit_loan.copy()

df_asset_deposit_loan_log[['Total Asset','Total Liabilities','Total Deposit']]=df_asset_deposit_loan_log[['Total Asset','Total Liabilities','Total Deposit']].apply(np.log)

df_asset_deposit_loan_log.tail()

px.scatter(df_asset_deposit_loan_log,x='Total Asset',y='Total Deposit',size='Total Liabilities',color='Total loans held for trading',width=800,trendline="ols")
df_ci_loans=pd.read_csv(f'{path}/FFIEC CDR Call Schedule RCCI 03312020.txt',delimiter="\t")

df_ci_loans_new=df_ci_loans[['IDRSSD','RCON1766']]

df_ci_loans_new=df_ci_loans_new.dropna(how='all',thresh=2)
df_deposit_loans=pd.merge(df_asset_deposit,df_ci_loans_new,on='IDRSSD',how='left')

df_deposit_loans.columns=['IDRSSD','Total Asset','Total Liabilities','Total Deposit','C&I Loans']

df_deposit_loans['IDRSSD']=df_deposit_loans['IDRSSD'].astype(float)

df_deposit_loans.tail()

#px.scatter(df_deposit_loans,x='Total Deposit',y='C&I Loans',width=800)

df_deposit_loans.dropna(how='all',thresh=5)
df_rc_totalasset.tail()
pd.merge(df_rc_totalasset,df_ci_loans_new,on='IDRSSD',how='left')
df_ci_loans_new[df_ci_loans_new['IDRSSD']==12311]
import seaborn as sns

df_RI=pd.read_csv(f'{path}/FFIEC CDR Call Schedule RI 03312020.txt',delimiter="\t")

df_netincome=df_RI[['IDRSSD','RIAD4340']]

df_netincome.dropna(how='all',thresh=2,inplace=True)
df_asset_deposit_loan_income_log=pd.merge(df_asset_deposit_loan_log,df_netincome,on='IDRSSD')

df_asset_deposit_loan_income_log.rename(columns={'RIAD4340':'Net Income'},inplace=True)

df_asset_deposit_loan_income_log['Net Income']=df_asset_deposit_loan_income_log['Net Income'].astype(float).apply(np.log)

df_asset_deposit_loan_income_log.tail()

df_RCR=pd.read_csv(f'{path}/FFIEC CDR Call Schedule RCRI 03312020.txt',delimiter="\t")

df_tier1=df_RCR[['IDRSSD','RCFAP742']].dropna(how='all',thresh=2)
df_asset_deposit_loan_income_regulatory_log=pd.merge(df_asset_deposit_loan_income_log,df_tier1,on='IDRSSD')

df_asset_deposit_loan_income_regulatory_log.rename(columns={'RCFAP742':'Tier 1 Capital'},inplace=True)

df_asset_deposit_loan_income_regulatory_log['Tier 1 Capital']=df_asset_deposit_loan_income_regulatory_log['Tier 1 Capital'].astype(float).apply(np.log)

df_asset_deposit_loan_income_regulatory_log.tail()
df_capital=df_rc[['IDRSSD','RCON3210']].dropna(how='all',thresh=2)

df_capital['RCON3210']=df_capital['RCON3210'].astype(float).apply(np.log)

df_capital.rename(columns={'RCON3210':'Total Equity Capital'},inplace=True)

pd.merge(df_asset_deposit_loan_income_regulatory_log,df_capital,on='IDRSSD')
sns.set(style="darkgrid",palette='deep',color_codes=False)

df_plot=df_asset_deposit_loan_income_regulatory_log[['Total Asset','Total Liabilities','Total Deposit','Total loans held for trading','Net Income','Tier 1 Capital']]

sns.pairplot(df_plot, hue="Total loans held for trading", size=2, diag_kind="kde")
df_Deposit_Liabilities_foreign=pd.read_csv(f'{path}/FFIEC CDR Call Schedule RCEII 03312020.txt',delimiter="\t")

df_Deposit_Liabilities_domestic=pd.read_csv(f'{path}/FFIEC CDR Call Schedule RCEI 03312020.txt',delimiter="\t")
df_totaldomestic_deposit=df_Deposit_Liabilities_domestic[['IDRSSD','RCON2215']].dropna(thresh=2)

print(df_Deposit_Liabilities_foreign['RCFN2200'].dropna())

df_totalforeign_deposit=df_Deposit_Liabilities_foreign[['IDRSSD','RCFN2200']].dropna()

df_totalforeign_deposit.tail()
df_deposit_fd=pd.merge(df_totalforeign_deposit,df_totaldomestic_deposit,on='IDRSSD')

df_deposit_fd=df_deposit_fd.astype(float)

print(df_deposit_fd)

print("\n")

print("The number of zero in Foreign Deposit:\n")

print(len(df_deposit_fd[df_deposit_fd['RCFN2200']==0]))
deposit_ftod_ratio=df_deposit_fd['RCFN2200']/df_deposit_fd['RCON2215']

deposit_ftod_ratio=pd.DataFrame(deposit_ftod_ratio)

deposit_ftod_ratio['IDRSSD']=df_deposit_fd['IDRSSD']

deposit_ftod_ratio.columns=['Ratio','IDRSSD']

deposit_ftod_ratio.tail()
#deposit_ftod_ratio=df_deposit_fd['RCFN2200']/df_deposit_fd['RCON2215']

px.bar(deposit_ftod_ratio['Ratio'],title='Foreign/Domestic',width=600,height=400)
fig=px.scatter_polar(deposit_ftod_ratio, r="Ratio", theta="IDRSSD",color="IDRSSD")

fig.show()
df_ci_loans_new.tail()
df_rcd=pd.read_csv(f'{path}/FFIEC CDR Call Schedule RCD 03312020.txt',delimiter="\t")

df_mbs=df_rcd[['IDRSSD','RCFDG379','RCFDG380','RCFDG381','RCFDK197','RCFDK198']]

df_mbs.dropna(thresh=6).tail()
df_rcd=pd.read_csv(f'{path}/FFIEC CDR Call Schedule RCD 03312020.txt',delimiter="\t")

df_rcd=df_rcd.dropna(how='all',axis=1)

df_mbs=df_rcd['RCFDG379']+df_rcd['RCFDG380']+df_rcd['RCFDG381']+df_rcd['RCFDK197']+df_rcd['RCFDK198']

#df_mbs['IDRSSD']=df_rcd['IDRSSD']

#df_mbs.dropna(how='all',thresh=2)