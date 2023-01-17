import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
data=pd.read_csv('../input/startup_funding.csv')
data.head()
data.info()
def format_date(s):
    if "." in s:
        s=s.replace("." , "/")
    elif "//" in s:
        s=s.replace("//","/")
    return s
data['Date']=data['Date'].apply(format_date)
data['Date']=pd.to_datetime(data['Date'], format='%d/%m/%Y')
data_am=data['Date'].groupby([data.Date.dt.year]).agg('count')
fig, axes = plt.subplots(figsize=(12,5))
plt.grid(True)
axes.plot(data_am,marker="o",mfc="red",ms=15,alpha=0.7)
axes.set_xlabel('Year Started')
axes.set_ylabel('Count')
axes.set_title('Year wise distribution')
data_ct=data['CityLocation'].groupby([data.CityLocation]).agg('count')
data_ct.plot(kind="bar",figsize=(16,9),grid=True,title="City wise distribution",cmap='rainbow')
data['AmountInUSD']=data['AmountInUSD'].apply(lambda x:float(str(x).replace(",","")))
dt_amo=data['IndustryVertical'].groupby([data.IndustryVertical]).agg('count').nlargest(10)
dt_amo.plot(kind="bar",figsize=(16,9),grid=True,title="Industry wise distribution",cmap='rainbow')
dt_inv=data['InvestorsName'].groupby([data.InvestorsName]).agg('count').nlargest(10)
dt_inv.plot(kind="bar",figsize=(16,9),grid=True,title="Industry wise distribution",cmap='rainbow')
plt.figure(figsize=(14,6))
sns.set_style('whitegrid')
sns.distplot(data['AmountInUSD'].dropna(),bins=80,kde=False,hist_kws={'edgecolor':'blue'})


