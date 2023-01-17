mind= 1       # min number of deaths or confirmed cases
numd= 20      # range of days for growth (after) and for average UV radiation (before)
data="deaths" # confirmed or deaths
ds1='/kaggle/input/uv-biologically-effective-dose-from-cams/' #dataset 1
ds2='/kaggle/input/corona-virus-time-series-dataset/'         #dataset 2

import pandas as pd
from numpy import nan
from datetime import timedelta

df= pd.read_csv(ds1+"uvbed.csv",header=0)
df["date"]= pd.to_datetime(df.date,format="%Y%m%d")
dflu= pd.read_csv(ds1+"LookUp_Table.csv",header=0)
df= df.merge(dflu,on="UID")
df.head()

#deaths / confirmed  US
dfu= pd.read_csv(ds2+"COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_%s_US.csv"%data,header=0,index_col=0)
dfu= dfu.loc[~dfu.index.isnull() & ~dfu.Lat.isnull() & ~dfu.Long_.isnull()]
dfu.head()

def calculate_growth(ser,mind,numd):
    ser= ser.tail(-10)
    if "Population" in ser.index: ser= ser.tail(-1)
    g,d=0,""
    ser= ser.loc[ser>mind]
    vals= ser.values
    if len(vals)>=numd:
        g= vals[numd-1]/vals[0]
        d= ser.index[0]
    return g,d

g,d,j= [],[],[]
for i in dfu.index:
    gr,da= calculate_growth(dfu.loc[i],mind,numd)
    g.append(gr)
    d.append(da)
    if da == "":
        j.append(nan)
        continue
    da= pd.to_datetime(da,infer_datetime_format=True)
    da0= da - timedelta(numd)
    mask= (df["UID"] == int(i)) & (df.date >= da0) & (df.date <= da)
    uv= df.loc[mask]["uvbed[W/m2]"].median()
    j.append(uv)
    #print(i,"%40s"%dfu.Combined_Key[i],da,"growth=%6.1f"%gr,"uv=%6.4fW/m2"%uv)

dfu["growth"]= g
dfu["1st_date"]= d
dfu["uvbed[W/m2]"]= j

mask= (~dfu.growth.isnull()) & (~dfu["uvbed[W/m2]"].isnull())
dfu= dfu.loc[mask]

#Population
if not "Population" in dfu.columns:
        dfp= pd.read_csv(ds2+"COVID-19/csse_covid_19_data/UID_ISO_FIPS_LookUp_Table.csv",index_col=0)
        dfp= dfp[["Population"]]
        dfu= dfu.join(dfp)
dfu.head()

#remove locations with less than 1000 habitants
mask= (dfu.Population > 1000) & (~dfu.Population.isnull()) 
dfu= dfu.loc[mask]

#save only relevant columns
dfu= dfu[["Combined_Key","Lat","Long_","Population","1st_date","growth","uvbed[W/m2]"]]
dfu.to_csv("/kaggle/working/US_maxuv_growth.csv")
dfu.head()
df= pd.read_csv("/kaggle/working/US_maxuv_growth.csv",header=0,index_col=0)
g= data+" growth"
df[g]= df["growth"]

import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
plt.figure(figsize=(18,12))
sns.scatterplot(hue="uvbed[W/m2]",size=g,y="Lat",x="Long_",data=df)
plt.show()
from numpy import log
df["log10(growth)"]= log(df.growth)/log(10)

plt.figure(figsize=(18,12))
p= sns.regplot(y="log10(growth)",x="uvbed[W/m2]",data=df)
p.set(yscale="log")
plt.show()
from datetime import datetime
df["1st_date"]= pd.to_datetime(df["1st_date"])
df["first date with %d %s or more"%(mind,data)]= df["1st_date"].apply(datetime.timestamp)

plt.figure(figsize=(18,12))
X= "first date with %d %s or more"%(mind,data)
Y= "log10(growth)"
Z= "uvbed[W/m2]"
cmap= sns.diverging_palette(220, 20, as_cmap=True)
p= sns.scatterplot(y=Y,x=X,hue=Z,size=Z,data=df,palette=cmap,alpha=1)
xticks = p.get_xticks()
xticks_dates = [datetime.fromtimestamp(x).strftime('%Y-%m-%d') for x in xticks]
p.set_xticklabels(xticks_dates)
#for i in df.index:
#    p.text(df[X][i], df[Y][i], df.Combined_Key[i].lower(), horizontalalignment='left', size='small', color='black')

from numpy import percentile
upper_thres= percentile(df[Z],67)
lower_thres= percentile(df[Z],33)
lower_uv= df.loc[df[Z]<lower_thres]
upper_uv= df.loc[df[Z]>upper_thres]

# Draw the two density plots
sns.kdeplot(upper_uv[X],upper_uv[Y],cmap="Reds", shade=True, shade_lowest=True,alpha=0.3)
sns.kdeplot(lower_uv[X],lower_uv[Y],cmap="Blues", shade=True, shade_lowest=True,alpha=0.3)
plt.show()
from datetime import datetime
df["ts"]= pd.to_datetime(df["1st_date"]).apply(datetime.timestamp)

from scipy.stats import kendalltau,pearsonr
for t in [kendalltau,pearsonr]:
    print("%s TEST"%t.__name__.upper())
    for c in ["uvbed[W/m2]","Population","Lat","Long_","ts"]:
        tau,p_value= t(df[c].values,df.growth.values)
        print("%15s cor.coef=%6.3f p-value=%.10f"%(c,tau,p_value))
from numpy import percentile
tercile=33
Q= [percentile(df["log10(growth)"],n) for n in range(0,101,tercile)]
def label(x):
    for i in range(len(Q)-1):
        if Q[i]<=x<=Q[i+1] or i==len(Q)-2: return "%6d .. %-6d"%(10**Q[i],10**Q[i+1])

r= g
df[r]= df["log10(growth)"].apply(label)

Q= [percentile(df["ts"],n) for n in range(0,101,tercile)]
def dlabel(x):
    for i in range(len(Q)-1):
        if Q[i]<=x<=Q[i+1] or i==len(Q)-2: return "%s .. %s"%(datetime.fromtimestamp(Q[i]).strftime("%Y-%m-%d"),datetime.fromtimestamp(Q[i+1]).strftime("%Y-%m-%d"))

df["date ranges"]= df["ts"].apply(dlabel)
label_order= sorted(set(df["date ranges"].values))
hue_order= sorted(set(df[r].values))
plt.figure(figsize=(15,10))
sns.violinplot(x="date ranges", y="uvbed[W/m2]", hue=r,hue_order=hue_order,data=df, palette="muted",scale="count",scale_hue=False,order=label_order)
plt.show()
def dlabel(x):
    for i in range(len(Q)-1):
        if Q[i]<=x<=Q[i+1] or i==len(Q)-2: return "%8d .. %-8d"%(Q[i],Q[i+1])

for c in ["Population","Lat","Long_"]:
    Q= [percentile(df[c],n) for n in range(0,101,tercile)]
    df[c+" ranges"]= df[c].apply(dlabel)
    label_order= sorted(set(df[c+" ranges"].values))
    plt.figure(figsize=(15,10))
    sns.violinplot(x=c+" ranges", y="uvbed[W/m2]", hue=r,hue_order=hue_order,data=df, palette="muted",scale="count",scale_hue=False,order=label_order)
    plt.show()