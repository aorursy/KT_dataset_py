import warnings

warnings.filterwarnings('ignore')



#eda

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import statsmodels.stats as stats

import statsmodels.api as sm

import copy

import scipy

import pylab

import statsmodels

#from pandas_profiling import ProfileReport



#modeling

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

%matplotlib inline
df=pd.read_csv('../input/innercity.csv')
df.shape
df.head()
df.columns
from datetime import datetime

def clean(i):

  return datetime.strptime(i[:8], '%Y%m%d')
df.dayhours=df.dayhours.apply(clean)

df_org=copy.deepcopy(df)
df.head()
df.info()
df.describe()
df.columns
df.basement
df.drop('cid',1,inplace=True)
#Creating a new column called Base 
df['base']=df.basement.where(cond=df.basement==0,other=1)
df['base']
df['base'].value_counts()
df.yr_renovated
df['renovated']=df.yr_renovated.where(df.yr_renovated==0,1)

df.renovated
date_min=df.dayhours.min()

def to_days(i):

  return (i-date_min).days

date_min
df['dayhours']=df.dayhours.apply(to_days)
sns.distplot(df.dayhours)
sns.scatterplot('dayhours','living_measure',data=df)
plt.plot(df.price.groupby(df.dayhours).sum().index,df.price.groupby(df.dayhours).sum(),'o')
plt.plot(df.price.groupby(df.dayhours).mean().index,df.price.groupby(df.dayhours).mean(),'o')
plt.plot(df.price.groupby(df.dayhours).count().index,df.price.groupby(df.dayhours).count(),'o')
np.corrcoef(df.dayhours,df.price)[0,1]
plt.figure(figsize=(16,8))

sns.heatmap(df.corr(),annot=True,center=0,cmap='coolwarm')

plt.show()
df.price.describe().apply(lambda x: format(x, 'f'))
sns.distplot(df.price)

plt.show()
sns.boxplot('price',data=df)

plt.show()
q1=df.price.describe()['25%']

q3=df.price.describe()['75%']

iqr=q3-q1

ll=q1-(1.5*iqr)

ul=q3+(1.5*iqr)

df_price=df[(df.price<ul) & (df.price>ll)]

df_price.price.describe().apply(lambda x: format(x, 'f'))
print('number of outliers in price:{} out of {} which is {}%'.format(df.price.count()-df_price.price.count(),df.price.count(),(df.price.count()-df_price.price.count())*100/df.price.count()))
sns.distplot(df_price.price)
sns.distplot(np.log(df_price.price))
sns.boxplot(df_price.price)
df.room_bed.value_counts()
df_room_bed=df[df.room_bed.isin([1,2,3,4,5,6])]
df.room_bed.count()-df_room_bed.room_bed.count()
#sns.distplot(df.price[df.room_bed==0],label='0')

#sns.distplot(df.price[df.room_bed==1],label='1')

sns.distplot(df.price[df.room_bed==2],label='2')

sns.distplot(df.price[df.room_bed==3],label='3')

sns.distplot(df.price[df.room_bed==4],label='4')

sns.distplot(df.price[df.room_bed==5],label='5')

#sns.distplot(df.price[df.room_bed==6],label='6')

#sns.distplot(df.price[df.room_bed==7],label='7')

#sns.distplot(df.price[df.room_bed==8],label='8')

#sns.distplot(df.price[df.room_bed==9],label='9')

#sns.distplot(df.price[df.room_bed==10],label='10')

plt.legend()

plt.show()
sns.boxplot('room_bed','price', data=df_price)
df.room_bath.value_counts()
sns.distplot(df.room_bath)
df_room_bath=df[(df.room_bath<=4.5) & (df.room_bath>=1)]
df_room_bath.room_bath.count()-df.room_bath.count()
sns.scatterplot('price','room_bath',data=df)
sns.scatterplot('price','room_bath',data=df_price)
df.price.groupby(df.room_bath).sum().plot(kind='bar')
df.price.groupby(df.room_bath).mean().plot(kind='bar')
df_room_bath.price.groupby(df_room_bath.room_bath).mean().plot(kind='bar') #shows us that price of a house increases with room_bath
sns.boxplot('room_bath', 'price', data=df)
sns.boxplot('room_bath', 'price', data=df_room_bath)
sns.boxplot('room_bath', 'price', data=df_price)
df.living_measure.describe().apply(lambda x: format(x, 'f'))
sns.distplot(df.living_measure)
q1=df.living_measure.describe()['25%']

q3=df.living_measure.describe()['75%']

iqr=q3-q1

ll=q1-(1.5*iqr)

ul=q3+(1.5*iqr)

df_living_measure=df[(df.living_measure<ul) & (df.living_measure>ll)]

df_living_measure.living_measure.describe().apply(lambda x: format(x, 'f'))
sns.distplot(df_living_measure.living_measure)
sns.distplot(np.log(df_living_measure.living_measure))
sns.jointplot('living_measure','price', data=df,kind='reg')

sns.jointplot('living_measure','price', data=df_living_measure,kind='reg')
df.lot_measure.describe().apply(lambda x: format(x, 'f'))
sns.distplot(df.lot_measure)
q1=df.lot_measure.describe()['25%']

q3=df.lot_measure.describe()['75%']

iqr=q3-q1

ll=q1-(1.5*iqr)

ul=q3+(1.5*iqr)

df_lot_measure=df[(df.lot_measure<ul) & (df.lot_measure>ll)]

df_lot_measure.lot_measure.describe().apply(lambda x: format(x, 'f'))
q1=df_price.lot_measure.describe()['25%']

q3=df_price.lot_measure.describe()['75%']

iqr=q3-q1

ll=q1-(1.5*iqr)

ul=q3+(1.5*iqr)

df_price_lot_measure=df_price[(df_price.lot_measure<ul) & (df_price.lot_measure>ll)]
sns.distplot(df_lot_measure.lot_measure)
sns.scatterplot('price', 'lot_measure', data=df)
sns.jointplot('lot_measure', 'price', data=df_price_lot_measure,kind='reg')
df.ceil.value_counts()
sns.distplot(df.ceil)
sns.distplot(np.log(df.price[df.ceil==1]),label='1')

sns.distplot(np.log(df.price[df.ceil==1.5]),label='1.5')

sns.distplot(np.log(df.price[df.ceil==2]),label='2')

sns.distplot(np.log(df.price[df.ceil==2.5]),label='2.5')

plt.legend()

plt.show()
sns.boxplot('ceil', 'price', data=df)
sns.boxplot('ceil', 'price', data=df_price)
df4=df[(df.ceil<3)]

np.corrcoef(df4.price,df4.ceil)
df4.shape[0]/df.shape[0]
df.coast.value_counts()
sns.distplot((df.price[df.coast==0]),label='0')

sns.distplot((df.price[df.coast==1]),label='1')

plt.legend()

plt.show()
df.coast.value_counts()
sns.boxplot('coast', 'price', data=df_price)
sns.distplot(df.price[df.coast==0],label='0')

sns.distplot(df.price[df.coast==1],label='1')

plt.legend()

plt.plot()
sns.distplot(df_price.price[df_price.coast==0],label='0')

sns.distplot(df_price.price[df_price.coast==1],label='1')

plt.legend()

plt.plot()
df.sight.value_counts()
sns.boxplot('sight', 'price', data=df_price)
sns.distplot(df.price[df.sight==0],label='0')

sns.distplot(df.price[df.sight==1],label='1')

sns.distplot(df.price[df.sight==2],label='2')

sns.distplot(df.price[df.sight==3],label='3')

sns.distplot(df.price[df.sight==4],label='4')

plt.legend()

plt.show()
sns.distplot(df_price.price[df_price.sight==0],label='0')

sns.distplot(df_price.price[df_price.sight==1],label='1')

sns.distplot(df_price.price[df_price.sight==2],label='2')

sns.distplot(df_price.price[df_price.sight==3],label='3')

sns.distplot(df_price.price[df_price.sight==4],label='4')

plt.legend()

plt.show()
df.price.groupby(df.sight).mean().plot(kind='bar')
df_price.price.groupby(df_price.sight).mean().plot(kind='bar')
df.condition.value_counts()
sns.distplot(df.condition)
sns.boxplot('condition', 'price',data=df_price)
df.quality.value_counts()
df.quality.value_counts(sort= False).plot(kind='bar')
sns.boxplot('quality', 'price', data=df)
sns.boxplot('quality', 'price', data=df_price)
sns.distplot(df.price[df.quality==5],label='5')

sns.distplot(df.price[df.quality==6],label='6')

sns.distplot(df.price[df.quality==7],label='7')

sns.distplot(df.price[df.quality==8],label='8')

sns.distplot(df.price[df.quality==9],label='9')

sns.distplot(df.price[df.quality==10],label='10')

sns.distplot(df.price[df.quality==11],label='11')

plt.legend()

plt.show()
sns.distplot(df_price.price[df_price.quality==5],label='5')

sns.distplot(df_price.price[df_price.quality==6],label='6')

sns.distplot(df_price.price[df_price.quality==7],label='7')

sns.distplot(df_price.price[df_price.quality==8],label='8')

sns.distplot(df_price.price[df_price.quality==9],label='9')

sns.distplot(df_price.price[df_price.quality==10],label='10')

sns.distplot(df_price.price[df_price.quality==11],label='11')

plt.legend()

plt.show()
df.ceil_measure.describe().apply(lambda x:format(x,'f'))
sns.distplot(df.ceil_measure)
q1=df.ceil_measure.describe()['25%']

q3=df.ceil_measure.describe()['75%']

iqr=q3-q1

ll=q1-(1.5*iqr)

ul=q3+(1.5*iqr)

df_ceil_measure=df[(df.ceil_measure<ul) & (df.ceil_measure>ll)]

df_ceil_measure.ceil_measure.describe().apply(lambda x: format(x, 'f'))
sns.jointplot('ceil_measure', 'price', data=df,kind='reg')
sns.jointplot('ceil_measure', 'price', data=df_ceil_measure,kind='reg')
df.basement.describe().apply(lambda x: format(x,'f'))
plt.pie(df.base.value_counts(),labels=['13126 houses without_basement','8487 houses with_basement' ])
df.basement.value_counts()
sns.distplot(df.basement)
q1=df[df.basement!=0].basement.describe()['25%']

q3=df[df.basement!=0].basement.describe()['75%']

iqr=q3-q1

ll=q1-(1.5*iqr)

ul=q3+(1.5*iqr)

df_basement=df[df.basement!=0][(df[df.basement!=0].basement<ul) & (df[df.basement!=0].basement>ll)]

df_basement.basement.describe().apply(lambda x: format(x, 'f'))
q1=df_price[df_price.basement!=0].basement.describe()['25%']

q3=df_price[df_price.basement!=0].basement.describe()['75%']

iqr=q3-q1

ll=q1-(1.5*iqr)

ul=q3+(1.5*iqr)

df_price_basement=df_price[df_price.basement!=0][(df_price[df_price.basement!=0].basement<ul) & (df_price[df_price.basement!=0].basement>ll)]
(df.basement>ul).sum()
df_basement.shape[0]/df.shape[0]
sns.distplot(df_basement.basement)
sns.jointplot('basement','price', data=df_price_basement,kind='reg')
sns.distplot(df.price[df.basement==0],label='no basement')

sns.distplot(df.price[df.basement!=0],label='basement')

plt.legend()

plt.show()
sns.distplot(df_price.price[df_price.basement==0],label='no basement')

sns.distplot(df_price.price[df_price.basement!=0],label='basement')

plt.legend()

plt.show()
df.base.value_counts()
sns.boxplot('base','price',data=df)
sns.boxplot('base','price',data=df_price)
df.yr_built.describe().apply(lambda x: format(x,'f'))
sns.distplot(df.yr_built)
sns.scatterplot('yr_built','price', data =df)
plt.figure(figsize=(20,8))

sns.jointplot('yr_built','price', data =df_price)

plt.show()
df.yr_built.nunique()
plt.plot(df.price.groupby(df.yr_built).mean().index,df.price.groupby(df.yr_built).mean(),'o')
plt.plot(df.price.groupby(df.yr_built).count().index,df.price.groupby(df.yr_built).count(),'o')
plt.plot(df.living_measure.groupby(df.yr_built).count().index,df.living_measure.groupby(df.yr_built).mean(),'o')
df.yr_renovated.describe()
df.renovated.value_counts()
(df.living_measure!=df.living_measure15).sum()
sns.distplot(df.price[df.renovated==0],label='0')

sns.distplot(df.price[df.renovated==1],label='1')

plt.legend()

plt.show()
sns.distplot(df_price.price[df_price.renovated==0],label='0')

sns.distplot(df_price.price[df_price.renovated==1],label='1')

plt.legend()

plt.show()
sns.scatterplot('yr_renovated','price',data=df[df.renovated==1])
sns.jointplot('yr_renovated','price',data=df_price[df_price.renovated==1],kind='reg')
sns.jointplot(df[df.renovated==1].price.groupby(df[df.renovated==1].yr_renovated).mean().index,df[df.renovated==1].price.groupby(df[df.renovated==1].yr_renovated).mean(),kind='reg')
plt.plot(df_price[df_price.renovated==1].price.groupby(df_price[df_price.renovated==1].yr_renovated).mean().index,df_price[df_price.renovated==1].price.groupby(df_price[df_price.renovated==1].yr_renovated).mean(),'o')
df.price[df.renovated==0].mean()
df.zipcode.nunique()
plt.plot(df.price.groupby(df.zipcode).mean().index,df.price.groupby(df.zipcode).mean(),'o')
plt.plot(df.price.groupby(df.zipcode).count().index,df.price.groupby(df.zipcode).count(),'o') #cant see any relation
plt.plot(df.price.groupby(df.zipcode).sum().index,df.price.groupby(df.zipcode).sum(),'o') #cant see any relation
sns.jointplot('long','price', data=df_price)
sns.jointplot('lat','price', data=df_price)
df.living_measure15.describe()
q1=df.living_measure15.describe()['25%']

q3=df.living_measure15.describe()['75%']

iqr=q3-q1

ll=q1-(1.5*iqr)

ul=q3+(1.5*iqr)

df_living_measure15=df[(df.living_measure15<ul) & (df.living_measure15>ll)]

df_living_measure15.living_measure15.describe().apply(lambda x: format(x, 'f'))
sns.jointplot('living_measure15','price',data=df_price,kind='reg')
df.lot_measure15.describe()
q1=df.lot_measure15.describe()['25%']

q3=df.lot_measure15.describe()['75%']

iqr=q3-q1

ll=q1-(1.5*iqr)

ul=q3+(1.5*iqr)

df_lot_measure15=df[(df.lot_measure15<ul) & (df.lot_measure15>ll)]

df_lot_measure15.lot_measure15.describe().apply(lambda x: format(x, 'f'))
q1=df_price.lot_measure15.describe()['25%']

q3=df_price.lot_measure15.describe()['75%']

iqr=q3-q1

ll=q1-(1.5*iqr)

ul=q3+(1.5*iqr)

df_price_lot_measure15=df_price[(df_price.lot_measure15<ul) & (df_price.lot_measure15>ll)]
sns.jointplot('lot_measure15','price',data=df_price_lot_measure15,kind='reg')
sns.jointplot('lot_measure15','price',data=df_lot_measure15)
df.furnished.value_counts()
sns.boxplot('furnished','price',data=df_price)
sns.distplot(df.price[df.furnished==0],label='0')

sns.distplot(df.price[df.furnished==1],label='1')

plt.legend()

plt.show()
sns.distplot(df_price.price[df_price.furnished==0],label='0')

sns.distplot(df_price.price[df_price.furnished==1],label='1')

plt.legend()

plt.show()
df.total_area.describe().apply(lambda x: format(x,'f'))
q1=df.total_area.describe()['25%']

q3=df.total_area.describe()['75%']

iqr=q3-q1

ll=q1-(1.5*iqr)

ul=q3+(1.5*iqr)

df_total_area=df[(df.total_area<ul) & (df.total_area>ll)]

df_total_area.total_area.describe().apply(lambda x: format(x, 'f'))
q1=df_price.total_area.describe()['25%']

q3=df_price.total_area.describe()['75%']

iqr=q3-q1

ll=q1-(1.5*iqr)

ul=q3+(1.5*iqr)

df_price_total_area=df_price[(df_price.total_area<ul) & (df_price.total_area>ll)]
sns.jointplot('total_area','price',data=df_price_total_area,kind='reg')
df.room_bed.value_counts()
sns.distplot(df.room_bed,kde=False)
df.room_bed.describe()
df2=df[df['room_bed'].isin([2,3,4,5,6,7,8])]
df2.shape
np.corrcoef(df.room_bed,df.price)
#sns.distplot(df.price[df.room_bed==0],label='0')

#sns.distplot(df.price[df.room_bed==1],label='1')

sns.distplot(df.price[df.room_bed==2],label='2')

sns.distplot(df.price[df.room_bed==3],label='3')

sns.distplot(df.price[df.room_bed==4],label='4')

sns.distplot(df.price[df.room_bed==5],label='5')

#sns.distplot(df.price[df.room_bed==6],label='6')

#sns.distplot(df.price[df.room_bed==7],label='7')

#sns.distplot(df.price[df.room_bed==8],label='8')

#sns.distplot(df.price[df.room_bed==9],label='9')

#sns.distplot(df.price[df.room_bed==10],label='10')

plt.legend()

plt.show()
df5=df[(df.room_bed<6) & (df.room_bed>1)]

np.corrcoef(df5.price,df5.room_bed)
from scipy.stats import f_oneway

g2=df.price[df.room_bed==2]

g3=df.price[df.room_bed==3]

g4=df.price[df.room_bed==4]

g5=df.price[df.room_bed==5]

print(g2.mean(),g3.mean(),g4.mean(),g5.mean())

print(g2.std(),g3.std(),g4.std(),g5.std())
sns.distplot(g2)

sns.distplot(g3)
from scipy.stats import shapiro, anderson
shapiro(g2),shapiro(g3)
from scipy.stats import mannwhitneyu
mannwhitneyu(g2,g3)
f_oneway(g2,g3)
df.price.describe()
q1=df.price.describe()['25%']

q3=df.price.describe()['75%']

iqr=q3-q1

ll=q1-(1.5*iqr)

ul=q3+(1.5*iqr)
df1=df[(df.price<ul) & (df.price>ll)]

df1.price.describe()
shapiro(np.log(df1.price))
gr2=df1.price[df1.room_bed==2]

gr3=df1.price[df1.room_bed==3]

gr4=df1.price[df1.room_bed==4]

gr5=df1.price[df1.room_bed==5]
sns.distplot(np.log(gr2),label='2')

sns.distplot(np.log(gr3),label='3')

#sns.distplot(np.log(gr4),label='4')

#sns.distplot(np.log(gr5),label='5')

plt.legend()

plt.show()
shapiro(np.log(g2)),shapiro(np.log(g3))
g2.mean(),g3.mean()
mannwhitneyu(np.log(gr2),np.log(gr3))
df1.shape
statsmodels.graphics.gofplots.qqplot(np.log(gr5),line='s')

plt.show()
sns.distplot(df.living_measure[df.room_bed==2],label='2')

sns.distplot(df.living_measure[df.room_bed==3],label='3')

sns.distplot(df.living_measure[df.room_bed==4],label='4')

sns.distplot(df.living_measure[df.room_bed==5],label='5')

plt.legend()

plt.show()
sns.distplot(df1.living_measure[df1.room_bed==2],label='2')

sns.distplot(df1.living_measure[df1.room_bed==3],label='3')

#sns.distplot(df1.living_measure[df1.room_bed==4],label='4')

#sns.distplot(df1.living_measure[df1.room_bed==5],label='5')

plt.legend()

plt.show()
plt.figure(figsize=(16,8))

sns.heatmap(pd.crosstab(df_room_bed.room_bed,df_room_bed.furnished,values=df_room_bed.price,aggfunc='mean'),annot=True,cmap='coolwarm')

plt.show()
sns.distplot(df.room_bed)
sns.distplot(df.room_bath)
df6=pd.DataFrame()
df.living_measure.describe()
sns.distplot(df.living_measure)

plt.show()
q1=df.living_measure.describe()['25%']

q3=df.living_measure.describe()['75%']

iqr=q3-q1

ll=q1-(1.5*iqr)

ul=q3+(1.5*iqr)

df3=df[(df.living_measure<ul) & (df.living_measure>ll)]

df3.living_measure.describe()
sns.distplot(np.log(df3.living_measure))

plt.show()
anderson(df3.living_measure),anderson(np.log(df3.living_measure))
#log transformation giving better results towards normality than outlier treatment

anderson(df.living_measure),anderson(np.log(df.living_measure))
sns.scatterplot('living_measure','lot_measure',hue='ceil',data=df,palette='ocean')
sns.scatterplot('living_measure','price',data=df) # can see the correlation but this suffers hetroschedasticity
df_price['diff']=df_price.living_measure-df_price.living_measure15
df_price['diff2']=df_price['diff'].where(df_price['diff']==0,df_price['diff']/df_price['diff'].apply(abs))
sns.boxplot('diff2','price',data=df_price)
sns.jointplot('diff','price',df_price[df_price.diff2==1],kind='reg')
sns.scatterplot('living_measure','living_measure15',data=df[df.basement==0],hue='renovated')
df.columns
df.info()
for i in df.columns:

  if i in ['room_bed','room_bath','ceil','coast','sight','condition','quality','zipcode','furnished','base','renovated']:

    df[i]=df[i].apply(str)
cat=['room_bed','room_bath','ceil','coast','sight','condition','quality','zipcode','furnished','base','renovated']

cont=['dayhours','price','living_measure','lot_measure','ceil_measure','basement','yr_built','yr_renovated','lat','long','living_measure15','lot_measure15','total_area']
df.info()
for i in cat:

  print(i,df[i].nunique())
df_mod=pd.get_dummies(df,drop_first=True)
df_mod.shape
df_mod.head(3)
df_mod.columns.tolist()
df_out=pd.DataFrame()

df_out['dayhours']=np.ones((1,21613),dtype=bool).ravel()
q1=df_mod.price.describe()['25%']

q3=df_mod.price.describe()['75%']

iqr=q3-q1

ll=q1-(1.5*iqr)

ul=q3+(1.5*iqr)

df_out['price']=(df_mod.price<ul) & (df_mod.price>ll)
df_out.price.head()
q1=df_mod.living_measure.describe()['25%']

q3=df_mod.living_measure.describe()['75%']

iqr=q3-q1

ll=q1-(1.5*iqr)

ul=q3+(1.5*iqr)

df_out['living_measure']=(df_mod.living_measure<ul) & (df_mod.living_measure>ll)
q1=df_mod.lot_measure.describe()['25%']

q3=df_mod.lot_measure.describe()['75%']

iqr=q3-q1

ll=q1-(1.5*iqr)

ul=q3+(1.5*iqr)

df_out['lot_measure']=(df_mod.lot_measure<ul) & (df_mod.lot_measure>ll)
q1=df_mod.ceil_measure.describe()['25%']

q3=df_mod.ceil_measure.describe()['75%']

iqr=q3-q1

ll=q1-(1.5*iqr)

ul=q3+(1.5*iqr)

df_out['ceil_measure']=(df_mod.ceil_measure<ul) & (df_mod.ceil_measure>ll)
q1=df_mod[df_mod.basement!=0].basement.describe()['25%']

q3=df_mod[df_mod.basement!=0].basement.describe()['75%']

iqr=q3-q1

ll=q1-(1.5*iqr)

ul=q3+(1.5*iqr)

df_out['basement']=(df_mod.basement<ul) & (df_mod.basement>ll)
df_out['yr_built']=np.ones((1,21613),dtype=bool).ravel()
df_out['yr_renovated']=np.ones((1,21613),dtype=bool).ravel()
df_out['lat']=np.ones((1,21613),dtype=bool).ravel()
df_out['long']=np.ones((1,21613),dtype=bool).ravel()
q1=df_mod.living_measure15.describe()['25%']

q3=df_mod.living_measure15.describe()['75%']

iqr=q3-q1

ll=q1-(1.5*iqr)

ul=q3+(1.5*iqr)

df_out['living_measure15']=(df_mod.living_measure15<ul) & (df_mod.living_measure15>ll)
q1=df_mod.lot_measure15.describe()['25%']

q3=df_mod.lot_measure15.describe()['75%']

iqr=q3-q1

ll=q1-(1.5*iqr)

ul=q3+(1.5*iqr)

df_out['lot_measure15']=(df_mod.lot_measure15<ul) & (df_mod.lot_measure15>ll)
q1=df_mod.total_area.describe()['25%']

q3=df_mod.total_area.describe()['75%']

iqr=q3-q1

ll=q1-(1.5*iqr)

ul=q3+(1.5*iqr)

df_out['total_area']=(df_mod.total_area<ul) & (df_mod.total_area>ll)
for i in ['room_bed_1',

 'room_bed_10',

 'room_bed_11',

 'room_bed_2',

 'room_bed_3',

 'room_bed_33',

 'room_bed_4',

 'room_bed_5',

 'room_bed_6',

 'room_bed_7',

 'room_bed_8',

 'room_bed_9',

 'room_bath_0.5',

 'room_bath_0.75',

 'room_bath_1.0',

 'room_bath_1.25',

 'room_bath_1.5',

 'room_bath_1.75',

 'room_bath_2.0',

 'room_bath_2.25',

 'room_bath_2.5',

 'room_bath_2.75',

 'room_bath_3.0',

 'room_bath_3.25',

 'room_bath_3.5',

 'room_bath_3.75',

 'room_bath_4.0',

 'room_bath_4.25',

 'room_bath_4.5',

 'room_bath_4.75',

 'room_bath_5.0',

 'room_bath_5.25',

 'room_bath_5.5',

 'room_bath_5.75',

 'room_bath_6.0',

 'room_bath_6.25',

 'room_bath_6.5',

 'room_bath_6.75',

 'room_bath_7.5',

 'room_bath_7.75',

 'room_bath_8.0',

 'ceil_1.5',

 'ceil_2.0',

 'ceil_2.5',

 'ceil_3.0',

 'ceil_3.5',

 'coast_1',

 'sight_1',

 'sight_2',

 'sight_3',

 'sight_4',

 'condition_2',

 'condition_3',

 'condition_4',

 'condition_5',

 'quality_10',

 'quality_11',

 'quality_12',

 'quality_13',

 'quality_3',

 'quality_4',

 'quality_5',

 'quality_6',

 'quality_7',

 'quality_8',

 'quality_9',

 'zipcode_98002',

 'zipcode_98003',

 'zipcode_98004',

 'zipcode_98005',

 'zipcode_98006',

 'zipcode_98007',

 'zipcode_98008',

 'zipcode_98010',

 'zipcode_98011',

 'zipcode_98014',

 'zipcode_98019',

 'zipcode_98022',

 'zipcode_98023',

 'zipcode_98024',

 'zipcode_98027',

 'zipcode_98028',

 'zipcode_98029',

 'zipcode_98030',

 'zipcode_98031',

 'zipcode_98032',

 'zipcode_98033',

 'zipcode_98034',

 'zipcode_98038',

 'zipcode_98039',

 'zipcode_98040',

 'zipcode_98042',

 'zipcode_98045',

 'zipcode_98052',

 'zipcode_98053',

 'zipcode_98055',

 'zipcode_98056',

 'zipcode_98058',

 'zipcode_98059',

 'zipcode_98065',

 'zipcode_98070',

 'zipcode_98072',

 'zipcode_98074',

 'zipcode_98075',

 'zipcode_98077',

 'zipcode_98092',

 'zipcode_98102',

 'zipcode_98103',

 'zipcode_98105',

 'zipcode_98106',

 'zipcode_98107',

 'zipcode_98108',

 'zipcode_98109',

 'zipcode_98112',

 'zipcode_98115',

 'zipcode_98116',

 'zipcode_98117',

 'zipcode_98118',

 'zipcode_98119',

 'zipcode_98122',

 'zipcode_98125',

 'zipcode_98126',

 'zipcode_98133',

 'zipcode_98136',

 'zipcode_98144',

 'zipcode_98146',

 'zipcode_98148',

 'zipcode_98155',

 'zipcode_98166',

 'zipcode_98168',

 'zipcode_98177',

 'zipcode_98178',

 'zipcode_98188',

 'zipcode_98198',

 'zipcode_98199',

 'furnished_1',

 'base_1',

 'renovated_1']:

       df_out[i]=np.ones((1,21613),dtype=bool).ravel()
df_out.head()
df_mod.shape
df_mod.head()
df_null = pd.DataFrame(np.array([np.nan]*(21613*151)).reshape(21613,151))

df_null.head()
df_out =pd.DataFrame(np.where(df_out,df_mod,df_null),columns=df_mod.columns)
df_out.isnull().sum()
df_out['out']=df_out.isnull().sum(axis=1) #each row wise total sum of the nulls in a row
df_out.out
df_out.out.value_counts() 

# sum of the 0 nulls in a row in the dataset are 17,786

# sum of the 3 nulls in a row in the dataset are 1651

#sum of the 1 null in a row in the dataset are 944 

# sum of the 2 nulls in a roe are in the dataset 665
df_out['ind']=df.index.tolist()
df_out.ind.shape
df_out=df_out[df_out.out.isin([0,1,2,3])]

seq=df_out.ind
seq.head()
df_out.shape
df_out.drop(['out','ind'],1,inplace=True)
df_out.head()
!pip install impyute
from impyute.imputation.cs import mice
df_ao=mice(df_out)
df_ao.columns=df_mod.columns

df_ao.head()
df_ao=df_ao.iloc[:,:13]

df_ao.head()
#df_ao['room_bed']=df.iloc[seq].room_bed.values

#df_ao.loc[:,['price']]
df_ao=df_ao.iloc[:,:13]

df_ao['room_bed']=df.iloc[seq].room_bed.values

df_ao['room_bath']=df.iloc[seq].room_bath.values

df_ao['ceil']=df.iloc[seq].ceil.values

df_ao['coast']=df.iloc[seq].coast.values

df_ao['sight']=df.iloc[seq].sight.values

df_ao['condition']=df.iloc[seq].condition.values

df_ao['quality']=df.iloc[seq].quality.values

df_ao['zipcode']=df.iloc[seq].zipcode.values

df_ao['furnished']=df.iloc[seq].furnished.values

df_ao['base']=df.iloc[seq].base.values

df_ao['renovated']=df.iloc[seq].renovated.values

df_ao.head()
df_ao.columns.tolist()
from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn import model_selection

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor

from sklearn.linear_model import BayesianRidge

from sklearn.linear_model import Ridge, Lasso
x=df_ao.drop(['price'],1)

y=df_ao.price

x=pd.get_dummies(x,drop_first=True)

n,p=x.shape
#ss=StandardScaler()

#sx=ss.fit_transform(x)
#pca=PCA(9)

#pcasx1=pca.fit_transform(sx1)
#l=[]

#for i in range(1,13):

#  l.append(pca.explained_variance_ratio_[:i].sum())

#l
#xf=np.concatenate((pcasx1,x2),1)
#xf.shape
kfold = model_selection.KFold(n_splits=10, random_state=100)
lr=LinearRegression()

results = model_selection.cross_val_score(lr, x, y, cv=kfold)

results.mean(),results.var(),1-((1-results.mean())*(n-1)/(n-p-1))
from statsmodels.stats.outliers_influence import variance_inflation_factor
x.shape
for i in range(x.shape[1]):

  if x.columns[i] in cont:

    print(x.columns[i], variance_inflation_factor(x.values, i))
x=df_ao.drop(['price','lot_measure','lot_measure15','ceil_measure','basement','total_area','living_measure15','yr_built','yr_renovated','dayhours','long','lat',

              'room_bath','room_bed','furnished','base','renovated','coast','condition','ceil'],1)

x.head()
y=df_ao.price

print(x.columns)

x=pd.get_dummies(x,drop_first=True)

n,p=x.shape

print(p)

lr=LinearRegression()

results = model_selection.cross_val_score(lr, x, y, cv=kfold)

print(results.mean(),results.var())

adjr2=1-((1-results.mean())*(n-1)/(n-p-1))

print('adjr2',adjr2)
x.columns.tolist()
x=df_ao.drop(['price','lot_measure15'],1)

x.head()
y=df_ao.price

print(x.columns)

x=pd.get_dummies(x,drop_first=True)

n,p=x.shape

print(p)

lr=LinearRegression()

results = model_selection.cross_val_score(lr, x, y, cv=kfold)

print(results.mean(),results.var())

adjr2=1-((1-results.mean())*(n-1)/(n-p-1))

print('adjr2',adjr2)
for i in ['lot_measure','lot_measure15','ceil_measure','basement','total_area','living_measure15','yr_built','yr_renovated','dayhours','long','lat',

              'room_bath','room_bed','furnished','base','renovated','coast','condition','ceil']:

              l=['price','lot_measure','lot_measure15','ceil_measure','basement','total_area','living_measure15','yr_built','yr_renovated','dayhours','long','lat',

              'room_bath','room_bed','furnished','base','renovated','coast','condition','ceil']

              l.remove(i)

              x=df_ao.drop(l,1)

              y=df_ao.price

              x=pd.get_dummies(x,drop_first=True)

              lr=LinearRegression()

              results = model_selection.cross_val_score(lr, x, y, cv=kfold)

              adjr2=1-((1-results.mean())*(n-1)/(n-p-1))

              print(i,'adjr2',adjr2)
x.shape
for i in range(x.shape[1]):

  print(x.columns[i],variance_inflation_factor(x.values, i))
df_ao.columns
x=df_ao.drop(['price','lot_measure15'],1)

y=df_ao.price

x=pd.get_dummies(x,drop_first=True)

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=123)

lr=LinearRegression()

lr.fit(xtrain,ytrain)

print(lr.score(xtrain,ytrain),lr.score(xtest,ytest))

ypred=lr.predict(xtest)
sns.jointplot(ypred,ytest)
from sklearn.feature_selection import RFE
selector = RFE(lr, 5, step=1)

selector = selector.fit(x, y)

selector.support_ 
selector.ranking_
dt=DecisionTreeRegressor(max_depth=7)

rf=RandomForestRegressor()

ab=AdaBoostRegressor(dt)

br=BayesianRidge()
results = model_selection.cross_val_score(dt, x, y, cv=kfold)

print(results.mean())
results = model_selection.cross_val_score(rf, x, y, cv=kfold)

print(results.mean())
results = model_selection.cross_val_score(br, x, y, cv=kfold)

print(results.mean(),results.var())
dt=DecisionTreeRegressor()

ab=AdaBoostRegressor(dt)

results = model_selection.cross_val_score(ab, x, y, cv=kfold)

print(results.mean(),results.var())
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=47)

ss=StandardScaler()

scale=ss.fit(xtrain)

sxtrain=scale.transform(xtrain)

sxtest=scale.transform(xtest)
lr=LinearRegression()

lr.fit(sxtrain,ytrain)

c1=lr.coef_

c1
lr=Lasso()

lr.fit(sxtrain,ytrain)

c2=lr.coef_

c2
lr=Ridge()

lr.fit(sxtrain,ytrain)

c3=lr.coef_

c3
xtrain.columns
plt.figure(figsize=(16,5))

#plt.plot(xtrain.columns,c1,'o-',label='linear')

plt.plot(xtrain.columns,c2,'o-',label='Lasso')

plt.plot(xtrain.columns,c3,'o-',label='ridge')

plt.xticks(rotation=90)

plt.legend()

plt.show()
(df.living_measure==df.living_measure15).count()
df_org.cid.nunique()-df_org.shape[0]
df_org.cid.value_counts()
df_org[df_org.cid==2724049222]
df.columns
df2=df1.loc[:, ['room_bed', 'room_bath', 'living_measure','living_measure15', 'coast', 'sight', 'quality', 'basement', 'furnished']]
#df2['basement']=pd.Series(np.zeros(21613)).where(df2.basement==0,1)
sdf2=StandardScaler().fit_transform(df2)
from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score
kmeans=KMeans(n_clusters=3)

kmeans.fit(df2)

df2['cluster']=kmeans.labels_
df2.cluster.value_counts()
df2.columns
df2['price']=df.price
sns.scatterplot('price','living_measure',data=df2,hue='cluster')
df2.price[df2.cluster==0].mean(),df2.price[df2.cluster==1].mean(),df2.price[df2.cluster==2].mean()
d1=pd.DataFrame(np.array([1,2,3,4,5,6,7,8]).reshape(4,2),columns='A B'.split())
k=['a','b','c','d'].remove('a')

k
d2=pd.DataFrame(np.array([True,False,True,False,True,False,False,True]).reshape(4,2),columns='A B'.split())

d2
d1[~d2]
np.ones((1,21613),dtype=bool)
len('0.6666666666666666')
df_org.cid.value_counts().value_counts()
df_org.cid.nunique()
df_org.cid.count()
df.dayhours.max()
df.dayhours.value_counts()
1-(20454/21436)
df.columns
df_cont=df[['price','living_measure','lot_measure','ceil_measure','basement','yr_built','yr_renovated','living_measure15','lot_measure15','total_area']]
corr=df_cont.corr()
mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

sns.heatmap(corr, mask=mask, cmap='coolwarm', vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5},annot=True)
sns.pairplot(df_cont)