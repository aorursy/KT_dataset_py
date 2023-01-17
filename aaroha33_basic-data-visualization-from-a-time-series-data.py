# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn.preprocessing import LabelEncoder
# Any results you write to the current directory are saved as output.

# Visualisation libraries
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()
import pycountry
import plotly.express as px
from plotly.offline import init_notebook_mode, iplot 
import plotly.graph_objs as go
import plotly.offline as py
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot

from scipy import stats
from scipy.stats import norm, skew #for some statistics
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting fl
#df = pd.read_excel('../kaggle/input/anurag/Graph_Anurag.xlsx',sheet_name="Sheet1")
df = pd.read_csv('../input/anurag/Graph_Anurag.csv')
df.head(10)
df.isnull().sum()
df.info()
df.describe()
df1=df['DL_Throughput'].mean()
df1
df['DL_Throughput'].fillna(df['DL_Throughput'].mean(), inplace =True)
df.isnull().sum()
df1=df[['Data Availability (%)','Num of Cells','DL_Throughput']]
h = df1.hist(bins=25,figsize=(16,16),xlabelsize='10',ylabelsize='10',xrot=-15)
sns.despine(left=True, bottom=True)
[x.title.set_size(12) for x in h.ravel()];
[x.yaxis.tick_left() for x in h.ravel()];
df['RecordDate'] = pd.to_datetime(df['RecordDate'],infer_datetime_format=True) 
df.drop_duplicates(inplace=True)
df.info()
#def highlight_max(s):
#    is_max = s == s.max()
#    return ['background-color: pink' if v else '' for v in is_max]
#df.style.apply(highlight_max,subset=['DL_Throughput', 'Num of Cells','Data Availability (%)','RecordDate'])
x = df.groupby('RecordDate')['DL_Throughput'].sum().sort_values(ascending=False).to_frame()
x.style.background_gradient(cmap='Reds')
sns.distplot(df['Num of Cells'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(df['Num of Cells'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('Num of Cells distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(df['Num of Cells'], plot=plt)
plt.show()
sns.set(style="whitegrid", font_scale=1)

from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure(figsize=(9.5,6.25))
ax=fig.add_subplot(1,1,1, projection="3d")
ax.scatter(df['Num of Cells'],df['Data Availability (%)'],df['DL_Throughput'],c="darkgreen",alpha=.5)
ax.set(xlabel='\nNum of Cells',ylabel='\nData Availability (%)',zlabel='\nDL_Throughput');
features = ['Num of Cells','Data Availability (%)','DL_Throughput']

mask = np.zeros_like(df[features].corr(), dtype=np.bool) 
mask[np.triu_indices_from(mask)] = True 

f, ax = plt.subplots(figsize=(16, 12))
plt.title('Pearson Correlation Matrix',fontsize=25)

sns.heatmap(df[features].corr(),linewidths=0.25,vmax=0.7,square=True,cmap="BuGn", #"BuGn_r" to reverse 
            linecolor='w',annot=True,annot_kws={"size":8},mask=mask,cbar_kws={"shrink": .9});
# !pip install pyEx
df['Num of Cells'].plot()
df['Data Availability (%)'].plot()
df['DL_Throughput'].plot()
la = LabelEncoder()
#df['DL_Throughput']=la.fit_transform(df['DL_Throughput'])
#df['Num of Cells']=la.fit_transform(df['Num of Cells'])
corrmat = df.corr()
f, ax = plt.subplots(figsize=(20, 9))
sns.heatmap(corrmat, vmax=.8, annot=True);
df1=df[['RecordDate','DL_Throughput']]
df1.info()
df['DL_Throughput']=np.log(df['DL_Throughput'])
df['DL_Throughput'].plot()
plt.plot(df['RecordDate'][0:250],df['Num of Cells'][0:250])
plt.xticks(rotation=45)
plt.plot(df['RecordDate'][0:250],df['DL_Throughput'][0:250])
plt.xticks(rotation=45)
plt.plot(df['RecordDate'][0:250],df['Data Availability (%)'][0:250])
plt.xticks(rotation=45)
