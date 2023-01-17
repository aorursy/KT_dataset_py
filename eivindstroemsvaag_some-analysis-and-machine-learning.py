import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
# Read the data
tex = pd.read_csv('../input/texture.csv',index_col=0)

# Visualizing the missing data with heatmap
sns.heatmap(tex.isnull(),yticklabels=False,cbar=False,cmap='viridis')
# Filling the NaN values with 0.00
clean = tex.fillna(0.00)
clean.head()
# Check the columns.
clean.info()
sns.heatmap(clean.isnull(),yticklabels=False,cbar=False,cmap='viridis')
# Value counts for the Lithology 
top = clean['LITHOLOGY'].value_counts().head(10)
top.head()
clean ['AREA'].value_counts().head(10)
# Singeling out the Georges Bank columns. 
Georges = clean[clean['AREA']=='GEORGES BANK']
# Finding correlations.
sns.heatmap(Georges.corr(method='pearson'))
#Here is a better visuals for latitude and longitude.
sns.jointplot(x='LATITUDE',y='LATITUDE',
              data=Georges,kind='reg',
              color='b')
sns.set(style='white',color_codes=True)
# More detailed correlations with scikit 
from sklearn.preprocessing import LabelEncoder
labe = LabelEncoder()
dic = {}

labe.fit(Georges.MONTH_COLL.drop_duplicates())
dic['MONTH_COLL'] = list(labe.classes_)
Georges.MONTH_COLL = labe.transform(Georges.MONTH_COLL)
cor = ['LATITUDE','LONGITUDE','DEPTH_M','T_DEPTH','B_DEPTH']
kor = np.corrcoef(Georges[cor].values.T)
sns.set(font_scale=1.5)
map = sns.heatmap(kor,cbar=True,
                  cmap="YlGnBu",
                  annot = True, 
                  square= True,
                  fmt = '.1f',
                  annot_kws = {'size':10}, 
                 yticklabels = cor,
                 xticklabels = cor)
clean ['AREA'].value_counts().head(10)
gulf = clean[clean['AREA']=='GULF OF MEXICO']
gulf.head()
sns.heatmap(gulf.corr())
#the correlations look the same for both gulf and Georges bank
sns.jointplot(x='LATITUDE',y='LATITUDE',
              data=gulf,kind='reg',
              color='b')
sns.set(style='white',color_codes=True)
# We see that some of the correlation values are different. 
from sklearn.preprocessing import LabelEncoder
labe = LabelEncoder()
dic = {}

labe.fit(gulf.MONTH_COLL.drop_duplicates())
dic['MONTH_COLL'] = list(labe.classes_)
gulf.MONTH_COLL = labe.transform(gulf.MONTH_COLL)

cor = ['LATITUDE','LONGITUDE','DEPTH_M','T_DEPTH','B_DEPTH']
kor = np.corrcoef(gulf[cor].values.T)
sns.set(font_scale=1.5)
map = sns.heatmap(kor,cbar=True,
                  cmap="YlGnBu",
                  annot = True, 
                  square= True,
                  fmt = '.1f',
                  annot_kws = {'size':10}, 
                 yticklabels = cor,
                 xticklabels = cor)