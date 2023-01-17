# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
#properties_2016 contains some of the data in train_2016.csv

prop_data=pd.read_csv('../input/properties_2016.csv') 
missing_data = pd.DataFrame(prop_data.isnull().sum() / float(len(prop_data)),columns=['NullPct'])

missing_data.sort_values(by='NullPct',ascending=False).head(20)
#use only these columns for the prop_data, they have low % of NaN

selected_features=['parcelid', 'assessmentyear',

       'bedroomcnt', 'calculatedfinishedsquarefeet', 'fips', 'fullbathcnt',

       'latitude', 'longitude', 'lotsizesquarefeet','bathroomcnt','landtaxvaluedollarcnt','structuretaxvaluedollarcnt',

       'propertycountylandusecode', 'propertylandusetypeid',

       'propertyzoningdesc', 'rawcensustractandblock', 'regionidcity',

       'regionidzip', 'roomcnt', 'taxamount', 'taxvaluedollarcnt',

       'yearbuilt']

prop_subset=prop_data[selected_features]
#look at number of unique values for each column

prop_subset.nunique()
#Specify numerical columns of interest to perform further analysis on numerical columns

num_columns = ['assessmentyear', 'bathroomcnt', 'bedroomcnt', 'fullbathcnt',

               'landtaxvaluedollarcnt','lotsizesquarefeet', 'regionidzip', 'roomcnt', 

               'structuretaxvaluedollarcnt', 'taxamount','taxvaluedollarcnt', 'yearbuilt']
num_data=prop_subset[num_columns]
from patsy import dmatrices

import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor
df = num_data.dropna()
df.columns
col=['assessmentyear', 'bathroomcnt',  'fullbathcnt',

       'landtaxvaluedollarcnt', 'lotsizesquarefeet', 'regionidzip', 'roomcnt',

       'structuretaxvaluedollarcnt', 'taxvaluedollarcnt',

       'yearbuilt']

features="+".join(col)

features
# using taxamount as dependent variable, fit the model using OLS method

y, X = dmatrices("taxamount ~" + features, data=df, return_type="dataframe")
# For each column, calculate VIF

vif = pd.DataFrame()

vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif["features"] = X.columns
#from statsmodel: One recommendation is that if VIF is greater than 5, then the explanatory variable given by exog_idx is highly collinear with the other explanatory variables, 

#and the parameter estimates will have large standard errors because of this.

vif
prop_subset.drop(['bathroomcnt', 'landtaxvaluedollarcnt','structuretaxvaluedollarcnt'], axis=1, inplace=True)
training=pd.read_csv('../input/train_2016_v2.csv')
merge_data=pd.merge(training, prop_subset, how='inner', on='parcelid')
merge_data.tail()
g = sns.PairGrid(merge_data[['taxamount','roomcnt','calculatedfinishedsquarefeet','logerror']])

g.map(plt.scatter);
merge_data.shape
merge_data.columns
fig = plt.figure()

plt.rcParams['figure.figsize'] = (40, 20)



ax1 = fig.add_subplot(321)

ax2 = fig.add_subplot(322)

ax3 = fig.add_subplot(323)

ax4 = fig.add_subplot(324)

ax5 = fig.add_subplot(325)

ax6 = fig.add_subplot(326)



sns.distplot(merge_data['yearbuilt'].dropna(),color='g', ax=ax1)

sns.distplot(merge_data['taxamount'].dropna(), color='purple',ax=ax2)

sns.distplot(merge_data['calculatedfinishedsquarefeet'].dropna(), color='gray', ax=ax3)

sns.distplot(merge_data['roomcnt'].dropna(), color='b',ax=ax4 )

sns.distplot(merge_data['lotsizesquarefeet'].dropna(), color='red',ax=ax5)

sns.distplot(merge_data['fullbathcnt'].dropna(), color='black',ax=ax6)



plt.show()
map1={31.0: 'Commercial',

     46.0: 'MultiStory Store', 

     47.0: 'Store/Office',

     246.0: 'Duplex',

     247.0: 'Triplex',

     248.0:'Quadruplex', 

    260.0: 'Residential General',

    261.0: 'Single Family Residential',

    262.0: 'Rural Residence',

    263.0: 'Mobile Home',

    264.0: 'Townhouse',

    265.0: 'Cluster Home',

    266.0: 'Condominium',

    267.0: 'Cooperative',

    268.0: 'Row House',

    269.0: 'Planned Unit Development',

    270.0: 'Residential Common Area',

    271.0: 'Timeshare',

    273.0: 'Bungalow',

    274.0: 'Zero Lot Line',

    275.0: 'Manufactured/Modular Homes',

    276.0: 'Patio Home',

    279.0: 'Inferred Single Family',

    290.0: 'Vacant Land',

    291.0: 'Vacant Land' }

merge_data['propertylandusetypeid']=merge_data['propertylandusetypeid'].map(map1)
merge_data['transactiondate']=pd.to_datetime(merge_data['transactiondate'])

merge_data['year'] = merge_data['transactiondate'].dt.year

merge_data['month'] = merge_data['transactiondate'].dt.month
transaction_month=merge_data.groupby(['month', 'propertylandusetypeid']).size()

transaction_summary=pd.DataFrame({'count':transaction_month}).reset_index()

transaction_summary.tail()
#bar plots for type of property transaction throughout the year 2016

plt.rcParams['figure.figsize'] = (30, 15)



pivotPlot=transaction_summary.pivot(index='month', columns='propertylandusetypeid')

p1=pivotPlot.plot(kind='bar', stacked=True,colormap='Spectral', fontsize=20,title="Number of property transactions from Jan-Dec 2016")
(sns

 .FacetGrid(merge_data, hue='logerror', palette="coolwarm", size=10)

 .map(plt.scatter, 'longitude', 'latitude')

 .set( 

    xlabel='Longitude(west/east)',

    ylabel='Latitude(north/south)',

 ))

sns.plt.title('Blue denotes negative log error, red denotes positive', size=24)
g=sns.barplot(x='propertylandusetypeid', y='logerror', data=merge_data)

plt.xticks(rotation=45)

plt.title('Mean log error of the different property types', size=20)