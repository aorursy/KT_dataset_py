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
import warnings

warnings.filterwarnings("ignore")
path='../input/immigration-dataset'
ctyeconprofile=pd.read_csv(f'{path}/ctyeconprofile.csv',encoding='latin-1')

soi_migration_clean=pd.read_csv(f'{path}/soi_migration_clean.csv',encoding='latin-1')



crime_cov=pd.read_csv(f'{path}/acs_crime_clean.csv',encoding='latin-1')

demo_cov=pd.read_csv(f'{path}/socialexplorer_acs_1yr_clean.csv',encoding='latin-1')

fhfa_cov=pd.read_csv(f'{path}/HPI_AT_BDL_county_clean.csv',encoding='latin-1')
f_d_cov=pd.merge(fhfa_cov,demo_cov,on=['fips','year'],how='outer')
crime_cov_clean=crime_cov.dropna(how='any')

crime_cov_clean['year']=crime_cov_clean['year'].astype(int)

crime_cov_clean['fips']=crime_cov_clean['fips'].astype(float).astype(int)

f_d_c_cov=pd.merge(f_d_cov,crime_cov_clean,on=['fips','year'],how='outer')
soi_migration_clean['year']=soi_migration_clean['year1']

f_d_c_c_cov=pd.merge(f_d_c_cov,ctyeconprofile,on=['fips','year'],how='outer')
smc=soi_migration_clean

length=[]

for i in range(1995,2016):

    length.append(len(smc.loc[(smc['year1']==i) & (smc['year2']==i+1)]))



length=pd.DataFrame(length).T

length.columns=np.arange(1995,2016)

length
results=[]

for i in range(1995,2016):

    results.append(smc.loc[(smc['year1']==i) & (smc['year2']==i+1)])
total_pop=ctyeconprofile[['fips','year','ctyname','tot_pop']]

results_with_pop=[]

for i in range(1995,2016):

    results_with_pop.append(pd.merge(results[i-1995],total_pop.loc[total_pop['year']==i],on='fips'))
for i in range(21):

    temp=results_with_pop[i]

    temp['gross migration']=temp['inmig']+temp['outmig']

    temp['gross migration rate']=temp['gross migration']*100/temp['tot_pop']
df=results_with_pop[0]

for i in range(1,21):

    df=df.append(results_with_pop[i])
df.rename(columns={"year1":"year"},inplace=True)
len(df['fips'].unique())
df_total=pd.merge(df,f_d_c_c_cov,on=['fips','year'],how='outer')
df_total['gross migration rate']=df_total['gross migration rate'].fillna('999999')



a=df_total[(df_total['gross migration rate']=='999999')].index.tolist()



df_total_clean=df_total.drop(a)



df_total_clean['gross migration rate'].isnull().value_counts()
def mean_value_cal_bytime(df,var):

    df_new=df[['fips','year',var]]

    results=[]

    for i in range(1995,2016):

        df_cal=df_new[df_new['year']==i]

        temp=df_cal.apply(np.mean)[var]

        results.append(temp)

    results=pd.DataFrame(results)

    results.columns=[var]

    return results
df_total_clean.columns
lists=['gross migration rate','frac_employed','rate_crime','avearnings',

       'average_hh_income','frac_married','median_hh_income', 'popdens',

       'hpi','tot_violent', 'tot_property','tot_crime','pcincome','pcunempb','avwage',

      'frac_black','frac_hsgrad', 'frac_collegegrad','frac_labforce']
results=[]

for i in lists:

    temp=mean_value_cal_bytime(df_total_clean,i)[i].T.values

    results.append(temp)

    

results=pd.DataFrame(results).T

results.columns=lists

results.index=range(1995,2016)
results_new=results.drop(['tot_violent','tot_property'],axis=1)
import seaborn as sns

from subprocess import check_output
#sns.heatmap(results.corr(), cmap = cmap, linewidths = 0.05, linecolor= 'blue', fmt= '.2f',annot=True,ax=ax)
import matplotlib.pyplot as plt

f, ax = plt.subplots(figsize = (16,12))

cmap = sns.cubehelix_palette(start = 1, rot = 3, gamma=0.8, as_cmap = True) 



sns.heatmap(results.corr(), cmap = cmap, linewidths = 0.05, linecolor= 'blue', fmt= '.2f',annot=True,ax=ax)



#ax.set_title('Amounts per kind and region')

#ax.set_xlabel('covariates')

#ax.set_ylabel('kind')
sns_plot = sns.jointplot('rate_crime','gross migration rate',df_total_clean[df_total_clean['year']==2015],kind='reg',color='red')



sns_plot = sns.jointplot('rate_crime','gross migration rate',df_total_clean[df_total_clean['year']==2010],kind='reg')
results['year']=results.index
results[['gross migration rate','rate_crime',

                    'average_hh_income','popdens','pcincome','tot_property','frac_collegegrad','avwage']].tail()
results_plot=results[['gross migration rate','rate_crime','average_hh_income',

                      'popdens','pcincome','tot_property','frac_collegegrad','avwage']]



sns.set(rc={"figure.figsize": (16, 6)}); 

sns.set(style="white",palette='deep',color_codes=False)



fig,axes=plt.subplots(1,2)

results_plot.plot(style=['+-', 'd--', 'o-.', '.-', 'v:','*-','X--','|-.'],xticks=range(1990,2016,2),ax=axes[0],title='Covariates cross time')



results_plot_percentage=results_plot/results_plot.max()

#results_plot_percentage['gross migration rate']=results_plot['gross migration rate']

results_plot_percentage.plot(style=['+-', 'd--', 'o-.', '.-', 'v:','*-','X--','|-.'],xticks=range(1990,2016,2), logy=True,ax=axes[1],title='Covariates percentage cross time')

sns.despine()
df_plot=df_total_clean[(df_total_clean['year']==2010)|(df_total_clean['year']==2013)|(df_total_clean['year']==2015)][['year','rate_crime',

                                                                                      'average_hh_income','popdens','pcincome','tot_property','frac_collegegrad','avwage']]

sns.pairplot(data=df_plot,hue='year', dropna=True)
df_total_clean.tail()
df_total_clean_1015=df_total_clean[(df_total_clean['year']==2010) | (df_total_clean['year']==2011) | (df_total_clean['year']==2012)| (df_total_clean['year']==2013)

              | (df_total_clean['year']==2014)| (df_total_clean['year']==2015)][['year','fips','gross migration rate',

                                                                                 'rate_crime','average_hh_income','popdens','pcincome','tot_property','frac_collegegrad','avwage']]

mi_data = df_total_clean_1015.set_index(['fips','year'])

mi_data.tail()
!pip install linearmodels

from linearmodels import PanelOLS

mod = PanelOLS( mi_data['gross migration rate'],mi_data[['rate_crime','average_hh_income','popdens','pcincome','tot_property','frac_collegegrad','avwage']], entity_effects=True)

print(mod.fit())
mod = PanelOLS( mi_data['gross migration rate'],mi_data[['rate_crime','popdens','tot_property','frac_collegegrad','avwage']], entity_effects=True)

print(mod.fit())
import plotly.express as px
df_total_clean.columns
df_total_clean_plot=df_total_clean[(df_total_clean['year']==2010) | (df_total_clean['year']==2011) | (df_total_clean['year']==2012)| (df_total_clean['year']==2013)

              | (df_total_clean['year']==2014)| (df_total_clean['year']==2015)][['year','state','pop','fips','gross migration rate','hpi',

                                                                                 'rate_crime','average_hh_income','popdens','pcincome','tot_property','frac_collegegrad','avwage']]

px.scatter(df_total_clean_plot.dropna(), x="avwage", y="gross migration rate", 

           color="state",facet_col ="year",size="pop", size_max=60,trendline='ols')

px.scatter(df_total_clean_plot.dropna(), x="frac_collegegrad", y="gross migration rate", 

           color="state",facet_col ="year",size="pop", size_max=60,trendline='ols')
px.scatter(df_total_clean_plot.dropna(), x="rate_crime", y="gross migration rate", 

           color="state",facet_col ="year",size="pop", size_max=60,trendline='ols')
px.scatter(df_total_clean_plot.dropna(), x="hpi", y="gross migration rate", 

           color="state",facet_col ="year",size="pop", size_max=60,trendline='ols')