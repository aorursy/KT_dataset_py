import pandas as pd
import missingno as msno
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df = pd.read_excel("../input/boroughpop.xlsx")
df2 = pd.read_excel("../input/londonimddecile.xlsx")
df_map = pd.read_excel("../input/mmap.xlsx")
df.head()
df2.head()
df_map.head()
# Change name of column 'ladname' to 'ladnm' for merging data
df2 = df2.rename(columns={'ladname':'ladnm'})
# First merge (df and df2)
data = pd.merge(df,df2, on='ladnm',how='outer')
# Second merge (merged df,df2 and map data)
data = pd.merge(data,df_map, on=['ladnm','lsoacode'],how='outer')
# Overall information about merged data
data.info()
# Look at first 5 rows of the merged data
data.head()
msno.bar(data)
# Number of unique boroughs in London
data.ladnm.nunique()
df.sort_values('population', ascending=False).set_index('ladnm').plot.barh(figsize=(10,8))
# Distribution of Index of Multiple Deprivation
sns.distplot(data[data.imddecile.notnull()].imddecile)
# Proportion of boroughs where gangs are present (In Orange)
(data.groupby('ladnm').first().gangpresent.value_counts()*100 / data.groupby('ladnm').first().gangpresent.value_counts().sum()).plot('bar')
# Chi-Squared test of goodness-of-fit 
from scipy import stats
stats.chisquare(f_obs= data.groupby('ladnm').first().gangpresent.value_counts()*100 / data.groupby('ladnm').first().gangpresent.\
                value_counts().sum(),
                f_exp= [0.5,0.5])
f, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
ax1 = sns.distplot(data[data.vicage.notnull()].vicage,ax=axes[0],color='darkred')
ax2 = sns.distplot(data[data.susage.notnull()].susage,ax=axes[1],color='darkblue')
# Victim sex proportion
labels='Male','Female'
plt.pie(data['vicsex'].value_counts(), labels=labels,autopct='%1.1f%%', startangle = 150, shadow=True)
# Suspect sex proportion
labels='Male','Female'
plt.pie(data['sussex'].value_counts(), labels=labels,autopct='%1.1f%%', startangle = 150,shadow=True)
data.weapon.value_counts().plot('bar')
(data.Status.value_counts() * 100 / data.Status.value_counts().sum()).plot('bar')
plt.title("% proportion homicide cases by Status")
data['date_year'] = data.date.dt.year
data['date_month'] = data.date.dt.month
data.groupby('date_year').size().plot()
plt.title("Number of homicides in London over time")
data.groupby('date_month').size().plot('bar')
plt.title("Number of homocides by month")
f, axes = plt.subplots(1,2, figsize=(10,4),sharey=True)

ax1 = (data[data.sussex=='M'].weapon.value_counts() * 100 / data[data.sussex=='M'].weapon.value_counts().sum()).plot('bar', ax=axes[0])
ax1.set_ylabel("% Proportion")
ax1.set_title("Proportion of Male Suspects by weapon")

ax2 = (data[data.sussex=='F'].weapon.value_counts() * 100 / data[data.sussex=='F'].weapon.value_counts().sum()).plot('bar', ax=axes[1])
ax2.set_ylabel("% Proportion")
ax2.set_title("Proportion of Female Suspects by weapon")
plt.tight_layout()
from scipy.stats import chi2_contingency
table = pd.crosstab(data.sussex, data.weapon)
stat, p, dof, expected = chi2_contingency(table)
print("p-value: ",p) 
# Add information about number of homicides in which victims were females to the original dataframe
data = data.merge(pd.DataFrame(data[data.vicsex=='F'].groupby('ladnm').size(),columns=['num_of_female_vic_cases']).reset_index(),
           how='left',on=['ladnm'])
# Number of homicides in which victims were females by gang present status
data.groupby('ladnm').first().groupby('gangpresent').mean().\
reset_index().plot(y='num_of_female_vic_cases',x= 'gangpresent', kind='barh')
# Variance of Number of homicides in which victims were females in gang present boroughs
data.groupby('ladnm').first().groupby('gangpresent').get_group('Y').num_of_female_vic_cases.var()
# Variance of Number of homicides in which victims were females in gang non-present boroughs
data.groupby('ladnm').first().groupby('gangpresent').get_group('N').num_of_female_vic_cases.var()
from scipy.stats import ttest_ind
d1,d2 = data.groupby('ladnm').first().groupby('gangpresent').get_group('Y').num_of_female_vic_cases,\
data.groupby('ladnm').first().groupby('gangpresent').get_group('N').num_of_female_vic_cases

ttest_ind(d1,d2, equal_var=False)
# Chi Squared Test of Indepenence on Victims' ethnicity and sex
chi2_contingency(pd.crosstab(data.vicethnic, data.vicsex))
data2 = data.copy() # Make a copy of data just for this test
data2 = data2.join(pd.get_dummies(data2.vicethnic)) # Join the dummy variables onto the original dataframe
# Chi Squared test of indepence on relationship between victims having any other ethnic appearance and their sex
chi2_contingency(pd.crosstab(data2['Any Other Ethnic Appearance'], data2.vicsex))
# Chi Squared test of indepence on relationship between victims being White or White British and their sex
chi2_contingency(pd.crosstab(data2['White or White British'], data2.vicsex))
# Chi Squared test of indepence on relationship between victims being Black or Black British and their sex
chi2_contingency(pd.crosstab(data2['Black or Black British'], data2.vicsex))
# Chi Squared test of indepence on relationship between victims being Asian or Asian British and their sex
chi2_contingency(pd.crosstab(data2['Asian or Asian British'], data2.vicsex))
# Percentage Proportion normalized by row
pd.crosstab(data.vicethnic, data.vicsex,normalize='index')
# Chi Squared Test of Indepenence on suspects' sex and status
chi2_contingency(pd.crosstab(data.sussex, data.Status))
# Chi Squared Test of Indepenence on choice of weapons and status
chi2_contingency(pd.crosstab(data.weapon, data.Status))
pd.crosstab(data.weapon, data.Status, normalize='columns')
# Chi Squared Test of Indepenence on victims' ethnicity and status
chi2_contingency(pd.crosstab(data.vicethnic, data.Status))
pd.crosstab(data.vicethnic, data.Status,normalize='index')
# Chi Squared Test of Indepenence on Victims' sex and status
chi2_contingency(pd.crosstab(data.vicsex, data.Status))
pd.crosstab(data.vicsex, data.Status, normalize='index')
# Victims' ethnicity in various boroughs(location)
sns.scatterplot(x='latitude',y='longitude',data=data,hue='vicethnic')
plt.legend(loc='best', bbox_to_anchor=(1,1))
# Victims' sex in various boroughs(location)
sns.scatterplot(x='latitude',y='longitude',data=data,hue='vicsex')
plt.legend(loc=1)
# Suspects' sex in various boroughs(location)
sns.scatterplot(x='latitude',y='longitude',data=data,hue='sussex')
sns.pairplot(data.groupby('ladnm').first().reset_index(), vars=['population','imddecile','num_of_female_vic_cases'], hue='ladnm')