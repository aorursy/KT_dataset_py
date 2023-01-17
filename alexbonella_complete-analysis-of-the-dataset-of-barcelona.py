import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
%matplotlib inline 
df_births=pd.read_csv('../input/births.csv')
df_births.info()
df_births.head()
df_births['District Name'].value_counts()
Number_by_year=df_births[['Year','Gender','Number']].groupby(['Year']).sum()
Number_by_year.sort_values(by=['Number'],ascending=False)
df_births[['Year','Gender','Number']].groupby(['Year','Gender']).sum()
Dist_birth=df_births[['District Name','Gender','Number']].groupby(['District Name']).sum()
test=pd.DataFrame(Dist_birth)
New_Dist_birth=test.reset_index(inplace=False) 
New_Dist_birth.sort_values(by=['Number'],ascending=False)
plt.figure(figsize=(14,8))
plt.title('Births by district')
sns.set_style(style='darkgrid')
sns.barplot(x="Number", y="District Name", data=New_Dist_birth.sort_values(by=['Number'],ascending=False))

df_births.describe()
df_births[df_births['Number']==283]
df_births[df_births['Number']==0]
Max_neig=df_births[['District Name','Number']].groupby(['District Name']).max()
test=pd.DataFrame(Max_neig)
New_Max_neig=test.reset_index(inplace=False)
New_Max_neig
barrio=pd.DataFrame()
for i in range(11): 
    barrio=barrio.append(df_births[(df_births['Number']==New_Max_neig['Number'][i])
                & (df_births['District Name']==New_Max_neig['District Name'][i])])

barrio[['Year','District Name','Neighborhood Name','Gender','Number']]

df_death=pd.read_csv('../input/deaths.csv')
df_death.head()
df_death.info()
death_byAge=df_death[['Age','Number']].groupby(['Age']).sum()
test=pd.DataFrame(death_byAge)
New_death_byAge=test.reset_index(inplace=False)
New_death_byAge
plt.figure(figsize=(14,8))
plt.title('Deaths by Age Range ')
sns.set_style(style='darkgrid')
sns.barplot(x="Number", y="Age", data=New_death_byAge.sort_values(by=['Number'],ascending=False))
death_byDist=df_death[['District.Name','Number']].groupby(['District.Name']).sum()
test=pd.DataFrame(death_byDist)
New_death_byDist=test.reset_index(inplace=False)
New_death_byDist.sort_values(by=['Number'],ascending=False)
plt.figure(figsize=(14,8))
plt.title('Deaths by district')
sns.set_style(style='darkgrid')
sns.barplot(x="Number", y="District.Name", data=New_death_byDist.sort_values(by=['Number'],ascending=False))
Max_Death_neig=df_death[['District.Name','Number']].groupby(['District.Name']).max()
test=pd.DataFrame(Max_Death_neig)
New_Max_Death_neig=test.reset_index(inplace=False)
New_Max_Death_neig
death_barrio=pd.DataFrame()
for i in range(10): 
    death_barrio=death_barrio.append(df_death[(df_death['Number']==New_Max_Death_neig['Number'][i])
                & (df_death['District.Name']==New_Max_Death_neig['District.Name'][i])])
death_barrio[['Year','District.Name','Neighborhood.Name','Number']]
df_imigrants=pd.read_csv('../input/immigrants_by_nationality.csv')
df_imigrants.info()
df_imigrants.head()
df_imigrants['Nationality'].nunique()
#  There are a total of 177 nationalities
Number_immi=df_imigrants[['Nationality','Number']].groupby(['Nationality']).sum()
test=pd.DataFrame(Number_immi)
New_Number_immi=test.reset_index(inplace=False)
New_Number_immi.sort_values(by=['Number'],ascending=False).head(15)
plt.figure(figsize=(14,8))
plt.title('IMIGRANTS BARCELONA')
sns.set_style(style='darkgrid')
sns.barplot(x="Number", y="Nationality", data=New_Number_immi.sort_values(by=['Number'],ascending=False).head(15))
immi_by_Dist=df_imigrants[['District Name','Number']].groupby(['District Name']).max()
test=pd.DataFrame(immi_by_Dist)
New_immi_by_neig=test.reset_index(inplace=False)
New_immi_by_neig
immi_barrio=pd.DataFrame()
for i in range(11): 
    immi_barrio=immi_barrio.append(df_imigrants[(df_imigrants['Number']==New_immi_by_neig['Number'][i])
                & (df_imigrants['District Name']==New_immi_by_neig['District Name'][i])])
immi_barrio[['Year','District Name','Neighborhood Name','Nationality','Number']]

df_immi_By_Age=pd.read_csv('../input/immigrants_emigrants_by_age.csv')
df_immi_By_Age.info()
df_immi_By_Age.head()
immi_By_Age=df_immi_By_Age[['Age','Immigrants','Emigrants']].groupby(['Age']).sum()
test=pd.DataFrame(immi_By_Age)
New_immi_By_Age=test.reset_index(inplace=False)
New_immi_By_Age
# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(14, 8))

sns.set_color_codes("muted")
sns.set_style(style='darkgrid')
sns.barplot(x="Immigrants", y="Age", data=New_immi_By_Age,label="IMMIGRANTS",color='r')

sns.set_color_codes("pastel")
plt.title('EMIGRANTS Vs IMMIGRANTS')
sns.set_style(style='darkgrid')
sns.barplot(x="Emigrants", y="Age", data=New_immi_By_Age,label="EMIGRANTS",color='g')

# Add a legend and informative axis label
ax.legend(ncol=2, loc="center right", frameon=True)
sns.despine(left=True, bottom=True)
df_iemmi_by_dest=pd.read_csv('../input/immigrants_emigrants_by_destination.csv')
df_iemmi_by_dest.head()
df_iemmi_by_dest['from'].value_counts().head(3)
df_iemmi_by_dest[df_iemmi_by_dest['from']=='Barcelona'].sort_values(by='weight',ascending=False).head(5)

df_iemmi_by_dest['to'].value_counts().head(3)
df_iemmi_by_dest[df_iemmi_by_dest['to']=='Barcelona'].sort_values(by='weight',ascending=False).head(5)
df_iemmi_by_dest2=pd.read_csv('../input/immigrants_emigrants_by_destination2.csv')
df_iemmi_by_dest2.head() # Barrios de Barcelona 2017
df_iemmi_by_dest2['from'].nunique()
df_iemmi_by_dest2.groupby(['from']).sum().sort_values(by='weight',ascending=False).head(10)
df_iemmi_by_dest2['to'].nunique()
df_iemmi_by_dest2.groupby(['to']).sum().sort_values(by='weight',ascending=False).head(10)

immi_By_sex=pd.read_csv('../input/immigrants_emigrants_by_sex.csv')
immi_By_sex.head()
IE_Bysex=immi_By_sex[['Gender','Immigrants','Emigrants']].groupby(['Gender']).sum()
test=pd.DataFrame(IE_Bysex)
New_IE_Bysex=test.reset_index(inplace=False)
New_IE_Bysex
New_IE_Bysex.plot(kind='bar',x='Gender',colormap='rainbow')
iemmi_bar=immi_By_sex[['District Name','Gender','Immigrants','Emigrants']].groupby(['District Name']).sum().sort_values(by='Immigrants',ascending=False)
test=pd.DataFrame(iemmi_bar)
New_iemmi_bar=test.reset_index(inplace=False)
New_iemmi_bar
# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(14, 8))
plt.title('IMMIGRANTS Vs EMIGRANTS BY DISTRICT')

sns.set_color_codes("muted")
sns.barplot(data=New_iemmi_bar,x='Immigrants',y='District Name',label='Immigrants',color='r')
sns.set_color_codes("pastel")
sns.barplot(data=New_iemmi_bar,x='Emigrants',y='District Name',label='Emigrants',color='g')

# Add a legend and informative axis label
ax.legend(ncol=2, loc="center right", frameon=True)
sns.despine(left=True, bottom=True)

df_baby_names=pd.read_csv('../input/most_frequent_baby_names.csv')
df_baby_names.info()
df_baby_names['Name'].nunique()
# There are 97 types of names between men and women
df_baby_names[df_baby_names['Gender']=='Female']['Name'].nunique()
# There are 50 types of names for girls
df_baby_names[df_baby_names['Gender']=='Male']['Name'].nunique()
# There are 47 types of names for children
df_baby_names['Order'].nunique()

Most_freq=df_baby_names[df_baby_names['Gender']=='Female'][['Name','Frequency']].groupby(['Name']).sum().sort_values(by='Frequency',ascending=False).head(10)
test=pd.DataFrame(Most_freq)
New_Most_freq=test.reset_index(inplace=False)
New_Most_freq
sns.barplot(data=New_Most_freq,x='Frequency',y='Name')
Most_freqM=df_baby_names[df_baby_names['Gender']=='Male'][['Name','Frequency']].groupby(['Name']).sum().sort_values(by='Frequency',ascending=False).head(10)
test=pd.DataFrame(Most_freqM)
New_Most_freqM=test.reset_index(inplace=False)
New_Most_freqM
sns.barplot(data=New_Most_freqM,x='Frequency',y='Name')

df_mf_names=pd.read_csv('../input/most_frequent_names.csv')
df_mf_names.info()
df_mf_names['Decade'].unique()
df_mf_names['Name'].nunique()
# There are 276 different names among men and women in total
df_mf_names[df_mf_names['Gender']=='Female']['Name'].nunique()
# There are 153 different names among women
df_mf_names[df_mf_names['Gender']=='Male']['Name'].nunique()
# There are 123 different names among men
mf_namesW=df_mf_names[df_mf_names['Gender']=='Female'][['Name','Frequency']].groupby('Name').sum().sort_values(by='Frequency',ascending=False).head(10)
test=pd.DataFrame(mf_namesW)
New_mf_namesW=test.reset_index(inplace=False)
New_mf_namesW
plt.title('TOP 10 NAMES WOMEN ')
sns.barplot(data=New_mf_namesW,x='Frequency',y='Name')
mf_namesM=df_mf_names[df_mf_names['Gender']=='Male'][['Name','Frequency']].groupby('Name').sum().sort_values(by='Frequency',ascending=False).head(10)
test=pd.DataFrame(mf_namesM)
New_mf_namesM=test.reset_index(inplace=False)
New_mf_namesM
plt.title('TOP 10 NAMES MEN ')
sns.barplot(data=New_mf_namesM,x='Frequency',y='Name')
mf_by_dec=df_mf_names[['Name','Gender','Decade','Frequency']]
Women_dec=mf_by_dec[mf_by_dec['Gender']=='Female'][['Decade','Frequency']].groupby(['Decade']).max()
test=pd.DataFrame(Women_dec)
New_women_dec=test.reset_index(inplace=False)
New_women_dec
women_mf=pd.DataFrame()
for i in range(11):
    women_mf=women_mf.append(df_mf_names[(df_mf_names['Frequency']==New_women_dec['Frequency'][i]) & (df_mf_names['Gender']=='Female')])
women_mf[['Decade','Name','Frequency']]
Men_dec=mf_by_dec[mf_by_dec['Gender']=='Male'][['Decade','Frequency']].groupby(['Decade']).max()
test=pd.DataFrame(Men_dec)
New_Men_dec=test.reset_index(inplace=False)
New_Men_dec
men_mf=pd.DataFrame()
for i in range(11):
    men_mf=men_mf.append(df_mf_names[(df_mf_names['Frequency']==New_Men_dec['Frequency'][i]) & (df_mf_names['Gender']=='Male')])
men_mf[['Decade','Name','Frequency']]

df_popu=pd.read_csv('../input/population.csv')
df_popu.head()
popu=df_popu[['Age','Number']].groupby(['Age']).sum()
test=pd.DataFrame(popu)
New_popu=test.reset_index(inplace=False)

plt.figure(figsize=(12,8))
sns.barplot(data=New_popu.sort_values(by='Number',ascending=False),x='Number',y='Age')

Unemplo=pd.read_csv('../input/unemployment.csv')
Unemplo.head()
Unemplo.info()
Unemplo['Demand_occupation'].value_counts()
Do_unemplo=Unemplo[['Gender','Demand_occupation','Number']].groupby('Demand_occupation').sum()
Do_unemplo
plt.pie(Do_unemplo
        ,autopct='%1.1f%%',shadow=True,labels=['Reg_Unmp','Unmp_Dem'],startangle=90,radius=0.8)
plt.legend(loc=4)
Unemplo[Unemplo['Gender']=='Male'][['Demand_occupation','Number']].groupby(['Demand_occupation']).sum()
Unemplo[Unemplo['Gender']=='Female'][['Demand_occupation','Number']].groupby(['Demand_occupation']).sum()
Unemplo[Unemplo['Demand_occupation']=='Registered unemployed'][['District Name','Neighborhood Name','Number']].groupby(['District Name']).sum().sort_values(by='Number',ascending=False).head(5)
Unemplo[Unemplo['Demand_occupation']=='Unemployment demand'][['District Name','Neighborhood Name','Number']].groupby(['District Name']).sum().sort_values(by='Number',ascending=False).head(5)

