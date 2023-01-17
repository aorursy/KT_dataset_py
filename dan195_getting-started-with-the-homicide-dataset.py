

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns 

import datetime 

import scipy.stats as stats

import math

import warnings

warnings.simplefilter('ignore')



df = pd.read_csv("../input/Serial Killers Data.csv",encoding = "ISO-8859-1")

df.head()
print(df.shape)
del_list = ['PerfIQ','VerbalIQ','LocAbduct','Nickname','IQ2','Chem','Fired','GetToCrime','XYY','Killed',

           'Combat','BedWet','AgeEvent','IQ1','PsychologicalDiagnosis','Otherinformation','News',

           'DadSA','DadStable','LivChild','LocKilling','PoliceGroupie','MomSA','ParMarSta','Fire',

           'AppliedCop','Attrac','LiveWith','Educ','FamEvent','MomStable','Mental','Animal',

           'HeadInj','PartSex']



for i in del_list:

    del df[i]
df.KillMethod.unique()
Kill_Methods = []

for i in df['KillMethod']:

    if ',' in str(i):

        i.split(',')

        for x in i:

            Kill_Methods.append(x)

    elif i == 'liveabortions':

        Kill_Methods.append(27)

    else:

        try:

            Kill_Methods.append(int(i))

        except:

            continue



Kill_Methods[:] = [int(m) for m in Kill_Methods if m != ',']

#Returns all unique values after splitting observations with multiple methods

set(Kill_Methods)
methods_dict = {0:'Unknown', 1: 'Bludgeon' , 2:'Gun', 3:'Poison',

              4:'Stabbing', 5:'Strangulation', 6:'Pills', 7:'Bomb',

             8:'Suffication',9:'Gassed',10:'Drown',11:'Fire',12:'Starved/Neglect',13:'Shaken',

             14:'Axed',15:'Hanging',18:'RanOver',21:'AlcoholPoisoning',22:'DrugOverdose',

             25:'WithdrewTreatment',27:'LiveAbortions'}
len(Kill_Methods)
ethnicities = ['White','Black','Hispanic','Asian','NativeAmerican','Aboriginal']

ethnicity_dict = {}

gender_dict = {}

for x,y in zip(range(1,7,1),ethnicities):

    ethnicity_dict[x] = y



Gender = ['Male','Female']

for a,b in zip(range(1,3,1),Gender):

    gender_dict[a] = b

    

birthcat_dict = {}

BirthOrder = ['Oldest','Middle','Youngest','Only']

for c,d in zip(range(1,5,1),BirthOrder):

    birthcat_dict[c] = d
for i in df['City'].iloc[:30]:

    if ',' in str(i):

        print(i)
multiple_cities = [x for x in df['City'] if ',' in str(x)]

print(len(multiple_cities),len(df['City'].loc[df.City == 'Multiplecities']))
low_counts = df.City.value_counts()

print(sum(low_counts[low_counts < 2]),sum(low_counts[low_counts > 1]),len(multiple_cities))
def city_totals(str_city):

    Total_City = [x for x in multiple_cities if str(str_city) in x in multiple_cities]

    Total_City = len(Total_City)

    Total_City += low_counts[str_city]

    return Total_City





city_totals("Detroit"),low_counts["Detroit"]
print([x for x in multiple_cities if 'Detroit' in x in multiple_cities])
df.state.value_counts(ascending=False).iloc[:25].plot(kind='barh').invert_yaxis()

plt.title("Serial Homicide Offenders by State")
print("There are {} unique regions in the dataset. \n\n\n{}".format(len(set(df['state'])),df['state'].value_counts(ascending=True)[:20]))
df.City.value_counts().iloc[:30].plot(kind="barh").invert_yaxis()
Actual_City_Totals = {}

Top_Cities = ['Chicago','Houston','LosAngeles','Washington','Philadelphia','Miami','London','NewOrleans',

             'Jacksonville','Richmond','Moscow','NYC-Brooklyn','Phoenix','Detroit','Baltimore','LasVegas',

             'Atlanta','KansasCity','Tampa','NYC-Manhattan','Indianapolis','FortWorth',

             'SanFrancisco','St.Petersburg','Oakland','St.Louis','OklahomaCity','Memphis','Birmingham',

             'Boston','Johannesburg','Cleveland','Portland','Austin','Dallas','Nashville',

              'Cincinnati','Louisville','Sydney','Vienna','Vancouver','CapeTown','Milwaukee','Seattle',

             'Columbus','Wichita','Sacramento','Milan']



for x in Top_Cities:

    Actual_City_Totals[x] = city_totals(x)


import operator

sorted_city_values = sorted(Actual_City_Totals.items(), key=operator.itemgetter(1),reverse=True)

sorted_city_values
plt.style.use(['fivethirtyeight'])

sns.countplot(df['Sex'].map(gender_dict),palette='Blues_d')

plt.title("Gender Makeup")
schoolprob_dict = {0:'No',1:'Yes'}

teased_dict = {0: 'No',1:'Yes'}

sns.countplot(df['SchoolProb'].map(schoolprob_dict),hue=df['Teased'].map(teased_dict))

plt.title("Did the individual have problems in school?")
alc_dict = {1: "Yes",0:"No"}

sns.countplot(df['Killerabusealcohol'].map(alc_dict))

plt.title("Did the killer abuse alcohol?")
drug_dict = {1:"Yes",0:"No"}

sns.countplot(df['Killerabusedrugs'].map(drug_dict))

plt.title("Did the killer abuse drugs?")
Kill_Methods = pd.Series(Kill_Methods).dropna().map(methods_dict)

Kill_Methods.value_counts().sort_values(ascending=True).plot(kind='barh')

plt.title("Methods of killing")
df['Race'].map(ethnicity_dict).value_counts().plot(kind='pie')
df.Race.isnull().sum()
df['BirthCat'].map(birthcat_dict).value_counts().plot(kind='pie')


df.BirthCat.isnull().sum()
sns.distplot(df['YearsBetweenFirstandLast'].dropna(),rug=True)
df.plot(

    x="YearsBetweenFirstandLast",

    y="NumVics",

    kind="scatter"

)
df['NumVics'].dropna().describe()
df.loc[df['NumVics']<150].plot(

    x="YearsBetweenFirstandLast",

    y="NumVics",

    kind="scatter"

)



#sns.regplot(x='YearsBetweenFirstandLast',y='NumVics',data=df.loc[df['NumVics'] < 150])
sns.distplot(df['YearFinal'].dropna(),rug=True)
sns.distplot(df['YearFinal'].dropna(),rug=True).set(xlim=(1900, 2016))
sns.distplot(df['Age1stKill'].dropna(),rug=True)
sns.distplot(df['AgeLastKill'].dropna(),rug=True)
print("Age at first kill: \n{} \n\n\nAge at last kill: \n{} \n\n{} individuals were minors when they killed for the first time.".format(df['Age1stKill'].dropna().describe(),

                                                                                                                                         df['AgeLastKill'].dropna().describe(),

                                                                                                                                         len(df.loc[df['Age1stKill']<18])))
Minor_dict = {1:'Minor',0:'Adult'}

df['Minor'] = np.where(df['Age1stKill'] < 18,1,0)

facetgrid = sns.FacetGrid(df,hue='Minor',size=6)

facetgrid.map(sns.kdeplot,'NumVics',shade=True)

facetgrid.set(xlim=(0,df['NumVics'].max()))

facetgrid.add_legend()
print("Test Statistic:\n{} \n\n\nDescriptive Statistics:\n{}".format(stats.ttest_ind(a= df['NumVics'].dropna().loc[df['Minor']==1],

                b= df['NumVics'].dropna().loc[df['Minor']==0],

                equal_var=False), df['NumVics'].dropna().groupby(df['Minor'].map(Minor_dict)).describe()))



print("Test Statistic:\n{} \n\n\nDescriptive Statistics:\n{}".format(stats.ttest_ind(a= df['YearsBetweenFirstandLast'].dropna().loc[df['Minor']==1],

                b= df['YearsBetweenFirstandLast'].dropna().loc[df['Minor']==0],

                equal_var=False), df['YearsBetweenFirstandLast'].dropna().groupby(df['Minor'].map(Minor_dict)).describe()))
facetgrid = sns.FacetGrid(df,hue='Minor',size=6)

facetgrid.map(sns.kdeplot,'YearsBetweenFirstandLast',shade=True)

facetgrid.set(xlim=(0,df['YearsBetweenFirstandLast'].max()))

facetgrid.add_legend()

VicSex_dict = {1.0:"Men",2.0:"Women",3.0:"Both"}

df['NumVics'].dropna().loc[df['VicSex'] != 9].groupby(df['VicSex'].map(VicSex_dict)).describe()
stats.ttest_ind(a= df['NumVics'].dropna().loc[df['VicSex']==3.0],

                b=df['NumVics'].dropna().loc[df['VicSex']!=3.0],

                equal_var=False)
print(len(df['NumVics'].dropna().loc[df['VicSex']==3.0]),df['NumVics'].dropna().mean())
stats.t.ppf(q=0.025, 

            df=1423)
print(

stats.ttest_ind(a= df['NumVics'].dropna().loc[df['VicSex']==3.0],

                b= df['NumVics'].dropna().loc[df['VicSex']==2.0],

                equal_var=False),



stats.ttest_ind(a= df['NumVics'].dropna().loc[df['VicSex']==3.0],

                b= df['NumVics'].dropna().loc[df['VicSex']==2.0],

                equal_var=False))
df.loc[(df['NumVics']>150) & ((df['VicSex']==2) | (df['VicSex']==1))]
print(

stats.ttest_ind(a= df['NumVics'].dropna().loc[df['VicSex']==2.0],

                b= df['NumVics'].dropna().loc[df['VicSex']==1.0],

                equal_var=False))
print(

stats.ttest_ind(a= df['AgeSeries'].dropna().loc[df['VicSex']==2.0],

                b= df['AgeSeries'].dropna().loc[df['VicSex']==1.0],

                equal_var=False))
Facetgrid = sns.FacetGrid(df,hue='VicSex',size=6)

Facetgrid.map(sns.kdeplot,'AgeSeries',shade=True)

Facetgrid.set(xlim=(0,df['AgeSeries'].max()))

Facetgrid.add_legend()