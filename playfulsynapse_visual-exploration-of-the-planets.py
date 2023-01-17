import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline 

from matplotlib_venn import venn3 



#ignore some anoying warnings from the seaborn plot

import warnings

warnings.filterwarnings('ignore')
#Import the data

planets_all_obs=pd.read_csv('../input/oec.csv')

planets_all_obs.shape

print("Number of observations:",planets_all_obs.shape[0] )

print("Number of features:",planets_all_obs.shape[1] )
planets_all_obs.describe()
selectedColumns=['PlanetaryMassJpt', 'RadiusJpt','PeriodDays', 'SemiMajorAxisAU', 'Eccentricity',

                 'SurfaceTempK','HostStarMassSlrMass', 'HostStarRadiusSlrRad',

                 'HostStarMetallicity','HostStarTempK',]
corrmat = planets_all_obs[selectedColumns].corr()

sns.heatmap(corrmat, vmax=.8, square=True)
# drop NA values for each column, and create a sett of the planet identifiers for

# each of the remaining values in each attribute.

periodDays_db=planets_all_obs[['PlanetIdentifier','PeriodDays']].dropna()

periodDays_set=set(periodDays_db['PlanetIdentifier'].values.flatten())



axis_db=planets_all_obs[['PlanetIdentifier','SemiMajorAxisAU']].dropna()

axis_set=set(axis_db['PlanetIdentifier'].values.flatten())



starmass_db=planets_all_obs[['PlanetIdentifier','HostStarMassSlrMass']].dropna()

starmass_set=set(starmass_db['PlanetIdentifier'].values.flatten())



venn3([periodDays_set, axis_set, starmass_set], ('PeriodDays', 'SemiMajorAxisAU', 'HostStarMassSlrMass'))

plt.show()
# take a closer look at the correlation of the features in cluster1

selected_cluster1=['PeriodDays', 'SemiMajorAxisAU','HostStarMassSlrMass']

cluster1_noNA=planets_all_obs[selected_cluster1].dropna() #remove the NA values



print("Number of observations: ",cluster1_noNA.shape[0])
sns.pairplot(cluster1_noNA)
# take a closer look at the correlation of the features in cluster2

selected_cluster2=['SurfaceTempK', 'HostStarTempK', 'Eccentricity','RadiusJpt']

cluster2_noNA=planets_all_obs[selected_cluster2].dropna() #remove the NA values



print("Number of observations: ",cluster2_noNA.shape[0])
sns.pairplot(cluster2_noNA)
#Lets examine the relationship between the orbital period and the semi major axis of the orbit.

selectedColumns2=['SemiMajorAxisAU','PeriodDays'] #make a list of the columns we want to examine

planets_noNA=planets_all_obs[selectedColumns2].dropna() #remove the NA values



#add a new column with years instead of days, just to make it easier to compare with the earth

#The discance is meassured in AU = distance from Earth to the sun

planets_noNA['PeriodYears']=planets_noNA['PeriodDays']/365.25

print("Number of observations: ",planets_noNA.shape[0])
sns.set(style="darkgrid", color_codes=True)

g = sns.jointplot("SemiMajorAxisAU", "PeriodYears", data=planets_noNA, kind="reg",xlim=(0, 200), ylim=(0, 1000), color="b", size=7)
#Lets examine the relationship between the surface temperature and the Eccentricity

selectedColumns2=['SurfaceTempK','Eccentricity'] #make a list of the columns we want to examine

planets_noNA2=planets_all_obs[selectedColumns2].dropna() #remove the NA values

print('Number of observations: ', planets_noNA2.shape[0])
sns.set(style="darkgrid", color_codes=True)

g = sns.jointplot("Eccentricity", "SurfaceTempK", data=planets_noNA2, kind="reg",xlim=(0, 1.2), ylim=(0, 4000), color="b", size=7)
selectedColumns4=['SemiMajorAxisAU', 'SurfaceTempK','HostStarTempK']

selectedColumns5=['PeriodDays', 'SurfaceTempK','HostStarTempK']

planets_noNA4=planets_all_obs[selectedColumns4].dropna() #remove the NA values

planets_noNA5=planets_all_obs[selectedColumns5].dropna() #remove the NA values

print('Number of observations using SemiMajorAxisAU: ', planets_noNA4.shape[0])

print('Number of observations using PeriodDays     : ', planets_noNA5.shape[0])
#Lets examine the relationship between the surface temperature and the orbital distance from the planet to the star

sns.set(style="darkgrid", color_codes=True)

g = sns.jointplot("SemiMajorAxisAU", "SurfaceTempK", data=planets_noNA4, kind="scatter",xlim=(0, 7), ylim=(0, 3000), color="b", size=7)
#Lets examine the relationship between the surface temperature and the temperature of the star

sns.set(style="darkgrid", color_codes=True)

g = sns.jointplot("HostStarTempK", "SurfaceTempK", data=planets_noNA4, kind="scatter",xlim=(0, 12000), ylim=(0, 3000), color="b", size=7)
g = sns.jointplot(x="HostStarMassSlrMass", y="HostStarTempK", xlim=(0, 5), ylim=(0, 20000),data=planets_all_obs)