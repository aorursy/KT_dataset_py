#Import libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
colnames = ["Name","Country","Operator","OperatorCountry","Users","Purpose","PurposeDetail","OrbitClass","OrbitType",

           "OrbitLong","Perigree","Apogee","Eccentricity","Inclination","Period","LaunchMass","DryMass","Power",

            "LaunchDate","ExpLifetime","Contractor","ContractorCountry","LaunchSite","LaunchVehicle","COSPAR","NORAD"]

sat = pd.read_csv("../input/database.csv",names = colnames,header=0)
#Remove records with too many NaN values

droprow = [796]

sat = sat.drop(droprow)

sat = sat.reset_index(drop=True)
sat.head()
sat.shape
sat.dtypes
sat.Name.describe()
name_counts = sat['Name'].value_counts()
name_counts[name_counts == 2]
sat.Country.describe()
sat.Country.value_counts()
fig, ax = plt.subplots(figsize=(14,6))

sns.countplot(sat['Country'], order = sat.Country.value_counts().iloc[:20].index)

plt.xticks(rotation=80)

plt.title("Top 20 countries sending out satellites", fontdict = {'fontsize':20}, pad = 30.0)

plt.xlabel("Country")

plt.ylabel("Number of satellites")
sat.Operator.describe()
fig, ax = plt.subplots(figsize=(14,6))

sns.countplot(sat['Operator'], order = sat.Operator.value_counts().iloc[:20].index)

plt.xticks(rotation=80)

plt.title("Top 20 operators managing satellites", fontdict = {'fontsize':20}, pad = 30.0)

plt.xlabel("Operator")

plt.ylabel("Number of satellites")
sat.OperatorCountry.describe()
fig, ax = plt.subplots(figsize=(14,6))

sns.countplot(sat['OperatorCountry'], order = sat.OperatorCountry.value_counts().iloc[:20].index)

plt.xticks(rotation=80)

plt.title("Top 20 countries of operators managing satellites", fontdict = {'fontsize':20}, pad = 30.0)

plt.xlabel("Country of operator")

plt.ylabel("Number of satellites")
sat.Users.describe()
#Clean up 'Users' data

for i in sat.index:

    if sat.loc[i,'Users'] == 'Commercial/Government':

        sat.loc[i,'Users'] = 'Government/Commercial'

    if sat.loc[i,'Users'] == 'Military/Government':

        sat.loc[i,'Users'] = 'Government/Military'

    if sat.loc[i,'Users'] == 'Commerical':

        sat.loc[i,'Users'] = 'Commercial'

    if sat.loc[i,'Users'] == 'Civil/Government':

        sat.loc[i,'Users'] = 'Government/Civil'

    if sat.loc[i,'Users'] == 'Military ':

        sat.loc[i,'Users'] = 'Military'
fig, ax = plt.subplots(figsize=(14,6))

sns.countplot(sat['Users'], order = sat.Users.value_counts().index)

plt.xticks(rotation=80)

plt.title("Users of satellites", fontdict = {'fontsize':20}, pad = 30.0)

plt.xlabel("User")

plt.ylabel("Number of satellites")
#Create Boolean variables for Users

users = ['Commercial','Government','Military','Civil']



for i in users:

    sat[i] = 0
for i in sat.index:

    if sat.loc[i,'Users'] == 'Commercial':

        sat.loc[i,'Commercial'] = 1

    if sat.loc[i,'Users'] == 'Government':

        sat.loc[i,'Government'] = 1

    if sat.loc[i,'Users'] == 'Military':

        sat.loc[i,'Military'] = 1

    if sat.loc[i,'Users'] == 'Government/Commercial':

        sat.loc[i,'Government'] = 1

        sat.loc[i,'Commercial'] = 1

    if sat.loc[i,'Users'] == 'Civil':

        sat.loc[i,'Civil'] = 1

    if sat.loc[i,'Users'] == 'Military/Commercial':

        sat.loc[i,'Military'] = 1

        sat.loc[i,'Commercial'] = 1

    if sat.loc[i,'Users'] == 'Government/Military':

        sat.loc[i,'Government'] = 1

        sat.loc[i,'Military'] = 1

    if sat.loc[i,'Users'] == 'Government/Civil':

        sat.loc[i,'Government'] = 1

        sat.loc[i,'Civil'] = 1

    if sat.loc[i,'Users'] == 'Military/Civil':

        sat.loc[i,'Military'] = 1

        sat.loc[i,'Civil'] = 1

    if sat.loc[i,'Users'] == 'Commercial/Gov/Mil':

        sat.loc[i,'Commercial'] = 1

        sat.loc[i,'Government'] = 1

        sat.loc[i,'Military'] = 1
user_counts = {}

for i in users:

    user_counts[i] = sat[i].sum()
fig, ax = plt.subplots(figsize=(14,6))

plt.bar(user_counts.keys(),user_counts.values())

plt.title("Users of satellites", fontdict = {'fontsize':20}, pad = 30.0)

plt.xlabel("User")

plt.ylabel("Number of satellites")
sat.Purpose.describe()
fig, ax = plt.subplots(figsize=(14,6))

sns.countplot(sat['Purpose'], order = sat.Purpose.value_counts().iloc[:8].index)

plt.xticks(rotation=80)

plt.title("Purposes of satellites", fontdict = {'fontsize':20}, pad = 30.0)

plt.xlabel("Purpose")

plt.ylabel("Number of satellites")
purposes = ['Communications','Earth Observation/Science','Tech Development','Navigation','Space Science']



for i in purposes:

    sat[i] = 0
for i in sat.index:

    if sat.loc[i,'Purpose'] == 'Communications':

        sat.loc[i,'Communications'] = 1

    if sat.loc[i,'Purpose'] == 'Earth Observation':

        sat.loc[i,'Earth Observation/Science'] = 1

    if sat.loc[i,'Purpose'] == 'Technology Development':

        sat.loc[i,'Tech Development'] = 1

    if sat.loc[i,'Purpose'] == 'Navigation/Global Positioning':

        sat.loc[i,'Navigation'] = 1

    if sat.loc[i,'Purpose'] == 'Space Science':

        sat.loc[i,'Space Science'] = 1

    if sat.loc[i,'Purpose'] == 'Communications/Technology Development':

        sat.loc[i,'Communications'] = 1

    if sat.loc[i,'Purpose'] == 'Naviation/Regional Positioning':

        sat.loc[i,'Navigation'] = 1
purpose_counts = {}

for i in purposes:

    purpose_counts[i] = sat[i].sum()
fig, ax = plt.subplots(figsize=(14,6))

plt.bar(purpose_counts.keys(),purpose_counts.values())

plt.title("Purpose of satellites", fontdict = {'fontsize':20}, pad = 30.0)

plt.xlabel("Purpose")

plt.ylabel("Number of satellites")
sat.PurposeDetail.describe()
fig, ax = plt.subplots(figsize=(14,6))

sns.countplot(sat['PurposeDetail'], order = sat.PurposeDetail.value_counts().index)

plt.xticks(rotation=80)

plt.title("Detailed purpose of satellites", fontdict = {'fontsize':20}, pad = 30.0)

plt.xlabel("Purpose")

plt.ylabel("Number of satellites")
sat.OrbitClass.describe()
sat[sat['OrbitClass'] == 'LEO ']
sat['OrbitClass'].loc[1327] = 'LEO'
sat.OrbitClass.describe()
fig, ax = plt.subplots(figsize=(14,6))

sns.countplot(sat['OrbitClass'], order = sat.OrbitClass.value_counts().index)

plt.xticks(rotation=80)

plt.title("Orbit class of satellites", fontdict = {'fontsize':20}, pad = 30.0)

plt.xlabel("Orbit class")

plt.ylabel("Number of satellites")
sat.OrbitType.describe()
sat[sat['OrbitType'].isnull()]
fig, ax = plt.subplots(figsize=(14,6))

sns.countplot(sat['OrbitType'], order = sat.OrbitType.value_counts().index)

plt.xticks(rotation=80)

plt.title("Orbit type of satellites", fontdict = {'fontsize':20}, pad = 30.0)

plt.xlabel("Orbit type")

plt.ylabel("Number of satellites")
sat.OrbitLong.describe()
sat[sat['OrbitLong'].isnull()]
fig, ax = plt.subplots(figsize=(8,8))

plt.boxplot(sat[sat['OrbitLong'].notnull()].OrbitLong)

plt.title("Boxplot of satellite orbit longitude", fontdict = {'fontsize':20}, pad = 30.0)

plt.ylabel("Orbit Longitude")



#The boxplot reveals that the large majority of satellites have an Orbit Longitude of 0 degrees in the dataset.

#This is likely because a 0 is used to mark the Orbit Longitude for satellites that are not geosynchroneous.
fig, ax = plt.subplots(figsize=(12,6))

plt.hist(sat['OrbitLong'].dropna(), bins = 8)

plt.title("Histogram of satellite orbit longitude", fontdict = {'fontsize':20}, pad = 30.0)

plt.xlabel("Orbit Longitude")
sat.Perigree.describe()
fig, ax = plt.subplots(figsize=(8,8))

plt.boxplot(sat[sat['Perigree'].notnull()].Perigree)

plt.title("Boxplot of satellite perigree", fontdict = {'fontsize':20}, pad = 30.0)

plt.ylabel("Perigree")
fig, ax = plt.subplots(figsize=(12,6))

plt.hist(sat['Perigree'].dropna(), bins = 8)

plt.title("Histogram of satellite perigree", fontdict = {'fontsize':20}, pad = 30.0)

plt.xlabel("Perigree")
sat.Apogee.describe()
fig, ax = plt.subplots(figsize=(8,8))

plt.boxplot(sat[sat['Apogee'].notnull()].Apogee)

plt.title("Boxplot of satellite apogee", fontdict = {'fontsize':20}, pad = 30.0)

plt.ylabel("Apogee")
fig, ax = plt.subplots(figsize=(12,6))

plt.hist(sat['Apogee'].dropna(), bins = 8)

plt.title("Histogram of satellite apogee", fontdict = {'fontsize':20}, pad = 30.0)

plt.xlabel("Apogee")
sat.Eccentricity.describe()
fig, ax = plt.subplots(figsize=(8,8))

plt.boxplot(sat[sat['Eccentricity'].notnull()].Eccentricity)

plt.title("Boxplot of satellite eccentricity", fontdict = {'fontsize':20}, pad = 30.0)

plt.ylabel("Eccentricity")
fig, ax = plt.subplots(figsize=(12,6))

plt.hist(sat['Eccentricity'].dropna(), bins = 8)

plt.title("Histogram of satellite eccentricity", fontdict = {'fontsize':20}, pad = 30.0)

plt.xlabel("Eccentricity")
sat.Inclination.describe()
fig, ax = plt.subplots(figsize=(8,8))

plt.boxplot(sat[sat['Inclination'].notnull()].Inclination)

plt.title("Boxplot of satellite inclination", fontdict = {'fontsize':20}, pad = 30.0)

plt.ylabel("Inclination")
fig, ax = plt.subplots(figsize=(12,6))

plt.hist(sat['Inclination'].dropna(), bins = 8)

plt.title("Histogram of satellite inclination", fontdict = {'fontsize':20}, pad = 30.0)

plt.xlabel("Inclination")
sat.Period.describe()
#Replace non-numeric values

for i in sat.index:

    if sat.loc[i,'Period'] == '8 days':

        sat.loc[i,'Period'] = 8*60*24



sat.Period = pd.to_numeric(sat.Period)
sat.Period.describe()
fig, ax = plt.subplots(figsize=(8,8))

plt.boxplot(sat[sat['Period'].notnull()].Period)

plt.title("Boxplot of satellite period", fontdict = {'fontsize':20}, pad = 30.0)

plt.ylabel("Period")
fig, ax = plt.subplots(figsize=(12,6))

plt.hist(sat['Period'].dropna(), bins = 8)

plt.title("Histogram of satellite period", fontdict = {'fontsize':20}, pad = 30.0)

plt.xlabel("Period")
sat.LaunchMass.describe()
#Replace non-numeric values

for i in sat.index:

    if sat.loc[i,'LaunchMass'] == '5,000+':

        sat.loc[i,'LaunchMass'] = 5000

        

sat.LaunchMass = pd.to_numeric(sat.LaunchMass)
sat.LaunchMass.describe()
sat[sat['LaunchMass'].isnull()]
fig, ax = plt.subplots(figsize=(8,8))

plt.boxplot(sat[sat['LaunchMass'].notnull()].LaunchMass)

plt.title("Boxplot of satellite launch mass", fontdict = {'fontsize':20}, pad = 30.0)

plt.ylabel("Launch Mass")
fig, ax = plt.subplots(figsize=(12,6))

plt.hist(sat['LaunchMass'].dropna(), bins = 8)

plt.title("Histogram of satellite launch mass", fontdict = {'fontsize':20}, pad = 30.0)

plt.xlabel("Launch Mass")
sat.DryMass.describe()
#Replace non-numeric values

for i in sat.index:

    if sat.loc[i,'DryMass'] == "2,316 (BOL)":

        sat.loc[i,'DryMass'] = 2316

    if sat.loc[i,'DryMass'] == "3,010 (BOL)":

        sat.loc[i,'DryMass'] = 3010

    if sat.loc[i,'DryMass'] == "2,500 (BOL)":

        sat.loc[i,'DryMass'] = 2500

    if sat.loc[i,'DryMass'] == "1,050 (BOL)":

        sat.loc[i,'DryMass'] = 1050

    if sat.loc[i,'DryMass'] == "1,500-1,900":

        sat.loc[i,'DryMass'] = 1700

    if sat.loc[i,'DryMass'] == "2,510 (BOL)":

        sat.loc[i,'DryMass'] = 2510

    if sat.loc[i,'DryMass'] == "2,389 (BOL)":

        sat.loc[i,'DryMass'] = 2389

    if sat.loc[i,'DryMass'] == " ":

        sat.loc[i,'DryMass'] = np.nan

    if sat.loc[i,'DryMass'] == "1,700 (BOL)":

        sat.loc[i,'DryMass'] = 1700

        

sat.DryMass = pd.to_numeric(sat.DryMass)
sat.DryMass.describe()
fig, ax = plt.subplots(figsize=(8,8))

plt.boxplot(sat[sat['DryMass'].notnull()].DryMass)

plt.title("Boxplot of satellite dry mass", fontdict = {'fontsize':20}, pad = 30.0)

plt.ylabel("Dry Mass")
fig, ax = plt.subplots(figsize=(12,6))

plt.hist(sat['DryMass'].dropna(), bins = 8)

plt.title("Histogram of satellite dry mass", fontdict = {'fontsize':20}, pad = 30.0)

plt.xlabel("Dry Mass")
sat.Power.describe()
#Replace non-numeric values

for i in sat.index:

    if isinstance(sat.loc[i,'Power'], str):

        sat.loc[i,'Power'] = sat.loc[i,'Power'].strip(" (EOL)")

        sat.loc[i,'Power'] = sat.loc[i,'Power'].strip(" (B")

        sat.loc[i,'Power'] = sat.loc[i,'Power'].replace(",","")

    if sat.loc[i,'Power'] == "500-700":

        sat.loc[i,'Power'] = 600



sat.Power = pd.to_numeric(sat.Power)
sat.Power.describe()
fig, ax = plt.subplots(figsize=(8,8))

plt.boxplot(sat[sat['Power'].notnull()].Power)

plt.title("Boxplot of satellite power", fontdict = {'fontsize':20}, pad = 30.0)

plt.ylabel("Power")
fig, ax = plt.subplots(figsize=(12,6))

plt.hist(sat['Power'].dropna(), bins = 8)

plt.title("Histogram of satellite power", fontdict = {'fontsize':20}, pad = 30.0)

plt.xlabel("Power")
sat.LaunchDate.describe()
sat.LaunchDate = pd.to_datetime(sat.LaunchDate)
sat.LaunchDate.describe()
sat[sat['LaunchDate'].isnull()]
sat['LaunchYear'] = sat['LaunchDate'].dt.year
sat.LaunchYear.describe()
sat.dtypes
fig, ax = plt.subplots(figsize=(14,6))

sat[sat['LaunchYear'].notnull()].LaunchYear.groupby(sat['LaunchYear']).count().plot()

plt.xticks(np.arange(1974,2017,step=2))

plt.title("Time line of satellite launches", fontdict = {'fontsize':20}, pad = 30.0)

plt.ylabel("Number of launches")

plt.xlabel("Year")
sat.ExpLifetime.describe()
#Clean up ExpLifetime data

for i in sat.index:

    if isinstance(sat.loc[i,'ExpLifetime'], str):

        sat.loc[i,'ExpLifetime'] = sat.loc[i,'ExpLifetime'].strip(" yr.")

        sat.loc[i,'ExpLifetime'] = sat.loc[i,'ExpLifetime'].strip("yrs.")

        sat.loc[i,'ExpLifetime'] = sat.loc[i,'ExpLifetime'].strip(" trs,")

        sat.loc[i,'ExpLifetime'] = sat.loc[i,'ExpLifetime'].strip(" hrs.")

        if "-" in sat.loc[i,'ExpLifetime']:

            sat.loc[i,'ExpLifetime'] = (float(sat.loc[i,'ExpLifetime'].split("-")[0]) + 

                                        float(sat.loc[i,'ExpLifetime'].split("-")[1])) / 2

            

sat.ExpLifetime = pd.to_numeric(sat.ExpLifetime)
fig, ax = plt.subplots(figsize=(14,6))

sat[sat['ExpLifetime'].notnull()].ExpLifetime.groupby(sat['ExpLifetime']).count().plot(kind="bar")

plt.title("Expected lifetime of satellites", fontdict = {'fontsize':20}, pad = 30.0)

plt.ylabel("Number of satellites")

plt.xlabel("Expected lifetime (in years)")
fig, ax = plt.subplots(figsize=(12,6))

plt.hist(sat['ExpLifetime'].dropna(), bins = 8)

plt.title("Histogram of satellite expected lifetime", fontdict = {'fontsize':20}, pad = 30.0)

plt.xlabel("Expected Lifetime")
sat.Contractor.describe()
fig, ax = plt.subplots(figsize=(14,6))

sns.countplot(sat['Contractor'], order = sat.Contractor.value_counts().iloc[:20].index)

plt.xticks(rotation=80)

plt.title("Top 20 contractors launching satellites", fontdict = {'fontsize':20}, pad = 30.0)

plt.xlabel("Contractor")

plt.ylabel("Number of satellites")
sat.ContractorCountry.describe()
fig, ax = plt.subplots(figsize=(14,6))

sns.countplot(sat['ContractorCountry'], order = sat.ContractorCountry.value_counts().iloc[:20].index)

plt.xticks(rotation=80)

plt.title("Top 20 countries of contractors launching satellites", fontdict = {'fontsize':20}, pad = 30.0)

plt.xlabel("Country of contractor")

plt.ylabel("Number of satellites")
sat.LaunchSite.describe()
fig, ax = plt.subplots(figsize=(14,6))

sns.countplot(sat['LaunchSite'], order = sat.LaunchSite.value_counts().index)

plt.xticks(rotation=80)

plt.title("Satellites launch sites", fontdict = {'fontsize':20}, pad = 30.0)

plt.xlabel("Launch Site")

plt.ylabel("Number of satellites")
sat.LaunchVehicle.describe()
fig, ax = plt.subplots(figsize=(14,6))

sns.countplot(sat['LaunchVehicle'], order = sat.LaunchVehicle.value_counts().iloc[:20].index)

plt.xticks(rotation=80)

plt.title("Top 20 launch vehicles for satellites", fontdict = {'fontsize':20}, pad = 30.0)

plt.xlabel("Launch Vehicle")

plt.ylabel("Number of satellites")
sat.COSPAR.describe()
sat.NORAD.describe()
fig, ax = plt.subplots(figsize=(7,7))

sns.heatmap(sat.corr(), cmap = 'Spectral', center = 0)

plt.title("Heatmap of correlations", fontdict = {'fontsize':20}, pad = 30.0)
fig, ax = plt.subplots(figsize=(8,6))

sns.scatterplot(sat.OrbitLong, sat.ExpLifetime)

plt.title("Scatterplot of Expected lifetime against Orbit Longitude", fontdict = {'fontsize':20}, pad = 30.0)

plt.xlabel("Orbit Longitude")

plt.ylabel("Expected lifetime")
fig, ax = plt.subplots(figsize=(8,6))

sns.scatterplot(sat.Perigree, sat.ExpLifetime)

plt.title("Scatterplot of Expected lifetime against Perigree", fontdict = {'fontsize':20}, pad = 30.0)

plt.xlabel("Perigree")

plt.ylabel("Expected lifetime")
fig, ax = plt.subplots(figsize=(8,6))

sns.scatterplot(sat.Apogee, sat.ExpLifetime)

plt.title("Scatterplot of Expected lifetime against Apogee", fontdict = {'fontsize':20}, pad = 30.0)

plt.xlabel("Apogee")

plt.ylabel("Expected lifetime")
fig, ax = plt.subplots(figsize=(8,6))

sns.scatterplot(sat.Eccentricity, sat.ExpLifetime)

plt.title("Scatterplot of Expected lifetime against Eccentricity", fontdict = {'fontsize':20}, pad = 30.0)

plt.xlabel("Eccentricity")

plt.ylabel("Expected lifetime")
fig, ax = plt.subplots(figsize=(8,6))

sns.scatterplot(sat.Period, sat.ExpLifetime)

plt.title("Scatterplot of Expected lifetime against Period", fontdict = {'fontsize':20}, pad = 30.0)

plt.xlabel("Period")

plt.ylabel("Expected lifetime")
fig, ax = plt.subplots(figsize=(8,6))

sns.scatterplot(sat.LaunchMass, sat.ExpLifetime)

plt.title("Scatterplot of Expected lifetime against LaunchMass", fontdict = {'fontsize':20}, pad = 30.0)

plt.xlabel("LaunchMass")

plt.ylabel("Expected lifetime")
fig, ax = plt.subplots(figsize=(8,6))

sns.scatterplot(sat.DryMass, sat.ExpLifetime)

plt.title("Scatterplot of Expected lifetime against DryMass", fontdict = {'fontsize':20}, pad = 30.0)

plt.xlabel("DryMass")

plt.ylabel("Expected lifetime")
fig, ax = plt.subplots(figsize=(8,6))

sns.scatterplot(sat.Power, sat.ExpLifetime)

plt.title("Scatterplot of Expected lifetime against Power", fontdict = {'fontsize':20}, pad = 30.0)

plt.xlabel("Power")

plt.ylabel("Expected lifetime")
fig, ax = plt.subplots(figsize=(12,7))

topCountries = sat.Country.value_counts().head(10).index.tolist()

sns.boxplot(x=sat[sat['Country'].isin(topCountries)].Country, y=sat.ExpLifetime, order = topCountries, showfliers=False)

plt.xticks(rotation=80)

plt.title("Boxplots of expected lifetime for sattelites from top 10 countries",

          fontdict = {'fontsize':20}, pad = 30.0)

plt.xlabel("Country")

plt.ylabel("Expected lifetime")
fig, ax = plt.subplots(figsize=(12,7))

topOperators = sat[sat['ExpLifetime'].notnull()].Operator.value_counts().head(5).index.tolist()

sns.boxplot(x=sat[((sat['Operator'].isin(topOperators)) & (sat['ExpLifetime'].notnull()))].Operator,

            y=sat.ExpLifetime, order = topOperators, showfliers=False)

plt.xticks(rotation=80)

plt.title("Boxplots of expected lifetime for sattelites from top 5 operators",

          fontdict = {'fontsize':20}, pad = 30.0)

plt.xlabel("Operator")

plt.ylabel("Expected lifetime")
fig, ax = plt.subplots(figsize=(12,7))

topOperatorCountries = sat[sat['ExpLifetime'].notnull()].OperatorCountry.value_counts().head(5).index.tolist()

sns.boxplot(x=sat[((sat['OperatorCountry'].isin(topOperatorCountries)) & (sat['ExpLifetime'].notnull()))].OperatorCountry,

            y=sat.ExpLifetime, order = topOperatorCountries, showfliers=False)

plt.xticks(rotation=80)

plt.title("Boxplots of expected lifetime for sattelites from top 5 operator countries",

          fontdict = {'fontsize':20}, pad = 30.0)

plt.xlabel("Operator Country")

plt.ylabel("Expected lifetime")
fig, ax = plt.subplots(figsize=(12,7))

sns.boxplot(x=sat.Users, y=sat.ExpLifetime, showfliers=False)

plt.xticks(rotation=80)

plt.title("Boxplots of expected lifetime for sattelites from different users",

          fontdict = {'fontsize':20}, pad = 30.0)

plt.xlabel("Users")

plt.ylabel("Expected lifetime")
fig, ax = plt.subplots(figsize=(7,7))

sns.boxplot(x=sat.Commercial, y=sat.ExpLifetime, showfliers=False)

plt.title("Boxplots of expected lifetime for commercial vs non-commercial satellites",

          fontdict = {'fontsize':20}, pad = 30.0)

plt.xticks(np.arange(2), ("Non-Commercial","Commercial"))

plt.xlabel("Commercial vs non-commercial")

plt.ylabel("Expected lifetime")
fig, ax = plt.subplots(figsize=(7,7))

sns.boxplot(x=sat.Government, y=sat.ExpLifetime, showfliers=False)

plt.title("Boxplots of expected lifetime for government vs non-government satellites",

          fontdict = {'fontsize':20}, pad = 30.0)

plt.xticks(np.arange(2), ("Non-Government","Government"))

plt.xlabel("Government vs non-government")

plt.ylabel("Expected lifetime")
fig, ax = plt.subplots(figsize=(7,7))

sns.boxplot(x=sat.Military, y=sat.ExpLifetime, showfliers=False)

plt.title("Boxplots of expected lifetime for military vs non-military satellites",

          fontdict = {'fontsize':20}, pad = 30.0)

plt.xticks(np.arange(2), ("Non-Military","Military"))

plt.xlabel("Military vs non-military")

plt.ylabel("Expected lifetime")
fig, ax = plt.subplots(figsize=(7,7))

sns.boxplot(x=sat.Civil, y=sat.ExpLifetime, showfliers=False)

plt.title("Boxplots of expected lifetime for civil vs non-civil satellites",

          fontdict = {'fontsize':20}, pad = 30.0)

plt.xticks(np.arange(2), ("Non-civil","Civil"))

plt.xlabel("Civil vs non-civil")

plt.ylabel("Expected lifetime")
fig, ax = plt.subplots(figsize=(12,7))

topPurpose = sat[sat['ExpLifetime'].notnull()].Purpose.value_counts().head(8).index.tolist()

sns.boxplot(x=sat[((sat['Purpose'].isin(topPurpose)) & (sat['ExpLifetime'].notnull()))].Purpose,

            y=sat.ExpLifetime, order = topPurpose, showfliers=False)

plt.xticks(rotation=80)

plt.title("Boxplots of expected lifetime for sattelites from top 8 purposes",

          fontdict = {'fontsize':20}, pad = 30.0)

plt.xlabel("Purposes")

plt.ylabel("Expected lifetime")
fig, ax = plt.subplots(figsize=(12,7))

sns.boxplot(x=sat.OrbitClass, y=sat.ExpLifetime, showfliers=False)

plt.xticks(rotation=80)

plt.title("Boxplots of expected lifetime for sattelites from different orbit classes",

          fontdict = {'fontsize':20}, pad = 30.0)

plt.xlabel("Orbit Class")

plt.ylabel("Expected lifetime")
fig, ax = plt.subplots(figsize=(12,7))

sns.boxplot(x=sat.OrbitType, y=sat.ExpLifetime, showfliers=False)

plt.xticks(rotation=80)

plt.title("Boxplots of expected lifetime for sattelites from different orbit types",

          fontdict = {'fontsize':20}, pad = 30.0)

plt.xlabel("Orbit Type")

plt.ylabel("Expected lifetime")
fig, ax = plt.subplots(figsize=(14,6))

sat[sat['LaunchDate'].notnull()].ExpLifetime.groupby(sat['LaunchDate'].dt.year).mean().plot()

plt.xticks(np.arange(1974,2017,step=2))

plt.title("Average expected lifetime per launch year", fontdict = {'fontsize':20}, pad = 30.0)

plt.ylabel("Average expected lifetime")

plt.xlabel("Year")
fig, ax = plt.subplots(figsize=(12,7))

topContractors = sat[sat['ExpLifetime'].notnull()].Contractor.value_counts().head(8).index.tolist()

sns.boxplot(x=sat[((sat['Contractor'].isin(topContractors)) & (sat['ExpLifetime'].notnull()))].Contractor,

            y=sat.ExpLifetime, showfliers=False)

plt.xticks(rotation=80)

plt.title("Boxplots of expected lifetime for sattelites from different contractors",

          fontdict = {'fontsize':20}, pad = 30.0)

plt.xlabel("Contractor")

plt.ylabel("Expected lifetime")
fig, ax = plt.subplots(figsize=(12,7))

topContractorCountries = sat[sat['ExpLifetime'].notnull()].ContractorCountry.value_counts().head(8).index.tolist()

sns.boxplot(x=sat[((sat['ContractorCountry'].isin(topContractorCountries)) & (sat['ExpLifetime'].notnull()))].ContractorCountry,

            y=sat.ExpLifetime, showfliers=False)

plt.xticks(rotation=80)

plt.title("Boxplots of expected lifetime for sattelites from different contractor countries",

          fontdict = {'fontsize':20}, pad = 30.0)

plt.xlabel("Contractor country")

plt.ylabel("Expected lifetime")
fig, ax = plt.subplots(figsize=(12,7))

topLaunchSites = sat[sat['ExpLifetime'].notnull()].LaunchSite.value_counts().head(8).index.tolist()

sns.boxplot(x=sat[sat['LaunchSite'].isin(topLaunchSites)].LaunchSite, y=sat.ExpLifetime, showfliers=False)

plt.xticks(rotation=80)

plt.title("Boxplots of expected lifetime for sattelites from different contractor countries",

          fontdict = {'fontsize':20}, pad = 30.0)

plt.xlabel("Contractor country")

plt.ylabel("Expected lifetime")
fig, ax = plt.subplots(figsize=(7,7))

sns.boxplot(x=sat.Communications, y=sat.ExpLifetime, showfliers=False)

plt.title("Boxplots of expected lifetime for Communications purpose vs non-communication satellites",

          fontdict = {'fontsize':20}, pad = 30.0)

plt.xticks(np.arange(2), ("Non-communications","Communications"))

plt.xlabel("Communications vs non-communications")

plt.ylabel("Expected lifetime")
fig, ax = plt.subplots(figsize=(7,7))

sns.boxplot(x=sat["Earth Observation/Science"], y=sat.ExpLifetime, showfliers=False)

plt.title("Boxplots of expected lifetime for Earth Observation/Science purpose vs non-earth observation/science satellites",

          fontdict = {'fontsize':20}, pad = 30.0)

plt.xticks(np.arange(2), ("Non-Earth Observation/Science","Earth Observation/Science"))

plt.xlabel("Earth Observation/Science vs non-earth observation/science")

plt.ylabel("Expected lifetime")
fig, ax = plt.subplots(figsize=(7,7))

sns.boxplot(x=sat["Tech Development"], y=sat.ExpLifetime, showfliers=False)

plt.title("Boxplots of expected lifetime for Tech Development purpose vs non-tech development satellites",

          fontdict = {'fontsize':20}, pad = 30.0)

plt.xticks(np.arange(2), ("Non-Tech Development","Tech Development"))

plt.xlabel("Tech Development vs non-Tech Development")

plt.ylabel("Expected lifetime")
fig, ax = plt.subplots(figsize=(7,7))

sns.boxplot(x=sat.Navigation, y=sat.ExpLifetime, showfliers=False)

plt.title("Boxplots of expected lifetime for Navigation purpose vs non-navigation satellites",

          fontdict = {'fontsize':20}, pad = 30.0)

plt.xticks(np.arange(2), ("Non-Navigation","Navigation"))

plt.xlabel("Navigation vs non-Navigation")

plt.ylabel("Expected lifetime")
fig, ax = plt.subplots(figsize=(7,7))

sns.boxplot(x=sat["Space Science"], y=sat.ExpLifetime, showfliers=False)

plt.title("Boxplots of expected lifetime for Space Science purpose vs non-Space Science satellites",

          fontdict = {'fontsize':20}, pad = 30.0)

plt.xticks(np.arange(2), ("Non-Space Science","Space Science"))

plt.xlabel("Space Science vs non-Space Science")

plt.ylabel("Expected lifetime")
from sklearn.linear_model import LinearRegression
features = ['Perigree','Inclination','LaunchMass','Commercial','Government','Military','Civil','Communications',

            'Earth Observation/Science','Tech Development','Navigation','Space Science']



feat = sat[features]

target = sat['ExpLifetime']
nan_rows = np.where(np.asanyarray(np.isnan(feat)))[0].tolist()

feat = feat.drop(nan_rows).reset_index(drop=True)

target = target.drop(nan_rows).reset_index(drop=True)
nan_y_rows = np.where(np.asanyarray(np.isnan(target)))[0].tolist()

feat = feat.drop(nan_y_rows).reset_index(drop=True)

target = target.drop(nan_y_rows).reset_index(drop=True)
X_train, X_test, y_train, y_test = train_test_split(feat,target,test_size = 0.33)
lr = LinearRegression()

lr.fit(X_train,y_train)

lr.score(X_train,y_train)
lr.score(X_test,y_test)
lr.coef_