# Objectives of this Notebook is to Analyse India Covid 19 Data,

#  Present notebook is Analysed India position in Corona Virus Epedimic.

#  How Datewise cases increases in India.

#  Discover which states are affected more which are less.

#  How Total Samples versus Positive cases behaves in India.

#  Statewise total sampling position in India

#  What is a position of Testing labs in India.

#  What is a position of Hospitals in India and How prepare we are.



# 1.0 Call libraries



# 1.1 For data manipulations

import numpy as np

import pandas as pd

# 1.2 For plotting

import matplotlib.pyplot as p

#import matplotlib

#import matplotlib as mpl     # For creating colormaps

import seaborn as sns

import matplotlib.pyplot as plt

# 1.3 For data processing

from sklearn.preprocessing import StandardScaler

# 1.4 OS related

import os



# 1.5 Go to folder containing data file

os.chdir("../input")
# 1.6 Read file Covid 19 India



df_agegroup = pd.read_csv("AgeGroupDetails.csv")

df_coviddata = pd.read_csv("covid_19_india.csv")

df_Hospitalbeds =pd.read_csv("HospitalBedsIndia.csv")

df_Testingdetails = pd.read_csv("ICMRTestingDetails.csv")

df_Testinglabs = pd.read_csv("ICMRTestingLabs.csv")

df_Individualdetails = pd.read_csv("IndividualDetails.csv")

df_Population_master= pd.read_csv("population_india_census2011.csv")

df_statewise_testing = pd.read_csv("StatewiseTestingDetails.csv")



#1.7 Cleaning of Covid Data Death field i.e. removal of # character and change to intger datatype

df_coviddata['Deaths'] = df_coviddata['Deaths'].str.strip('#')

df_coviddata['Deaths'] = df_coviddata['Deaths'].astype(int)
#1.8 How the Date Wise Positive cases behave

grp_datewise = df_statewise_testing.groupby(['Date'],as_index =False)

grp_datewise.groups

gp = grp_datewise.agg({'Negative':np.sum,'Positive':np.sum,'TotalSamples':np.sum})

grpgraph =sns.barplot(x = "Date",y="Positive",data = gp)

grpgraph.set_xticklabels(grpgraph.get_xticklabels(), rotation=90,horizontalalignment='center')

# How datewise Confirmed Cases data behave.



sns.distplot(gp.Positive)

#2.0 How Total Samples vs Positive Sample data behave using point plot



graph_sample_positive =sns.pointplot(x="TotalSamples",y="Positive",data =gp)

graph_sample_positive.set_xticklabels(graph_sample_positive.get_xticklabels(), rotation=90,horizontalalignment='center')





#2.1 What is a sample position of different states on a particular date

df_statewise_testing['StateSubStr']= df_statewise_testing.State.str.slice(0,20)

df_datedata = df_statewise_testing.loc[df_statewise_testing['Date'] == '2020-04-27'] 



samplevspositive=sns.barplot(x="StateSubStr",y="TotalSamples",data=df_datedata) 

samplevspositive.set_xticklabels(samplevspositive.get_xticklabels(), rotation=90,horizontalalignment='center')

#2.2 How the Behaviour of Confirmed cases versus Death cases

df_grpcoviddata= df_coviddata.groupby(['State/UnionTerritory'],as_index =False)

df_grpcoviddata.groups

df_grpcoviddata = df_grpcoviddata.agg({'Confirmed':np.sum,

                                          'Cured':np.sum,

                                          'Deaths':np.sum

                                          })



sns.jointplot(x="Confirmed",y="Deaths",data = df_grpcoviddata)

#2.2 Bar Plot of Statewise Confirmed Cases

statewisegraph =sns.barplot(x="State/UnionTerritory",y="Confirmed",data =df_grpcoviddata)

statewisegraph.set_xticklabels(statewisegraph.get_xticklabels(), rotation=90,horizontalalignment='right')

#2.3 How every state prepared with Testing labs

grp_testing_details = df_Testinglabs.groupby(['state'],as_index =False)

grp_testing_details = grp_testing_details.count()



labwisegraph= sns.barplot(x="state",y="lab",data = grp_testing_details)

labwisegraph.set_xticklabels(labwisegraph.get_xticklabels(), rotation=90,horizontalalignment='center')



#2.4 Statewise Confirmed versus position of Public Beds statewise

df_merge =pd.merge(df_grpcoviddata,

         df_Hospitalbeds, 

         left_on='State/UnionTerritory', 

         right_on='State/UT')



#Inserting an exta Column of ratio of Confirm Vs Bed state wise.

df_merge['Confirm_vs_Bed']=df_merge.Confirmed/df_merge.NumPublicBeds_HMIS



Graph_Confirm_vs_Bed =sns.barplot(x='State/UnionTerritory',y='Confirm_vs_Bed',data=df_merge)

Graph_Confirm_vs_Bed.set_xticklabels(Graph_Confirm_vs_Bed.get_xticklabels(), rotation=90,horizontalalignment='center')

#2.5 Covid Data Comparison with respect to Population Density in India

df_Merge_Pop_Hos =pd.merge(df_Population_master,

         df_Hospitalbeds, 

         left_on='State / Union Territory', 

         right_on='State/UT')



df_output=  df_Merge_Pop_Hos.loc[:,['State / Union Territory','Population','NumPublicBeds_HMIS']]

df_output['StatePopulationVsBeds']= df_output.NumPublicBeds_HMIS/df_output.Population



Graph_StatePopulationVsBeds =sns.barplot(x='State / Union Territory',y='StatePopulationVsBeds',data=df_output)

Graph_StatePopulationVsBeds.set_xticklabels(Graph_StatePopulationVsBeds.get_xticklabels(), rotation=90,horizontalalignment='center')
