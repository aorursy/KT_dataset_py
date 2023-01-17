#importing the necessary libraries

#for mathematical opeartions and data manipulation
import numpy as np
import pandas as pd

#for data visualisation
import matplotlib.pyplot as plt
import seaborn as sns

#warnings
import warnings
warnings.filterwarnings('ignore')
#Reading the dataset
data_nat = pd.read_csv("../input/mpi/MPI_national.csv")
data_subnat = pd.read_csv("../input/mpi/MPI_subnational.csv")

#Displaying the shape of both the datasets
print("The shape of MPI_national dataset is: ",data_nat.shape)
print("The shape of MPI_subnational is: ",data_subnat.shape)
#The first five rows of the MPI_national dataset
data_nat.head()
# The first five rows of the MPI_subnational dataset
data_subnat.head()
#maximum Headcount ratio of Urban Areas
data_nat['Headcount Ratio Urban'].max()
#Looking at the data when Headcount Ratio Urban is equal to 82.5
data_nat[data_nat['Headcount Ratio Urban']==82.5]
#we created a new column which contain Headcount Ratio of both Urban as well as Rural Area
data_nat['Total Headcount Ratio'] = data_nat['Headcount Ratio Urban']+data_nat['Headcount Ratio Rural']
#Returns the country having maximum HeadCount Ratio
data_nat[data_nat['Total Headcount Ratio']==data_nat['Total Headcount Ratio'].max()] [['Country']]
#Returns the country having minimum Headcount Ratio
data_nat[data_nat['Total Headcount Ratio']==data_nat['Total Headcount Ratio'].min()] [['Country']]
#merges both the datasets together on Country column
data = pd.merge(data_nat,data_subnat, on='Country',how='outer')
#Checking the head of the newly merged dataset
data.head()
#Value counts for World Region
data['World region'].value_counts()
data['World region'] = data['World region'].replace(('East Asia and the Pacific','South Asia','Europe and Central Asia'),('Asia'))
data['World region'].value_counts()
data.tail()
'''We sorted the "Intensity of Deprivation" column in ascending order for only Asian Countries'''

data[data['World region']=='Asia']['Intensity of Deprivation Urban'].sort_values(ascending=True)
#grouped by country and aggregated by maximum of Total Headcount Ratio
d = data_nat.groupby('Country')['Total Headcount Ratio'].agg(max)
d = d.reset_index()
d.sort_values(by='Total Headcount Ratio',ascending=False).head(10).style.background_gradient(cmap='copper')
#Returns the country having highest MPI Rural
data[data['MPI Rural']==data['MPI Rural'].max()][['Country']]
#Creates a new dataframe which contains poverty details of only Afghanistan
afg = data[data['Country']=='Afghanistan']
afg
#checking for the subnational region which has highest MPI
afg[afg['MPI Regional']==afg['MPI Regional'].max()][['Sub-national region']]
afg.groupby('Sub-national region')['MPI Regional'].agg(max).sort_values(ascending=False)
afg.groupby('Sub-national region')['Headcount Ratio Regional'].agg(max).sort_values(ascending=False)
afg.groupby('Sub-national region')['Intensity of deprivation Regional'].agg(max).sort_values(ascending=False)
data_nat[data_nat['Country']=='India']
data_nat[data_nat['Country']=='Afghanistan']
