# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #data visualization



import warnings            

warnings.filterwarnings("ignore") 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Load dataset

data=pd.read_csv('../input/Hospital_Inpatient_Discharges__SPARCS_De-Identified___2015.csv')

#data includes how many rows and columns

data.shape

print("Our data has {} rows and {} columns".format(data.shape[0],data.shape[1]))

#Features name in data

data.columns
#diplay first 5 rows

data.head()
#display last 5 rows

data.tail()
print("Data Type:")

data.dtypes
#column name change

data.columns=[each.replace(" ","_") for each in data.columns]



#remove dollar sign

data.Total_Charges=[each.replace("$","") for each in data.Total_Charges]

data.Total_Costs=[each.replace("$","") for each in data.Total_Costs]



#lets convert object to float

data["Total_Charges"]=data["Total_Charges"].astype('float')

data["Total_Costs"]=data["Total_Costs"].astype('float')



#Delete the + sign

data.Length_of_Stay=[each.replace("+","") if(each=="120 +") else each for each in data.Length_of_Stay]

#lets convert object to int

data["Length_of_Stay"]=data["Length_of_Stay"].astype('int')
#Let's look again

data.dtypes
data.loc[:,["Total_Costs","Total_Charges","Birth_Weight","Length_of_Stay"]].describe()
#checking for missing values

print("Are there missing values? {}".format(data.isnull().any().any()))

#missing value control in features

data.isnull().sum()
assert data["Hospital_County"].notnull().all()
#we found out how many Type of Admission

print("Type of Admission in Dataset:\n")

print(data.Type_of_Admission.unique())

#we found out how many Age group

print("\n\nAge Group in Dataset:\n")

print(data.Age_Group.unique())

#we found out how many ARP Risk of Mortality

print("\n\nARP Risk of Mortality:\n")

print(data.APR_Risk_of_Mortality.unique())

#we found out how many hospital country in our data

print("\n\nHospital Country in Dataset:\n")

print("There are {} different values\n".format(len(data.Hospital_County.unique())))

print(data.Hospital_County.unique())

#we found out how many ARP MDC Description

print("\n\nARP MDC Description(disease diagnosis) in Dataset:\n")

print("There are {} different values\n".format(len(data.APR_MDC_Description.unique())))

print(data.APR_MDC_Description.unique())
#We group features by data numbers

#show it if missing value(dropna=False)

data["Type_of_Admission"].value_counts(dropna=False)
#number of patients by age groups

#show it if missing value(dropna=False)

data["Age_Group"].value_counts(dropna=False)
#show it if missing value(dropna=False)

print("Patients with or without abortion:\n")

print(data["Abortion_Edit_Indicator"].value_counts(dropna=False))
#filtering

data_newborn=data['Type_of_Admission']=='Newborn'

print("Total Newborns:",data_newborn.count())

data[data_newborn].head()
#grouping of mortality risk values

#show it if missing value(dropna=False)

data["APR_Severity_of_Illness_Description"].value_counts(dropna=False)


data_new = data.head()

melted = pd.melt(frame = data_new, id_vars = 'APR_MDC_Description', value_vars = ['Age_Group','Type_of_Admission'])

melted
#firstly lets create 2 data frame

data1=data['APR_MDC_Description'].tail()

data2=data['Age_Group'].tail()



conc_data_col=pd.concat([data1,data2],axis=1)

conc_data_col
#data frames from dictionary

Hospital=list(data["Hospital_County"].head())

Facility=list(data["Facility_Name"].head())

Year=list(data["Discharge_Year"].head())

Costs=list(data["Total_Costs"].head())



list_label=["hospital_country","facility_name","discharge_year","total_costs"]

list_col=[Hospital,Facility,Year,Costs]

zipped=list(zip(list_label,list_col))

data_dict=dict(zipped)



df=pd.DataFrame(data_dict)

df
#add new column

data["Entry_Year"]=0

data.head()
#ploting

data1=data.loc[:,["Total_Costs","Total_Charges","Birth_Weight","Length_of_Stay"]]

data1.plot()

plt.show()

#this is complete
#To solve the above complexity

#subplot

data1.plot(subplots=True)

plt.show()
#histogram

data1.plot(kind="hist",y="Total_Costs",bins=50,range=(0,250),normed=True)

plt.show()
#histogram subplot with non cumulative an cumulative

fig,axes=plt.subplots(nrows=2,ncols=1)



data1.plot(kind="hist",y="Total_Costs",bins=50,range=(0,250),normed=True,ax=axes[0])

data1.plot(kind="hist",y="Total_Costs",bins=50,range=(0,250),normed=True,ax=axes[1],cumulative=True)



plt.savefig("Graph.png")

plt.show()
print(df["discharge_year"])

df.discharge_year=pd.to_datetime(df["discharge_year"])

#lets make discharge_year as index

df=df.set_index("discharge_year")

df
df.resample("A").mean()

#lets resample with month

#df.resample("M").mean()
#indexing data frame

#using loc accessor

print(data.loc[85,['APR_DRG_Description']])

#selecting only some columns

data[["APR_DRG_Description","Age_Group","Length_of_Stay"]].head(20)
#silincing and indexing data series

print(data.loc[1:10,"Race":"Length_of_Stay"])

#from something to end

data.loc[1:10,"Gender":]
first_filter=data.Gender=="F"

second_filter=data.Abortion_Edit_Indicator=="Y"

data[first_filter & second_filter].head()

#filtering columns based others

#data.Gender[data.Race=="Black/African American"]
#Defining column using other columns

data["Average_Costs"]=data.Total_Costs.mean()

data.head()



#print(data.Total_Costs.apply(lambda n:n/2))
print("Total hospitalization times for patients admitted to the hospital as Urgent:",

      data['Length_of_Stay'][data['Type_of_Admission']=='Urgent'].sum())



#The first value of unique races of patients coming to the hospital

data.groupby("Race").first()
