# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
no_govt_hospital = pd.read_csv("/kaggle/input/hospitals-and-beds-in-india/Number of Government Hospitals and Beds in Rural and Urban Areas .csv",header=None)



no_govt_hospital = no_govt_hospital[2:38]



no_govt_hospital = no_govt_hospital.rename(columns = {0:"State",1:"Rural Hospitals",2:"Rural Beds",3:"Urban Hospitals",4:"Urban Beds",5:"Date"})

no_govt_hospital = no_govt_hospital.reset_index(drop=True)
no_govt_hospital
%matplotlib inline
no_govt_hospital["Rural Hospitals"] =no_govt_hospital["Rural Hospitals"].astype(int)

no_govt_hospital["Rural Beds"] =no_govt_hospital["Rural Beds"].astype(int)

no_govt_hospital["Urban Hospitals"] =no_govt_hospital["Urban Hospitals"].astype(int)

no_govt_hospital["Urban Beds"] =no_govt_hospital["Urban Beds"].astype(int)

no_govt_hospital["Total Beds"] = no_govt_hospital["Rural Beds"] + no_govt_hospital["Urban Beds"]
no_govt_hospital.sort_values(by=["Total Beds"],ascending=False).plot.bar(x='State', rot=90, title='Government Hospital and Beds', figsize=(20,10),logy=True)
hospital_beds = pd.read_csv("/kaggle/input/hospitals-and-beds-in-india/Hospitals_and_Beds_statewise.csv")



hospital_beds = hospital_beds.rename(columns={'Unnamed: 0':"State",'Unnamed: 6':"Total Beds"})
hospital_beds = hospital_beds.fillna(0)

hospital_beds = hospital_beds[:36]
hospital_beds.head()
hospital_beds["PHC"] =hospital_beds["PHC"].astype(int)

hospital_beds["CHC"] =hospital_beds["CHC"].astype(int)

hospital_beds["SDH"] =hospital_beds["SDH"].astype(int)

hospital_beds["DH"] =hospital_beds["DH"].astype(int)

hospital_beds["Total Beds"] =hospital_beds["Total Beds"].astype(int)

#hospital_beds["Total"] =hospital_beds["Total Beds"].astype(int)

hospital_beds.sort_values(by=["Total Beds"],ascending=False).plot.bar(x='State', rot=90, title='Hospital and Beds', figsize=(20,10),logy=True)
hospital_beds_defence = pd.read_csv("/kaggle/input/hospitals-and-beds-in-india/Hospitals and Beds maintained by Ministry of Defence.csv")



hospital_beds_defence = hospital_beds_defence.drop(columns=["S. No."])

hospital_beds_defence =hospital_beds_defence[:29]
hospital_beds_defence.head()
hospital_beds_defence["No. of Hospitals"] =hospital_beds_defence["No. of Hospitals"].astype(int)

hospital_beds_defence["No. of beds"] =hospital_beds_defence["No. of beds"].astype(int)

hospital_beds_defence.sort_values(by=["No. of beds"],ascending=False).plot.bar(x='Name of State', rot=90, title='Defence Hospital and Beds', figsize=(20,10),logy=True)
esic_hospital = pd.read_csv("/kaggle/input/hospitals-and-beds-in-india/Employees State Insurance Corporation .csv")

esic_hospital = esic_hospital.rename(columns={"Employees State Insurance Corporation Hospitals and beds (as on 31.03.2017)":1,"Unnamed: 1":"State","Unnamed: 2":"Total Hospital","Unnamed: 3":"Total Beds"})

esic_hospital = esic_hospital.drop(columns=[1])[1:30].reset_index(drop= True)
esic_hospital.head()
esic_hospital["Total Hospital"] =esic_hospital["Total Hospital"].astype(int)

esic_hospital["Total Beds"] =esic_hospital["Total Beds"].astype(int)

esic_hospital.sort_values(by=["Total Beds"],ascending=False).plot.bar(x='State', rot=90, title='ESIC Hospital and Beds', figsize=(20,10),logy=True)
railway_hospitals = pd.read_csv("/kaggle/input/hospitals-and-beds-in-india/Hospitals and beds maintained by Railways.csv")

railway_hospitals = railway_hospitals.rename(columns={"Number of Hospitals and beds in Railways (as on 21/03/2018)":1,"Unnamed: 1":"Zone","Unnamed: 2":"No of Hospitals","Unnamed: 3":"No of Beds"})

railway_hospitals = railway_hospitals[1:26]

railway_hospitals = railway_hospitals.drop(columns=[1])
railway_hospitals.head()

railway_hospitals["No of Hospitals"] =railway_hospitals["No of Hospitals"].astype(int)

railway_hospitals["No of Beds"] =railway_hospitals["No of Beds"].astype(int)

railway_hospitals.sort_values(by=["No of Beds"],ascending=False).plot.bar(x='Zone', rot=90, title='Railway Hospital and Beds', figsize=(20,10),logy=True)
ayuush_hospitals = pd.read_csv("/kaggle/input/hospitals-and-beds-in-india/AYUSHHospitals.csv",header=None ,index_col=0)
ayuush_hospitals.iloc[0:2] = ayuush_hospitals.iloc[0:2].fillna(method='ffill',axis=1)

ayuush_hospitals.iloc[0:2] = ayuush_hospitals.iloc[0:2].fillna('')

ayuush_hospitals.columns =ayuush_hospitals.iloc[0:2].apply(lambda x: '.'.join([y for y in x if y]),axis =0)

ayuush_hospitals = ayuush_hospitals.iloc[3:39]

ayuush_hospitals =ayuush_hospitals.reset_index(drop=True)
ayuush_hospitals.head()
ayuush_hospitals["Number of Beds.Total"] =ayuush_hospitals["Number of Beds.Total"].astype(int)

ayuush_hospitals["Number of Hospitals.Total"] =ayuush_hospitals["Number of Hospitals.Total"].astype(int)

ayuush_hospitals.sort_values(by=["Number of Beds.Total"],ascending=False).plot.bar(x='State / UT', rot=90, title='AYUUSH Hospital and Beds', figsize=(20,10),logy=True)
pop = pd.read_csv("/kaggle/input/india-population-and-density-2011/Population and density.csv")

pop = pop.astype(int,errors='ignore')

pop = pop.drop(columns=["Rank"])
pop.columns
pop.sort_values(by=["Population\n(%)"], ascending=False).plot.bar(x="State or union territory",y=["Population\n(%)","Rural population\n(%)","Urban population\n(%)"], rot=90, title='Population by States', figsize=(20,10),logy=True)
pop.sort_values(by=["Density[a]"], ascending=False).plot.bar(x="State or union territory",y=["Density[a]"], rot=90, title='Population by States', figsize=(20,10),logy=True)