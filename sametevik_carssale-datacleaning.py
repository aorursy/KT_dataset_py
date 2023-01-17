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
dataset = pd.read_excel("/kaggle/input/car-sales/CarSales.xlsx")

dataset_backup = dataset.copy()
dataset.head()
print("Shape of The Dataset","\nColumns : " ,dataset.shape[1],"\nRows : ",dataset.shape[0])
dataset.isnull().sum()
dataset.drop(axis=0, index=dataset[dataset["Price"].isnull()] \

             .index,inplace=True)
dataset.info()
#Droping listing date

dataset.drop("Listing Date", axis = 1, inplace = True)
dataset.iloc[[1,3,5,9], :]
counter = 0

abnormal = []

for value in dataset["Price"].apply(lambda val : val[:-4].replace(".","")):

    counter +=1

    if value.isdigit() == False:

        abnormal.append(counter-1)



print("Number of Abnormal Price : " ,len(abnormal))
#Adjusting the number of max rows.

pd.options.display.max_rows = 1611#If you want to see all of the rows, you can activate this line.

dataset_abnormal = dataset.iloc[abnormal,:]

dataset_abnormal.head(10)
#Taking the last values

dataset["Price"] = dataset["Price"].apply(lambda data : data[:-3].split(" ")[0].replace(".","") if "\r\n\r\n" in  data else data[:-3].replace(".","") )



#some values include usd or eur. I have to change these values to TL

def exchange(value):

    if "USD" in value:

        return str(int(value[:-3]) * 7.44)

    elif "EUR" in value:

        return str(int(value[:-3]) * 8.80) 

    else:

        return value

dataset["Price"] = dataset["Price"].apply(exchange)

dataset.head(3) 

#Changing the data type of Price

dataset["Price"] = dataset["Price"].astype("int")
# Checking how many abnormal values there are.

counter = 0

index_list = []

for i in dataset["Km"]:

    if type(i) != int and type(i) != float:

        print("Index : " + str(counter),len(str(i)))

        index_list.append(counter)

    counter += 1

print("Abnormal values : ",counter)
dataset.iloc[index_list,:]
abnormal_km = dataset.iloc[index_list,:]

dataset = dataset.drop(axis = 0, index = abnormal_km.index)
#Changing the data type of Price

dataset["Km"] = dataset["Km"].astype("int")
# Features For Fiat Albea ( model : 2010 - 2013, Price : 35000 TL - 45000 TL)

Fiat_Albea = dataset.loc[(dataset["Model"] >=2010) \

            & (dataset["Model"]<=2013)\

            & (dataset["Car"] == "Fiat Albea Sole 1.3 Multijet Premio Plus")\

            & (dataset["Price"] >=35000) \

            & (dataset["Price"] <=45000),:]["Km"].mean()

# Features For Fiat Linea ( model : 2009 - 2011,  Price : 67000 TL - 71000 TL)

Fiat_Linea = dataset.loc[(dataset["Model"] >=2009) \

            & (dataset["Model"]<=2011)\

            & (dataset["Car"] == "Fiat Linea 1.3 Multijet Dynamic")\

            & (dataset["Price"] >= 67000) \

            & (dataset["Price"] <= 71000),:]["Km"].mean()

# Features For Hyundai Accent Era ( model : 2010 - 2013, Price : 60000 TL - 65000 TL)

Hyundai_Accent = dataset.loc[(dataset["Model"] >=2010) \

            & (dataset["Model"]<=2013)\

            & (dataset["Car"] == "Hyundai Accent Era 1.5 CRDi-VGT Team")\

            & (dataset["Price"] >= 60000) \

            & (dataset["Price"] <= 65000),:]["Km"].mean()

# Features For Opel Astra  ( model : 2010 - 2013,  Price : 98000 TL - 101000 TL)

Opel_Astra = dataset.loc[(dataset["Model"] >=2010) \

            & (dataset["Model"]<=2013)\

            & (dataset["Car"] == "Opel Astra 1.6 Enjoy 111.Yıl")\

            & (dataset["Price"] >= 98000) \

            & (dataset["Price"] <= 101000),:]["Km"].mean()

# Features For Peugeot 206 ( model : 2005 - 2007, Price : 57000 TL - 63000 TL)

Peugeot_206 = dataset.loc[(dataset["Model"] >=2004) \

            & (dataset["Model"]<=2010)\

            & (dataset["Car"] == "Peugeot 206 1.4 HDi X-Design")\

            & (dataset["Price"] >= 30000) \

            & (dataset["Price"] <= 50000),:]["Km"].mean()

# Features For Renault Fluence ( model : 2013 - 2015, Price : 110000 TL - 115000 TL)

Renault = dataset.loc[(dataset["Model"] >=2013) \

            & (dataset["Model"]<=2015)\

            & (dataset["Car"] == "Renault Fluence 1.5 dCi Touch Plus")\

            & (dataset["Price"] >= 110000) \

            & (dataset["Price"] <= 115000),:]["Km"].mean()



#Creating a series to add in the abnormal_km dataset

km = data=[str(round(Fiat_Albea)),str(round(Fiat_Linea)),str(round(Hyundai_Accent)),

                     str(round(Opel_Astra)),str(round(Peugeot_206)),str(round(Renault))]



#Adding in abnormal_km

abnormal_km["Km"] = km



#Concatenating

dataset = pd.concat([dataset, abnormal_km], ignore_index=True)
#Changing the data type of Price

dataset["Km"] = dataset["Km"].astype("int")
#Dividing dataset depends on car

Alfa = [] ; Audi = [] ; BMW = [] ; Chevrolet = [] ; Citroen = [] ; Dacia = [] ; Fiat = [] ; Ford = [] ; Honda = []

Hyundai = [] ; Kia = [] ; Mazda = [] ; Mercedes = [] ; Opel = [] ; Peugeot = [] ; Renault = []

counter = 0

for value in dataset["Car"]:

    if "Alfa Romeo" in value:

        Alfa.append(counter)

    elif "Audi" in value:

        Audi.append(counter)

    elif "BMW" in value:

        BMW.append(counter)

    elif "Chevrolet" in value:

        Chevrolet.append(counter)

    elif "Citroen" in value:

        Citroen.append(counter)

    elif "Dacia" in value:

        Dacia.append(counter)

    elif "Fiat" in value:

        Fiat.append(counter)

    elif "Ford" in value:

        Ford.append(counter) 

    elif "Honda" in value:

        Honda.append(counter) 

    elif "Hyundai" in value:

        Hyundai.append(counter)

    elif "Kia" in value:

        Kia.append(counter)

    elif "Mazda" in value:

        Mazda.append(counter)

    elif "Mercedes" in value:

        Mercedes.append(counter)

    elif "Opel" in value:

        Opel.append(counter)

    elif "Peugeot" in value:

        Peugeot.append(counter)

    elif "Renault" in value:

        Renault.append(counter)

    

    counter += 1
Alfa_df = dataset.iloc[Alfa,:]

Audi_df = dataset.iloc[Audi,:]

BMW_df = dataset.iloc[BMW,:]

Chevrolet_df = dataset.iloc[Chevrolet,:]

Citroen_df = dataset.iloc[Citroen,:]

Dacia_df = dataset.iloc[Dacia,:]

Fiat_df = dataset.iloc[Fiat,:]

Ford_df = dataset.iloc[Ford,:]

Honda_df = dataset.iloc[Honda,:]

BMW_df = dataset.iloc[BMW,:]

Hyundai_df = dataset.iloc[Hyundai,:]

Kia_df = dataset.iloc[Kia,:]

Mazda_df = dataset.iloc[Mazda,:]

Mercedes_df = dataset.iloc[Mercedes,:]

Opel_df = dataset.iloc[Opel,:]

Peugeot_df = dataset.iloc[Peugeot,:]

Renault_df = dataset.iloc[Renault,:]