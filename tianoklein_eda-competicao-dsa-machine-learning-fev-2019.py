##IMPORTS

!pip install pixiedust

import pandas as pd

import pixiedust     #Visualização
csv = '../input/dataset_treino.csv'

workers_df = pd.read_csv(csv, header=0, encoding='utf-8', delimiter = ',')

meta = workers_df.info(verbose=True)

print(meta)
# Order: Identificador Único

print(workers_df['Order'].head())

print('Count:',workers_df['Order'].count())
print(workers_df['NYC Borough, Block and Lot (BBL) self-reported'].head())

print('Count:',workers_df['NYC Borough, Block and Lot (BBL) self-reported'].count())



# TODO:: - mais de uma informação no mesmo campo, separado por ponto e virgula.

#        - avaliar se o campo será necessário ou poderá ser desprezado.
print(workers_df['NYC Building Identification Number (BIN)'].head())

print('Count:',workers_df['NYC Building Identification Number (BIN)'].count())

# TODO:: - mais de uma informação no mesmo campo, separado por ponto e virgula.

#        - avaliar se o campo será necessário ou poderá ser desprezado.
print(workers_df['Property Id'].head())

print('Count:',workers_df['Property Id'].count())
print(workers_df['Property Name'].head())

print('Count:',workers_df['Property Name'].count())



## Existem dados duplicados!!

print('EXISTE VALORES DUPLICADOS:')

workers_df[workers_df['Property Name'].duplicated(keep=False)].head()
print(workers_df['Parent Property Id'].head())

print('Count:',workers_df['Parent Property Id'].count())



## Existem dados duplicados!!

print('EXISTE VALORES DUPLICADOS:')

workers_df[workers_df['Parent Property Id'].duplicated(keep=False)].head()
print(workers_df['Parent Property Name'].head())

print('Count:',workers_df['Parent Property Name'].count())



## Existem dados duplicados!!

print('EXISTE VALORES DUPLICADOS:')

workers_df[workers_df['Parent Property Name'].duplicated(keep=False)].head()
print(workers_df['Street Number'].head(10))

print('Count:',workers_df['Street Number'].count())
print(workers_df['Street Name'].head(10))

print('Count:',workers_df['Street Name'].count())



print(workers_df['Street Name'].value_counts().head())
print(workers_df['Postal Code'].head(10))

print('Count:',workers_df['Postal Code'].count())
print(workers_df['Borough'].head(10))

print('Count:',workers_df['Borough'].count())

print(list(workers_df['Borough'].unique()))



print(workers_df['Borough'].value_counts())
print(workers_df['DOF Benchmarking Submission Status'].head(10))

print('Count:',workers_df['DOF Benchmarking Submission Status'].count())

print(workers_df['DOF Benchmarking Submission Status'].value_counts())

print(workers_df['Primary Property Type - Self Selected'].head(10))

print('Count:',workers_df['Primary Property Type - Self Selected'].count())

print(workers_df['Primary Property Type - Self Selected'].value_counts())
print(workers_df['List of All Property Use Types at Property'].head(10))

print('Count:',workers_df['List of All Property Use Types at Property'].count())

print('------------------------------------------')

print(workers_df['List of All Property Use Types at Property'].value_counts().head(20))
print(workers_df['Largest Property Use Type'].head(10))

print('Count:',workers_df['Largest Property Use Type'].count())

print(list(workers_df['Largest Property Use Type'].unique()))

print('------------------------------------------')

print(workers_df['Largest Property Use Type'].value_counts().head(20))
print(workers_df['Largest Property Use Type - Gross Floor Area (ft²)'].head(10))

print('Count:', workers_df['Largest Property Use Type - Gross Floor Area (ft²)'].count())
print(workers_df['2nd Largest Property Use Type'].head(10))

print('Count:', workers_df['2nd Largest Property Use Type'].count())

print('------------------------------------------')

print(workers_df['2nd Largest Property Use Type'].value_counts().head(10))
print(workers_df['3rd Largest Property Use Type'].head(10))

print('Count:', workers_df['3rd Largest Property Use Type'].count())

print('------------------------------------------')

print(workers_df['3rd Largest Property Use Type'].value_counts().head(10))
print(workers_df['Largest Property Use Type - Gross Floor Area (ft²)'].head(10))

print('Count:', workers_df['Largest Property Use Type - Gross Floor Area (ft²)'].count())
workers_df[['Property Id', \

            'Primary Property Type - Self Selected', \

            'Largest Property Use Type', \

            '2nd Largest Property Use Type', \

            '3rd Largest Property Use Type', \

            'List of All Property Use Types at Property', \

            'ENERGY STAR Score']].tail(10)
print(workers_df['Year Built'].head(10))

print('Count:', workers_df['Year Built'].count())

print('------------------------------------------')

print(workers_df['Year Built'].value_counts().head(10))



workers_df['Year Built'].describe()
print(workers_df['Number of Buildings - Self-reported'].head(10))

print('Count:', workers_df['Number of Buildings - Self-reported'].count())

print('------------------------------------------')

print(workers_df['Number of Buildings - Self-reported'].value_counts().head(10))



workers_df['Number of Buildings - Self-reported'].describe()
print(workers_df['Occupancy'].head(10))

print('Count:', workers_df['Occupancy'].count())

print('------------------------------------------')

print(workers_df['Occupancy'].value_counts().head(10))



workers_df['Occupancy'].describe()
print(workers_df['Metered Areas (Energy)'].head(10))

print('Count:', workers_df['Metered Areas (Energy)'].count())

print('------------------------------------------')

print(workers_df['Metered Areas (Energy)'].value_counts().head(10))



workers_df['Metered Areas (Energy)'].describe()
print(workers_df['Metered Areas  (Water)'].head(10))

print('Count:', workers_df['Metered Areas  (Water)'].count())

print('------------------------------------------')

print(workers_df['Metered Areas  (Water)'].value_counts().head(10))



workers_df['Metered Areas  (Water)'].describe()
print(workers_df['Site EUI (kBtu/ft²)'].head(10))

print('Count:', workers_df['Site EUI (kBtu/ft²)'].count())

print('------------------------------------------')

print(workers_df['Site EUI (kBtu/ft²)'].value_counts().head(10))



workers_df['Site EUI (kBtu/ft²)'].describe()
print(workers_df['Weather Normalized Site EUI (kBtu/ft²)'].head(10))

print('Count:', workers_df['Weather Normalized Site EUI (kBtu/ft²)'].count())

print('------------------------------------------')

print(workers_df['Weather Normalized Site EUI (kBtu/ft²)'].value_counts().head(10))



workers_df['Weather Normalized Site EUI (kBtu/ft²)'].describe()
print(workers_df['Weather Normalized Site EUI (kBtu/ft²)'].head(10))

print('Count:', workers_df['Weather Normalized Site EUI (kBtu/ft²)'].count())

print('------------------------------------------')

print(workers_df['Weather Normalized Site EUI (kBtu/ft²)'].value_counts().head(10))

print(workers_df['Weather Normalized Site EUI (kBtu/ft²)'].describe())



print('####################################')

print(workers_df['Weather Normalized Site Electricity Intensity (kWh/ft²)'].head(10))

print('Count:', workers_df['Weather Normalized Site Electricity Intensity (kWh/ft²)'].count())

print('------------------------------------------')

print(workers_df['Weather Normalized Site Electricity Intensity (kWh/ft²)'].value_counts().head(10))

print(workers_df['Weather Normalized Site Electricity Intensity (kWh/ft²)'].describe())



print('####################################')

print(workers_df['Weather Normalized Site Natural Gas Intensity (therms/ft²)'].head(10))

print('Count:', workers_df['Weather Normalized Site Natural Gas Intensity (therms/ft²)'].count())

print('------------------------------------------')

print(workers_df['Weather Normalized Site Natural Gas Intensity (therms/ft²)'].value_counts().head(10))

workers_df['Weather Normalized Site Natural Gas Intensity (therms/ft²)'].describe()
print(workers_df['Weather Normalized Source EUI (kBtu/ft²)'].head(10))

print('Count:', workers_df['Weather Normalized Source EUI (kBtu/ft²)'].count())

print('------------------------------------------')

print(workers_df['Weather Normalized Source EUI (kBtu/ft²)'].value_counts().head(10))

print(workers_df['Weather Normalized Source EUI (kBtu/ft²)'].describe())



print('####################################')



print(workers_df['Source EUI (kBtu/ft²)'].head(10))

print('Count:', workers_df['Source EUI (kBtu/ft²)'].count())

print('------------------------------------------')

print(workers_df['Source EUI (kBtu/ft²)'].value_counts().head(10))

print(workers_df['Source EUI (kBtu/ft²)'].describe())
print(workers_df['Fuel Oil #2 Use (kBtu)'].head(10))

print('Count:', workers_df['Fuel Oil #2 Use (kBtu)'].count())

print('------------------------------------------')

print(workers_df['Fuel Oil #2 Use (kBtu)'].value_counts().head(10))

print(workers_df['Fuel Oil #2 Use (kBtu)'].describe())

print(workers_df['Diesel #2 Use (kBtu)'].head(10))

print('Count:', workers_df['Diesel #2 Use (kBtu)'].count())

print('------------------------------------------')

print(workers_df['Diesel #2 Use (kBtu)'].value_counts().head(10))

print(workers_df['Diesel #2 Use (kBtu)'].describe())



print('####################################')



print(workers_df['District Steam Use (kBtu)'].head(10))

print('Count:', workers_df['District Steam Use (kBtu)'].count())

print('------------------------------------------')

print(workers_df['District Steam Use (kBtu)'].value_counts().head(10))

print(workers_df['District Steam Use (kBtu)'].describe())
print(workers_df['Weather Normalized Site Natural Gas Use (therms)'].head(10))

print('Count:', workers_df['Weather Normalized Site Natural Gas Use (therms)'].count())

print('------------------------------------------')

print(workers_df['Weather Normalized Site Natural Gas Use (therms)'].value_counts().head(10))

print(workers_df['Weather Normalized Site Natural Gas Use (therms)'].describe())
print(workers_df['Electricity Use - Grid Purchase (kBtu)'].head(10))

print('Count:', workers_df['Electricity Use - Grid Purchase (kBtu)'].count())

print('------------------------------------------')

print(workers_df['Electricity Use - Grid Purchase (kBtu)'].value_counts().head(10))

print(workers_df['Electricity Use - Grid Purchase (kBtu)'].describe())
print(workers_df['Weather Normalized Site Electricity (kWh)'].head(10))

print('Count:', workers_df['Weather Normalized Site Electricity (kWh)'].count())

print('------------------------------------------')

print(workers_df['Weather Normalized Site Electricity (kWh)'].value_counts().head(10))

print(workers_df['Weather Normalized Site Electricity (kWh)'].describe())
print(workers_df['Total GHG Emissions (Metric Tons CO2e)'].head(10))

print('Count:', workers_df['Total GHG Emissions (Metric Tons CO2e)'].count())

print('------------------------------------------')

print(workers_df['Total GHG Emissions (Metric Tons CO2e)'].value_counts().head(10))

print(workers_df['Total GHG Emissions (Metric Tons CO2e)'].describe())
print(workers_df['Direct GHG Emissions (Metric Tons CO2e)'].head(10))

print('Count:', workers_df['Direct GHG Emissions (Metric Tons CO2e)'].count())

print('------------------------------------------')

print(workers_df['Direct GHG Emissions (Metric Tons CO2e)'].value_counts().head(10))

print(workers_df['Direct GHG Emissions (Metric Tons CO2e)'].describe())
print(workers_df['Indirect GHG Emissions (Metric Tons CO2e)'].head(10))

print('Count:', workers_df['Indirect GHG Emissions (Metric Tons CO2e)'].count())

print('------------------------------------------')

print(workers_df['Indirect GHG Emissions (Metric Tons CO2e)'].value_counts().head(10))

print(workers_df['Indirect GHG Emissions (Metric Tons CO2e)'].describe())
print(workers_df['Indirect GHG Emissions (Metric Tons CO2e)'].head(10))

print('Count:', workers_df['Indirect GHG Emissions (Metric Tons CO2e)'].count())

print('------------------------------------------')

print(workers_df['Indirect GHG Emissions (Metric Tons CO2e)'].value_counts().head(10))

print(workers_df['Indirect GHG Emissions (Metric Tons CO2e)'].describe())
print(workers_df['Property GFA - Self-Reported (ft²)'].head(10))

print('Count:', workers_df['Property GFA - Self-Reported (ft²)'].count())

print('------------------------------------------')

print(workers_df['Property GFA - Self-Reported (ft²)'].value_counts().head(10))

print(workers_df['Property GFA - Self-Reported (ft²)'].describe())
print(workers_df['Water Use (All Water Sources) (kgal)'].head(10))

print('Count:', workers_df['Water Use (All Water Sources) (kgal)'].count())

print('------------------------------------------')

print(workers_df['Water Use (All Water Sources) (kgal)'].value_counts().head(10))

print(workers_df['Water Use (All Water Sources) (kgal)'].describe())
print(workers_df['Water Intensity (All Water Sources) (gal/ft²)'].head(10))

print('Count:', workers_df['Water Intensity (All Water Sources) (gal/ft²)'].count())

print('------------------------------------------')

print(workers_df['Water Intensity (All Water Sources) (gal/ft²)'].value_counts().head(10))

print(workers_df['Water Intensity (All Water Sources) (gal/ft²)'].describe())
print(workers_df['Release Date'].head(10))

print('Count:', workers_df['Release Date'].count())

print('------------------------------------------')

print(workers_df['Release Date'].value_counts().head(10))

print(workers_df['Release Date'].describe())
print(workers_df['Latitude'].head(10))

print('Count:', workers_df['Latitude'].count())

print('------------------------------------------')

print(workers_df['Latitude'].value_counts().head(10))

print(workers_df['Latitude'].describe())



print(workers_df['Longitude'].head(10))

print('Count:', workers_df['Longitude'].count())

print('------------------------------------------')

print(workers_df['Longitude'].value_counts().head(10))

print(workers_df['Longitude'].describe())
print(workers_df['Community Board'].head(10))

print('Count:', workers_df['Community Board'].count())

print('------------------------------------------')

print(workers_df['Community Board'].value_counts().head(10))

print(workers_df['Community Board'].describe())



print(list(workers_df['Community Board'].unique()))

print(workers_df['Council District'].head(10))

print('Count:', workers_df['Council District'].count())

print('------------------------------------------')

print(workers_df['Council District'].value_counts().head(10))

print(workers_df['Council District'].describe())



print(list(workers_df['Council District'].unique()))
print(workers_df['Census Tract'].head(10))

print('Count:', workers_df['Census Tract'].count())

print('------------------------------------------')

print(workers_df['Census Tract'].value_counts().head(10))

print(workers_df['Census Tract'].describe())

#print(list(workers_df['Census Tract'].unique()))
print(workers_df['NTA'].head(10))

print('Count:', workers_df['NTA'].count())

print('------------------------------------------')

print(workers_df['NTA'].value_counts().head(10))

print(workers_df['NTA'].describe())
print(workers_df['Water Required?'].head(10))

print('Count:', workers_df['Water Required?'].count())

print('------------------------------------------')

print(workers_df['Water Required?'].value_counts().head(10))

print(workers_df['Water Required?'].describe())
#SCATTER - Year Build

display(workers_df)
#workers_df['Street Name']

display(workers_df)
# Lat/Long de 'ENERGY STAR Score' - NYC

display(workers_df)
# Histograma de ['ENERGY STAR Score']

display(workers_df)
# MEDIA DE 'ENERGY STAR Score' por 'Primary Property Type - Self Selected'

display(workers_df)
# Média de [ENERGY STAR SCORE] por [Year Built]

display(workers_df)
# Média de [ENERGY STAR SCORE] por ['Number of Buildings - Self-reported']

display(workers_df)
# Soma de [ENERGY STAR SCORE] por ['Occupancy']

display(workers_df)
# Média de 'ENERGY STAR Score' por ['Borough']

display(workers_df)
# 

display(workers_df)
corr = workers_df.corr()

corr.style.background_gradient()