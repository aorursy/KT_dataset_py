#Import Libraries 



import pandas as pd

import matplotlib.pyplot as plt 

%matplotlib inline



#Load in dataset from Kaggle 



data = pd.read_csv('../input/Indicators.csv')
# select CO2 emissions for the United States



#Filtering the information we would like to use 



filterIndicator1 = 'CO2 emissions \(metric'

filterIndicator2 = 'USA'



#Creating mask for the filters 



mask1 = data['IndicatorName'].str.contains(filterIndicator1) 

mask2 = data['CountryCode'].str.contains(filterIndicator2)



#Storing the masks in a new value 

USCarbon = data[mask1 & mask2]



#Preview the new dataframe (nice for seeing how you would like to clean it up if at all)

USCarbon.head()



#Drop unnecessary columns 



USCarbonFinal = USCarbon.drop(columns=['CountryName','CountryCode','IndicatorName','IndicatorCode'])



#View final dataframe



USCarbonFinal.head(10)

# select CO2 emissions for Central Europe and the Baltics

filterIndicator3 = 'CO2 emissions \(metric'

filterIndicator4 = 'CEB'



mask3 = data['IndicatorName'].str.contains(filterIndicator3) 

mask4 = data['CountryCode'].str.contains(filterIndicator4)



#create dateframe from filter masks

EuroCarbon = data[mask3 & mask4]



#Drop unwanted columns from dataframe

EuroCarbonFinal = EuroCarbon.drop(columns=['CountryName','CountryCode','IndicatorName','IndicatorCode'])



#View first 10 columns in dataframe post drop 

EuroCarbonFinal.head(10)
#Create a graph to compare the two countries Carbon Emissions 



#X & Y values

x = USCarbon['Year']

y = USCarbon['Value']  

Y = EuroCarbon['Value']



#Creating our plot using matplotlib function 



plt.plot(x, y, color='r', label='US')

plt.plot(x, Y, color='b', label='Euro')

plt.xlabel('Years')

plt.ylabel('Carbon Emissions')

plt.title('US vs. Euro Carbon Emissions')



#Add a legend

plt.gca().legend(('US','Euro'))

plt.show()
#Now let's look at fertility rates per country 



#US Fertility Rate 



filterIndicator5 = 'SP.DYN.TFRT.IN'

filterIndicator6 = 'USA'



mask5 = data['IndicatorCode'].str.contains(filterIndicator5) 

mask6 = data['CountryCode'].str.contains(filterIndicator6)



# Storing masks 

USFertility = data[mask5 & mask6]



#Drop unwanted Columns

USFertilityFinal = USFertility.drop(columns=['CountryName','CountryCode','IndicatorName','IndicatorCode'])



#View first 10 columns

USFertilityFinal.head(10)
#Fertility rates in Europe/Baltics



filterIndicator7 = 'SP.DYN.TFRT.IN'

filterIndicator8 = 'CEB'



mask7 = data['IndicatorCode'].str.contains(filterIndicator7) 

mask8 = data['CountryCode'].str.contains(filterIndicator8)



# Create a new dataframe for the European Fertility data 

EuroFertility = data[mask7 & mask8]



#Drop unwanted Columns

EuroFertilityFinal = EuroFertility.drop(columns=['CountryName','CountryCode','IndicatorName','IndicatorCode'])



#View first 10 columns

EuroFertilityFinal.head(10)
#Create a graph to compare the two countries fertility rates over the years.





#X & Y values 



x = USFertility['Year']

y = USFertility['Value']

Y = EuroFertility['Value']



#Creating a graph using matplotlib 



plt.plot(x, y, color='r', label='US')

plt.plot(x, Y, color='b', label='Euro')

plt.xlabel('Years')

plt.ylabel('Fertility')

plt.title('US vs. Euro Fertility Rates')



#Add a legend

plt.gca().legend(('US','Euro'))

plt.show()
#Is there a correlation between CO2 Emissions and fertility rates? 



#First let's get some more information from our datasets 



#US Carbon Emissions Data Set



#Mean, Std, min, max, counts (should be the same)

USCarbonFinal.describe()



#Count : Year = 52, Value = 52

#Mean : Value = 19.303472

#Standard Deviation : Value = 1.564525

#min : Value = 15.681256

#max : Value = 22.510582





#Euro Carbon Data Set



#Mean, Std, min, max, counts (should be the same)

EuroCarbonFinal.describe()



#Count : Year = 52, Value = 52

#Mean : Value = 8.237914

#Standard Deviation : Value = 1.805031

#min : Value = 5.114244

#max : Value = 11.285238
#US Fertility Data Set



#Mean, Std, min, max, counts (should be the same)

USFertilityFinal.describe()



#Count : Year = 54, Value = 54

#Mean : Value = 2.158602

#Standard Deviation : Value = 0.482290

#min : Value = 1.738000

#max : Value = 3.654000





#Euro Fertility Data Set



#Mean, Std, min, max, counts (should be the same)

EuroFertilityFinal.describe()



#Count : Year = 54, Value = 54

#Mean : Value = 1.893536

#Standard Deviation : Value = 0.432010

#min : Value = 1.251015

#max : Value = 2.498618
#Merge Dataframes together  



USMerge = USCarbon.merge(USFertility, on='Year', how='inner')



#Drop Columns

USMerge1 = USMerge.drop(columns=['CountryName_x','CountryName_y','IndicatorCode_x',

                                     'CountryName_y','CountryCode_y','IndicatorCode_y',

                                    'IndicatorName_x', 'IndicatorName_y'])



#Rename Columns 

USMerge1.rename(columns={'CountryCode_x': 'CountryCode','Value_x' : 'CO2_Value',

                          'Value_y':'Fertility_Value'}, inplace=True)

#preview

USMerge1.head()



#Reorder Columns so years is not in the middle of the data values 

USMergeFinal = USMerge1[['CountryCode', 'Year', 'CO2_Value', 

                             'Fertility_Value']]



#Final Dataframe

USMergeFinal.head()

#Merge Dataframes together 



#European Datasets 



EuroMerge = EuroCarbon.merge(EuroFertility, on='Year', how='inner')



#Drop Columns

EuroMerge1 = EuroMerge.drop(columns=['CountryName_x','CountryName_y','IndicatorCode_x',

                                     'CountryName_y','CountryCode_y','IndicatorCode_y',

                                    'IndicatorName_x', 'IndicatorName_y'])



#Rename Columns 

EuroMerge1.rename(columns={'CountryCode_x': 'CountryCode','Value_x' : 'CO2_Value',

                          'Value_y':'Fertility_Value'}, inplace=True)



#preview data 



EuroMerge1.head()



#Reorder Columns so years is not in the middle of the data values 

EuroMergeFinal = EuroMerge1[['CountryCode', 'Year', 'CO2_Value', 

                             'Fertility_Value']]

#Final View of data



EuroMergeFinal.head()

#Final Graphs 



#x & y values

x1 = USCarbon['Year']

y1 = USCarbon['Value']

Y1 = EuroCarbon['Value']



#Carbon

plt.subplot(1, 2, 1)

plt.plot(x1, y1, color='r', label='US')

plt.plot(x1, Y1, color='b', label='Euro')

plt.title('Fertility & Carbon Emissions')

plt.ylabel('Carbon Emissions')



####



#x & y values

x2 = USFertility['Year']

y2 = USFertility['Value']

Y2 = EuroFertility['Value']



#Fertility

plt.subplot(1, 2, 2)

plt.plot(x2, y2, color='r', label='US')

plt.plot(x2, Y2, color='b', label='Euro')

plt.xlabel('Years')

plt.ylabel('Fertility Rates')



plt.gca().legend(('US','Euro'))

plt.show()


