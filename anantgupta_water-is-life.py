import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

data=pd.read_csv('../input/AguaH.csv')



# First goal is to find out for which particular area, the water consumption 

# has increased dramatically compared to other regions and to check whether there are areaso which are

# not getting enough water



# We will convert the dataset into a list of time series

import datetime

dateList=[]

for year in range(2009,2016):

    for month in range(1,13):

        dateList.append(datetime.date(year,month,1))
# For starters, let us aggregate data on the first and last month in the dataset, just to get an idea

import matplotlib.pyplot as plt



fig=plt.figure()

ax=fig.add_subplot(111)

ax2=ax.twinx()



data.groupby(['USO2013'])['f.1_ENE_09'].sum().plot(label="JAN 2009",ax=ax,kind='bar',color='r',alpha=0.5,position=0,width=0.4)

data.groupby(['USO2013'])['f.1_DIC_15'].sum().plot(label="DEC 2015",ax=ax,kind='bar',color='b',alpha=0.5,position=1,width=0.4)



data.groupby(['USO2013'])['f.1_ENE_09'].mean().plot(label="JAN 2009",ax=ax2,color='r',alpha=0.5)

data.groupby(['USO2013'])['f.1_DIC_15'].mean().plot(label="DEC 2015",ax=ax2,color='b',alpha=0.5)

ax.set_ylabel("TOTAL WATER CONSUMPTION")

ax2.set_ylabel("AVERAGE WATER CONSUMPTION")



plt.show()



# We can see that there is a huge difference. We will drill down further and find more patterns
#Variable USO2013

#Explanation: this is the type of land use

#	Levels:  

#		Vegetation area				"AVD" 

#		Downtown				"CU"   

#		Parks					"EQ" 

#		Housing low density			"H1"  

#		Housing mid density 			"H2"  

#		Housing high density 			"H3"  

#		Infrastructure				"IN"  

#		High Risk Industry 			"IRA" 

#		Low Risk Industry 			"IRB" 

#		Medium Risk Industry 			"IRM" 

#		Mixed (commerce, housing, industry) 	"MX"  

#		Government Reserve 			"RG" 

#		Reserve Housing Conditioned		"RHC" 

#		Reserve Industry Conditioned 		"RIC"



#Variable TU

#Explanation: Type of user

#	Levels:

#		"COMERCIAL" = Commerce

#		"DOMESTICO BAJA" = Low Income 

#		"DOMESTICO MEDIO" = Median Income

#		"DOMESTICO RESIDENCIAL" = High Income 

#		"ESPECIAL"   = Big consumer of water (not industry, e.g. carwash)

#		"INDUSTRIAL" = Industry

#		"SOCIAL = Social Welfare



#Variable:CU

#Explanation: The diameter of the pipe of the house that is connected to the public grid

#	levels in inches 

#		"0.5"  

#		"0.75" 

#		"1"    

#		"1.5"  

#		"2"    

#		"3"    

#		"4"    

#		"10"  

#Variable: M

#Explanation: type of vendor of the device that measure the consumption in the house

#Variable: UL

#Explanation: cubic meters of consumption in January 2016

#The rest of the variables are labelled in the following manner f.1_XXX_XX. You can ignore the first fourth letters (f.1_) because it is an error

#of the conversion, Im sorry. The first three X referes to the month (ENE means Enero, January in Spanish, Feb means Febrero, February in Spanish and so on).  
# Let us analyse a particular section of the dataset

# The most impactful point is Housing in the DOMESTICO MEDIO space



# Since we have embedded time series data, we will have to think of a way to incorporate that



# 1. Identify the increase in number of households

# 2. Collated time series of increase in consumption