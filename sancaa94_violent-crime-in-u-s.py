import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import os

print(os.listdir("../input"))
df = pd.read_csv('../input/Crime in the United States.csv')
df.columns 
df.columns = ['Year', 'Population', 'Violent crime', 'Violent crime rate',

       'Murder and nonnegligent manslaughter',

       'Murder and nonnegligent manslaughter rate',

       'Rape (revised definition3)', 'Rape (revised definition3) Rate',

       'Rape(legacy definition)', 'Rape(legacy definition) Rate',

       'Robbery', 'Robbery rate', 'Aggravated assault',

       'Aggravated assault rate', 'Property crime',

       'Property crime rate', 'Burglary', 'Burglary rate',

       'Larceny theft', 'Larceny theft rate', 'Motor vehicle theft',

       'Motor vehicle theft rate']
df.head(20)
Year = [] #for loop rmoves that extra digit and makes list into int

for i in df['Year'].dropna().apply(int).apply(str):

    nYear = i[:4]

    nYear = int(nYear)

    Year.append(nYear)

print(len(Year))

print(Year) 
Population = [] #this for loop removes commas and changes to int

for i in df['Population'].dropna():

    i = i.replace(',', '')

    i = int(i)

    Population.append(i)

print(len(Population))

print(Population) 
#this for loop removes commas and changes to int

Violentcrime = [] 

for i in df['Violent crime'].dropna():

    i = i.replace(',', '')

    i = int(i)

    Violentcrime.append(i)

print(len(Violentcrime))

print(Violentcrime)



# this converst rates from a string to a float

df['Violent crime rate'].dropna()

crimerate = []

for i in df['Violent crime rate'].dropna():

    if [i] == [' ']:

        print('Yes')

    else:

        i = float(i)

        crimerate.append(i)

print(len(crimerate))

print(crimerate)  
Murder = [] 

for i in df['Murder and nonnegligent manslaughter'].dropna():

    i = i.replace(',', '')

    i = int(i)

    Murder.append(i)

print(len(Murder))

print(Murder)



Murderrate = []

for i in df['Murder and nonnegligent manslaughter rate'].dropna():

    if [i] == [' ']:

        print('Yes')

    else:

        i = float(i)

        Murderrate.append(i)

print(len(Murderrate))

print(Murderrate) 
#Average

x = 0

Rape = []

for i in df['Rape (revised definition3)'].fillna(x)[:20]:

    if [i] == [' ']:

        i = 0

        Rape.append(i)

    elif isinstance(i, str):

        i = i.replace(',', '')

        i = int(i)

        Rape.append(i)

    else:

        Rape.append(i)

print(Rape)

print(len(Rape))



RapeC = []

for i in Rape:

    if i == 0:

        Avg = sum(Rape) / 4#len(Rape) 

        i = round(Avg)

        RapeC.append(i)

    else:

        RapeC.append(i)

    

print(RapeC)

print(len(RapeC))  
#df['Rape (revised definition3) Rate']

x = 0

RapeRate = []

for i in df['Rape (revised definition3) Rate'].fillna(x)[:20]:

    if [i] == [' ']:

        i = 0

        RapeRate.append(i)

    else:

        RapeRate.append(i)

        

print(RapeRate)

print(len(RapeRate))

        

RapeRateC = []

for i in RapeRate:

    if i == 0:

        Avg = sum(RapeRate) / 4#len(RapeRate) 

        i = Avg

        RapeRateC.append(i)

    else:

        RapeRateC.append(i)

        

print(RapeRateC)

print(len(RapeRateC))

    
RapeL = [] #this for loop removes commas and changes to int

for i in df['Rape(legacy definition)'][:20]:

    i = i.replace(',', '')

    i = int(i)

    RapeL.append(i)

print(len(RapeL))

print(RapeL)



RapeLrate = df['Rape(legacy definition) Rate'][:20]

print(RapeLrate)

print(len(RapeLrate))

#df['Robbery'] df['Robbery rate']

Robbery = [] 

for i in df['Robbery'][:20]:

    i = i.replace(',', '')

    i = int(i)

    Robbery.append(i)

print(len(Robbery))

print(Robbery)



RobberyRate = df['Robbery rate'][:20]

print(len(RobberyRate))

print(RobberyRate)

#Year Population Violentcrime crimerate Murder Murderrate RapeC RapeRateC RapeL RapeLrate Robbery RobberyRate
  

#df['Aggravated assault'] df['Aggravated assault rate']

assault = [] 

for i in df['Aggravated assault'][:20]:

    i = i.replace(',', '')

    i = int(i)

    assault.append(i)

print(len(assault))

print(assault)



assaultrate = df['Aggravated assault rate'][:20]

print(len(assaultrate))

print(assaultrate)
#df['Property crime'] df['Property crime rate']

Property = [] 

for i in df['Property crime'][:20]:

    i = i.replace(',', '')

    i = int(i)

    Property.append(i)

print(len(Property))

print(Property)



PropertyRate = df['Property crime rate'][:20]

print(len(PropertyRate))

print(PropertyRate)
#df['Burglary'] df['Burglary rate']

Burglary = [] 

for i in df['Burglary'][:20]:

    i = i.replace(',', '')

    i = int(i)

    Burglary.append(i)

print(len(Burglary))

print(Burglary)



BurglaryRate = df['Burglary rate'][:20]

print(len(BurglaryRate))

print(BurglaryRate)



#df['Larceny theft'] df['Larceny theft rate']

Larceny = [] 

for i in df['Burglary'][:20]:

    i = i.replace(',', '')

    i = int(i)

    Larceny.append(i)

print(len(Larceny))

print(Larceny)



LarcenyRate = df['Larceny theft rate'][:20]

print(len(LarcenyRate))

print(LarcenyRate)



#df['Motor vehicle theft'] df['Motor vehicle theft rate']

Motor = [] 

for i in df['Motor vehicle theft'][:20]:

    i = i.replace(',', '')

    i = int(i)

    Motor.append(i)

print(len(Motor))

print(Motor)



MotorRate = df['Motor vehicle theft rate'][:20]

print(len(MotorRate))

print(MotorRate)





# Make a new data frame with cleaned columns

# dictionary where Keys are column, and values are lists

NewData = {

    'Year' : Year,

    'Population' : Population,

    'Violent Crime' : Violentcrime,

    'Violent Crime Rate' : crimerate,

    'Murder': Murder,

    'Murder Rate' : Murderrate,

    'Rape' : RapeC,

    'Rape Rate': RapeRateC,

    'Rape(legacy Def)': RapeL,

    'Rape Rate(legacy Def)': RapeLrate,

    'Robbery': Robbery,

    'Robbery Rate': RobberyRate,

    'Assault': assault,

    'Assault Rate' : assaultrate,

    #'Property' : Property,

    #'Property Rate' : PropertyRate,

    'Burglary': Burglary,

    'Burglary Rate' : BurglaryRate,

    'Larceny': Larceny,

    'Larceny Rate' : LarcenyRate,

    'Motor': Motor,

    'Motor Rate': MotorRate

}



df = pd.DataFrame(NewData, columns =['Year','Population', 'Violent Crime','Violent Crime Rate','Murder','Murder Rate',

                                     'Rape','Rape Rate', 'Rape(legacy Def)', 'Rape Rate(legacy Def)', 'Robbery', 

                                     'Robbery Rate', 'Assault', 'Assault Rate', 'Burglary',

                                     'Burglary Rate', 'Larceny', 'Larceny Rate', 'Motor', 'Motor Rate']) 



#I took out property damage 'Property', 'Property Rate'



df.head(21)

#df['Rape']==122115



#df[df.Year == 2016]



#df.loc[(df['Year']) & (df['Population']) & (df['Violent Crime']), [ 'Year', 'Population', 'Violent Crime']]

#df.reindex((df['Murder']) & (df['Rape']), ['Murder', 'Rape'])



#Crime = (df['Murder'], df['Rape'], df['Rape(legacy Def)'], df['Robbery'], 

#df['Assault'], df['Property'], df['Burglary'], ['Larceny Motor'])

valueninetyseven = df.loc[0, ['Violent Crime', 'Murder', 'Rape', 'Rape(legacy Def)', 'Robbery', 'Assault', 'Burglary', 'Larceny', 'Motor']]#1997

ninetyseven = []

for i in valueninetyseven:

    ninetyseven.append(i)

ninetyseven



valueosix = df.loc[9, ['Violent Crime', 'Murder', 'Rape', 'Rape(legacy Def)', 'Robbery', 'Assault', 'Burglary', 'Larceny', 'Motor']]#2006

osix = []

for i in valueosix:

    osix.append(i)

osix



valuesixteen = df.loc[19, ['Violent Crime', 'Murder', 'Rape', 'Rape(legacy Def)', 'Robbery', 'Assault', 'Burglary', 'Larceny', 'Motor']]#2016

sixteen = []

for i in valuesixteen:

    sixteen.append(i)

sixteen



x = ['Violent Crime', 'Murder', 'Rape', 'Rape(legacy Def)', 'Robbery', 'Assault', 'Burglary', 'Larceny', 'Motor']

# Bar Graphs are biased, property damage was the largest bar, removed to show other crimes



plt.bar(x,ninetyseven)

plt.xlabel('Category of Crime')

plt.ylabel('Total')

plt.yticks([500000,1000000,1500000,2000000,2500000],

           ['500K','100K','150K','200K','250K'])

plt.xticks(fontsize=7, rotation=30)

plt.title('Total Crime 1997')

plt.show()



plt.bar(x,osix)

plt.xlabel('Category of Crime')

plt.ylabel('Total')

plt.yticks([500000,1000000,1500000,2000000,2500000],

           ['500K','100K','150K','200K','250K'])

plt.xticks(fontsize=7, rotation=30)

plt.title('Total Crime 2006')

plt.show()



plt.bar(x,sixteen)

plt.xlabel('Category of Crime')

plt.ylabel('Total')

plt.yticks([500000,1000000,1500000,2000000,2500000],

           ['500K','100K','150K','200K','250K'])

plt.xticks(fontsize=7, rotation=30)

plt.title('Total Crime 2016')

plt.show()



#line graph

plt.plot(Year, df['Violent Crime'])

#plt.plot(Year, df['Murder'])

#plt.plot(Year, Population)

plt.title('Violent Crime 1997 to 2016')

plt.xlabel('years')

plt.ylabel('Violent Crime')

plt.show()
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import train_test_split
violentcrime  = df['Violent Crime'] 

year = Year

x = np.unique(year).reshape(-1,1) 

y = violentcrime



x_train, x_test, y_train, y_test = train_test_split(x,y, test_size =0.3, random_state=42)



reg = LinearRegression()

reg.fit(x_train, y_train)

y_pred = reg.predict(x_test)
# Plot outputs, shows that viloent crime in general is declining over the years

plt.scatter(x_test, y_test,  color='black')

plt.plot(x_test, y_pred, color='blue', linewidth=3)

plt.xlabel('Years')

plt.ylabel('Violent Crime Rate')



plt.show()