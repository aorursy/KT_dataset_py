import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



file ="../input/all.csv"

df=pd.read_csv(file,sep=",")
df.columns
df.head()
df.describe()
plt.figure(figsize=(12,8))

count =df['State'].value_counts()

count.plot("bar")

plt.xlabel("State name",size=20,color='g')

plt.ylabel("No. of cities", size=20,color='orange')

plt.title("Cities per state", size= 20)

plt.savefig("census1.jpg")
person=df['Persons']

temp = df.groupby('State').sum()

print(temp.columns)

temp
count= count.reindex(sorted(count.index))

#print(count)

density =temp['Persons']/count.values/10**5

#temp['Persons']
plt.figure(figsize=(12,12))

plt.title("Totals Persons per city in a State (City density)", size= 20)



plt.pie(density.values,labels=density.index, shadow=True)

plt.savefig("census2.jpg")
#	Household.size..per.household.	Sex.ratio..females.per.1000.males.	Sex.ratio..0.6.years.	Persons..literate	Males..Literate	...	Imp.Town.3.Population	Total.Inhabited.Villages	Drinking.water.facilities	Safe.Drinking.water	Electricity..Power.Supply.	Primary.school	Post..telegraph.and.telephone.facility	Permanent.House	Semi.permanent.House	Temporary.House

plt.figure(figsize=(14,12))

plt.title("Number of households state wise", size= 20)



plt.pie(temp['Number.of.households'].values/100,labels=density.index, shadow=True)

plt.savefig("census2.jpg")
plt.figure(figsize=(12,12))

plt.title("Sex.ratio..females.per.1000.males", size= 20)



plt.pie(temp['Sex.ratio..females.per.1000.males.'].values/100,labels=density.index, shadow=True)

plt.show()
#	Household.size..per.household.	.	Sex.ratio..0.6.years.		Males..Literate	...	Imp.Town.3.Population	Total.Inhabited.Villages	Drinking.water.facilities	Safe.Drinking.water	Electricity..Power.Supply.	Primary.school	Post..telegraph.and.telephone.facility	Permanent.House	Semi.permanent.House	Temporary.House

plt.figure(figsize=(12,12))

plt.title("Persons..literate Per person", size= 20)



plt.pie(temp['Persons..literate'].values/temp['Persons'],labels=density.index,autopct='%1.1f%%', shadow=True)

plt.show()
#	Household.size..per.household.	.	Sex.ratio..0.6.years.		Males..Literate	...	Imp.Town.3.Population	Total.Inhabited.Villages	Drinking.water.facilities	Safe.Drinking.water	Electricity..Power.Supply.	Primary.school	Post..telegraph.and.telephone.facility	Permanent.House	Semi.permanent.House	Temporary.House

plt.figure(figsize=(12,12))

plt.title("Drinking.water.facilities per Person", size= 20, color= 'm')



plt.pie(temp['Drinking.water.facilities'].values/temp['Persons']*10**4,labels=density.index, shadow=True)

plt.show()
#	Household.size..per.household.	.	Sex.ratio..0.6.years.		Males..Literate	...	Imp.Town.3.Population	Total.Inhabited.Villages	Drinking.water.facilities	Safe.Drinking.water	Electricity..Power.Supply.	Primary.school	Post..telegraph.and.telephone.facility	Permanent.House	Semi.permanent.House	Temporary.House

plt.figure(figsize=(12,12))

plt.title("Safe.Drinking.water per Person", size= 20)



plt.pie(temp['Safe.Drinking.water'].values/temp['Persons']*10**4,labels=density.index, shadow=True)

plt.show()
plt.figure(figsize=(12,12))

plt.title("Total.Inhabited.Villages", size= 20, color= 'm')

plt.pie(temp['Total.Inhabited.Villages'].values/100,labels=density.index, shadow=True)

plt.show()
l =np.array(temp[['Permanent.House','Semi.permanent.House','Temporary.House']])

l.shape
#	Household.size..per.household.	.	Sex.ratio..0.6.years.		Males..Literate	...	Imp.Town.3.Population	Total.Inhabited.Villages	Electricity..Power.Supply.	Primary.school	Post..telegraph.and.telephone.facility	Permanent.House	Semi.permanent.House	Temporary.House

df2 = pd.DataFrame(l, columns= ['Permanent.House','Semi.permanent.House','Temporary.House'], index=temp.index)

df2.plot.bar(figsize=(12,8),stacked=True)

df2 = pd.DataFrame(np.array(temp[['Males','Females']]), columns= ['Males','Females'], index=temp.index)

df2.plot.bar(figsize=(12,8),color = ['navy','red'])
df2 = pd.DataFrame(np.array(temp[['Below.Primary', 'Primary', 'Middle',

       'Matric.Higher.Secondary.Diploma', 'Graduate.and.Above',]]), columns= ['Below.Primary', 'Primary', 'Middle',

       'Matric.Higher.Secondary.Diploma', 'Graduate.and.Above'], index=temp.index)

df2.plot.bar(figsize=(12,8),stacked = True)
df2 = pd.DataFrame(np.array(temp[['X5...14.years', 'X15...59.years', 

                                  'X60.years.and.above..Incl..A.N.S..']]),columns=['X5...14.years', 'X15...59.years', 

                                  'X60.years.and.above..Incl..A.N.S..'], index=temp.index)

df2.plot.barh(figsize=(12,8),stacked = True,title="State wise age groups")
df2 = pd.DataFrame(np.array(temp[['Males..Literate', 'Females..Literate']]),

                   columns=['Males..Literate', 'Females..Literate'], index=temp.index)

df2.plot.bar(figsize=(12,8),stacked = True)
df2 = pd.DataFrame(temp['Post..telegraph.and.telephone.facility']/temp['Persons']*1000,

                   columns=['Post telegraph and telephone facility per Person'], index=temp.index)

df2.plot.bar(figsize=(12,8),stacked = True)
df2 = pd.DataFrame(np.array(temp[['Main.workers', 'Marginal.workers', 'Non.workers']]),

                   columns=['Main.workers', 'Marginal.workers', 'Non.workers'], index=temp.index)

df2.plot.bar(figsize=(12,8),stacked = True)
l =df[['State','SC.1.Name', 'SC.1.Population', 'SC.2.Name', 'SC.2.Population',

       'SC.3.Name', 'SC.3.Population', 'Religeon.1.Name',

       'Religeon.1.Population', 'Religeon.2.Name', 'Religeon.2.Population',

       'Religeon.3.Name', 'Religeon.3.Population', 'ST.1.Name',

       'ST.1.Population', 'ST.2.Name', 'ST.2.Population', 'ST.3.Name',

       'ST.3.Population']]



l.head(20)
l = l.groupby('State').sum()

l.head()
# Scatter plot



plt.figure(figsize=(13,8))

plt.scatter(x = temp['Sex.ratio..females.per.1000.males.'], y = temp['Persons..literacy.rate'],  s = np.array(temp['Persons'])/10**5, alpha = 0.8,color='g')

plt.xlabel('Sex ratio females per 1000 males')

plt.ylabel('Persons literacy rate')

plt.title('India Development')

#plt.xticks([1000,10000,100000], ['1k','10k','100k'])



plt.grid(True)

# Show the plotHousehold.size..per.household.

plt.show()
# Scatter plot



plt.figure(figsize=(13,8))

plt.scatter(x = temp['Household.size..per.household.'], y = temp['Persons..literacy.rate'],  s = np.array(temp['Persons'])/10**5, alpha = 0.8,color='g')

plt.xlabel('Household size per household')

plt.ylabel('Persons literacy rate')

plt.title('India Development')

#plt.xticks([1000,10000,100000], ['1k','10k','100k'])



plt.grid(True)

# Show the plot

plt.show()
set(df['Religeon.2.Name'])