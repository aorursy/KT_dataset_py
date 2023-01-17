import pandas as pd
import matplotlib.pyplot as plt

edu = pd.read_csv('../input/dataset/dataset.csv',
na_values = ':',
usecols = ["Favorite Color","Favorite Music Genre","Favorite Beverage","Favorite Soft Drink","Gender", "Salary"])
print(edu.head())
print(edu.describe())

print(edu['Favorite Music Genre'])


edu = edu.append ({"Favorite Color":'Warm',"Favorite Music Genre":'Rock',"Favorite Beverage":'Vodka',"Favorite Soft Drink":'Fanta',"Gender":'M', "Salary":18000},
ignore_index = True)
print(edu.tail())

edu1=edu.sort_values(by = 'Favorite Music Genre', ascending = False,
inplace = False)

print(edu1.head())


edu1=edu[edu['Gender'] == "M"]


s=edu["Salary"]/1000
print(s)

edu2=edu.sort_values(by = 'Favorite Music Genre', ascending = False,
inplace = False)



edu3=edu.drop(["Favorite Color","Favorite Beverage","Favorite Soft Drink"], axis=1, inplace=False)
edu3.head()




plt.hist(edu3['Favorite Music Genre'])

pivedu = pd.pivot_table(edu3, values = 'Salary',
index = ['Gender'],
columns = ['Favorite Music Genre'])




edu3=edu.drop(["Favorite Color","Favorite Beverage","Favorite Soft Drink"], axis=1, inplace=False)
edu3.head()
pivedu = pd.pivot_table(edu3, values = 'Salary',
index = ['Gender'],
columns = ['Favorite Music Genre'])
pivedu.head()
pivedu = pd.pivot_table(edu3, values = 'Salary',
index = ['Gender'],
columns = ['Favorite Music Genre'])

pivedu1 = pivedu.rename(index = {'F':'Female'})
pivedu1 = pivedu.rename(index = {'M':'Male'})
pivedu1.head()

totalSum = pivedu.sum(axis = 1)
totalSum.rank(ascending=False,method='dense')
totalSum.sort_values().head()
totalSum = pivedu.sum(axis = 1)
totalSum.sort_values(ascending = False)
totalSum.plot(kind = 'bar', style = 'b', alpha = 0.4,
title = "Total Salary for Gender")
my_colors = ['b', 'r', 'g', 'y', 'm', 'c', 'black']
ax = pivedu.plot(kind = 'barh', stacked = True,
color = my_colors)
ax.legend(loc = 'center left', bbox_to_anchor = (1, .5))