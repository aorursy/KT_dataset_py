# importing data from csv using pandas
import pandas as pd
data = pd.read_csv('../input/countries of the world.csv')
data.info()
#correlation table
data.corr()
# correlation map with seaborn and matplotlib
import seaborn as sns  # visualization tool
import matplotlib.pyplot as plt

f,ax = plt.subplots(figsize=(9, 9))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
data.head(10)
data.columns
dt = data[data["Region"].str.contains("ASIA")]
dt = dt[["Country","Population","Area (sq. mi.)","Pop. Density (per sq. mi.)","GDP ($ per capita)","Industry"]]
dt
# make col names look better
dt.columns = ['Country', 'Population', 'Area', 'PopDensity', 'GDPpC', 'Industry']
dt
dt.corr()
# fix wrong decimal seperators in our data
dt["PopDensity"] = dt["PopDensity"].str.replace(',', '').astype(float)
dt["Industry"] = dt["Industry"].str.replace(',', '').astype(float)
dt.corr()
dt.PopDensity.plot(kind = 'line', color = 'g',label = 'Population',linewidth=1,alpha = 1,grid = True,linestyle = ':')
dt.GDPpC.plot(color = 'r',label = 'Area',linewidth=1, alpha = 1,grid = True,linestyle = '-.')
plt.legend(loc='upper right')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Line Plot')
plt.show()
dt.plot(kind='scatter', x='Area', y='GDPpC',alpha = 0.5,color = 'red')
plt.xlabel('Area')              # label = name of label
plt.ylabel('GDPpC')
plt.title('Attack Defense Scatter Plot')            # title = title of p
dt.Industry.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()
dictionary = {'numpy' : 'math','pandas' : 'data analysis', 'matplotlib':'graphics'}
print(dictionary.keys())
print(dictionary.values())

dictionary['numpy'] = "maths"    # update existing entry
print(dictionary)
dictionary['python'] = "language"       # Add new entry
print(dictionary)
del dictionary['python']              # remove entry with key 'spain'
print(dictionary)
print('matplotlib' in dictionary)        # check include or not
dictionary.clear()                   # remove all entries in dict
print(dictionary)
# del dictionary
# print(dictionary)
series = dt.Population        # data['Defense'] = series
print(type(series))
data_frame = dt[['Population']]  # data[['Defense']] = data frame
print(type(data_frame))
#countries with more than 100 Million pop
oneMPop = dt[dt.Population > 100000000]
oneMPop
i=1
while i<6:
    print(i)
    i += 1
lis = []

i=1
while i<6:
    lis.append(i)
    i += 1
    
for x in lis:
    print(x)
germanNums = {"1" : "ein", "2": "zwei", "3":"drei", "4":"vier", "5":"fünf"}

for key,value in germanNums.items():
    print("{0} in german is {1}".format(key, value))
for index, data in dt[["Country","Population"]].iterrows():
    print("population of {0} is {1}".format(data.Country, data.Population))