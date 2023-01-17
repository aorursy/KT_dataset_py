# Importing the libraries for Analysis

import pandas as pd

import matplotlib 

import matplotlib.pyplot as plt

import plotly.graph_objs as go

import plotly.plotly as py

import seaborn as sns

import numpy as np 
# Loading the csv file into a variables

ofns = pd.read_csv("../input/nfl_offense_cleaned.csv")
ofns = ofns.dropna(how='all')

ofns.head()
ofns_t = ofns.drop(ofns.columns[[0]], axis=1)

ofns = ofns.drop(ofns.columns[[0,3,5,8,9,11,12,14]], axis=1)
ofns = ofns[ofns["POS"] ==" QB"]
ofns["PLAYER"].value_counts()
players = ["Eli Manning","Aaron Rodgers","Carson Palmer","Ben Roethlisberger","Drew Brees","Tom Brady"]
Manning = ofns['PLAYER'].map(lambda x: x.startswith(players[0]))

manning = ofns[Manning]

manning = manning.sort_values('YEAR')
Aron = ofns['PLAYER'].map(lambda x: x.startswith(players[1]))

aron = ofns[Aron]

aron = aron.sort_values('YEAR')
Carson = ofns['PLAYER'].map(lambda x: x.startswith(players[2]))

carson = ofns[Carson]

carson = carson.sort_values('YEAR')
Ben = ofns['PLAYER'].map(lambda x: x.startswith(players[3]))

ben = ofns[Ben]

ben = ben.sort_values('YEAR')
Drew = ofns['PLAYER'].map(lambda x: x.startswith(players[4]))

drew = ofns[Drew]

drew = drew.sort_values('YEAR')
Tom = ofns['PLAYER'].map(lambda x: x.startswith(players[5]))

tom = ofns[Tom]

tom = tom.sort_values('YEAR')
# Ploting the graph for count of Yards based on Year

objects = ("2007","2008","2009","2010","2011","2012","2013","2014","2015","2016","2017")



y_pos = np.arange(len(objects))



performance0 = manning["YDS"]

performance1 = aron["YDS"]

performance2 = carson["YDS"]

performance3 = ben["YDS"]

performance4 = drew["YDS"]

performance5 = tom["YDS"]



plt.plot(y_pos, performance0, label = players[0],color='aqua')

plt.plot(y_pos, performance1, label = players[1],color='black')

plt.plot(y_pos, performance2, label = players[2],color='blue')

plt.plot(y_pos, performance3, label = players[3],color='brown')

plt.plot(y_pos, performance4, label = players[4],color='darkgreen')

plt.plot(y_pos, performance5, label = players[5],color='gold')



plt.xticks(y_pos, objects,rotation=90)

plt.xlabel('YEAR')

plt.ylabel('YARDS')

plt.title('NFL QuaterBack analysis based on YARDs')

plt.legend()

plt.show()
ofns_t.head()
# Findn onlly the years which contains 

yr2017 = ofns_t['YEAR'].map(lambda x: str(x).startswith("2017"))

# Creating a Data frame for the year 

year_2017 = ofns_t[yr2017]

# Droping unnecessary  columns

year_2017 = year_2017.drop(year_2017.columns[[0,1,2,14]],axis=1)

# Croping it to 70 so that all the years have the same length 

year_2017 = year_2017.iloc[:70]
yr2016 = ofns_t['YEAR'].map(lambda x: str(x).startswith("2016"))

year_2016 = ofns_t[yr2016]

year_2016 = year_2016.drop(year_2016.columns[[0,1,2,14]],axis=1)

year_2016 = year_2016.iloc[:70]
yr2015 = ofns_t['YEAR'].map(lambda x: str(x).startswith("2015"))

year_2015 = ofns_t[yr2015]

year_2015 = year_2015.drop(year_2015.columns[[0,1,2,14]],axis=1)

year_2015 = year_2015.iloc[:70]
yr2014 = ofns_t['YEAR'].map(lambda x: str(x).startswith("2014"))

year_2014 = ofns_t[yr2014]

year_2014 = year_2014.drop(year_2014.columns[[0,1,2,14]],axis=1)

year_2014 = year_2014.iloc[:70]
yr2013 = ofns_t['YEAR'].map(lambda x: str(x).startswith("2013"))

year_2013 = ofns_t[yr2013]

year_2013 = year_2013.drop(year_2013.columns[[0,1,2,14]],axis=1)

year_2013 = year_2013.iloc[:70]
yr2012 = ofns_t['YEAR'].map(lambda x: str(x).startswith("2012"))

year_2012 = ofns_t[yr2012]

year_2012 = year_2012.drop(year_2012.columns[[0,1,2,14]],axis=1)

year_2012 = year_2012.iloc[:70]
yr2011 = ofns_t['YEAR'].map(lambda x: str(x).startswith("2011"))

year_2011 = ofns_t[yr2011]

year_2011 = year_2011.drop(year_2011.columns[[0,1,2,14]],axis=1)

year_2011 = year_2011.iloc[:70]
yr2010 = ofns_t['YEAR'].map(lambda x: str(x).startswith("2010"))

year_2010 = ofns_t[yr2010]

year_2010 = year_2010.drop(year_2010.columns[[0,1,2,14]],axis=1)

year_2010 = year_2010.iloc[:70]
yr2009 = ofns_t['YEAR'].map(lambda x: str(x).startswith("2009"))

year_2009 = ofns_t[yr2009]

year_2009 = year_2009.drop(year_2009.columns[[0,1,2,14]],axis=1)

year_2009 = year_2009.iloc[:70]
yr2008 = ofns_t['YEAR'].map(lambda x: str(x).startswith("2008"))

year_2008 = ofns_t[yr2008]

year_2008 = year_2008.drop(year_2008.columns[[0,1,2,14]],axis=1)

year_2008 = year_2008.iloc[:70]
yr2007 = ofns_t['YEAR'].map(lambda x: str(x).startswith("2007"))

year_2007 = ofns_t[yr2007]

year_2007 = year_2007.drop(year_2007.columns[[0,1,2,14]],axis=1)

year_2007 = year_2007.iloc[:70]
line = plt.figure()

plt.plot(year_2007['PCT'], year_2008['PCT'], "o")

plt.xlabel('2007')

plt.ylabel('2008')

plt.title('NFL QuaterBack analysis based on Percentage Completion (2007 vs 2008)')

plt.show()

plt.plot(year_2009['PCT'], year_2010['PCT'], "o")

plt.xlabel('2009')

plt.ylabel('2010')

plt.title('NFL QuaterBack analysis based on Percentage Completion (2009 vs 2010)')

plt.show()

plt.plot(year_2011['PCT'], year_2012['PCT'], "o")

plt.xlabel('2011')

plt.ylabel('2012')

plt.title('NFL QuaterBack analysis based on Percentage Completion (2011 vs 2012)')

plt.show()

plt.plot(year_2013['PCT'], year_2014['PCT'], "o")

plt.xlabel('2013')

plt.ylabel('2014')

plt.title('NFL QuaterBack analysis based on Percentage Completion (2013 vs 2014)')

plt.show()

plt.plot(year_2015['PCT'], year_2016['PCT'], "o")

plt.xlabel('2015')

plt.ylabel('2016')

plt.title('NFL QuaterBack analysis based on Percentage Completion (2015 vs 2016)')

plt.show()

plt.plot(year_2016['PCT'], year_2017['PCT'],"o")

plt.xlabel('2016')

plt.ylabel('2017')

plt.title('NFL QuaterBack analysis based on Percentage Completion (2016 vs 2017)')

plt.show()