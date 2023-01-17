%matplotlib inline
# The data comes as the raw data files, a transformed CSV file, and a SQLite database
import matplotlib.pyplot as pl
%matplotlib inline
import pandas as pd
import sqlite3
import numpy as np


# You can read in the SQLite datbase like this
import sqlite3
con = sqlite3.connect('../input/database.sqlite')
State = pd.read_sql_query("""
SELECT STABBR,
       COSTT4_A AverageCostOfAttendance,
       Year
FROM Scorecard
""", con)

State = State.fillna(0)

    




# You can read a CSV file like this
#scorecard = pd.read_csv("../input/Scorecard.csv")

# It's yours to take from here!

sample = pd.read_sql_query("""
SELECT STABBR,
       COSTT4_A AverageCostOfAttendance,
       Year
FROM Scorecard
WHERE INSTNM='KS'""", con)





# You can read in the SQLite datbase like this
#called pell grant percentage, region, state abbrieviation, institution name, and year
import sqlite3
con = sqlite3.connect('../input/database.sqlite')
Schooldata = pd.read_sql_query("""
SELECT STABBR,INSTNM,region,
       Year, PCTPELL
FROM Scorecard
WHERE Year >2009

""", con)
#
'''regional = Schooldata.pivot_table(index = 'region', values = 'PCTPELL', columns = 'Year', aggfunc = np.mean)
ax = regional.plot(kind = 'bar', legend = False)
patches, labels = ax.get_legend_handles_labels()
ax.legend(patches, labels, loc = 'best')
ax = regional.plot(kind = 'bar')
pl.show()
'''

Schooldata['region'].head()
#See the unique values in the column
Schooldata['region'].unique()
#define a dictionary that we will use to map the new values
regions = {'Southwest (AZ, NM, OK, TX)': 'Southwest', 'Southeast (AL, AR, FL, GA, KY, LA, MS, NC, SC, TN, VA, WV)':'Southeast', 'Far West (AK, CA, HI, NV, OR, WA)':'Far West', 'Plains (IA, KS, MN, MO, NE, ND, SD)':'Plains', 'Rocky Mountains (CO, ID, MT, UT, WY)':'Rocky Mountains', 'New England (CT, ME, MA, NH, RI, VT)':'New England', 'Mid East (DE, DC, MD, NJ, NY, PA)':'Mid East', 'Great Lakes (IL, IN, MI, OH, WI)':'Great Lakes', 'U.S. Service Schools':'U.S. Service Schools', 'Outlying Areas (AS, FM, GU, MH, MP, PR, PW, VI)': 'Outlying Areas'} 
#Define a new column that we will use to define the new values, you could just do it in place but you would change the data
Schooldata['RGN'] = Schooldata['region']
#the replace function will map the dictioary keys and replace them with the dictionary values specified
Schooldata['RGN'].replace(regions, inplace = True)
#The new values, they are easier to fit in the graph
Schooldata['RGN'].unique()
#pivot table for the data
regional = Schooldata.pivot_table(index = 'RGN', values = 'PCTPELL', columns = 'Year', aggfunc = np.mean)
#set up the bar plot, turn the legen off
ax = regional.plot(kind = 'bar', legend = False)
#this will define the legend handles and associated labels
handles, labels = ax.get_legend_handles_labels()
#then this adds the plot legend
ax.legend(handles, labels, loc = 'best')
pl.title('Regional average of Pell grant recipient percentage')
#What is going in the outlying areas? PREDDEG
"""
Predominant degree awarded
0        Not classified
1        Predominantly certificate-degree granting
2        Predominantly associate's-degree granting
3        Predominantly bachelor's-degree granting
4        Entirely graduate-degree granting

CONTROL
1 public
2 private nonprofit
3 private for profit


LOCALE
11	City: Large (population of 250,000 or more)
12	City: Midsize (population of at least 100,000 but less than 250,000)
13	City: Small (population less than 100,000)
21	Suburb: Large (outside principal city, in urbanized area with population of 250,000 or more)
22	Suburb: Midsize (outside principal city, in urbanized area with population of at least 100,000 but less than 250,000)
23	Suburb: Small (outside principal city, in urbanized area with population less than 100,000)
31	Town: Fringe (in urban cluster up to 10 miles from an urbanized area)
32	Town: Distant (in urban cluster more than 10 miles and up to 35 miles from an urbanized area)
33	Town: Remote (in urban cluster more than 35 miles from an urbanized area)
41	Rural: Fringe (rural territory up to 5 miles from an urbanized area or up to 2.5 miles from an urban cluster)
42	Rural: Distant (rural territory more than 5 miles but up to 25 miles from an urbanized area or more than 2.5 and up to 10 miles from an urban cluster)
43	Rural: Remote (rural territory more than 25 miles from an urbanized area and more than 10 miles from an urban cluster)


avg_net_price.public	integer	NPT4_PUB
avg_net_price.private	integer	NPT4_PRIV
avg_net_price.program_year	integer	NPT4_PROG
avg_net_price.other_academic_year	integer	NPT4_OTHER
"""
Outlying = pd.read_sql_query("""

SELECT STABBR,INSTNM,region,
       Year, PCTPELL, NPT4_PUB, NPT4_PRIV, NPT4_PROG, NPT4_OTHER, LOCALE, CONTROL, PREDDEG
FROM Scorecard
WHERE Year >2009

""", con)
Outlying = Outlying.fillna(0)
#We have to change the column values again
Outlying['RGN'] = Outlying['region']
Outlying['RGN'].replace(regions, inplace = True)
Outlying['RGN'].head()
# only select the region Outlying areas
Outlying = Outlying[Outlying['RGN']=='Outlying Areas']
Outlying.head()
Outlying.columns
#what kind of degrees are prevelant
degrees = Outlying.pivot_table(index = 'PREDDEG', values = 'INSTNM', aggfunc = np.count_nonzero)
degrees.head()
degrees.plot(kind = 'bar')
#look at average cost of year of study
costByDegree = Outlying.pivot_table(index = 'PREDDEG', values = ['NPT4_PRIV', 'NPT4_PUB', 'NPT4_PROG', 'NPT4_OTHER', 'PCTPELL'], aggfunc = np.mean)
costByDegree
costByDegree.columns

costByDegree[['NPT4_OTHER', 'NPT4_PRIV', 'NPT4_PROG', 'NPT4_PUB']].plot(kind='bar')
