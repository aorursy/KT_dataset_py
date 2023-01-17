import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.optimize import curve_fit

from mpl_toolkits.basemap import Basemap



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

h1bdat = pd.read_csv('../input/h1b_kaggle.csv')

#Note: Column "Unnamed: 0" (another indexing starting from 1) will be removed

h1bdat = h1bdat.drop(["Unnamed: 0"], axis=1)
print('Number of entries:', h1bdat.shape[0])

print('Number of missing data in each column:')

print(h1bdat.isnull().sum())
ax1 = h1bdat['EMPLOYER_NAME'][h1bdat['YEAR'] == 2011].groupby(h1bdat['EMPLOYER_NAME']).count().sort_values(ascending=False).head(10).plot(kind='barh', title="Top 10 Applicants, 2011")

ax1.set_ylabel("")

plt.show()

ax2 = h1bdat['EMPLOYER_NAME'][h1bdat['YEAR'] == 2016.0].groupby(h1bdat['EMPLOYER_NAME']).count().sort_values(ascending=False).head(10).plot(kind='barh', title="Top 10 Applicant, 2016")

ax2.set_ylabel("")

plt.show()

ax3 = h1bdat['EMPLOYER_NAME'].groupby([h1bdat['EMPLOYER_NAME']]).count().sort_values(ascending=False).head(10).plot(kind='barh', title="Top 10 Applicant over 2011 to 2016")

ax3.set_ylabel("")

plt.show()
topEmp = list(h1bdat['EMPLOYER_NAME'][h1bdat['YEAR'] >= 2015.0].groupby(h1bdat['EMPLOYER_NAME']).count().sort_values(ascending=False).head(10).index)

byEmpYear = h1bdat[['EMPLOYER_NAME', 'YEAR', 'PREVAILING_WAGE']][h1bdat['EMPLOYER_NAME'].isin(topEmp)]

byEmpYear = byEmpYear.groupby([h1bdat['EMPLOYER_NAME'],h1bdat['YEAR']])
markers=['o','v','^','<','>','d','s','p','*','h','x','D','o','v','^','<','>','d','s','p','*','h','x','D']

fig = plt.figure(figsize=(12,7))

for company in topEmp:

    tmp = byEmpYear.count().loc[company]

    plt.plot(tmp.index.values, tmp["PREVAILING_WAGE"].values, label=company, linewidth=2,marker=markers[topEmp.index(company)])

plt.xlabel("Year")

plt.ylabel("Number of Applications")

plt.legend()

plt.title('Number of Applications of Top 10 Applicants')

plt.show()
fig = plt.figure(figsize=(10,7))

for company in topEmp:

    tmp = byEmpYear.mean().loc[company]

    plt.plot(tmp.index.values, tmp["PREVAILING_WAGE"].values, label=company, linewidth=2,  marker=markers[topEmp.index(company)])



plt.xlabel("Year")

plt.ylabel("Average Salary Offer (USD)")

plt.legend()

plt.title("Average Salary of Top 10 Applicants")

plt.show()
for company in ['IBM INDIA PRIVATE LIMITED','ACCENTURE LLP']:

    print(h1bdat[['EMPLOYER_NAME','PREVAILING_WAGE','YEAR']][h1bdat['EMPLOYER_NAME']==company].sort_values(['PREVAILING_WAGE'], ascending=False).head(15))
h1bdat = h1bdat[h1bdat['PREVAILING_WAGE'] <= 500000]

byEmpYear = h1bdat[['EMPLOYER_NAME', 'YEAR', 'PREVAILING_WAGE']][h1bdat['EMPLOYER_NAME'].isin(topEmp)]

byEmpYear = byEmpYear.groupby([h1bdat['EMPLOYER_NAME'],h1bdat['YEAR']])
fig = plt.figure(figsize=(10,7))

for company in topEmp:

    tmp = byEmpYear.mean().loc[company]

    plt.plot(tmp.index.values, tmp["PREVAILING_WAGE"].values, label=company, linewidth=2,  marker=markers[topEmp.index(company)])

        

plt.ylim(50000,110000)

plt.xlabel("Year")

plt.ylabel("Average Salary Offer (USD)")

plt.legend()

plt.title("Salary From Top 10 Applicants")

plt.show()
PopJobs = h1bdat[['JOB_TITLE', 'EMPLOYER_NAME', 'PREVAILING_WAGE']][h1bdat['EMPLOYER_NAME'].isin(topEmp)].groupby(['JOB_TITLE'])

topJobs = list(PopJobs.count().sort_values(by='EMPLOYER_NAME', ascending=False).head(30).index)

df = PopJobs.count().loc[topJobs].assign(mean_wage=PopJobs.mean().loc[topJobs])

fig = plt.figure(figsize=(10,12))

ax1 = fig.add_subplot(111)

ax2 = ax1.twiny()

width = 0.35

df.EMPLOYER_NAME.plot(kind='barh', ax=ax1, color='C0', width=0.4, position=0, label='# of Applications')

df.mean_wage.plot(kind='barh', ax=ax2, color='C7', width=0.4, position=1, label='Mean Salary')

ax1.set_xlabel('Number of Applications')

ax1.set_ylabel('')

ax1.legend(loc=(0.75,0.55))

ax2.set_xlabel('Mean Salary')

ax2.set_ylabel('Job Title')

ax2.legend(loc=(0.75,0.50))

plt.show()
ax = h1bdat[['JOB_TITLE', 'EMPLOYER_NAME', 'PREVAILING_WAGE']][h1bdat['EMPLOYER_NAME'].isin(topEmp)  & h1bdat['JOB_TITLE'].isin(topJobs)]['PREVAILING_WAGE'].hist(bins=100)

ax.set_ylabel('Offering Wage (USD/year)')

plt.title('Offering Salary Distribution of Popular Jobs from Top Applicants')

plt.show()
PopJobsAll = h1bdat[['JOB_TITLE', 'EMPLOYER_NAME', 'PREVAILING_WAGE']].groupby(['JOB_TITLE'])

topJobsAll = list(PopJobsAll.count().sort_values(by='EMPLOYER_NAME', ascending=False).head(30).index)

dfAll = PopJobsAll.count().loc[topJobsAll].assign(mean_wage=PopJobsAll.mean().loc[topJobsAll])

fig = plt.figure(figsize=(10,12))

ax1 = fig.add_subplot(111)

ax2 = ax1.twiny()

width = 0.35

dfAll.EMPLOYER_NAME.plot(kind='barh', ax=ax1, color='C0', width=0.4, position=0, label='# of Applications')

dfAll.mean_wage.plot(kind='barh', ax=ax2, color='C7', width=0.4, position=1, label='Mean Salary')

ax1.set_xlabel('# of Applications')

ax1.set_ylabel('')

ax1.legend(loc=(0.75,0.55))

ax2.set_xlabel('Mean wage')

ax2.set_ylabel('Job Title')

ax2.legend(loc=(0.75,0.50))

plt.show()
dsj = h1bdat[['JOB_TITLE','YEAR']][h1bdat['JOB_TITLE'] == "DATA SCIENTIST"].groupby('YEAR').count()['JOB_TITLE']

X = np.array(dsj.index)

Y = dsj.values

def func(x, a, b, c):

    return a*np.power(x-2011,b)+c



popt, pcov = curve_fit(func, X, Y)

X1 = np.linspace(2011,2018,9)

X2 = np.linspace(2016,2018,3)

X3 = np.linspace(2017,2018,2)

fig = plt.figure(figsize=(7,5))

plt.scatter(list(dsj.index), dsj.values, c='C0', marker='o', s=120, label='Data')

plt.plot(X1, func(X1,*popt), color='C0', label='')

plt.plot(X2, func(X2,*popt), color='C5', linewidth=3, marker='s', markersize=1, label='')

plt.plot(X3, func(X3,*popt), color='C5', marker='s', markersize=10, label='Prediction')

plt.legend()

plt.title('Number of Data Scientist Jobs')

plt.xlabel('Year')

plt.show()
ax1 = h1bdat[h1bdat['JOB_TITLE'] == "DATA SCIENTIST"]['EMPLOYER_NAME'].groupby(h1bdat['EMPLOYER_NAME']).count().sort_values(ascending=False).head(10).plot(kind='barh', title="Top 10 Data Scientist Hiring Companies, 2011-2016")

ax1.set_ylabel("")

plt.show()

ax2 = h1bdat[h1bdat['JOB_TITLE'] == "DATA SCIENTIST"]['EMPLOYER_NAME'][h1bdat['YEAR'] == 2016.0].groupby(h1bdat['EMPLOYER_NAME']).count().sort_values(ascending=False).head(10).plot(kind='barh', title="Top 10 Data Scientist Hiring Companies, 2016")

ax2.set_ylabel("")

plt.show()

ax3 = h1bdat[h1bdat['JOB_TITLE'] == "DATA SCIENTIST"]['WORKSITE'].groupby(h1bdat['WORKSITE']).count().sort_values(ascending=False).head(10).plot(kind='barh', title="Top City Data Scientist Work In, 2011-2016")

ax3.set_ylabel("")

plt.show()
PopJobs2016 = h1bdat[['JOB_TITLE', 'EMPLOYER_NAME', 'PREVAILING_WAGE','YEAR']]

PopJobs2016 = PopJobs2016[PopJobs2016['YEAR']==2016].groupby(['JOB_TITLE'])

topJobs2016 = list(PopJobs2016.count().sort_values(by='EMPLOYER_NAME', ascending=False).head(30).index)

df2016 = PopJobs2016.count().loc[topJobs2016].assign(mean_wage=PopJobs2016.mean().loc[topJobs2016]['PREVAILING_WAGE'])

fig = plt.figure(figsize=(10,12))

ax1 = fig.add_subplot(111)

ax2 = ax1.twiny()

width = 0.35

df2016.EMPLOYER_NAME.plot(kind='barh', ax=ax1, color='C0', width=0.4, position=0, label='# of Applications')

df2016.mean_wage.plot(kind='barh', ax=ax2, color='C7', width=0.4, position=1, label='Mean Salary')

ax1.set_xlabel('Number of Applications')

ax1.set_ylabel('')

ax1.legend(loc=(0.75,0.55))

ax2.set_xlabel('Mean Salary')

ax2.set_ylabel('Job Title')

ax2.legend(loc=(0.75,0.50))

plt.show()
# I don't know how to add additional file to the folder here so I paste whole dictionary for state capitol positions here ^^

capitolpos = {'ALABAMA': (-86.79113, 32.806671), 'ALASKA': (-152.404419, 61.370716), 'ARIZONA': (-111.431221, 33.729759),

 'ARKANSAS': (-92.373123, 34.969704), 'CALIFORNIA': (-119.681564, 36.116203), 'COLORADO': (-105.311104, 39.059811),

 'CONNECTICUT': (-72.755371, 41.597782), 'DELAWARE': (-75.507141, 39.318523), 'DISTRICT OF COLUMBIA': (-77.026817, 38.897438),

 'FLORIDA': (-81.686783, 27.766279), 'GEORGIA': (-83.643074, 33.040619), 'HAWAII': (-157.498337, 21.094318),

 'IDAHO': (-114.478828, 44.240459), 'ILLINOIS': (-88.986137, 40.349457), 'INDIANA': (-86.258278, 39.849426),

 'IOWA': (-93.210526, 42.011539), 'KANSAS': (-96.726486, 38.5266), 'KENTUCKY': (-84.670067, 37.66814),

 'LOUISIANA': (-91.867805, 31.169546), 'MAINE': (-69.381927, 44.693947), 'MARYLAND': (-76.802101, 39.063946),

 'MASSACHUSETTS': (-71.530106, 42.230171), 'MICHIGAN': (-84.536095, 43.326618), 'MINNESOTA': (-93.900192, 45.694454),

 'MISSISSIPPI': (-89.678696, 32.741646), 'MISSOURI': (-92.288368, 38.456085), 'MONTANA': (-110.454353, 46.921925),

 'NEBRASKA': (-98.268082, 41.12537), 'NEVADA': (-117.055374, 38.313515), 'NEW HAMPSHIRE': (-71.563896, 43.452492),

 'NEW JERSEY': (-74.521011, 40.298904), 'NEW MEXICO': (-106.248482, 34.840515), 'NEW YORK': (-74.948051, 42.165726),

 'NORTH CAROLINA': (-79.806419, 35.630066), 'NORTH DAKOTA': (-99.784012, 47.528912), 'OHIO': (-82.764915, 40.388783),

 'OKLAHOMA': (-96.928917, 35.565342), 'OREGON': (-122.070938, 44.572021), 'PENNSYLVANIA': (-77.209755, 40.590752),

 'RHODE ISLAND': (-71.51178, 41.680893), 'SOUTH CAROLINA': (-80.945007, 33.856892), 'SOUTH DAKOTA': (-99.438828, 44.299782),

 'TENNESSEE': (-86.692345, 35.747845), 'TEXAS': (-97.563461, 31.054487), 'UTAH': (-111.862434, 40.150032),

 'VERMONT': (-72.710686, 44.045876), 'VIRGINIA': (-78.169968, 37.769337), 'WASHINGTON': (-121.490494, 47.400902),

 'WEST VIRGINIA': (-80.954453, 38.491226), 'WISCONSIN': (-89.616508, 44.268543), 'WYOMING': (-107.30249, 42.755966)}
h1bdat = h1bdat.assign(state=h1bdat['WORKSITE'].str.split(',').str.get(1).str.strip())

stlist = list(capitolpos.keys())

h1bdat[~h1bdat.state.isin(stlist)]['state'].value_counts()

h1bdat = h1bdat[h1bdat.state.isin(capitolpos.keys())]
sbystate = h1bdat[['state','PREVAILING_WAGE']].groupby(h1bdat['state']).count()['state']

X = []

Y = []

for state in list(sbystate.index):

    (lon, lan) = capitolpos[state]

    X.append(lon)

    Y.append(lan)

fig = plt.figure(figsize=(16,8))

m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,

        projection='lcc',lat_1=32,lat_2=45,lon_0=-95)

m.drawcoastlines()

m.drawcountries(linewidth=3, color='C0')

m.drawstates(linewidth=1.5)

xmap, ymap = m(X, Y)

ax = m.scatter(xmap, ymap, s=1000, c=sbystate.values/1000, cmap='cool', vmin=sbystate.values.min()/1000, vmax=sbystate.values.max()/1000)

cb = plt.colorbar(ax)

plt.legend()

plt.title('Number of Applications by State (x1000)')

#plt.show()

sbystate = h1bdat[['state','PREVAILING_WAGE']].groupby(h1bdat['state']).mean()['PREVAILING_WAGE']

X = []

Y = []

for state in list(sbystate.index):

    (lon, lan) = capitolpos[state]

    X.append(lon)

    Y.append(lan)

fig = plt.figure(figsize=(16,8))

m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,

        projection='lcc',lat_1=32,lat_2=45,lon_0=-95)

m.drawcoastlines()

m.drawcountries(linewidth=3, color='C0')

m.drawstates(linewidth=1.5)

xmap, ymap = m(X, Y)

ax = m.scatter(xmap, ymap, s=1000, c=sbystate.values/1000, cmap='cool', vmin=sbystate.values.min()/1000, vmax=sbystate.values.max()/1000)

cb = plt.colorbar(ax)

plt.legend()

plt.title('Average Salary by State (x1000 USD/year)')

plt.show()
print("Number of petitions: top 5 states having least peritions")

print(h1bdat['state'].groupby(h1bdat['state']).count().sort_values().head(5))

print("")

print('Top high salary jobs in Wyoming')

print(h1bdat[h1bdat['state']=='WYOMING'][['JOB_TITLE','PREVAILING_WAGE']].groupby(h1bdat['JOB_TITLE']).mean().sort_values(by='PREVAILING_WAGE',ascending=False).head(5))

print("")

print('Top high salary jobs in North Dakota')

print(h1bdat[h1bdat['state']=='NORTH DAKOTA'][['JOB_TITLE','PREVAILING_WAGE']].groupby(h1bdat['JOB_TITLE']).mean().sort_values(by='PREVAILING_WAGE',ascending=False).head(5))
sbystate = h1bdat[h1bdat['JOB_TITLE']=='PROGRAMMER ANALYST'][['state','PREVAILING_WAGE']].groupby(h1bdat['state']).mean()

sbystate2 = h1bdat[h1bdat['JOB_TITLE']=='PROGRAMMER ANALYST'][['state','PREVAILING_WAGE']].groupby(h1bdat['state']).count()

X = []

Y = []

for state in list(sbystate.index):

    (lon, lan) = capitolpos[state]

    X.append(lon)

    Y.append(lan)

fig = plt.figure(figsize=(16,8))

m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,

        projection='lcc',lat_1=32,lat_2=45,lon_0=-95)

m.drawcoastlines()

m.drawcountries(linewidth=3, color='C0')

m.drawstates(linewidth=1.5)

xmap, ymap = m(X, Y)

ax = m.scatter(xmap, ymap, s=1.5*sbystate2.state.values/np.min(sbystate2.state.values), c=sbystate.PREVAILING_WAGE.values/1000, cmap='cool', vmin=sbystate.values.min()/1000, vmax=sbystate.values.max()/1000)



cb = plt.colorbar(ax)

plt.legend()

plt.title('Average "Programmer Analyst" Salary by State (x1000 USD/year), 2011-2016')

####

sbystate2 = h1bdat[h1bdat['JOB_TITLE']=='SOFTWARE ENGINEER'][['state','PREVAILING_WAGE']].groupby(h1bdat['state']).count()

X = []

Y = []

for state in list(sbystate.index):

    (lon, lan) = capitolpos[state]

    X.append(lon)

    Y.append(lan)



fig = plt.figure(figsize=(16,8))

m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,

        projection='lcc',lat_1=32,lat_2=45,lon_0=-95)

m.drawcoastlines()

m.drawcountries(linewidth=3, color='C0')

m.drawstates(linewidth=1.5)

xmap, ymap = m(X, Y)

ax = m.scatter(xmap, ymap, s = 1.5*sbystate2.state.values/np.min(sbystate2.state.values), c=sbystate.PREVAILING_WAGE.values/1000, cmap='cool', vmin=sbystate.values.min()/1000, vmax=sbystate.values.max()/1000)

cb = plt.colorbar(ax)

plt.legend()

plt.title('Average "Software Engineer" Salary by State (x1000 USD/year), 2011-2016')

plt.show()
# Dictionary of 2015 RPP:

rpp2015 = {'ALABAMA': 0.868, 'ALASKA': 1.056, 'ARIZONA': 0.962, 'ARKANSAS': 0.874, 'CALIFORNIA': 1.134,

 'COLORADO': 1.032, 'CONNECTICUT': 1.087, 'DELAWARE': 1.004, 'DISTRICT OF COLUMBIA': 1.17, 'FLORIDA': 0.995,

 'GEORGIA': 0.926, 'HAWAII': 1.188, 'IDAHO': 0.934, 'ILLINOIS': 0.997, 'INDIANA': 0.907,

 'IOWA': 0.903, 'KANSAS': 0.904, 'KENTUCKY': 0.886, 'LOUISIANA': 0.906, 'MAINE': 0.98,

 'MARYLAND': 1.096, 'MASSACHUSETTS': 1.069, 'MICHIGAN': 0.935, 'MINNESOTA': 0.974, 'MISSISSIPPI': 0.862,

 'MISSOURI': 0.893, 'MONTANA': 0.948, 'NEBRASKA': 0.906, 'NEVADA': 0.98, 'NEW HAMPSHIRE': 1.05,

 'NEW JERSEY': 1.134, 'NEW MEXICO': 0.944, 'NEW YORK': 1.153, 'NORTH CAROLINA': 0.912, 'NORTH DAKOTA': 0.923,

 'OHIO': 0.892, 'OKLAHOMA': 0.899, 'OREGON': 0.992, 'PENNSYLVANIA': 0.979, 'RHODE ISLAND': 0.987,

 'SOUTH CAROLINA': 0.903, 'SOUTH DAKOTA': 0.882, 'TENNESSEE': 0.899, 'TEXAS': 0.968, 'UNITED STATES': 1.0,

 'UTAH': 0.97, 'VERMONT': 1.016, 'VIRGINIA': 1.025, 'WASHINGTON': 1.048, 'WEST VIRGINIA': 0.889,

 'WISCONSIN': 0.931, 'WYOMING': 0.962}
sbystate = h1bdat[h1bdat['JOB_TITLE']=='PROGRAMMER ANALYST']

sbystate = sbystate[sbystate['YEAR'] == 2015][['state','PREVAILING_WAGE']].groupby(h1bdat['state']).mean()

sbystate2 = h1bdat[h1bdat['JOB_TITLE']=='PROGRAMMER ANALYST']

sbystate2 = sbystate2[sbystate2['YEAR'] ==2015][['state','PREVAILING_WAGE']].groupby(h1bdat['state']).count()

X = []

Y = []

for state in list(sbystate.index):

    (lon, lan) = capitolpos[state]

    X.append(lon)

    Y.append(lan)

fig = plt.figure(figsize=(16,8))

m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,

        projection='lcc',lat_1=32,lat_2=45,lon_0=-95)

m.drawcoastlines()

m.drawcountries(linewidth=3, color='C0')

m.drawstates(linewidth=1.5)

xmap, ymap = m(X, Y)

ax = m.scatter(xmap, ymap, s=1.5*sbystate2.state.values/np.min(sbystate2.state.values), c=sbystate.PREVAILING_WAGE.values/1000, cmap='cool', vmin=sbystate.values.min()/1000, vmax=sbystate.values.max()/1000)

cb = plt.colorbar(ax)

plt.legend()

plt.title('Average "Programmer Analyst" Salary by State in 2015 (x1000 USD/year)')

########

sbystate3 = h1bdat[h1bdat['JOB_TITLE']=='PROGRAMMER ANALYST']

sbystate3 = sbystate3[sbystate3['YEAR'] ==2015][['state','PREVAILING_WAGE']]

sbystate3 = sbystate3.assign(adj_salary=sbystate3.apply(lambda x: x.PREVAILING_WAGE/rpp2015[x['state']], axis=1))

sbystate3 = sbystate3[['state','adj_salary']].groupby(h1bdat['state']).mean()

X = []

Y = []

for state in list(sbystate.index):

    (lon, lan) = capitolpos[state]

    X.append(lon)

    Y.append(lan)

fig = plt.figure(figsize=(16,8))

m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,

        projection='lcc',lat_1=32,lat_2=45,lon_0=-95)

m.drawcoastlines()

m.drawcountries(linewidth=3, color='C0')

m.drawstates(linewidth=1.5)

xmap, ymap = m(X, Y)

ax = m.scatter(xmap, ymap, s=1.5*sbystate2.state.values/np.min(sbystate2.state.values), c=sbystate3.adj_salary.values/1000, cmap='cool', vmin=sbystate.values.min()/1000, vmax=sbystate.values.max()/1000)

cb = plt.colorbar(ax)

plt.legend()

plt.title('Adjusted Average "Programmer Analyst" Salary by State in 2015 (x1000 USD/year)')

plt.show()
sbystate = h1bdat[h1bdat['JOB_TITLE']=='SOFTWARE ENGINEER']

sbystate = sbystate[sbystate['YEAR'] == 2015][['state','PREVAILING_WAGE']].groupby(h1bdat['state']).mean()

sbystate2 = h1bdat[h1bdat['JOB_TITLE']=='SOFTWARE ENGINEER']

sbystate2 = sbystate2[sbystate2['YEAR'] ==2015][['state','PREVAILING_WAGE']].groupby(h1bdat['state']).count()

X = []

Y = []

for state in list(sbystate.index):

    (lon, lan) = capitolpos[state]

    X.append(lon)

    Y.append(lan)

fig = plt.figure(figsize=(16,8))

m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,

        projection='lcc',lat_1=32,lat_2=45,lon_0=-95)

m.drawcoastlines()

m.drawcountries(linewidth=3, color='C0')

m.drawstates(linewidth=1.5)

xmap, ymap = m(X, Y)

ax = m.scatter(xmap, ymap, s=1.5*sbystate2.state.values/np.min(sbystate2.state.values), c=sbystate.PREVAILING_WAGE.values/1000, cmap='cool', vmin=sbystate.values.min()/1000, vmax=sbystate.values.max()/1000)

cb = plt.colorbar(ax)

plt.legend()

plt.title('Average "Software Engineer" Salary by State in 2015 (x1000 USD/year)')

########

sbystate3 = h1bdat[h1bdat['JOB_TITLE']=='SOFTWARE ENGINEER']

sbystate3 = sbystate3[sbystate3['YEAR'] ==2015][['state','PREVAILING_WAGE']]

sbystate3 = sbystate3.assign(adj_salary=sbystate3.apply(lambda x: x.PREVAILING_WAGE/rpp2015[x['state']], axis=1))

sbystate3 = sbystate3[['state','adj_salary']].groupby(h1bdat['state']).mean()

X = []

Y = []

for state in list(sbystate.index):

    (lon, lan) = capitolpos[state]

    X.append(lon)

    Y.append(lan)

fig = plt.figure(figsize=(16,8))

m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,

        projection='lcc',lat_1=32,lat_2=45,lon_0=-95)

m.drawcoastlines()

m.drawcountries(linewidth=3, color='C0')

m.drawstates(linewidth=1.5)

xmap, ymap = m(X, Y)

ax = m.scatter(xmap, ymap, s=1.5*sbystate2.state.values/np.min(sbystate2.state.values), c=sbystate3.adj_salary.values/1000, cmap='cool', vmin=sbystate.values.min()/1000, vmax=sbystate.values.max()/1000)

cb = plt.colorbar(ax)

plt.legend()

plt.title('Adjusted Average "Software Engineer" Salary by State in 2015 (x1000 USD/year)')

plt.show()