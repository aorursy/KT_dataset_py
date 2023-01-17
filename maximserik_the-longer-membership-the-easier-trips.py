import numpy as np 

import pandas as pd 

import datetime as datetime

import matplotlib.pyplot as plt

import folium

pd.options.display.max_rows = 100
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
trips = pd.read_csv('../input/austin_bikeshare_trips.csv')

print('Shape: ', trips.shape)

trips.head()
stations = pd.read_csv('../input/austin_bikeshare_stations.csv')

print('Shape: ', stations.shape)

stations.head()
print('Stations missing values:')

stations.isnull().sum()
print('Trips missing values: ')

trips.isnull().sum()
# transforming start_time to dateTime Object, adding 'end_time' column

trips['start_time'] = pd.to_datetime(trips['start_time'])

deltas = trips['duration_minutes'].values

endTimes = []

for i in range(trips.shape[0]):

    value = trips['start_time'][i] + datetime.timedelta(minutes=int (trips['duration_minutes'][i]))

    endTimes.append(value)

trips['end_time'] = endTimes
# year missing values

trips['year'] = trips['year'].fillna(trips['start_time'].dt.year)



# month missing values

trips['month'] = trips['month'].fillna(trips['start_time'].dt.month)



# subscriber_type missing values 2077 (0.3%) - filling with the most frequent 'Walk Up'

trips['subscriber_type'] = trips['subscriber_type'].fillna('Walk Up')



# bikeid missing values. 723 (0.1%), fill all n/a with -1, unknown value

trips['bikeid'] = trips['bikeid'].fillna(-1)
trips[trips.start_station_id.isnull()]['start_station_name'].value_counts()
def fillMissingStartId(missingName, correctedName):

    ind = trips[(trips.start_station_id.isnull()) & 

       (trips.start_station_name == missingName)].index

    trips.loc[ind, 'start_station_id'] = stations[stations.name ==correctedName]['station_id'].values[0]



missingCorrNames = []

missingCorrNames.append(('Zilker Park at Barton Springs & William Barton Drive', 'Zilker Park'))

missingCorrNames.append(('ACC - West & 12th', 'ACC - West & 12th Street'))

missingCorrNames.append(( 'Convention Center/ 3rd & Trinity', 'Convention Center / 3rd & Trinity'))

missingCorrNames.append(('Mobile Station', 'Convention Center / 4th St. @ MetroRail'))

missingCorrNames.append(('East 11th Street at Victory Grill',  'East 11th St. at Victory Grill'))

missingCorrNames.append(('Red River @ LBJ Library', 'Red River & LBJ Library'))

missingCorrNames.append(('Mobile Station @ Bike Fest', '4th & Congress'))

missingCorrNames.append(('Main Office', 'OFFICE/Main/Shop/Repair'))

missingCorrNames.append(('Shop', 'OFFICE/Main/Shop/Repair'))

    

for missingCorrName in missingCorrNames:

    fillMissingStartId(missingCorrName[0], missingCorrName[1])



# All the other (130 - 0.02%) na fill the most frequent. 2575 - Riverside @ S. Lamar 

trips['start_station_id'] = trips['start_station_id'].fillna(2575)

trips[trips.end_station_id.isnull()]['end_station_name'].value_counts()
def fillMissingEndId(missingName, correctedName):

    ind = trips[(trips.end_station_id.isnull()) & 

       (trips.end_station_name == missingName)].index

    trips.loc[ind, 'end_station_id'] = stations[stations.name ==correctedName]['station_id'].values[0]



missingCorrNames = []

missingCorrNames.append(('Zilker Park at Barton Springs & William Barton Drive', 'Zilker Park'))

missingCorrNames.append(('ACC - West & 12th', 'ACC - West & 12th Street'))

missingCorrNames.append(( 'Convention Center/ 3rd & Trinity',  'Convention Center / 3rd & Trinity'))

missingCorrNames.append(('Mobile Station', 'Convention Center / 4th St. @ MetroRail'))

missingCorrNames.append(('East 11th Street at Victory Grill', 'East 11th St. at Victory Grill'))

missingCorrNames.append(('Red River @ LBJ Library', 'Red River & LBJ Library'))

missingCorrNames.append(('Main Office', 'OFFICE/Main/Shop/Repair'))

missingCorrNames.append(( 'Customer Service', 'OFFICE/Main/Shop/Repair'))

missingCorrNames.append(('Repair Shop', 'OFFICE/Main/Shop/Repair'))

missingCorrNames.append(('Mobile Station @ Bike Fest', '5th & Bowie'))

missingCorrNames.append(('Shop', 'OFFICE/Main/Shop/Repair'))



for missingCorrName in missingCorrNames:

    fillMissingEndId(missingCorrName[0], missingCorrName[1])

    

# All the other (186 - 0.03%) na fill the most frequent. 2499 - City Hall / Lavaca & 2nd 

trips['end_station_id'] = trips['end_station_id'].fillna(2499)
trips.isnull().sum().any()
stations['status'].value_counts()
def getLatLon(allStations):

    res = []

    resNames = []    

    latitudes = allStations['latitude']

    longitudes = allStations['longitude']

    names = allStations['name']    

    for latitude, longitude, name in zip(latitudes,longitudes, names):

        res.append((latitude, longitude))

        resNames.append(name)        

    return res, resNames



activeStations = stations[stations.status == 'active']

movedStations = stations[stations.status == 'moved']

closedStations = stations[stations.status == 'closed']

ACLStations = stations[stations.status == 'ACL only']



latlonActive, namesStationsActive = getLatLon(activeStations)

latlonMoved, namesStationsMoved = getLatLon(movedStations)

latlonClosed, namesStationsClosed = getLatLon(closedStations)

latlonACL, namesStationsACL = getLatLon(ACLStations)



mapStations = folium.Map( location=[30.26754, -97.74154], zoom_start=14 )

for latlon, names, color in zip((latlonActive, latlonMoved, latlonClosed, latlonACL), 

                                 (namesStationsActive, namesStationsMoved, 

                                  namesStationsClosed, namesStationsACL),

                                ('green', 'blue', 'red', 'purple')):

    i=0

    for coord in latlon:

        folium.Marker( location=[ coord[0], coord[1]], icon=folium.Icon(color=color), 

                      popup=names[i]).add_to( mapStations )

        i += 1

mapStations
station_ids = stations['station_id'].values

first_trip = []

last_trip = []

workDays = []

for station_id in station_ids:

    wDays = len(trips[(trips.start_station_id == station_id) | (trips.end_station_id == station_id)]['start_time'].dt.date.value_counts())

    start = trips[(trips['start_station_id'] == station_id) | 

                  (trips['end_station_id'] == station_id)].sort_values(by='start_time')

    start_date = start['start_time'].iloc[0]

    end = trips[(trips['start_station_id'] == station_id) | 

                  (trips['end_station_id'] == station_id)].sort_values(by='end_time')

    end_date = end['end_time'].iloc[len(end) - 1]

    

    first_trip.append(start_date)

    last_trip.append(end_date)

    workDays.append(wDays)

stations['First trip'] = first_trip

stations['Last trip'] = last_trip

stations['First trip'] = stations['First trip'].dt.date

stations['Last trip'] = stations['Last trip'].dt.date

stations['Working Days'] = workDays
stations.sort_values(by='Working Days', ascending=False)[:5]
stations[(stations.name == 'Guadalupe & 6th') | (stations.name == 'Guadalupe & 21st') 

         |(stations.name == 'Pease Park')]
stations.loc[stations.name == 'Guadalupe & 6th', 'status'] = 'moved'

stations.loc[stations.name == 'Guadalupe & 21st', 'status'] = 'active'

stations.loc[stations.name == 'Pease Park', 'status'] = 'closed'



stations = stations.sort_values(by='name')

stations['docks_total'] = [12, 13, 12, 15, 13, 15, 15, 9, 11, 13, 14, 11, 5, 11, 11, 16,

                          13, 9, 9, 14, 13, 18, 19, 11, 13, 17, 19, 11, 11, 10, 9, 11,

                          13, 5, 13, 11, 13, 12, 13, 11, 19, 11, 11, 13, 15, 13, 19, 12,

                          14, 17, 13, 13, 9, 13, 13, 15, 13, 10, 13, 13, 9, 11, 13, 18,

                          12, 13, 11, 15, 13, 11, 16, 6]

stations = stations.sort_index()

stations.head()
colorsSt = {'active':'green', 'closed':'red', 'moved':'blue','ACL only': 'purple'}

station_names = stations['name'].values



def genTicks(size):

    res = []

    res.append(0)

    step = size / (size - 1)

    total = step

    for i in range(size-1):

        res.append(total)

        total += step

    return res



def showStationOpenDays(wDays):

    colors = stations['status'].map(colorsSt)

    sortedStationNames = [x for (y,x) in sorted(zip(wDays, station_names))]

    colSorted = [x for (y,x) in sorted(zip(wDays, colors))]

    sortedWdays = sorted(wDays)

    

    

    x = np.linspace(0, len(sortedWdays), len(sortedWdays))

    y = sortedWdays

  

    plt.figure(figsize=(15,10))

    plt.bar(x, y, color=colSorted)



    ticksX = genTicks(len(sortedStationNames))

    plt.xticks(ticksX, sortedStationNames, rotation=90)

    plt.ylabel('Days')

    plt.title('Open days by station')

#     plt.legend(loc='upper left')

    plt.show()

    

showStationOpenDays(stations['Working Days'])
plt.xlabel('Docks')

plt.ylabel('Stations')

plt.title('Number of docks per station')

plt.hist(stations['docks_total'], bins=13);

stations.head()
trips.head()
def loadByStation(tr, station_id):

    beg = len(tr[tr['start_station_id'] == station_id])

    end = len(tr[tr['end_station_id'] == station_id]) 

    return beg, end



def showLoadTotal(tr, title, workingDays = []):    

    tripsTotalStations = []

    for i in range(len(stations)):

        startSt, endSt = loadByStation(tr, stations['station_id'].values[i])

        if len(workingDays) == 0:

            tripsTotalStations.append(startSt + endSt)

        else:

            tripsTotalStations.append((startSt + endSt) / workingDays[i])

     

    colors = stations['status'].map(colorsSt)

    colSorted = [x for (y,x) in sorted(zip(tripsTotalStations, colors))]

    sortedStationNames = [x for (y,x) in sorted(zip(tripsTotalStations, station_names))]   

    sortedTotalStations = sorted(tripsTotalStations)    

    

    x = np.linspace(0, len(sortedTotalStations), len(sortedTotalStations))

    y = sortedTotalStations

    plt.figure(figsize=(15,10))

    plt.bar(x, y, color=colSorted)

    ticksX = genTicks(len(sortedStationNames))

    plt.xticks(ticksX, sortedStationNames, rotation=90)

    plt.ylabel('Load')

    plt.title(title)

    plt.show()

    

    return tripsTotalStations



totalLoads = showLoadTotal(trips, 'Total stations load')

stations['Total load'] = totalLoads
dailyLoads = showLoadTotal(trips, 'Average Daily stations load', stations['Working Days'].values)

stations['Daily load'] = dailyLoads
def getLoadWorkingDays(tr):

    loadStart = tr[(tr.start_time.dt.weekday >=0) & (tr.start_time.dt.weekday < 5)]

    return loadStart



def getLoadNotWorkingDays(tr):   

    loadStart = tr[(tr.start_time.dt.weekday >=5) & (tr.start_time.dt.weekday <= 6)]

    return loadStart



def getLoadByMonth(tr, month):

    loadRes = tr[tr.month == month]

    return loadRes



def getLoadBySeason(tr, season):

    res = []

    if season == 1:

        res = tr[(tr.month == 1) | (tr.month == 2) | (tr.month == 12)]

    elif season == 2:

        res = tr[(tr.month == 3) | (tr.month == 4) | (tr.month == 5)]

    elif season == 3:

        res = tr[(tr.month == 6) | (tr.month == 7) | (tr.month == 8)]

    elif season == 4:

        res = tr[(tr.month == 9) | (tr.month == 10) | (tr.month == 11)]

    return res



# year 

def showLoadByYear(ax, tr, station_id=2575, allSt=False):

    if allSt:

        tr = tr

    else:

        tr = tr[(tr.start_station_id == station_id) | (tr.end_station_id == station_id)]

    

    tr_work = getLoadWorkingDays(tr)

    tr_hol = getLoadNotWorkingDays(tr)

    

    len_year_work = []

    len_year_hol = []

    

    for year in range(2013,2018):

        len_year_work.append(len(tr_work[tr_work.year == year]['start_time'].dt.date.value_counts()))

        len_year_hol.append(len(tr_hol[tr_hol.year == year]['start_time'].dt.date.value_counts()))

    

    tripsByYearInd_work = []

    tripsByYearVal_work = []

    tripsByYearInd_hol = []

    tripsByYearVal_hol = []

    

    for year in range(2013,2018):

        if allSt:

            st_work = len(tr_work[tr_work.year == year])

            end_work = len(tr_work[tr_work.year == year])

            st_hol = len(tr_hol[tr_hol.year == year])

            end_hol = len(tr_hol[tr_hol.year == year])

        else:

            st_work = len(tr_work[(tr_work.year == year) & (tr_work.start_station_id == station_id)])

            end_work = len(tr_work[(tr_work.year == year) & (tr_work.end_station_id == station_id)])

            st_hol = len(tr_hol[(tr_hol.year == year) & (tr_hol.start_station_id == station_id)])

            end_hol = len(tr_hol[(tr_hol.year == year) & (tr_hol.end_station_id == station_id)])

        tot_work = st_work + end_work

        tot_hol = st_hol + end_hol

        tripsByYearInd_work.append(year)

        tripsByYearVal_work.append(tot_work)

        tripsByYearInd_hol.append(year)

        tripsByYearVal_hol.append(tot_hol)

        

    for i in range(len(tripsByYearVal_work)):

        if len_year_work[i] != 0:

            tripsByYearVal_work[i] = tripsByYearVal_work[i] / len_year_work[i]

        else:

            tripsByYearVal_work[i] = 0

        if len_year_hol[i] != 0:

            tripsByYearVal_hol[i] = tripsByYearVal_hol[i] / len_year_hol[i]

        else:

            tripsByYearVal_hol[i] = 0

    

    sortedTripsByYearVal_work = [x for (y,x) in sorted(zip(tripsByYearInd_work, tripsByYearVal_work))]

    sortedTripsByYearInd_work= sorted(tripsByYearInd_work)

    sortedTripsByYearVal_hol = [x for (y,x) in sorted(zip(tripsByYearInd_hol, tripsByYearVal_hol))]

    sortedTripsByYearInd_hol= sorted(tripsByYearInd_hol)

    

    n_groups = 5

    index = np.arange(n_groups)

    bar_width = 0.4

#     fig, ax = plt.subplots(1,1, figsize=(16,8))

        

    ax.bar(sortedTripsByYearInd_work, sortedTripsByYearVal_work, bar_width, label = 'Work')

    ax.bar(np.array(sortedTripsByYearInd_hol) + bar_width, sortedTripsByYearVal_hol, bar_width, label='Holiday')    

    ax.set_ylabel('Daily Load')

    

    ax.set_xticks(sortedTripsByYearInd_work)

    ax.set_xticklabels(['2013', '2014', '2015', '2016', '2017'], rotation=90);

    ax.legend()

    

    if allSt:

        ax.set_title('Daily Average Load by year, All Stations')

    else:

        ax.set_title('Daily Average Load by year, Station ' + str(station_id))

        

    return ax
fig, ax = plt.subplots(1,1, figsize=(16,10))

showLoadByYear(ax, trips, allSt=True);
stations.sort_values(by='Daily load')[['station_id', 'name', 'Daily load']][-11:]
fig, ax = plt.subplots(1,1, figsize=(16,10))

showLoadByYear(ax, trips, 1006);
highestLoads = stations.sort_values(by='Daily load')['station_id'].values[-11:-1]

fig, ax = plt.subplots(5,2, figsize=(16,36), sharey=True)

num = 0

for i in range(5):

    for j in range(2):

        showLoadByYear(ax[i][j], trips, highestLoads[num])

        num += 1
# month 

def showLoadByMonth(ax, tr, station_id=2575, allSt=False):

    

    if allSt:

        tr = tr

    else:

        tr = tr[(tr.start_station_id == station_id) | (tr.end_station_id == station_id)]

    

    tr_work = getLoadWorkingDays(tr)

    tr_hol = getLoadNotWorkingDays(tr)

        

  

    len_month_work = []

    len_month_hol = []



    for month in range(1,13):

        len_month_work.append(len(tr_work[tr_work.month == month]['start_time'].dt.date.value_counts()))

        len_month_hol.append(len(tr_hol[tr_hol.month == month]['start_time'].dt.date.value_counts()))

    

    tripsByMonthInd_work = []

    tripsByMonthVal_work = []

    tripsByMonthInd_hol = []

    tripsByMonthVal_hol = []

    

    for month in range(1,13):

        if allSt:

            st_work = len(tr_work[tr_work.month == month])

            end_work = len(tr_work[tr_work.month == month])

            st_hol = len(tr_hol[tr_hol.month == month])

            end_hol = len(tr_hol[tr_hol.month == month])

        else:

            st_work = len(tr_work[(tr_work.month == month) & (tr_work.start_station_id == station_id)])

            end_work = len(tr_work[(tr_work.month == month) & (tr_work.end_station_id == station_id)])

            st_hol = len(tr_hol[(tr_hol.month == month) & (tr_hol.start_station_id == station_id)])

            end_hol = len(tr_hol[(tr_hol.month == month) & (tr_hol.end_station_id == station_id)])

        tot_work = st_work + end_work

        tot_hol = st_hol + end_hol

        tripsByMonthInd_work.append(month)

        tripsByMonthVal_work.append(tot_work)

        tripsByMonthInd_hol.append(month)

        tripsByMonthVal_hol.append(tot_hol)

    tripsByMonthVal_work = np.array(tripsByMonthVal_work) / np.array(len_month_work)

    tripsByMonthVal_hol = np.array(tripsByMonthVal_hol) / np.array(len_month_hol)



    sortedTripsByMonthVal_work = [x for (y,x) in sorted(zip(tripsByMonthInd_work, tripsByMonthVal_work))]

    sortedTripsByMonthInd_work= sorted(tripsByMonthInd_work)

    sortedTripsByMonthVal_hol = [x for (y,x) in sorted(zip(tripsByMonthInd_hol, tripsByMonthVal_hol))]

    sortedTripsByMonthInd_hol= sorted(tripsByMonthInd_hol)

    

    n_groups = 12

    index = np.arange(n_groups)

    bar_width = 0.4

#     fig, ax = plt.subplots(1,1, figsize=(16,8))

        

    ax.bar(sortedTripsByMonthInd_work, sortedTripsByMonthVal_work, bar_width, label = 'Work')

    ax.bar(np.array(sortedTripsByMonthInd_hol) + bar_width, sortedTripsByMonthVal_hol, bar_width, label='Holiday')    

    ax.set_ylabel('Daily Load')

    

    ax.set_xticks(sortedTripsByMonthInd_work)

    ax.set_xticklabels(['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',

                        'September', 'October', 'November', 'December'], rotation=90);

    ax.legend()

    

    if allSt:

        ax.set_title('Daily Average Load by month, All Stations')

    else:

        ax.set_title('Daily Average Load by month, Station ' + str(station_id))
fig, ax = plt.subplots(1,1, figsize=(16,10))

showLoadByMonth(ax, trips, allSt=True)

def getTotalHorLoad(tr):

    asStartinPoint = []

    asEndingPoint = []

    asStartinPointAv = []

    asEndingPointAv = []

    

    meansStart = []

    meansEnd = []

    stdStart = []

    stdEnd = []

    

    unStart = tr['start_time'].dt.date.value_counts().index

    totWorkingDays = unStart

    

    for hour in range(24):

        oneStart = len(tr[tr.start_time.dt.hour == hour])

        oneEnd = len(tr[tr.end_time.dt.hour == hour])

        

        oneStart_std = tr[tr.start_time.dt.hour == hour]['start_time'].dt.date.value_counts()

        oneEnd_std = tr[tr.end_time.dt.hour == hour]['end_time'].dt.date.value_counts()

        

        

        if (len(totWorkingDays) != 0):

            oneStartAv = oneStart / len(totWorkingDays)

            oneEndAv = oneEnd / len(totWorkingDays)

        else:

            oneStartAv = 0

            oneEndAv = 0

        

        a = list(set(totWorkingDays) - set(oneStart_std.index))

        s1 = pd.Series(np.zeros(len(a), int), index=a)

        oneStart_std = oneStart_std.append(s1)

        

        b = list(set(totWorkingDays) - set(oneEnd_std.index))

        s2 = pd.Series(np.zeros(len(b), int), index=b)

        oneEnd_std = oneEnd_std.append(s2)



        oneStd_start = np.std(oneStart_std)

        oneStd_end = np.std(oneEnd_std)    

        stdStart.append(oneStd_start)

        stdEnd.append(oneStd_end)

        

        asStartinPoint.append(oneStart)

        asEndingPoint.append(oneEnd)

        asStartinPointAv.append(oneStartAv)

        asEndingPointAv.append(oneEndAv)

    

    return asStartinPoint, asEndingPoint, asStartinPointAv, asEndingPointAv, len(totWorkingDays), stdStart, stdEnd



def showLoad(ax, stLoadAv, wDays, title, yerr=False):

    n_groups = 24

    index = np.arange(n_groups)

    bar_width = 0.4

    

    if yerr:

        rects1 = ax.bar(range(len(stLoadAv[0])), stLoadAv[0], bar_width, yerr=[np.zeros(len(stLoadAv[2]), int), stLoadAv[2]],

                 label='As start point: ' + str(np.round(np.sum(stLoadAv[0]),2)) + ' (per day)')



        rects2 = ax.bar(np.array(range(len(stLoadAv[1]))) + bar_width, stLoadAv[1], bar_width, yerr=[np.zeros(len(stLoadAv[3]), int), stLoadAv[3]],

                 label='As end point: ' + str(np.round(np.sum(stLoadAv[1]),2)) + ' (per day)')

    else:

        rects1 = ax.bar(range(len(stLoadAv[0])), stLoadAv[0], bar_width,

                 label='As start point: ' + str(np.round(np.sum(stLoadAv[0]),2)) + ' (per day)')



        rects2 = ax.bar(np.array(range(len(stLoadAv[1]))) + bar_width, stLoadAv[1], bar_width,

                 label='As end point: ' + str(np.round(np.sum(stLoadAv[1]),2)) + ' (per day)')

    

#     ax.set_xlabel('Hour')

    ax.set_ylabel('Daily Load')

    ax.set_title('Daily Average Load by hour (' + title +')')

    ax.set_xticks(range(24))

    ax.set_xticklabels(index)

    ax.legend()



def showTotalHourLoad(tr, st_id=-1):

    stationsHourLoad_total = []

    stationHourLoadAverage_total = []

    if st_id== -1:

        asStart, asEnd, asStartAv, asEndAv, totWorkingDays, std_start, std_end = getTotalHorLoad(tr)

    else:

        asStart, asEnd, asStartAv, asEndAv, totWorkingDays, std_start, std_end = getStationHourLoad(tr, st_id)



    stationsHourLoad_total.append([asStart, asEnd])

    stationHourLoadAverage_total.append([asStartAv, asEndAv, std_start, std_end])

    workingDays_total = totWorkingDays



    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16,10))

    if st_id == -1:

        title = ' all stations'

    else:

        title = 'Station ' + str(st_id)

    showLoad(ax, stationHourLoadAverage_total[0], workingDays_total, title, False)

    

showTotalHourLoad(trips)
# subscriber 

def showSuscriberTypes(tr, num=10):

      

    n_groups = num

    index = np.arange(n_groups)

    bar_width = 0.4

    fig, ax = plt.subplots(1, 2, figsize=(16,8), sharey=True)

    

    tr_work = getLoadWorkingDays(tr)

    tr_hol = getLoadNotWorkingDays(tr)

    work_days = len(getLoadWorkingDays(tr)['start_time'].dt.date.value_counts())

    hol_days = len(getLoadNotWorkingDays(tr)['start_time'].dt.date.value_counts())

    

    x_work = tr_work['subscriber_type'].value_counts().index

    x_work = x_work[::-1]

    x_work = x_work[-num:]

    

    y_work = tr_work['subscriber_type'].value_counts().values

    y_work = y_work[::-1]

    y_work = y_work[-num:]

    y_work = np.array(y_work) / work_days

    

    x_hol = tr_hol['subscriber_type'].value_counts().index

    x_hol = x_hol[::-1]

    x_hol = x_hol[-num:]

    

    y_hol = tr_hol['subscriber_type'].value_counts().values

    y_hol = y_hol[::-1]

    y_hol = y_hol[-num:]

    y_hol = np.array(y_hol) / hol_days

    

    ax[0].set_xticks(range(len(x_work)))

    ax[0].set_xticklabels(x_work, rotation=90);

    ax[0].bar(range(len(x_work)), y_work)

    ax[0].set_title('Daily Average load by Subscriber - Work')

    ax[0].set_ylabel('Trips')

    

    ax[1].set_xticks(range(len(x_hol)))

    ax[1].set_xticklabels(x_hol, rotation=90);

    ax[1].bar(range(len(x_hol)), y_hol, color='orange')

    ax[1].set_title('Daily Average load by Subscriber - Holiday')

    

showSuscriberTypes(trips, 6)

def calculateMinutesCat(minutes):

    categoriesMinutes = []

    for oneMinutes in minutes:

        if oneMinutes <=5:

            categoriesMinutes.append('0 - 5')

            continue

        if oneMinutes <=10:

            categoriesMinutes.append('5 - 10')

            continue

        if oneMinutes <=20:

            categoriesMinutes.append('10 - 20')

            continue

        if oneMinutes <=30:

            categoriesMinutes.append('20 - 30')

            continue

        if oneMinutes <=40:

            categoriesMinutes.append('30 - 40')

            continue

        if oneMinutes <=50:

            categoriesMinutes.append('40 - 50')

            continue

        if oneMinutes <=60:

            categoriesMinutes.append('50 - 60')

            continue

        if oneMinutes <=80:

            categoriesMinutes.append('60 - 80')

            continue

        if oneMinutes <=100:

            categoriesMinutes.append('80 - 100')

            continue

        if oneMinutes <=120:

            categoriesMinutes.append('100 - 120')

            continue

        if oneMinutes <=180:

            categoriesMinutes.append('120 - 180')

            continue

        if oneMinutes <=360:

            categoriesMinutes.append('180 - 360')

            continue

        if oneMinutes <=720:

            categoriesMinutes.append('360 - 720')

            continue

        if oneMinutes <=1440:

            categoriesMinutes.append('720 - 1440')

            continue

        categoriesMinutes.append('> 1440')

    return categoriesMinutes



trips['duration_minutes_cat'] = calculateMinutesCat(trips['duration_minutes'].values)



def findIndMinRes(indMin):

    indMinRes = []

    for i in indMin:

        one = i.split('-')[0].strip()

        if one == '> 1440':

            one = one.split('>')[1].strip()        

        indMinRes.append(int(one))

    return indMinRes





def showTripsDuration(ax, tr, title):

    tr_work = getLoadWorkingDays(tr)

    tr_hol = getLoadNotWorkingDays(tr)

    

    tr_work_len = len(getLoadWorkingDays(trips)['start_time'].dt.date.value_counts())

    tr_hol_len = len(getLoadNotWorkingDays(trips)['start_time'].dt.date.value_counts())

    

    categoriesMinutes = ['0 - 5', '5 - 10', '10 - 20', '20 - 30', '30 - 40', '40 - 50', '50 - 60',

                     '60 - 80', '80 - 100', '100 - 120', '120 - 180', '180 - 360', '360 - 720', '720 - 1440', '> 1440']

    x = categoriesMinutes

    indMin_work = tr_work['duration_minutes_cat'].value_counts().index

    valMin_work = tr_work['duration_minutes_cat'].value_counts().values

    indMinRes_work = findIndMinRes(indMin_work)

    sortedY_work = [x for (y,x) in sorted(zip(indMinRes_work, valMin_work))]

    sortedY_work = np.array(sortedY_work) / tr_work_len

    

    indMin_hol = tr_hol['duration_minutes_cat'].value_counts().index

    valMin_hol = tr_hol['duration_minutes_cat'].value_counts().values

    indMinRes_hol = findIndMinRes(indMin_hol)

    sortedY_hol = [x for (y,x) in sorted(zip(indMinRes_hol, valMin_hol))]

    sortedY_hol = np.array(sortedY_hol) / tr_hol_len

    

    n_groups = len(categoriesMinutes)

    index = np.arange(n_groups)

    bar_width = 0.4

    

    ax.bar(range(len(sortedY_work)), sortedY_work, bar_width, label = 'Work, Average duration: ' + str(np.round(np.mean(tr_work['duration_minutes']),2)))

    ax.bar(np.array(range(len(sortedY_hol))) + bar_width, sortedY_hol, bar_width, label = 'Holiday, Average duration: ' + str(np.round(np.mean(tr_hol['duration_minutes']),2)))

    ax.set_xticks(range(len(categoriesMinutes)))

    ax.set_xticklabels(categoriesMinutes, rotation=90);

    

    ax.set_xlabel('Minutes')

    ax.set_ylabel('Trips')

    ax.set_title('Daily Average Trip Duration (in minutes) - ' + title)

    ax.legend()

    return ax

fig, ax = plt.subplots(1,1,figsize=(16,10))

showTripsDuration(ax, trips[trips.subscriber_type =='Walk Up'], 'All Subscribers'); 
fig, ax = plt.subplots(3,2,figsize=(16,16))



num = 0

mostFreqSubscribers = trips['subscriber_type'].value_counts().index[:6]

for i in range(3):

    for j in range(2):

        showTripsDuration(ax[i][j], trips[trips.subscriber_type ==mostFreqSubscribers[num]], mostFreqSubscribers[num])

        num += 1

plt.tight_layout()

        
def showTripDurationBySubscriber(tr, num=10):

    n_groups = num

    index = np.arange(n_groups)

    bar_width = 0.4

    fig, ax = plt.subplots(1, 1, figsize=(12,8))

    

    tr_work = getLoadWorkingDays(tr)

    tr_hol = getLoadNotWorkingDays(tr)

    work_days = len(getLoadWorkingDays(tr)['start_time'].dt.date.value_counts())

    hol_days = len(getLoadNotWorkingDays(tr)['start_time'].dt.date.value_counts())

    

    x_work = tr_work['subscriber_type'].value_counts().index

    x_work = x_work[::-1]

    x_work = x_work[-num:]

    

    y_dur_work = []

    for oneSubs in x_work:

        one = tr_work[tr_work.subscriber_type == oneSubs]['duration_minutes'].values

        y_dur_work.append(np.mean(one))

        

    y_dur_hol = []

    for oneSubs in x_work:

        one = tr_hol[tr_hol.subscriber_type == oneSubs]['duration_minutes'].values

        y_dur_hol.append(np.mean(one))

    

    ax.set_xticks(range(len(x_work)))

    ax.set_xticklabels(x_work, rotation=90);

    ax.bar(range(len(x_work)), y_dur_work, bar_width, label='Work')

    ax.bar(np.array(range(len(x_work))) + bar_width, y_dur_hol, bar_width, label='Holiday')

    ax.set_title('Average trip duration by Subscriber')

    ax.set_ylabel('Minutes')

    ax.legend()

showTripDurationBySubscriber(trips, 6)
def showTripsByBikeId():

    x = range(len(trips['bikeid'].value_counts()))

    y = trips['bikeid'].value_counts()

    

    minutesBike = []

    for bike in y.index:

        one = np.sum(trips[trips.bikeid == bike]['duration_minutes'])

        minutesBike.append(one)

    

    fig, ax = plt.subplots(1,2, figsize=(16,8))

    ax[0].plot(x, y)

    ax[0].set_ylabel('Trips')

    ax[0].set_xlabel('BikeId')

    ax[0].set_title('Number of trips by bike')    

    ax[1].plot(x, minutesBike)

    ax[1].set_ylabel('Minutes')

    ax[1].set_xlabel('BikeId')

    ax[1].set_title('Total minutes by bike')  

showTripsByBikeId()
from math import radians, cos, sin, asin, sqrt

import urllib.request

def haversine(lon1, lat1, lon2, lat2):

    """

    Calculate the great circle distance between two points 

    on the earth (specified in decimal degrees)

    """

    # convert decimal degrees to radians 

    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 

    dlon = lon2 - lon1 

    dlat = lat2 - lat1 

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2

    c = 2 * asin(sqrt(a)) 

    km = 6367 * c

    return km



distance_stations = []

for i in range(len(stations)):

    oneSt = []

    lon1 = stations.iloc[i]['longitude']

    lat1 = stations.iloc[i]['latitude']

    for j in range(len(stations)):

        lon2 = stations.iloc[j]['longitude']

        lat2 = stations.iloc[j]['latitude']

        dist = haversine(lon1, lat1, lon2, lat2)

        oneSt.append(dist)

    distance_stations.append(oneSt)



av_distance = []

for i in range(len(distance_stations)):

    sum_one = 0

    for j in range(len(distance_stations)):

        sum_one += distance_stations[i][j]

    av_distance.append(sum_one / 71)

stations['av_distance'] = av_distance
plt.figure(figsize = (16,10))

plt.xlabel('Longitude')

plt.ylabel('Latitude')

plt.title('Average haversine distance for stations')

plt.scatter(x=stations['longitude'], y=stations['latitude'], s =stations['av_distance']*50, c=stations['av_distance'])

plt.colorbar()
# def elevation(lat, lng):

#     apikey = "AIzaSyDLwT37dkueEXk2U1wfnfPoMisFA0YBIW0" # you need to provide your own API key here

#     url = "https://maps.googleapis.com/maps/api/elevation/json"

#     request = urllib.request.urlopen(url+"?locations="+str(lat)+","+str(lng)+"&key="+apikey)

#     results = json.load(request).get('results')

#     if 0 < len(results):

#         elevation = results[0].get('elevation')           

#     return elevation



# elevations = []

# for i in range(len(stations)):

#     elev = elevation(stations.iloc[i]['latitude'], stations.iloc[i]['longitude'])

#     elevations.append(elev)

#     print(i)

    

# stations['elevation'] = elevations
stations = stations.sort_values(by='name')

elevations = [166.0672760009766, 159.7005615234375, 165.21728515625, 144.1920013427734, 142.1951446533203,

 146.9313659667969, 144.1340789794922, 143.1091766357422, 150.1666564941406, 149.4506988525391, 152.5222625732422,

 163.6567993164062, 157.7004699707031, 153.0662231445312, 137.1873779296875, 139.5315399169922, 138.5940093994141,

 136.789306640625, 149.4154052734375, 160.9741516113281, 139.0222778320312, 157.8820037841797, 143.2070007324219,

 141.8021545410156, 142.6670837402344, 142.8450622558594, 142.3974456787109, 159.2362213134766, 164.0555114746094,

 139.8280487060547, 145.1390228271484, 143.8266448974609, 144.6851959228516, 140.6478271484375, 172.4871215820312,

 151.3396453857422, 143.3318939208984, 154.7284851074219, 135.8721466064453, 150.1439056396484, 137.0628509521484,

 145.014404296875, 144.6419525146484, 162.0620574951172, 136.9876556396484, 149.5563659667969, 137.22412109375,

 144.1656951904297, 137.2400054931641, 137.1984710693359, 147.4045562744141, 178.4207458496094, 144.5030364990234,

 144.363037109375, 147.4530334472656, 137.4808959960938, 155.9356689453125, 151.7225952148438, 138.4862060546875,

 164.6349182128906, 152.5728302001953, 165.1692199707031, 164.1784057617188, 156.5611114501953, 141.0511169433594,

 139.8759765625, 148.5473175048828, 176.1220703125, 149.1334533691406, 141.8068084716797, 143.5822296142578,

 143.4732971191406]

stations['elevation'] = elevations

stations = stations.sort_index()
plt.hist(stations['elevation'])

plt.xlabel('Elevation')

plt.xlabel('Count')

plt.title('Stations elevation')
trips['elevation_start'] = '-'

trips['elevation_end'] = '-'

ids = stations['station_id']

for oneId in ids:

    trips.loc[trips.start_station_id == oneId, 'elevation_start'] = stations[stations.station_id == oneId]['elevation'].values[0]

    trips.loc[trips.end_station_id == oneId, 'elevation_end'] = stations[stations.station_id == oneId]['elevation'].values[0]



def showElevationBySuscriber(tr, num=10):

      

    n_groups = num

    index = np.arange(n_groups)

    bar_width = 0.4

    fig, ax = plt.subplots(1, 2, figsize=(16,8), sharey=False)

    

    tr_work = getLoadWorkingDays(tr)

    tr_hol = getLoadNotWorkingDays(tr)

    work_days = len(getLoadWorkingDays(tr)['start_time'].dt.date.value_counts())

    hol_days = len(getLoadNotWorkingDays(tr)['start_time'].dt.date.value_counts())

    

    x_work = tr_work['subscriber_type'].value_counts().index

    x_work = x_work[::-1]

    x_work = x_work[-num:]

    

    y_elev_work_start = [] 

    y_elev_work_end = []

    for oneSubs in x_work:

        one = tr_work[tr_work.subscriber_type == oneSubs]['elevation_start'].values

        two = tr_work[tr_work.subscriber_type == oneSubs]['elevation_end'].values

        y_elev_work_start.append(np.sum(one) / len(one))

        y_elev_work_end.append(np.sum(two) / len(two))

            

    y_elev_hol_start = [] 

    y_elev_hol_end = []

    for oneSubs in x_work:

        one = tr_hol[tr_hol.subscriber_type == oneSubs]['elevation_start'].values

        two = tr_hol[tr_hol.subscriber_type == oneSubs]['elevation_end'].values

        y_elev_hol_start.append(np.sum(one) / len(one))

        y_elev_hol_end.append(np.sum(two) / len(two))

    



    ax[0].set_xticks(range(len(x_work)))

    ax[0].set_xticklabels(x_work, rotation=90);

    ax[0].bar(range(len(x_work)), y_elev_work_start, bar_width, label='Start point')

    ax[0].bar(np.array(range(len(x_work))) + bar_width, y_elev_work_end, bar_width, label = 'End point')

    ax[0].set_title('Average Elevation by Subscriber - Work')

    ax[0].set_ylabel('Elevation')

    ax[0].set_ylim([140, 150])

    ax[0].legend()

    

    ax[1].set_xticks(range(len(x_work)))

    ax[1].set_xticklabels(x_work, rotation=90);

    ax[1].bar(range(len(x_work)), y_elev_hol_start, bar_width, label='Start point')

    ax[1].bar(np.array(range(len(x_work))) + bar_width, y_elev_hol_end, bar_width, label='End point')

    ax[1].set_title('Average Elevation by Subscriber - Holiday')

    ax[1].set_ylim([140, 150])

    ax[1].legend()



showElevationBySuscriber(trips, 6)