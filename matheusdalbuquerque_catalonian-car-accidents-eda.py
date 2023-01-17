import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sb

dt = pd.read_csv('../input/road-traffic-injuries-deaths-catalonia-201020/Accidents_de_tr_nsit_amb_morts_o_ferits_greus_a_Catalunya.csv')
dt.head(5)
dt.columns = ['Year','TypeArea','Date','TypeRoad','KilometerPoint','City','County','Demarcation','Deaths','SeriousInjuries',
              'MinorInjuries','Victims','VehiclesInvolved','PedestriansInvolved','BicyclesInvolved','MopedsInvolved','MotocyclesInvolved',
             'LightVehiclesInvolved','HeavyVehiclesInvolved','OtherUnitsInvolved','UnknownUnitsInvolved','SpeedLimit','Leak','FogPresence','TerrainType','SpecialLane',
             'SpecialTraffic','Climantology','TrackSpecial','AccidentSeverity','FogInfluenced','TerrainInfluenced','TrafficInfluenced',
             'WeatherInfluenced','WindInfluenced','LuminosityInfluenced','SpecialTrafficInfluenced','RoadObjectInfluenced','FurrowsDitchesInfluenced',
             'VisibilityInfluenced','Intersection','TrackSpeedLimitDisplay','LightConditions','RoadRegulation','TrackTrajectories',
             'AccidentClassification','TypeSection','AreaOccorrued','RoadState','RoadType','RoadOwnership','AltimetricLayout','WindClassification',
             'WorkingDay','Hour','TimeSlot','TypeAccident','DayOfWeek']
dt.info()
plt.rcParams["figure.figsize"] = (20,10)
(dt.groupby(['Year','TypeRoad','County'])['Victims']
    .sum()
    .nlargest(30,keep='last')
    .sort_values()
    .plot.barh(grid=True))

plt.title('Most Victims per Year, TypeRoad and County')
plt.xlabel('Victims')
plt.rcParams["figure.figsize"] = (20,10)
(dt.groupby(['Year','TypeRoad','County'])['Deaths']
    .sum()
    .nlargest(30,keep='last')
    .sort_values()
    .plot.barh(grid=True))

plt.title('Most Deaths per Year, TypeRoad and County')
plt.xlabel('Deaths')
plt.rcParams["figure.figsize"] = (20,10)
(dt.groupby(['Year','TypeRoad','County'])[['SeriousInjuries','MinorInjuries']].sum()
     .nlargest(30,columns=['SeriousInjuries','MinorInjuries'],keep='last')
     .sort_values(by=['SeriousInjuries','MinorInjuries'])
     .plot.barh(grid=True))

plt.title('Most Injuries Type per Year, TypeRoad and County')
plt.xlabel('Injuries')
plt.xticks(ticks=np.arange(0, 350, step=25))
plt.rcParams["figure.figsize"] = (15,10)
dt.groupby('Year')[['BicyclesInvolved', 'MopedsInvolved','MotocyclesInvolved',
    'LightVehiclesInvolved', 'HeavyVehiclesInvolved', 
    'OtherUnitsInvolved', 'UnknownUnitsInvolved']].sum().plot.line(grid=True)
plt.title('Number of Vechiles involved per Year')
plt.ylabel('Vechiles')
plt.yticks(ticks=np.arange(0, 2000, step=150))
plt.rcParams["figure.figsize"] = (20,10)
dt['SpeedLimit'] = dt['SpeedLimit'].apply(lambda x: np.nan if x in [999.,99.,0.] else x) # there are some values with 999, 99 and 0 SpeedLimit

dt1 = (dt.groupby(['SpeedLimit','TrackSpeedLimitDisplay'])[['BicyclesInvolved', 'MopedsInvolved','MotocyclesInvolved',
    'LightVehiclesInvolved', 'HeavyVehiclesInvolved', 
    'OtherUnitsInvolved', 'UnknownUnitsInvolved']]
     .sum().dropna())
gr = sb.heatmap(dt1,annot=True,fmt='d')
gr.set_yticklabels(gr.get_yticklabels(), rotation = 0, fontsize = 10)
gr.set_xticklabels(gr.get_xticklabels(), rotation = 0, fontsize = 8) # is there another way of doing this?
plt.rcParams["figure.figsize"] = (10,5)
g2 = (dt[['FogInfluenced','TerrainInfluenced','TrafficInfluenced','WeatherInfluenced','WindInfluenced','LuminosityInfluenced',
                     'SpecialTrafficInfluenced','RoadObjectInfluenced','FurrowsDitchesInfluenced' ,'VisibilityInfluenced']]
 .isin(['Si'])
 .sum(axis=0)
 .apply(lambda x : x / len(dt))
 .sort_values()
.plot.barh())
xticks = g2.get_xticks()
g2.set_xticklabels(['{:,.2%}'.format(x) for x in xticks])
plt.title('Percentage of influenced conditions on accidents')
plt.rcParams["figure.figsize"] = (20,5)
sb.countplot(x='TypeArea',hue='TypeAccident',data=dt)
plt.title('Most common type of accident by area')
plt.xlabel('Types of Areas')
sb.countplot(x='RoadOwnership',hue='TimeSlot',data=dt)
plt.rcParams["figure.figsize"] = (40,5)
g4 = sb.FacetGrid(data=dt,col='TimeSlot',row='TypeArea',height=5.,aspect=.85)
g4.map(plt.hist,'Victims')