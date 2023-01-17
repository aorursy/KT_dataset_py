import pandas as pd

import warnings

warnings.filterwarnings('ignore')
AccidentsDF = pd.read_csv('../input/Acc.csv')
# Full dataframe

AccidentsDF
# Number of rows

len(AccidentsDF)
# See Columns' names

AccidentsDF.columns
# Number of Columns

len(AccidentsDF.columns)
# Top rows

AccidentsDF.head()
# Bottom rows

AccidentsDF.tail()
# Find the null values from the dataset and get the column in which each null value exists

AccidentsDF_Null = AccidentsDF.isnull().sum()

AccidentsDF_Null[AccidentsDF_Null > 0].sort_values(ascending=False)
# Information on the columns

AccidentsDF.info()
# Get statistics on the columns

AccidentsDF.describe()
AccidentsDF.describe().transpose()
AccidentsDF.columns
AccidentsDF.columns = ['AccidentIndex', 'LocationEastingOSGR', 'LocationNorthingOSGR',

       'Longitude', 'Latitude', 'PoliceForce', 'AccidentSeverity',

       'NumberOfVehicles', 'NumberOfCasualties', 'Date', 'DayOfWeek',

       'Time', 'LocalAuthorityDistrict', 'LocalAuthorityHighway',

       '1stRoadClass', '1stRoadNumber', 'RoadType', 'SpeedLimit',

       'JunctionDetail', 'JunctionControl', '2ndRoadClass',

       '2ndRoadNumber', 'PedestrianCrossingHumanControl',

       'PedestrianCrossingPhysicalFacilities', 'LightConditions',

       'WeatherConditions', 'RoadSurfaceConditions',

       'SpecialConditionsAtSite', 'CarriagewayHazards',

       'UrbanOrRuralArea', 'PoliceOfficerAtScene',

       'AccidentLocationLSOA']
AccidentsDF.head()
AccDF = AccidentsDF[['AccidentIndex', 'PoliceForce', 'AccidentSeverity', 'NumberOfVehicles',

                   'Date', 'DayOfWeek', 'RoadType', 'SpeedLimit',

                   'JunctionDetail', 'LightConditions', 'WeatherConditions', 'RoadSurfaceConditions',

                   'SpecialConditionsAtSite', 'CarriagewayHazards', 'UrbanOrRuralArea']]
AccDF.columns
len(AccDF.columns)
AccDF.head()
AccDF['PoliceForce'] = AccDF['PoliceForce'].astype(str)



AccDF['PoliceForce'] = AccDF['PoliceForce'].replace({

        '1' : 'Metropolitan Police','3' : 'Cumbria','4' : 'Lancashire',

        '5' : 'Merseyside','6' : 'Greater Manchester','7' : 'Cheshire',

        '10' : 'Northumbria','11' : 'Durham','12' : 'North Yorkshire',

        '13' : 'West Yorkshire','14' : 'South Yorkshire','16' : 'Humberside',

        '17' : 'Cleveland','20' : 'West Midlands','21' : 'Staffordshire',

        '22' : 'West Mercia','23' : 'Warwickshire','30' : 'Derbyshire',

        '31' : 'Nottinghamshire','32' : 'Lincolnshire','33' : 'Leicestershire',

        '34' : 'Northamptonshire','35' : 'Cambridgeshire','36' : 'Norfolk',

        '37' : 'Suffolk','40' : 'Bedfordshire','41' : 'Hertfordshire',

        '42' : 'Essex','43' : 'ThamesValley','44' : 'Hampshire',

        '45' : 'Surrey','46' : 'Kent','47' : 'Sussex',

        '48' : 'City Of London','50' : 'Devon And Cornwall','52' : 'Avon And Somerset',

        '53' : 'Gloucestershire','54' : 'Wiltshire','55' : 'Dorset',

        '60' : 'North Wales','61' : 'Gwent','62' : 'South Wales','63' : 'Dyfed Powys',

        '91' : 'Northern','92' : 'Grampian','93' : 'Tayside','94' : 'Fife',

        '95' : 'Lothian And Borders','96' : 'Central','97' : 'Strathclyde',

        '98' : 'Dumfries And Galloway' })
AccDF.head()
AccDF['AccidentSeverity'] = AccDF['AccidentSeverity'].astype(str)



AccDF['AccidentSeverity'] = AccDF['AccidentSeverity'].replace({'1' : 'Fatal',

                                                               '2' : 'Serious',

                                                               '3' : 'Slight'})
AccDF.head()
AccDF['DayOfWeek'] = AccDF['DayOfWeek'].astype(str)



AccDF['DayOfWeek'] = AccDF['DayOfWeek'].replace({'1' : 'Sunday','2' : 'Monday', '3' : 'Tuesday',

                                                 '4' : 'Wednesday', '5' : 'Thursday', '6' : 'Friday',

                                                 '7' : 'Saturday'})                                                               
AccDF.head()
AccDF['RoadType'] = AccDF['RoadType'].astype(str)



AccDF['RoadType'] = AccDF['RoadType'].replace({

        '1' : 'Roundabout','2' : 'One Way', '3' : 'Dual Carriageway',

        '6' : 'Single Carriageway', '7' : 'Slip Road', '9' : 'Unknown',

        '12' : 'One Way / Slip Road', '-1' : 'Data Missing'})
AccDF.head()
AccDF['JunctionDetail'] = AccDF['JunctionDetail'].astype(str)



AccDF['JunctionDetail'] = AccDF['JunctionDetail'].replace({

        '0' : 'Not Junction Within 20 Meters',

        '1' : 'Roundabout','2' : 'Mini Roundabout', '3' :'T Junction',

        '5' : 'Slip Road', '6' : 'Croosroads', '7' : 'More than 4 Arms',

        '8' : 'Private Drive / Entrance', '9' : 'Other Junction', '-1' : 'Data Missing' })
AccDF.head()
AccDF['LightConditions'] = AccDF['LightConditions'].astype(str)



AccDF['LightConditions'] = AccDF['LightConditions'].replace({

        '1' : 'Daylight','4' : 'Darkness Lights Lit', '5' : 'Darkness Lights Unlit',

        '6' : 'Darkness No Lighting', '7' : 'Darkness Lighting Unknown','-1' : 'Data Missing'})
AccDF.head()
AccDF['WeatherConditions'] = AccDF['WeatherConditions'].astype(str)



AccDF['WeatherConditions'] = AccDF['WeatherConditions'].replace({

        '1' : 'Fine No Winds','2' : 'Raining No Winds', '3' : 'Snowing No Winds',

        '4' : 'Fine With Winds', '5' : 'Raining With Winds','6' : 'Snowing With Winds',

        '7' : 'Fog or Mist', '8' : 'Other', '9' : 'Unknown', '-1' : 'Data Missing' })
AccDF.head()
AccDF['RoadSurfaceConditions'] = AccDF['RoadSurfaceConditions'].astype(str)



AccDF['RoadSurfaceConditions'] = AccDF['RoadSurfaceConditions'].replace({

        '1' : 'Dry','2' : 'Wet or Damp', '3' : 'Snow',

        '4' : 'Frost or Ice', '5' : 'Flood Over 3cm','6' : 'Oil or Diesel',

        '7' : 'Mud', '-1' : 'Data Missing' })
AccDF.head()
AccDF['SpecialConditionsAtSite'] = AccDF['SpecialConditionsAtSite'].astype(str)



AccDF['SpecialConditionsAtSite'] = AccDF['SpecialConditionsAtSite'].replace({ '0' : 'None',

        '1' : 'Auto Traffic Signal Out','2' : 'Auto Traffic Signal Defective', '3' : 'Road Sign',

        '4' : 'Roadworks', '5' : 'Road Surface Defective','6' : 'Oil or Diesel',

        '7' : 'Mud', '-1' : 'Data Missing' })
AccDF.head()
AccDF['CarriagewayHazards'] = AccDF['CarriagewayHazards'].astype(str)



AccDF['CarriagewayHazards'] = AccDF['CarriagewayHazards'].replace({ '0' : 'None',

        '1' : 'Vehicle Load On Road','2' : 'Other Object On Road', '3' : 'Previous Accident',

        '4' : 'Dog On Road', '5' : 'Other Animal On Road','6' : 'Pedestrian In Carriageway',

        '7' : 'Animal In Carriageway', '-1' : 'Data Missing' })
AccDF.head()
AccDF['UrbanOrRuralArea'] = AccDF['UrbanOrRuralArea'].astype(str)



AccDF['UrbanOrRuralArea'] = AccDF['UrbanOrRuralArea'].replace({ 

        '1' : 'Urban','2' : 'Rural', '3' : 'Unallocated' })
AccDF.head()
AccDF.info()
AccDF.AccidentSeverity = AccDF.AccidentSeverity.astype('category')

AccDF.CarriagewayHazards = AccDF.CarriagewayHazards.astype('category')

AccDF.DayOfWeek = AccDF.DayOfWeek.astype('category')

AccDF.JunctionDetail = AccDF.JunctionDetail.astype('category')

AccDF.LightConditions = AccDF.LightConditions.astype('category')

AccDF.PoliceForce = AccDF.PoliceForce.astype('category')

AccDF.RoadSurfaceConditions = AccDF.RoadSurfaceConditions.astype('category')

AccDF.RoadType = AccDF.RoadType.astype('category')

AccDF.SpecialConditionsAtSite = AccDF.SpecialConditionsAtSite.astype('category')

AccDF.WeatherConditions = AccDF.WeatherConditions.astype('category')

AccDF.UrbanOrRuralArea = AccDF.UrbanOrRuralArea.astype('category')
AccDF.info()
AccDF.describe().transpose()
FatAccSat = AccDF[(AccDF.DayOfWeek == 'Saturday') & (AccDF.AccidentSeverity == 'Fatal')]
FatNumAccSat = float(len(FatAccSat))

print(FatNumAccSat)
NumAcc = float(len(AccDF))

print(NumAcc)
Rate = (FatNumAccSat/NumAcc)*100

print(Rate)
GrMan = AccDF[AccDF.PoliceForce == 'Greater Manchester']

float(len(GrMan)) #Number of accidents in Greater Manchester
SnowAcc = AccDF[(AccDF.WeatherConditions == 'Snowing No Winds') | 

                          (AccDF.WeatherConditions == 'Snowing With Winds')]

float(len(SnowAcc)) #Number of accidents when it was snowing
GrManSnowAcc = AccDF[(AccDF.PoliceForce == 'Greater Manchester') &

                     ((AccDF.WeatherConditions == 'Snowing No Winds') | 

                     (AccDF.WeatherConditions == 'Snowing With Winds'))]

float(len(GrManSnowAcc)) #Number of accidents in Greater Manchester when it was snowing
UrbAcc = AccDF[AccDF.UrbanOrRuralArea == 'Urban']

float(len(UrbAcc)) #Number of accidents in Urban area
HighSpeedAcc = AccDF[AccDF.SpeedLimit > 30]

float(len(HighSpeedAcc)) #Number of accidents with speed higher than 30
UrbHighSpeedAcc = AccDF[(AccDF.UrbanOrRuralArea == 'Urban') & (AccDF.SpeedLimit > 30)]

float(len(UrbHighSpeedAcc)) #Number of accidents in Urban area with speed higher than 30
Percentage = (float(len(UrbHighSpeedAcc))/float(len(UrbAcc)))*100

print(Percentage)
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

plt.rcParams['figure.figsize'] = 8,4

import warnings

warnings.filterwarnings('ignore')
AccDF.head()
plot1 = sns.distplot(AccDF.SpeedLimit, bins = 17)
plot2 = sns.boxplot(data = AccDF, x = 'AccidentSeverity', y = 'SpeedLimit')
plt.hist(AccDF[AccDF.AccidentSeverity == 'Fatal'].SpeedLimit, label = 'Fatal')

plt.hist(AccDF[AccDF.AccidentSeverity == 'Slight'].SpeedLimit, label = 'Slight')

plt.hist(AccDF[AccDF.AccidentSeverity == 'Serious'].SpeedLimit, label = 'Serious')

plt.legend()

plt.show()
plot3 = sns.lmplot(data = AccDF, x = 'SpeedLimit', y = 'NumberOfVehicles', \

                   fit_reg=False, hue = 'DayOfWeek')
plot4 = sns.countplot(data = AccDF, x = 'DayOfWeek', hue = 'AccidentSeverity')
plot5 = sns.countplot(data = AccDF, y = 'AccidentSeverity', hue = 'RoadSurfaceConditions')