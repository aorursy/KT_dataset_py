import IPython

url = 'https://www.kaggle.io/svf/468124/561b1aec1e15fe16c581e792d16c9466/output.html'

iframe = '<iframe src=' + url + ' width=700 height=525></iframe>'

IPython.display.HTML(iframe)
import pandas as pd

import numpy as np

import datetime





import warnings

warnings.filterwarnings("ignore")

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="white", color_codes=True)





# Good for interactive plots

import plotly.plotly as py

import plotly.graph_objs as go

from plotly import tools

from plotly.offline import iplot, init_notebook_mode

init_notebook_mode()









# Read data accident.csv



FILE="../input/accident.csv"

d=pd.read_csv(FILE)



# Reading in distract.csv

FILE = "../input/distract.csv"

dd = pd.read_csv(FILE, encoding = "ISO-8859-1")



# Reading in violations

FILE = "../input/extra/violatn.csv"

v = pd.read_csv(FILE, encoding = "ISO-8859-1")



# Reading in sequence of events

FILE = "../input/extra/vsoe.csv"

vsoe = pd.read_csv(FILE, encoding = "ISO-8859-1")



FILE = "../input/maneuver.csv"

m = pd.read_csv(FILE, encoding = "ISO-8859-1")



FILE = "../input/person.csv"

p = pd.read_csv(FILE, encoding = "ISO-8859-1")



FILE = "../input/vehicle.csv"

vehicle = pd.read_csv(FILE, encoding = "ISO-8859-1")



FILE = "../input/vindecode.csv"

vindecode = pd.read_csv(FILE, encoding = "ISO-8859-1")



FILE = "../input/vision.csv"

vision = pd.read_csv(FILE, encoding = "ISO-8859-1")



FILE = "../input/damage.csv"

damage = pd.read_csv(FILE, encoding = "ISO-8859-1")
def f(x):

    year = x[0]

    month = x[1]

    day = x[2]

    hour = x[3]

    minute = x[4]

    # Sometimes they don't know hour and minute

    if hour == 99:

        hour = 0

    if minute == 99:

        minute = 0

    s = "%02d-%02d-%02d %02d:%02d:00" % (year,month,day,hour,minute)

    c = datetime.datetime.strptime(s,'%Y-%m-%d %H:%M:%S')

    return c

 

d['crashTime']   = d[['YEAR','MONTH','DAY','HOUR','MINUTE']].apply(f, axis=1)

d['crashDay']    = d['crashTime'].apply(lambda x: x.date())

d['crashMonth']  = d['crashTime'].apply(lambda x: x.strftime("%B") )

d['crashMonthN'] = d['crashTime'].apply(lambda x: x.strftime("%d") ) # sorting

d['crashTime'].head()
weather = {0: 'No Additional Atmospheric Conditions', 1: 'Clear', 

           2: 'Rain', 3: 'Sleet, Hail', 

           4: 'Snow', 5: 'Fog, Smog, Smoke', 6: 'Severe Crosswinds', 

           7: 'Blowing Sand, Soil, Dirt', 

           8: 'Other', 10: 'Cloudy', 11: 'Blowing Snow', 

           12: 'Freezing Rain or Drizzle', 

           98: 'Not Reported', 99: 'Unknown'}



d['weather']=d['WEATHER'].apply(lambda x: weather[x])

d['weather1']=d['WEATHER1'].apply(lambda x: weather[x])

d['weather2']=d['WEATHER2'].apply(lambda x: weather[x])
distract = {0: 'Not Distracted', 1: 'Looked But Did Not See',

           3: 'By Other Occupant(s)', 4: 'By a Moving Object in Vehicle',

           5: 'While Talking or Listening to Cellular Phone',

           6: 'While Manipulating Cellular Phone',

           7: 'While Adjusting Audio or Climate Controls',

           9: 'While Using Other Component/Controls Integral to Vehicle',

           10: 'While Using or Reaching For Device/Object Brought Into Vehicle',

           12: 'Distracted by Outside Person, Object or Event',

           13: 'Eating or Drinking',

           14: 'Smoking Related',

           15: 'Other Cellular Phone Related',

           16: 'No Driver Present/Unknown if Driver Present',

           17: 'Distraction/Inattention',

           18: 'Distraction/Careless',

           19: 'Careless/Inattentive',

           92: 'Distraction (Distracted), Details Unknown',

           93: 'Inattention (Inattentive), Details Unknown',

           96: 'Not Reported',

           97: 'Lost In Thought/Day Dreaming',

           98: 'Other Distraction',

           99: 'Unknown if Distracted'}



dd['mdrdstrd'] = dd['MDRDSTRD'].apply(lambda x: distract[x])
soe = {1:"Rollover/Overturn",

2:"Fire/Explosion",

3:"Immersion or Partial Immersion",

4:"Gas Inhalation",

5:"Fell/Jumped from Vehicle",

6:"Injured in Vehicle (Non-Collision)",

7:"Other Non-Collision",

8:"Pedestrian",

9:"Pedalcyclist",

10:"Railway Vehicle",

11:"Live Animal",

12:"Motor Vehicle in Transport",

14:"Parked Motor Vehicle",

15:"Non-Motorist on Personal Conveyance",

16:"Thrown or Falling Object",

17:"Boulder",

18:"Other Object (Not Fixed)",

19:"Building",

20:"Impact Attenuator/Crash Cushion",

21:"Bridge Pier or Support",

23:"Bridge Rail (Includes Parapet)",

24:"Guardrail Face",

25:"Concrete Traffic Barrier",

26:"Other Traffic Barrier",

30:"Utility Pole/Light Support",

31:"Other Post",

32:"Culvert",

33:"Curb",

34:"Ditch",

35:"Embankment",

38:"Fence",

39:"Wall",

40:"Fire Hydrant",

41:"Shrubbery",

42:"Tree (Standing Only)",

43:"Other Fixed Object",

44:"Pavement Surface Irregularity (Ruts Potholes Grates etc.)",

45:"Working Motor Vehicle",

46:"Traffic Signal Support",

48:"Snow Bank",

49:"Ridden Animal or Animal-Drawn Conveyance",

50:"Bridge Overhead Structure",

51:"Jackknife (Harmful to This Vehicle)",

52:"Guardrail End",

53:"Mail Box",

54:"Motor Vehicle In-Transport Strikes or is Struck by Cargo Persons or Objects Set-in-Motion from/by Another Motor Vehicle In-Transport",

55:"Motor Vehicle in Motion Outside the Trafficway",

57:"Cable Barrier",

58:"Ground",

59:"Traffic Sign Support",

60:"Cargo/Equipment Loss or Shift (Non-Harmful)",

61:"Equipment Failure (Blown Tire",

62:"Separation of Units",

63:"Ran Off Road - Right",

64:"Ran Off Road - Left",

65:"Cross Median",

66:"Downhill Runaway",

67:"Vehicle Went Airborne",

68:"Cross Centerline",

69:"Re-Entering Highway",

70:"Jackknife (Non-Harmful)",

71:"End Departure",

72:"Cargo/Equipment Loss or Shift (Harmful To This Vehicle)",

73:"Object Fell From Motor Vehicle In-Transport",

79:"Ran Off Roadway - Direction Unknown",

99:"Unknown",}
vsoe['soe'] = vsoe['SOE'].apply(lambda x: soe[x])
maneuver = {0:"Driver Did Not Maneuver To Avoid",

            1:"Object",

            2:"Poor Road Conditions (Puddle, Ice, Pothole, etc.)",

            3:"Live Animal",

            4:"Motor Vehicle",

            5:"Pedestrian, Pedalcyclist or Other Non-Motorist",

            92:"Phantom/Non-Contact Motor Vehicle",

            95:"No Driver Present/Unknown if Driver Present",

            98:"Not Reported",

            99:"Unknown"}
m['mdrmanav'] = m['MDRMANAV'].apply(lambda x: maneuver[x])
vis = {0:"No Obstruction Noted",

       1:"Rain, Snow, Fog, Smoke, Sand, Dust",

       2:"Reflected Glare, Bright Sunlight, Headlights",

       3:"Curve, Hill, or Other Roadway Design Features",

       4:"Building, Billboard, or Other Structure",

       5:"Trees, Crops, Vegetation",

       6:"In-Transport Motor Vehicle (Including Load)",

       7:"Not-in-Transport Motor Vehicle (Parked, Working)",

       8:"Splash or Spray of Passing Vehicle",

       9:"Inadequate Defrost or Defog System",

       10:"Inadequate Vehicle Lighting System",

       11:"Obstructing Interior to the Vehicle",

       12:"External Mirrors",

       13:"Broken or Improperly Cleaned Windshield",

       14:"Obstructing Angles on Vehicle",

       95:"No Driver Present/Unknown if Driver Present",

       97:"Vision Obscured – No Details",

       98:"Other Visual Obstruction",

       99:"Unknown"}
vision['mvisobsc'] = vision['MVISOBSC'].apply(lambda x: vis[x])
ST_CASE = 420044



d = d[d['ST_CASE']==ST_CASE]

v = v[v['ST_CASE']==ST_CASE]

vsoe = vsoe[vsoe['ST_CASE']==ST_CASE]

m = m[m['ST_CASE']==ST_CASE]

dd = dd[dd['ST_CASE']==ST_CASE]

p = p[p['ST_CASE']==ST_CASE]

vehicle = vehicle[vehicle['ST_CASE']==ST_CASE]

vindecode = vindecode[vindecode['ST_CASE']==ST_CASE]

vision = vision[vision['ST_CASE']==ST_CASE]

damage = damage[damage['ST_CASE']==ST_CASE]
# What happened

vsoe
# Unknown whether any driver tried to maneuver

m
d[['ST_CASE','crashTime','VE_TOTAL','PERSONS','FATALS','ARR_HOUR','ARR_MIN','HOSP_MN']]


# 59 Traffic Sign Support

d[['HARM_EV']]
# 04 On Roadside

d[['REL_ROAD']]
d[['LGT_COND']]
notification_hr = d[['NOT_HOUR']].values[0][0]

notification_mn = d[['NOT_MIN']].values[0][0]



arrive_hr = d[['ARR_HOUR']].values[0][0]

arrive_mn = d[['ARR_MIN']].values[0][0]



hospital_hr = d[['HOSP_HR']].values[0][0]

hospital_mn = d[['HOSP_MN']].values[0][0]



print("Time for EMS to get to the scene:", arrive_mn - notification_mn," minutes")

print("Time from notification to hospital:",hospital_mn - notification_mn," minutes")

vindecode[['VEH_NO','NCICMAKE','VINMODEL_T','DRIVETYP_T']]
# Some basic stats

dd
# No violations

v
vehicle[['VEH_NO','VIN','DR_ZIP']]
p[['VEH_NO','PER_NO','AGE','SEX','SEAT_POS','DEATH_TM','EXTRICAT']]
vision
# Damage - these are clock points (12 is head on)

# If you look at the photo, Winstar is head on 12, and Mustang is 12 and 3

damage