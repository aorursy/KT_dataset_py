# Listing the files

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



print("../input/extra")

print(check_output(["ls", "../input/extra"]).decode("utf8"))
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





# You might want to get started with accident.csv 



# Read data accident.csv



FILE="../input/accident.csv"

d=pd.read_csv(FILE)


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
d.head()
d.count()[0]
# Take a look at breakdown by PERSONS (Motorists in the crash - don't assume killed)

d["PERSONS"].value_counts()
# Total

d["PERSONS"].sum()
# Broken down by FATALS

d["FATALS"].value_counts()
# Total

d["FATALS"].sum()
# Where are the 3 or more FATALS per incident?

#  Reference:

#    https://www.kaggle.com/mchirico/d/nhtsa/2015-traffic-fatalities/fatalities-3-or-more

import IPython

url = 'https://www.kaggle.io/svf/474808/8194665eeb7f21d92f65ffa5f17c285f/output.html'

iframe = '<iframe src=' + url + ' width=700 height=525></iframe>'

IPython.display.HTML(iframe)


# School Bus Fatalities 



import IPython

url = 'https://www.kaggle.io/svf/474975/0021b5e39cead137f450588c873eae28/output.html'

iframe = '<iframe src=' + url + ' width=700 height=525></iframe>'

IPython.display.HTML(iframe)
# Bicycle Fatalities  

import IPython

url = 'https://www.kaggle.io/svf/473865/4ba7ff04c62cb2b89155f486e0393ae7/output.html'

iframe = '<iframe src=' + url + ' width=700 height=525></iframe>'

IPython.display.HTML(iframe)
states = {1: 'Alabama', 2: 'Alaska', 4: 'Arizona', 5: 'Arkansas', 

          6: 'California', 8: 'Colorado', 9: 'Connecticut', 10: 'Delaware', 

          11: 'District of Columbia', 12: 'Florida', 13: 'Georgia', 15: 'Hawaii', 

          16: 'Idaho', 17: 'Illinois', 18: 'Indiana', 19: 'Iowa', 20: 'Kansas', 

          21: 'Kentucky', 22: 'Louisiana', 23: 'Maine', 24: 'Maryland', 

          25: 'Massachusetts', 26: 'Michigan', 27: 'Minnesota', 

          28: 'Mississippi', 29: 'Missouri', 30: 'Montana', 31: 'Nebraska', 

          32: 'Nevada', 33: 'New Hampshire', 34: 'New Jersey', 35: 'New Mexico', 

          36: 'New York', 37: 'North Carolina', 38: 'North Dakota', 39: 'Ohio', 

          40: 'Oklahoma', 41: 'Oregon', 42: 'Pennsylvania', 43: 'Puerto Rico', 

          44: 'Rhode Island', 45: 'South Carolina', 46: 'South Dakota', 47: 'Tennessee', 

          48: 'Texas', 49: 'Utah', 50: 'Vermont', 51: 'Virginia', 52: 'Virgin Islands', 

          53: 'Washington', 54: 'West Virginia', 55: 'Wisconsin', 56: 'Wyoming'}



d['state']=d['STATE'].apply(lambda x: states[x])


# Incident by state

d['state'].value_counts().to_frame()



# You can check these values against

#   https://crashstats.nhtsa.dot.gov/Api/Public/ViewPublication/812318

d.groupby(['state']).agg({'FATALS':sum})
# Deadlist Week by State?

# Deadlist Month by State?

p = pd.pivot_table(d, values='FATALS', 

                   index=['crashTime'], columns=['state'], aggfunc=np.sum)



# Sample by W Week

#pp=p.resample('W', how=[np.sum]).reset_index()



# Sample by M Month

pp=p.resample('M', how=[np.sum]).reset_index()

pp.sort_values(by='crashTime',ascending=False,inplace=True)



# Let's flatten the columns 

pp.columns = pp.columns.get_level_values(0)



# Show values

# Note, last week might not be a full week

pp
# Pick a particular state

pp[['crashTime','Pennsylvania']].sort_values(by=['Pennsylvania'],

                                             ascending=False,inplace=False).head(15)
import IPython

url = 'https://www.kaggle.io/svf/490446/e51617de25c0084fdc51496ce476d947/output.html'

iframe = '<iframe src=' + url + ' width=700 height=525></iframe>'

IPython.display.HTML(iframe)
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
d[['WEATHER','WEATHER1','WEATHER2']].head()
d[['weather','weather1','weather2']].head()
# Interesting.  Clear weather is the worst

d['weather'].value_counts()
drunk = d[d.DRUNK_DR == 1]

n_drunk = d[d.DRUNK_DR == 0]



# Careful, maintain order

drunk_dict = drunk['weather'].value_counts().to_dict()

not_drunk_dict = n_drunk['weather'].value_counts().to_dict()





fig = {

  "data": [

    {

      "values": list(drunk_dict.values()),

      "labels": list(drunk_dict.keys()),

      "domain": {"x": [0, .48]},

      "name": "Drunk",

      "hoverinfo":"label+percent+name",

      "hole": .48,

      "type": "pie"

    },     

    {

      "values": list(not_drunk_dict.values()),

      "labels": list(not_drunk_dict.keys()),

      "text":"Not Drunk",

      "textposition":"inside",

      "domain": {"x": [.52, 1]},

      "name": "Not Drunk",

      "hoverinfo":"label+percent+name",

      "hole": .48,

      "type": "pie"

    }],

  "layout": {

        "title":"Weather and Drunk Driving",

        "annotations": [

            {

                "font": {

                    "size": 12

                },

                "showarrow": False,

                "text": "Drunk",

                "x": 0.20,

                "y": 0.5

            },

            {

                "font": {

                    "size": 12

                },

                "showarrow": False,

                "text": "  Not Drunk",

                "x": 0.8,

                "y": 0.5

            }

        ]

    }

}

iplot(fig)
d['VE_TOTAL'].value_counts()
# Reading in the data

FILE = "../input/distract.csv"

dd = pd.read_csv(FILE, encoding = "ISO-8859-1")
dd.head()
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
dd['mdrdstrd'].value_counts()
# Careful, maintain order

table = dd['mdrdstrd'].value_counts().to_dict()







fig = {

  "data": [

    {

      "values": list(table.values()),

      "labels": list(table.keys()),

      "domain": {"x": [0, .58]},

      "name": "Distracted",

      "hoverinfo":"label+percent+name",

      "hole": .48,

      "type": "pie"

    }],

  "layout": {

        "title":"Distracted",

        "annotations": [

            {

                "font": {

                    "size": 20

                },

                "showarrow": False,

                "text": "",

                "x": 0.80,

                "y": 0.5

            }

        ]

    }

}

iplot(fig)
# Let's combine cell phones

distract2 = {0: 'Not Distracted', 1: 'Looked But Did Not See',

           3: 'By Other Occupant(s)', 4: 'By a Moving Object in Vehicle',

           5: 'Cell Phone',

           6: 'Cell Phone',

           7: 'While Adjusting Audio or Climate Controls',

           9: 'While Using Other Component/Controls Integral to Vehicle',

           10: 'While Using or Reaching For Device/Object Brought Into Vehicle',

           12: 'Distracted by Outside Person, Object or Event',

           13: 'Eating or Drinking',

           14: 'Smoking Related',

           15: 'Cell Phone',

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





dd['mdrdstrd2'] = dd['MDRDSTRD'].apply(lambda x: distract2[x])
dd['mdrdstrd2'].value_counts()
# Careful, maintain order

table = dd['mdrdstrd2'].value_counts().to_dict()







fig = {

  "data": [

    {

      "values": list(table.values()),

      "labels": list(table.keys()),

      "domain": {"x": [0, .58]},

      "name": "Distracted",

      "hoverinfo":"label+percent+name",

      "hole": .48,

      "type": "pie"

    }],

  "layout": {

        "title":"Distracted - Cell Phone Groups Combined",

        "annotations": [

            {

                "font": {

                    "size": 20

                },

                "showarrow": False,

                "text": "",

                "x": 0.80,

                "y": 0.5

            }

        ]

    }

}

iplot(fig)
p = pd.pivot_table(dd, values='VEH_NO', 

                   index=['ST_CASE'], columns=['mdrdstrd2'], aggfunc=lambda x: len(x.unique()))
p.fillna(0, inplace=True)

# Let's flatten the columns 

p.columns = p.columns.get_level_values(0)



p.head()
# Interesting... so few "Cell Phone"

p['Cell Phone'].value_counts()
print("Total Crashes with at least one known Cell Phone distraction:",430+9+3)

print("Percent:", (430+9+3.0)/(31724+430+9+3)  * 100)
# Let's check this.  

# Here's the question we're asking: Is there a crash where distraction from 

# Cell Phone is listed for 3 vehicles involved?

p[p['Cell Phone']==3]
# Here we're going back to check our data 



dd[dd['ST_CASE'] == 470227]
dd[dd['VEH_NO']==1]['mdrdstrd2'].value_counts()
table = dd[dd['VEH_NO']==1]['mdrdstrd2'].value_counts().to_dict()







fig = {

  "data": [

    {

      "values": list(table.values()),

      "labels": list(table.keys()),

      "domain": {"x": [0, .58]},

      "name": "Distracted",

      "hoverinfo":"label+percent+name",

      "hole": .48,

      "type": "pie"

    }],

  "layout": {

        "title":"Distracted VEH_NO 1 - Cell Phone Groups Combined",

        "annotations": [

            {

                "font": {

                    "size": 20

                },

                "showarrow": False,

                "text": "",

                "x": 0.80,

                "y": 0.5

            }

        ]

    }

}

iplot(fig)
# Reading in the data

FILE = "../input/extra/violatn.csv"

v = pd.read_csv(FILE, encoding = "ISO-8859-1")
v.head()
v['MVIOLATN'].value_counts().head(8)
# Reading in the data

FILE = "../input/extra/vsoe.csv"

vsoe = pd.read_csv(FILE, encoding = "ISO-8859-1")
vsoe.head()
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
vsoe[(vsoe['ST_CASE'] == 10001) & (vsoe['VEH_NO'] == 1)].sort_values(by='VEVENTNUM',ascending=True)
d[d['ST_CASE']==10001][['PERSONS','FATALS','DRUNK_DR','crashTime','state','weather']].head()
# Information on this person



# Reading in the data

FILE = "../input/person.csv"

person = pd.read_csv(FILE, encoding = "ISO-8859-1")

person[person['ST_CASE']==10001][['AGE','AIR_BAG','DRUGS','DRUG_DET',

                                  'DOA','RACE']]
# Reading in the data

FILE = "../input/vision.csv"

# watch name collusions use vision_df not vision

vision_df = pd.read_csv(FILE, encoding = "ISO-8859-1")
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
vision_df['mvisobsc'] = vision_df['MVISOBSC'].apply(lambda x: vis[x])
vision_df['mvisobsc'].value_counts()
# 

FILE = "../input/vindecode.csv"

v = pd.read_csv(FILE, encoding = "ISO-8859-1")



FILE = "../input/maneuver.csv"

m = pd.read_csv(FILE, encoding = "ISO-8859-1")



dv = pd.merge(d, v, how='left',left_on='ST_CASE', right_on='ST_CASE')

dm = pd.merge(d, v, how='left',left_on='ST_CASE', right_on='ST_CASE')
v.head()
dv[['state','VINMAKE_T','VINMODEL_T']].head()
m.head()
# Note, multiple Attribute Codes can be applied to a single ST_CASE

m['MDRMANAV'].value_counts().head()