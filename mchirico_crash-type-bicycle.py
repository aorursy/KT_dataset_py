import IPython

url = 'https://www.kaggle.io/svf/473002/358c9d1e02508cb5ce321bcef7e32cd8/output.html'

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









FILE="../input/accident.csv"

d=pd.read_csv(FILE)



FILE="../input/pbtype.csv"

b=pd.read_csv(FILE)



FILE="../input/person.csv"

person=pd.read_csv(FILE)



FILE="../input/extra/nmcrash.csv"

nmcrash=pd.read_csv(FILE)

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





db = pd.merge(d, b, how='right',left_on='ST_CASE', right_on='ST_CASE')

db[db['PBPTYPE']==6][['ST_CASE','crashTime','PER_NO','LATITUDE','LONGITUD','FATALS','DRUNK_DR']].head()
# 06 Bicyclist

per = person[person['PER_TYP']==6][['ST_CASE','PER_NO','STR_VEH','DEATH_TM']]

per.head()
k = pd.merge(per, db, how='left',left_on=['ST_CASE','PER_NO'], right_on=['ST_CASE','PER_NO'])
k.head()


# 06 Bicyclist

person[(person['PER_TYP']==6) & (person['DEATH_TM'] != 8888)].head()
# 

# 05 Pedestrian

# 06 Bicyclist

bic_killed  = person[(person['PER_TYP']==6) & (person['DEATH_TM'] != 8888)].count()[0]

bic_injured = person[(person['PER_TYP']==6) & (person['DEATH_TM'] == 8888)].count()[0]

print("Bicyclist killed: ",bic_killed," injured",bic_injured)
# 

# 05 Pedestrian

# 06 Bicyclist

ped_killed  = person[(person['PER_TYP']==5) & (person['DEATH_TM'] != 8888)].count()[0]

ped_injured = person[(person['PER_TYP']==5) & (person['DEATH_TM'] == 8888)].count()[0]

print("Pedestrian killed: ",ped_killed," injured",ped_injured)


k2=pd.merge(k, nmcrash, how='left',left_on=['ST_CASE','PER_NO'], right_on=['ST_CASE','PER_NO'])
action={0:"None Noted",

1:"Dart-Out",

2:"Failure to Yield Right-Of-Way",

3:"Failure to Obey Traffic Signs, Signals or Officer",

4:"In Roadway Improperly (Standing, Lying, Working, Playing)",

5:"Entering/Exiting Parked or Stopped Vehicle",

6:"Inattentive (Talking, Eating, etc.)",

7:"Improper Turn/Merge",

8:"Improper Passing",

9:"Wrong-Way Riding or Walking",

10:"Riding on Wrong Side of Road",

11:"Dash",

12:"Improper Crossing of Roadway or Intersection (Jaywalking)",

13:"Failing to Have Lights on When Required",

14:"Operating Without Required Equipment",

15:"Improper or Erratic Lane Changing",

16:"Failure to Keep in Proper Lane or Running Off Road",

17:"Making Improper Entry to or Exit from Trafficway",

18:"Operating in Other Erratic, Reckless, Careless or Negligent Manner",

19:"Not Visible (Dark Clothing, No Lighting, etc.)",

20:"Passing with Insufficient Distance or Inadequate Visibility or Failing to Yield to Overtaking Vehicle",

21:"Other",

99:"Unknown"}
k2['mtm_crsh']=k2['MTM_CRSH'].apply(lambda x: action[x])
# Clean this up...pie graph?

k2['mtm_crsh'].value_counts()