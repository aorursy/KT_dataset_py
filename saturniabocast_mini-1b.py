# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Loading the dataset
df = pd.read_csv("/kaggle/input/d14s1-irr-csv/D14S1_irr1.csv")
df
#I read this might be useful due to missing data but I don't think it actually did anything
df.interpolate()
#Kaggle seemed mad about the decimal places so I tried taking them out and working with whole integers. This worked.

tt = (949990)
if tt in range(940000, 960000):
    print ("Match")
#Next I tried to use the value from the data instead one entered value. This failed.
time = df['Begin Time - msec']
if time in range(940000, 960000):
    print ("Match")
#I tried a different format. Fail.
time = df['Begin Time - msec']
if time >= 940000 and time <= 960000:
    print ("Match")
#Tried just using one number instead of a range. Fail.
time = df['Begin Time - msec']
if time >= 940000:
    print ("Match")
#It really wants me to use a.something, so I tried that. 
time = df['Begin Time - msec']
if a.any(time) >= 940000:
    print ("Match")
#Having a hard time understanding this but trying something else found googling. This did not result in an error, so that's progress.
time = np.array(df['Begin Time - msec'])
if np.any(time>940000):
    print(time)
#now let's see if it will work with a range
time = np.array(df['Begin Time - msec'])
if np.any(time>940000) and np.any(time<960000):
    print(time)
#ok it responds to a range. Now to get it to work with another column)
time = np.array(df['Begin Time - msec'])
time2 = np.array(df['Begin Time - msec.1'])
if np.any(time>time2-10000) and np.any(time<time2+10000):
    print(time)
#I'll reduce the range and test the impact
time = np.array(df['Begin Time - msec'])
time2 = np.array(df['Begin Time - msec.1'])
if np.any(time>time2-100) and np.any(time<time2+100):
    print(time)
#I'm not actually interested in the numbers anyway so what will this do?
time = np.array(df['Begin Time - msec'])
time2 = np.array(df['Begin Time - msec.1'])
if np.any(time>time2-100) and np.any(time<time2+100):
    print("Match")
#I'll try all instead of any.
time = np.array(df['Begin Time - msec'])
time2 = np.array(df['Begin Time - msec.1'])
if np.all(time>time2-100) and np.all(time<time2+100):
    print("Match")
#trying again
np.seterr(invalid='ignore')
time = np.array(df['Begin Time - msec'])
time2 = np.array(df['Begin Time - msec.1'])
if np.all(time>time2-1000) and np.all(time<time2+1000):
    print("Match")
#trying again
np.seterr(invalid='ignore')
time = np.array(df['Begin Time - msec'])
time2 = np.array(df['Begin Time - msec.1'])
if np.any(time>time2-100) and np.any(time<time2+100):
    print("Match")
#Another idea, using this formula, mabye it will simplify things?
#((c - a) * (b - c) >= 0)

np.seterr(invalid='ignore')
time = np.array(df['Begin Time - msec'])
time2 = np.array(df['Begin Time - msec.1'])
if np.any((time - (time2-1000)) * ((time2+1000) - time) >= 0):
    print("Match")

#maybe eachItem is the solution

time = np.array(df['Begin Time - msec'])
time2 = np.array(df['Begin Time - msec.1'])
# Given range
X = time2-1000
Y = time2+1000

def checkRange(num):
   # using comaparision operator
   if X <= num <= Y:
       print('Match')
   else:
      print('No match')

for eachItem in time:
   checkRange(eachItem)
timelist = np.array([949990,951175,955400,956935,957715,959790,975260])
timelist2 = np.array([950000,951035,955405,956825,957595,959720,975160])
# Given range
X = timelist-1000
Y = timelist+1000

def checkRange(num):
   # using comaparision operator
   if X <= num <= Y:
       print('Match')
   else:
      print('No match')

for eachItem in timelist2:
   checkRange(eachItem)

#trying something simpler, going back a few steps.

timelist = np.array([949990,951175,955400,956935,957715,959790,975260])
# Given range
X = 940000
Y = 960000

def checkRange(num):
   # using comaparision operator
   if X <= num <= Y:
       print('Match')
   else:
      print('No match')

for eachItem in timelist:
   checkRange(eachItem)
#First I'll check if this will even work pulling directly from the data
timelist = np.array(df['Begin Time - msec'])
# Given range
X = 940000
Y = 960000

def checkRange(num):
   # using comaparision operator
   if X <= num <= Y:
       print('Match')
   else:
      print('No match')

for eachItem in timelist:
   checkRange(eachItem)
df['match'] = np.where((df['Begin Time - msec'] >= df['Begin Time - msec.1']-10) & (df['Begin Time - msec'] <= df['Begin Time - msec.1']+10)
                     , df['match'], np.nan)
df
#This is too much data right now, I'll reduce it
df2 = df.drop(['Duration - msec', 'End Time - msec', 'EG_P1_Divya', 'EG_P1_Rachel', 'End Time - msec.1','Duration - msec.1','EG_P1_Robbie','End Time - msec.2','Duration - msec.2','Patrick_G1','End Time - msec.3','Duration - msec.3'], axis=1)
df2
#trying something else
df2 = df2.drop(['match'], axis=1)
df2
#now to do this again but with the actual tolerance that was used

df2['match2'] = np.where((df2['Begin Time - msec'] >= df2['Begin Time - msec.1']-250) & (df2['Begin Time - msec'] <= df2['Begin Time - msec.1']+250)
                     , 'match', np.nan)
df2
#Going further...

#0 vs 1 (Comparing the first and second coders)
df2['close match'] = np.where((df2['Begin Time - msec'] >= df2['Begin Time - msec.1']-100) & (df2['Begin Time - msec'] <= df2['Begin Time - msec.1']+100)
                     , 'match', np.nan)
df2
#I've compared two columns, but that's not enough. I need to know how all four compare. 
#I already know that most of the numbers are within the 250msec tolerance, my students did a good job! But that means looking at that wont be very interesting.
#So I'll focus on closer agreements.

#0 vs 2

df2['second match'] = np.where((df2['Begin Time - msec'] >= df2['Begin Time - msec.2']-100) & (df2['Begin Time - msec'] <= df2['Begin Time - msec.2']+100)
                     , 'match', np.nan)
df2

#and again 0 vs 3
df2['third match'] = np.where((df2['Begin Time - msec'] >= df2['Begin Time - msec.3']-100) & (df2['Begin Time - msec'] <= df2['Begin Time - msec.3']+100)
                     , 'match', np.nan)
df2
#and again (1 vs 2)
df2['forth match'] = np.where((df2['Begin Time - msec.1'] >= df2['Begin Time - msec.2']-100) & (df2['Begin Time - msec.1'] <= df2['Begin Time - msec.2']+100)
                     , 'match', np.nan)
df2
#and again (1 vs 3)
df2['fifth match'] = np.where((df2['Begin Time - msec.1'] >= df2['Begin Time - msec.3']-100) & (df2['Begin Time - msec.1'] <= df2['Begin Time - msec.3']+100)
                     , 'match', np.nan)
df2
#one more (2 vs 3)
df2['sixth match'] = np.where((df2['Begin Time - msec.2'] >= df2['Begin Time - msec.3']-100) & (df2['Begin Time - msec.2'] <= df2['Begin Time - msec.3']+100)
                     , 'match', np.nan)
df2
#oops wrote six instead of sixth, fixing that
df2 = df2.drop(['six match'], axis=1)
df2
df2['total match'] = np.where((df2['Begin Time - msec'] >= df2['Begin Time - msec.1']-100) & (df2['Begin Time - msec'] <= df2['Begin Time - msec.1']+100) & (df2['Begin Time - msec'] >= df2['Begin Time - msec.2']-100) & (df2['Begin Time - msec'] <= df2['Begin Time - msec.2']+100) & (df2['Begin Time - msec'] >= df2['Begin Time - msec.3']-100) & (df2['Begin Time - msec'] <= df2['Begin Time - msec.3']+100) & (df2['Begin Time - msec.1'] >= df2['Begin Time - msec.2']-100) & (df2['Begin Time - msec.1'] <= df2['Begin Time - msec.2']+100) & (df2['Begin Time - msec.1'] >= df2['Begin Time - msec.3']-100) & (df2['Begin Time - msec.1'] <= df2['Begin Time - msec.3']+100) & (df2['Begin Time - msec.2'] >= df2['Begin Time - msec.3']-100) & (df2['Begin Time - msec.2'] <= df2['Begin Time - msec.3']+100) 
                     , 'match', np.nan)
df2
#forgot one
df2 = df2.drop(['match2'], axis=1)
df2
#So how many are there?
df2.groupby('total match').count()
#So what about the rest of the data?

df2['total match end'] = np.where((df['End Time - msec'] >= df['End Time - msec.1']-100) & (df['End Time - msec'] <= df['End Time - msec.1']+100) & (df['End Time - msec'] >= df['End Time - msec.2']-100) & (df['End Time - msec'] <= df['End Time - msec.2']+100) & (df['End Time - msec'] >= df['End Time - msec.3']-100) & (df['End Time - msec'] <= df['End Time - msec.3']+100) & (df['End Time - msec.1'] >= df['End Time - msec.2']-100) & (df['End Time - msec.1'] <= df['End Time - msec.2']+100) & (df['End Time - msec.1'] >= df['End Time - msec.3']-100) & (df['End Time - msec.1'] <= df['End Time - msec.3']+100) & (df['End Time - msec.2'] >= df['End Time - msec.3']-100) & (df['End Time - msec.2'] <= df['End Time - msec.3']+100) 
                     , 'match', np.nan)
df2
df2.groupby('total match end').count()
#One more

df2['Duration match'] = np.where((df['Duration - msec'] >= df['Duration - msec.1']-100) & (df['Duration - msec'] <= df['Duration - msec.1']+100) & (df['Duration - msec'] >= df['Duration - msec.2']-100) & (df['Duration - msec'] <= df['Duration - msec.2']+100) & (df['Duration - msec'] >= df['Duration - msec.3']-100) & (df['Duration - msec'] <= df['Duration - msec.3']+100) & (df['Duration - msec.1'] >= df['Duration - msec.2']-100) & (df['Duration - msec.1'] <= df['Duration - msec.2']+100) & (df['Duration - msec.1'] >= df['Duration - msec.3']-100) & (df['Duration - msec.1'] <= df['Duration - msec.3']+100) & (df['Duration - msec.2'] >= df['Duration - msec.3']-100) & (df['Duration - msec.2'] <= df['Duration - msec.3']+100) 
                     , 'match', np.nan)
df2
df2.groupby('Duration match').count()
df2['total match 250'] = np.where((df2['Begin Time - msec'] >= df2['Begin Time - msec.1']-250) & (df2['Begin Time - msec'] <= df2['Begin Time - msec.1']+250) & (df2['Begin Time - msec'] >= df2['Begin Time - msec.2']-250) & (df2['Begin Time - msec'] <= df2['Begin Time - msec.2']+250) & (df2['Begin Time - msec'] >= df2['Begin Time - msec.3']-250) & (df2['Begin Time - msec'] <= df2['Begin Time - msec.3']+250) & (df2['Begin Time - msec.1'] >= df2['Begin Time - msec.2']-250) & (df2['Begin Time - msec.1'] <= df2['Begin Time - msec.2']+250) & (df2['Begin Time - msec.1'] >= df2['Begin Time - msec.3']-250) & (df2['Begin Time - msec.1'] <= df2['Begin Time - msec.3']+250) & (df2['Begin Time - msec.2'] >= df2['Begin Time - msec.3']-250) & (df2['Begin Time - msec.2'] <= df2['Begin Time - msec.3']+250) 
                     , 'match', np.nan)
df2['total match end 250'] = np.where((df['End Time - msec'] >= df['End Time - msec.1']-250) & (df['End Time - msec'] <= df['End Time - msec.1']+250) & (df['End Time - msec'] >= df['End Time - msec.2']-250) & (df['End Time - msec'] <= df['End Time - msec.2']+250) & (df['End Time - msec'] >= df['End Time - msec.3']-250) & (df['End Time - msec'] <= df['End Time - msec.3']+250) & (df['End Time - msec.1'] >= df['End Time - msec.2']-250) & (df['End Time - msec.1'] <= df['End Time - msec.2']+250) & (df['End Time - msec.1'] >= df['End Time - msec.3']-250) & (df['End Time - msec.1'] <= df['End Time - msec.3']+250) & (df['End Time - msec.2'] >= df['End Time - msec.3']-250) & (df['End Time - msec.2'] <= df['End Time - msec.3']+250) 
                     , 'match', np.nan)
df2['Duration match 250'] = np.where((df['Duration - msec'] >= df['Duration - msec.1']-250) & (df['Duration - msec'] <= df['Duration - msec.1']+250) & (df['Duration - msec'] >= df['Duration - msec.2']-250) & (df['Duration - msec'] <= df['Duration - msec.2']+250) & (df['Duration - msec'] >= df['Duration - msec.3']-250) & (df['Duration - msec'] <= df['Duration - msec.3']+250) & (df['Duration - msec.1'] >= df['Duration - msec.2']-250) & (df['Duration - msec.1'] <= df['Duration - msec.2']+250) & (df['Duration - msec.1'] >= df['Duration - msec.3']-250) & (df['Duration - msec.1'] <= df['Duration - msec.3']+250) & (df['Duration - msec.2'] >= df['Duration - msec.3']-250) & (df['Duration - msec.2'] <= df['Duration - msec.3']+250) 
                     , 'match', np.nan)
df2

#Need to find a better way to get the counts than the groupby.count function
count = df2.groupby(['total match','total match end']).size() 
print(count) 
df2['total match'].value_counts()
df2['total match end'].value_counts()
# make a table of the new data
data = [['Start Time 250', 78,17], ['Start Time 100', 5,90], ['End Time 250', 74,21],['End Time 100',5,90],['Duration 250',70,25],['Duration 100',32,63]]

dftotals = pd.DataFrame(data, columns = ['Measurement/Tolerance in msec', 'Code Times Within Tolerance','Code Times Outside Tolerance'])

dftotals
#Now to graph it!
import matplotlib.pyplot as plt
dftotals[['Code Times Within Tolerance','Code Times Outside Tolerance']].plot(kind='barh', stacked=True, legend=True)
dftotals.set_index("Measurement/Tolerance in msec",inplace=True)
dftotals.plot.barh(stacked=True, figsize=(10, 6)).legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title('Eye Gaze Time Congruency Among 4 Coders')
plt.xlabel('Counts of 4-way Congruence within Tolerance')
plt.show()
#First I need to bring back the original data
df
#And reduce
df3 = df.drop(['Begin Time - msec','match','Begin Time - msec.1','Begin Time - msec.2','Begin Time - msec.3','Duration - msec', 'End Time - msec', 'End Time - msec.1','Duration - msec.1','End Time - msec.2','Duration - msec.2','End Time - msec.3','Duration - msec.3'], axis=1)
df3
#Comparing two is simple
df3['EG_P1_Divya'] == df3['EG_P1_Rachel']
#But adding the others to that doesn't work, so I'll need to find another way.
df3['EG_P1_Divya'] == df3['EG_P1_Rachel'] == df3['EG_P1_Robbie'] == df3['Patrick_G1']
#Found a code that should work for this
i = df3.nunique(axis = 1).eq(1)
i
#And now to add them
i.value_counts()
#Now I have the totals for all 4 coders. But pairwise data is useful too, so I'll go back and look at that.
p1 = df3['EG_P1_Divya'] == df3['EG_P1_Rachel'] 
p2 = df3['EG_P1_Divya'] == df3['EG_P1_Robbie']
p3 = df3['EG_P1_Divya'] == df3['Patrick_G1']
p4 = df3['EG_P1_Rachel'] == df3['EG_P1_Robbie']
p5 = df3['EG_P1_Rachel'] == df3['Patrick_G1']
p6 = df3['EG_P1_Robbie'] == df3['Patrick_G1']
print(p1.value_counts())
print(p2.value_counts())
print(p3.value_counts())
print(p4.value_counts())
print(p5.value_counts())
print(p6.value_counts())
#Now to make a new dataframe with those numbers. At this point I feel like just building it manually will be the fastest method so I'll do that.

data = [['Coder 1 & Coder 2', 88,7], ['Coder 1 & Coder 3', 86,9], ['Coder 1 & Coder 4', 74,21],['Coder 2 & Coder 3',90,5],['Coder 2 & Coder 4',74,21],['Coder 3 & Coder 4',71,24], ['Total',79,16]]

dfpw = pd.DataFrame(data, columns = ['Coder Pair', 'Same Codes','Different Codes'])

dfpw
#I'll sort them to see who were most in agreement.

pwsort = dfpw.sort_values('Same Codes',ascending=False)
pwsort
pwsort.set_index("Coder Pair",inplace=True)
pwsort.plot.barh(stacked=True, figsize=(10, 6)).legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title('Pairwise Agreement Among 4 Coders')
plt.xlabel('Agreement')
plt.show()
codes = pd.read_csv("/kaggle/input/d14s1-codes/D14S1_codes.csv")
codes
g = codes.nunique(axis = 1).eq(1)
g
g.value_counts()
a1 = codes['C1'] == codes['C2']
a2 = codes['C1'] == codes['C3']
a3 = codes['C1'] == codes['C4']
a4 = codes['C2'] == codes['C3']
a5 = codes['C2'] == codes['C4']
a6 = codes['C3'] == codes['C4']

print(a1.value_counts())
print(a2.value_counts())
print(a3.value_counts())
print(a4.value_counts())
print(a5.value_counts())
print(a6.value_counts())
data = [['Coder 1 & Coder 2', 93,2], ['Coder 1 & Coder 3', 92,3], ['Coder 1 & Coder 4', 88,7],['Coder 2 & Coder 3',92,3],['Coder 2 & Coder 4',90,5],['Coder 3 & Coder 4',88,7], ['Total',87,8]]

dfpw2 = pd.DataFrame(data, columns = ['Coder Pair', 'Same Codes','Different Codes'])
pwsort2 = dfpw2.sort_values('Same Codes',ascending=False)
dfpw2
#Graphing the new data
pwsort2.set_index("Coder Pair",inplace=True)
pwsort2.plot.barh(stacked=True, figsize=(10, 6)).legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title('Pairwise Agreement Among 4 Coders')
plt.xlabel('Agreement')
plt.show()