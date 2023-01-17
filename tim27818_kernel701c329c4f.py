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
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

import datetime as dt

#%matplotlib notebook

pd.options.display.max_rows=5000

pd.options.display.max_columns=50

import os

from IPython.display import display

%matplotlib inline

import statsmodels.api as sm
# open the first sheet of the excel file -- this file has homeless data in it, specifically about the number of beds, not the number of homeless!

excel_sheet='2008'

dftmp = pd.read_excel('/kaggle/input/homeless-in-america-20102020/2007-2019-HIC-Counts-by-State.xlsx',index_col=0,header=[0,1],sheet_name=excel_sheet) # multindex on cols, 1st col = rows
# show what's in the file - the file is from HUD, and the explanations for all the columns are at the HUD site (see readme.txt)

dftmp.head(2)
# this is too much data, so only work with the first column

# s2008 will be a series, the values are the total # of beds for each state, the index is the two-letter abbreviations for each state.

s2008=dftmp[('Total Beds (ES,TH,SH)','Total Year-Round Beds (ES,TH,SH)')] # this is the 1st column, now it's a series # multindex on cols, 1st col = rows

s2008.name='2008'
# now do the same for the rest of the sheets:

excel_sheet='2009'

dftmp = pd.read_excel('/kaggle/input/homeless-in-america-20102020/2007-2019-HIC-Counts-by-State.xlsx',index_col=0,header=[0,1],sheet_name=excel_sheet) 

s2009=dftmp[('Total Beds (ES,TH,SH)','Total Year-Round Beds (ES,TH,SH)')]

s2009.name='2009'



excel_sheet='2010'

dftmp = pd.read_excel('/kaggle/input/homeless-in-america-20102020/2007-2019-HIC-Counts-by-State.xlsx',index_col=0,header=[0,1],sheet_name=excel_sheet) 

s2010=dftmp[('Total Beds (ES,TH,SH)','Total Year-Round Beds (ES,TH,SH)')]

s2010.name='2010'



excel_sheet='2011'

dftmp = pd.read_excel('/kaggle/input/homeless-in-america-20102020/2007-2019-HIC-Counts-by-State.xlsx',index_col=0,header=[0,1],sheet_name=excel_sheet) 

s2011=dftmp[('Total Beds (ES,TH,SH)','Total Year-Round Beds (ES,TH,SH)')]

s2011.name='2011'



excel_sheet='2012'

dftmp = pd.read_excel('/kaggle/input/homeless-in-america-20102020/2007-2019-HIC-Counts-by-State.xlsx',index_col=0,header=[0,1],sheet_name=excel_sheet) 

s2012=dftmp[('Total Beds (ES,TH,SH)','Total Year-Round Beds (ES,TH,SH)')]

s2012.name='2012'



# sheet 2013 was a bit different in the header rows, so this set of statements is a bit different:

excel_sheet='2013'

dftmp = pd.read_excel('/kaggle/input/homeless-in-america-20102020/2007-2019-HIC-Counts-by-State.xlsx',index_col=0,skiprows=[0],sheet_name=excel_sheet) 

dftmp['tot']=dftmp['Total Year-Round Beds (ES,TH,RRH,SH)']-dftmp['Total Year-Round RRH Beds']

s2013=dftmp['tot']

s2013.name='2013'



# sheet 2014 and subsequent sheets had extra spaces in names, so these sets of statements are a bit different:

excel_sheet='2014'

dftmp = pd.read_excel('/kaggle/input/homeless-in-america-20102020/2007-2019-HIC-Counts-by-State.xlsx',index_col=0,header=[0,1],sheet_name=excel_sheet) 

s2014=dftmp[('Total Beds (ES, TH, SH)','Total Year-Round Beds (ES, TH, SH)')]

s2014.name='2014'



excel_sheet='2015'

dftmp = pd.read_excel('/kaggle/input/homeless-in-america-20102020/2007-2019-HIC-Counts-by-State.xlsx',index_col=0,header=[0,1],sheet_name=excel_sheet) 

s2015=dftmp[('Total Beds (ES, TH, SH)','Total Year-Round Beds (ES, TH, SH)')]

s2015.name='2015'



excel_sheet='2016'

dftmp = pd.read_excel('/kaggle/input/homeless-in-america-20102020/2007-2019-HIC-Counts-by-State.xlsx',index_col=0,header=[0,1],sheet_name=excel_sheet) 

s2016=dftmp[('Total Beds (ES, TH, SH)','Total Year-Round Beds (ES, TH, SH)')]

s2016.name='2016'



excel_sheet='2017'

dftmp = pd.read_excel('/kaggle/input/homeless-in-america-20102020/2007-2019-HIC-Counts-by-State.xlsx',index_col=0,header=[0,1],sheet_name=excel_sheet) 

s2017=dftmp[('Total Beds (ES, TH, SH)','Total Year-Round Beds (ES, TH, SH)')]

s2017.name='2017'



excel_sheet='2018'

dftmp = pd.read_excel('/kaggle/input/homeless-in-america-20102020/2007-2019-HIC-Counts-by-State.xlsx',index_col=0,header=[0,1],sheet_name=excel_sheet) 

s2018=dftmp[('Total Beds (ES, TH, SH)','Total Year-Round Beds (ES, TH, SH)')]

s2018.name='2018'



excel_sheet='2019'

dftmp = pd.read_excel('/kaggle/input/homeless-in-america-20102020/2007-2019-HIC-Counts-by-State.xlsx',index_col=0,header=[0,1],sheet_name=excel_sheet) 

s2019=dftmp[('Total Beds (ES, TH, SH)','Total Year-Round Beds (ES, TH, SH)')]

s2019.name='2019'
# combine all of them into a single dataframe, years are column headings:

dftmp = pd.merge(s2008,s2009,left_index=True,right_index=True)

dftmp = pd.merge(dftmp,s2010,left_index=True,right_index=True)

dftmp = pd.merge(dftmp,s2011,left_index=True,right_index=True)

dftmp = pd.merge(dftmp,s2012,left_index=True,right_index=True)

dftmp = pd.merge(dftmp,s2013,left_index=True,right_index=True)

dftmp = pd.merge(dftmp,s2014,left_index=True,right_index=True)

dftmp = pd.merge(dftmp,s2015,left_index=True,right_index=True)

dftmp = pd.merge(dftmp,s2016,left_index=True,right_index=True)

dftmp = pd.merge(dftmp,s2017,left_index=True,right_index=True)

dftmp = pd.merge(dftmp,s2018,left_index=True,right_index=True)

dftmp = pd.merge(dftmp,s2019,left_index=True,right_index=True)
# drop anything that's not recognizable as a state

state_list = ['AK','AL','AR','AZ','CA','CO','CT','DE','FL','GA',

             'HI','IA','ID','IL','IN','KS','KY','LA','MA','MD',

             'ME','MI','MN','MO','MS','MT','NC','ND','NE','NH',

             'NJ','NM','NV','NY','OH','OK','OR','PA','RI','SC',

             'SD','TN','TX','UT','VA','VT','WA','WI','WV','WY']

setA=set(state_list)

setB=set(dftmp.index)

drop_list = setB-setA # whatever is in setB but NOT in setA, get ready to drop it

dfbeds = dftmp.drop(drop_list)
# transpose and rename index/column -- index is still year, a string value

# I wanted the years to be the indices, not the state abbreviations, so I transposed here

dfbeds = dfbeds.T

dfbeds.index.name='year'

dfbeds.columns.name='state'

dfbeds.head(2)
# here I open the excel file that has data on the number of homeless per year per state. It has numerous sheets inside of it, each one is a year.

# I make series for each year. 



excel_sheet='2008'

dftmp = pd.read_excel('/kaggle/input/homeless-in-america-20102020/2007-2019-PIT-Counts-by-State.xlsx',index_col=0,header=[0],sheet_name=excel_sheet) 

s2008=dftmp['Overall Homeless, 2008']

s2008.name=excel_sheet



excel_sheet='2009'

dftmp = pd.read_excel('/kaggle/input/homeless-in-america-20102020/2007-2019-PIT-Counts-by-State.xlsx',index_col=0,header=[0],sheet_name=excel_sheet)

s2009=dftmp['Overall Homeless, 2009']

s2009.name=excel_sheet



excel_sheet='2010'

dftmp = pd.read_excel('/kaggle/input/homeless-in-america-20102020/2007-2019-PIT-Counts-by-State.xlsx',index_col=0,header=[0],sheet_name=excel_sheet)

s2010=dftmp['Overall Homeless, 2010']

s2010.name=excel_sheet



excel_sheet='2011'

dftmp = pd.read_excel('/kaggle/input/homeless-in-america-20102020/2007-2019-PIT-Counts-by-State.xlsx',index_col=0,header=[0],sheet_name=excel_sheet) 

s2011=dftmp['Overall Homeless, 2011']

s2011.name=excel_sheet



excel_sheet='2012'

dftmp = pd.read_excel('/kaggle/input/homeless-in-america-20102020/2007-2019-PIT-Counts-by-State.xlsx',index_col=0,header=[0],sheet_name=excel_sheet) 

s2012=dftmp['Overall Homeless, 2012']

s2012.name=excel_sheet



excel_sheet='2013'

dftmp = pd.read_excel('/kaggle/input/homeless-in-america-20102020/2007-2019-PIT-Counts-by-State.xlsx',index_col=0,header=[0],sheet_name=excel_sheet) 

s2013=dftmp['Overall Homeless, 2013']

s2013.name=excel_sheet



excel_sheet='2014'

dftmp = pd.read_excel('/kaggle/input/homeless-in-america-20102020/2007-2019-PIT-Counts-by-State.xlsx',index_col=0,header=[0],sheet_name=excel_sheet) 

s2014=dftmp['Overall Homeless, 2014']

s2014.name=excel_sheet



excel_sheet='2015'

dftmp = pd.read_excel('/kaggle/input/homeless-in-america-20102020/2007-2019-PIT-Counts-by-State.xlsx',index_col=0,header=[0],sheet_name=excel_sheet) 

s2015=dftmp['Overall Homeless, 2015']

s2015.name=excel_sheet



excel_sheet='2016'

dftmp = pd.read_excel('/kaggle/input/homeless-in-america-20102020/2007-2019-PIT-Counts-by-State.xlsx',index_col=0,header=[0],sheet_name=excel_sheet) 

s2016=dftmp['Overall Homeless, 2016']

s2016.name=excel_sheet



excel_sheet='2017'

dftmp = pd.read_excel('/kaggle/input/homeless-in-america-20102020/2007-2019-PIT-Counts-by-State.xlsx',index_col=0,header=[0],sheet_name=excel_sheet) 

s2017=dftmp['Overall Homeless, 2017']

s2017.name=excel_sheet



excel_sheet='2018'

dftmp = pd.read_excel('/kaggle/input/homeless-in-america-20102020/2007-2019-PIT-Counts-by-State.xlsx',index_col=0,header=[0],sheet_name=excel_sheet) 

s2018=dftmp['Overall Homeless, 2018']

s2018.name=excel_sheet



excel_sheet='2019'

dftmp = pd.read_excel('/kaggle/input/homeless-in-america-20102020/2007-2019-PIT-Counts-by-State.xlsx',index_col=0,header=[0],sheet_name=excel_sheet) 

s2019=dftmp['Overall Homeless, 2019']

s2019.name=excel_sheet
# combine all of them into a single dataframe, years are column headings:

dftmp = pd.merge(s2008,s2009,left_index=True,right_index=True)

dftmp = pd.merge(dftmp,s2010,left_index=True,right_index=True)

dftmp = pd.merge(dftmp,s2011,left_index=True,right_index=True)

dftmp = pd.merge(dftmp,s2012,left_index=True,right_index=True)

dftmp = pd.merge(dftmp,s2013,left_index=True,right_index=True)

dftmp = pd.merge(dftmp,s2014,left_index=True,right_index=True)

dftmp = pd.merge(dftmp,s2015,left_index=True,right_index=True)

dftmp = pd.merge(dftmp,s2016,left_index=True,right_index=True)

dftmp = pd.merge(dftmp,s2017,left_index=True,right_index=True)

dftmp = pd.merge(dftmp,s2018,left_index=True,right_index=True)

dftmp = pd.merge(dftmp,s2019,left_index=True,right_index=True)
# drop anything that's not recognizable as a state

# 'dfnohome' will be a dataframe containing homeless data

state_list = ['AK','AL','AR','AZ','CA','CO','CT','DE','FL','GA',

             'HI','IA','ID','IL','IN','KS','KY','LA','MA','MD',

             'ME','MI','MN','MO','MS','MT','NC','ND','NE','NH',

             'NJ','NM','NV','NY','OH','OK','OR','PA','RI','SC',

             'SD','TN','TX','UT','VA','VT','WA','WI','WV','WY']

setA=set(state_list)

setB=set(dftmp.index)

drop_list = setB-setA

dfnohome=dftmp.drop(drop_list)
# transpose and rename index/column -- index is still year, a string value

dfnohome = dfnohome.T

dfnohome.index.name='year'

dfnohome.columns.name='state'
# the values are strings, but should have been integers,converting:

dfnohome=dfnohome.astype(int)
dfnohome.head(2)
# form a dictionary of abbreviations for all 50 states

dftmp = pd.read_csv('/kaggle/input/homeless-in-america-20102020/state_abbrev_dict.csv',index_col=None,header=None,names=['state','abbrev'])

abbr_dict = dict(zip(dftmp.state,dftmp.abbrev))
# get the populations by state

my_cols=[0,3,4,5,6,7,8,9,10,11,12]

my_names=['state_name','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019']

dftmp=pd.read_excel('/kaggle/input/homeless-in-america-20102020/nst-est2019-01.xlsx',skiprows=8,skipfooter=5,index_col=None,usecols=my_cols,names=my_names)

# one row is a row of NaNs so get rid of it:

dftmp.dropna(inplace=True)

dftmp.head(2)
# every state name starts with a period. Want to replace those full names with periods with an abbreviation

x=dftmp.state_name

y=[]

for item in x:

    name = item[1:] # strip off the period at start

    val = abbr_dict.get(name,'NOTASTATE') # look in dictionary and assign abbreviation or else the string "NOTASTATE"

    y.append(val)

dftmp['state']=y

# now drop everthing that's not a state (e.g., Puerto Rico)

dftmp = dftmp[dftmp['state'] != 'NOTASTATE']

# now dump the 'state_name' column

dftmp.drop('state_name',axis=1,inplace=True)

# make the state col (really the state's abbreviated name) the index

dftmp.set_index('state',inplace=True)

# see McKinney book on 'period'

rng=pd.period_range('2010','2019',freq='Y')

dftmp.columns=rng

dfpop=dftmp.copy()
dfpop.head(2)
# I again want the years to be the rows, so transpose:

dfpop=dfpop.T

dfpop.head(2)
dfpop=dfpop.astype(int) # they were floats, but integers makes more sense
# It would be good to know the no. of homeless normalized by each state's population.

# obviously, there are less homeless in alaska than california, but how many do we have relative to state's residents?

# get homeless and population data into same shape -- the former has two more rows (years) than the latter:

dftmp1 = dfnohome[2:].copy()

# here's another temporary datafile:

dftmp2 = dfpop.copy()

# make a copy of dftmp1, but with the intention of overwriting every value, keeping only columns and index values:

dfhomepop = dftmp1.copy()

dfhomepop=dfhomepop.astype(float) # but the values I'm replacing will be floats, so change this here



# go through each row and column, form a new value, assign it -- this is the no. of homeless per 100,000 residents:

for i in range(0,len(dftmp1)):

    for j in range(0,len(dftmp1.columns)):

        xlabel=dftmp1.index[i]

        ylabel=dftmp1.columns[j]

        homeless = dftmp1.loc[xlabel].loc[ylabel]

        statepop = dftmp2.loc[xlabel].loc[ylabel]

        x = homeless/statepop*100000.0

        dfhomepop.loc[xlabel][ylabel]=x
dfhomepop.head(2)
dftmp3 = dfhomepop.iloc[-1] # get last year (2019) date only

dftmp4 = dftmp3.sort_values() # sort this series out

sns.set_style('whitegrid')

fig,axis=plt.subplots(nrows=1,ncols=1,figsize=(6,9))

sns.barplot(x=dftmp4.values,y=dftmp4.index,color='cyan',ax=axis)

axis.set_title('homeless per 100,000 residents (by state) (2019 only)')

axis.set_xlabel('')

axis.set_ylabel('');
# no. of shelter beds per homeless:

dftmp5 = dfbeds.copy()

# here's another temporary datafile:

dftmp6 = dfnohome.copy()

# make a copy of dftmp5, but with the intention of overwriting every value, keeping only columns and index values:

dfbedhome = dftmp5.copy()

dfbedhome=dfbedhome.astype(float) # but the values I'm replacing will be floats, so change this here



# go through each row and column, form a new value, assign it -- this is the no. of homeless per 100,000 residents:

for i in range(0,len(dftmp5)):

    for j in range(0,len(dftmp5.columns)):

        xlabel=dftmp5.index[i]

        ylabel=dftmp5.columns[j]

        beds = dftmp5.loc[xlabel].loc[ylabel]

        nohome = dftmp6.loc[xlabel].loc[ylabel]

        x = beds/nohome*1.0

        dfbedhome.loc[xlabel][ylabel]=x
dftmp7 = dfbedhome.iloc[-1] # get last year (2019) date only

dftmp8 = dftmp7.sort_values() # sort this series out

sns.set_style('whitegrid')

fig,axis=plt.subplots(nrows=1,ncols=1,figsize=(6,9))

sns.barplot(x=dftmp8.values,y=dftmp8.index,color='cyan',ax=axis)

axis.set_title('beds per homeless person (by state) (2019 only)')

axis.set_xlabel('')

axis.set_ylabel('');
# maybe states with the most homeless people per 100,000 residents have the lowest bed-to-homeless ratios?

s = dfhomepop.iloc[9] # get latest yr's data for homeless per 100,000

dftmp10 = pd.DataFrame(s) # convert to dataframe so merge can happen

s = dfbedhome.iloc[11] # beds per homeless as series

dftmp11 = pd.DataFrame(s) # as dataframe

dftmp12 = dftmp10.merge(dftmp11,left_index=True,right_index=True)

dftmp12.columns=['homelessper100000','bedsperhomeless'] # this is a dataframe with states as index and two columns

dftmp12.head()
fig,axis=plt.subplots(nrows=1,ncols=1,figsize=(6,6))

x=dftmp12.homelessper100000.astype('float')

y=dftmp12.bedsperhomeless.astype('float')

sns.regplot(x=x,y=y)

plt.xlabel('homeless per 100,000 residents')

plt.ylabel('beds per homeless person')

plt.annotate('HI',xy=(450,0.4),size=15)

plt.annotate('NY',xy=(450,0.9),size=15)

plt.annotate('OR',xy=(335,0.42),size=15)

plt.annotate('CA',xy=(337,0.28),size=15);
x2 = sm.add_constant(x) # if y intercept is assumed non-zero, then need this.

model = sm.OLS(y,x2).fit()

pred = model.predict(x2)

model.summary()
# stuff you can get from the linear model

b = model.params[0]

m = model.params[1]

rsqared=model.rsquared

rsqradj=model.rsquared_adj

pred =model.fittedvalues;

#model.params gets all of these, maybe
print(sm.stats.linear_rainbow.__doc__)
sm.stats.linear_rainbow(model) # if 2nd of these values is low (say less than 0.05, linear fit may be good)
# this shows that seaborn's regplot and statsmodels OLS fit both do the same thing.

fig,axis=plt.subplots(nrows=1,ncols=1,figsize=(6,6))



sns.regplot(x=x,y=y)

sns.lineplot(x=x,y=pred)



plt.xlabel('homeless per 100,000 residents')

plt.ylabel('beds per homeless person')

plt.annotate('HI',xy=(450,0.4),size=15)

plt.annotate('NY',xy=(450,0.9),size=15)

plt.annotate('OR',xy=(335,0.42),size=15)

plt.annotate('CA',xy=(337,0.28),size=15)

my_text = 'weak correlation:\nrsqradj = %.2f'%(rsqradj)

plt.annotate(my_text,xy=(200,1.2),size=15)