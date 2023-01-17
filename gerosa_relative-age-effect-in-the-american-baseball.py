# Import Libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%pylab inline
# Creating auxilary series for month names 

monthNames = pd.Series(

    ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    , index=range(1,13))



relativeMonthNames = pd.Series(

     ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul'])



# Auxiliary functions

def normalizeValues(s):

    '''returns the proportion of each value in relation to the sum of all values.'''

    return s / s.sum()

    
d = {

    'Year' : [1994,1994,1994,1994,1994,1994,1994,1994,1994,1994,1994,1994,1995,1995,1995,1995,1995,1995,1995,1995,1995,1995,1995,1995,1996,1996,1996,1996,1996,1996,1996,1996,1996,1996,1996,1996,1997,1997,1997,1997,1997,1997,1997,1997,1997,1997,1997,1997,1998,1998,1998,1998,1998,1998,1998,1998,1998,1998,1998,1998,1999,1999,1999,1999,1999,1999,1999,1999,1999,1999,1999,1999,2000,2000,2000,2000,2000,2000,2000,2000,2000,2000,2000,2000,2001,2001,2001,2001,2001,2001,2001,2001,2001,2001,2001,2001,2002,2002,2002,2002,2002,2002,2002,2002,2002,2002,2002,2002],

    'Month' : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],

    'Births' : [320705, 301327, 339736, 317392, 330295, 329737, 345862, 352173, 339223, 330172, 319397, 326748, 316013, 295094, 328503, 309119, 334543, 329805, 340873, 350737, 339103, 330012, 310817, 314970, 314283, 301763, 322581, 312595, 325708, 318525, 345162, 346317, 336348, 336346, 309397, 322469, 317211, 291541, 321212, 314230, 330331, 321867, 346506, 339122, 333600, 328657, 307282, 329335, 319340, 298711, 329436, 319758, 330519, 327091, 348651, 344736, 343384, 332790, 313241, 333896, 319182, 297568, 332939, 316889, 328526, 332201, 349812, 351371, 349409, 332980, 315289, 333251, 330108, 317377, 340553, 317180, 341207, 341206, 348975, 360080, 347609, 343921, 333811, 336787, 335198, 303534, 338684, 323613, 344017, 331085, 351047, 361802, 342564, 344074, 323746, 326569, 330674, 303977, 331505, 324432, 339007, 327588, 357669, 359417, 348814, 345814, 318573, 334256]}



generalBirths = pd.DataFrame(d)

generalBirths.head()
generalBirthsByMonth = generalBirths.groupby('Month').mean()['Births'] / 1000



# plot a line with the mean births per year

meanBirths = generalBirthsByMonth.sum() / len(generalBirthsByMonth)

plt.axhline(meanBirths, color='red', zorder=1)



# plot the distribution of births per months

generalBirthsByMonth.plot(kind='bar', zorder=2)

plt.xticks(range(13), monthNames, rotation=0)

plt.ylabel('Mean number of births (thousand)')

plt.title('U.S. month of birth distribution of the general population from 1994 to 2002');
master = pd.read_csv('../input/player.csv')

print(master.columns)

master.head()
batting = batting = pd.read_csv('../input/batting.csv')



# replace NaN with 0 so the batting stats can be used to calculate performance measures

batting = batting.fillna(value=0)

batting.head()
playerBirthsByMonth = master['birth_month'].value_counts().sort_index()

minYear = master['birth_year'].min()

maxYear = master['birth_year'].max()

total = playerBirthsByMonth.sum()



# plot a line indicating birth in a uniformly distribution

meanBirths = total / 12

plt.axhline(meanBirths, color='red', zorder=1)



# plot the distribution of births per months

playerBirthsByMonth.plot(kind='bar', zorder=2)

plt.xticks(range(13), monthNames, rotation=0)

plt.ylabel('Births')

plt.title('Month of birth distribution of {} american baseball players from {:4.0f} to {:4.0f}'.format(total, minYear, maxYear))



birthsBefore = playerBirthsByMonth.loc[5:7].sum()

birthsAfter = playerBirthsByMonth.loc[8:10].sum()

print('Players born in May, June or July: {}'.format(birthsBefore))

print('Players born in August, Septermber or October: {}'.format(birthsAfter))

print('Increase: {:4.2f}%'.format((birthsAfter - birthsBefore) / birthsBefore * 100))
# The relative month is calculated rotating left the month of birth by 8.



# master dataframe

master = master.assign(relativeMonth = (master.birth_month - 8) % 12)



# general population dataframe

generalBirths = generalBirths.assign(relativeMonth = (generalBirths.Month - 8) % 12)

generalBirths.head(n = 12)
generalMeanBirthsByRelativeMonth = generalBirths.groupby('relativeMonth').mean()['Births']

generalBirthRatesByRelativeMonth = normalizeValues(generalMeanBirthsByRelativeMonth)



playersBirthRatesByRelativeMonth = master['relativeMonth'].value_counts(normalize=True).sort_index()
generalBirthRatesByRelativeMonth.plot(label='General Population')

playersBirthRatesByRelativeMonth.plot(label='Baseball Players')



plt.legend()

plt.ylabel('Proportion')

plt.title('General population vs baseball players birth rates')

plt.xticks(range(12), relativeMonthNames, rotation=45);
deviations = playersBirthRatesByRelativeMonth - generalBirthRatesByRelativeMonth

deviations.plot(kind='bar')

plt.ylabel('Frequency deviation')

plt.title('Frequency deviation between baseball players birth rate and the general population birth rate')

plt.xticks(range(12), relativeMonthNames, rotation=0);
r = numpy.corrcoef(deviations, range(12))[0,1]

print('r = {}'.format(r))
#On Base Percentage = (H + BB + HBP)/ (AB + BB + HBP + SF)

batting = batting.assign(OBP = (batting.h + batting.bb +batting.hbp)/(batting.ab + batting.bb + batting.hbp + batting.sf))



meanOBPByMonth = batting.merge(master, on='player_id').groupby('relativeMonth').mean()['OBP']

meanOBPFrequencyByMonth = normalizeValues(meanOBPByMonth)
meanOBPFrequencyByMonth.plot(label='mean OBP')

playersBirthRatesByRelativeMonth.plot(label='Player Month of Birth')



plt.legend()

plt.xlabel('Month')

plt.ylabel('Proportion')

plt.title('Frequency distribution of mean OBP vs players months of birth')

plt.xticks(range(12), relativeMonthNames, rotation=45);