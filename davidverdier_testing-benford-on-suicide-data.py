"""Among most populations of numbers, a pattern will emerge in the first numbers. 

   Benford's Law predicts the leading number in most data sets is likely to be small.

   The occurence of '1' as the leading number will occur about 30% of the time. 

   Each number in sequence afterwards will decrease logarithmically with '9' occuring

   less than 5%.

   

   Does Benford's Law apply to the following data set?

"""



import numpy as np 

import pandas as pd 

from math import log10

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data_path = '../input/suicide-rates-overview-1985-to-2016/master.csv'

suicide_data = pd.read_csv(data_path, index_col=None)
suicide_data.head()
#remove columns with missing values.



missing_cols = [col for col in suicide_data.columns if suicide_data[col].isnull().any()]

reduced_suicide_data = suicide_data.drop(missing_cols, axis=1)

print(reduced_suicide_data.columns)
#function to calculate number of suicides by country and year



def combine_data(dataframe, row_year, country):

    country_data = pd.DataFrame(dataframe.loc[(dataframe['country'] == country) & (dataframe['year'] == row_year)])

    return country_data['suicides_no'].sum()

"""

    Trying to iterate list over function. Problem with saving result as a working dataframe.

    

"""

#list of countries from data set

country_list = ['Albania', 'Antigua and Barbuda', 'Argentina', 'Armenia', 'Aruba',

       'Australia', 'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain',

       'Barbados', 'Belarus', 'Belgium', 'Belize',

       'Bosnia and Herzegovina', 'Brazil', 'Bulgaria', 'Cabo Verde',

       'Canada', 'Chile', 'Colombia', 'Costa Rica', 'Croatia', 'Cuba',

       'Cyprus', 'Czech Republic', 'Denmark', 'Dominica', 'Ecuador',

       'El Salvador', 'Estonia', 'Fiji', 'Finland', 'France', 'Georgia',

       'Germany', 'Greece', 'Grenada', 'Guatemala', 'Guyana', 'Hungary',

       'Iceland', 'Ireland', 'Israel', 'Italy', 'Jamaica', 'Japan',

       'Kazakhstan', 'Kiribati', 'Kuwait', 'Kyrgyzstan', 'Latvia',

       'Lithuania', 'Luxembourg', 'Macau', 'Maldives', 'Malta',

       'Mauritius', 'Mexico', 'Mongolia', 'Montenegro', 'Netherlands',

       'New Zealand', 'Nicaragua', 'Norway', 'Oman', 'Panama', 'Paraguay',

       'Philippines', 'Poland', 'Portugal', 'Puerto Rico', 'Qatar',

       'Republic of Korea', 'Romania', 'Russian Federation',

       'Saint Kitts and Nevis', 'Saint Lucia',

       'Saint Vincent and Grenadines', 'San Marino', 'Serbia',

       'Seychelles', 'Singapore', 'Slovakia', 'Slovenia', 'South Africa',

       'Spain', 'Sri Lanka', 'Suriname', 'Sweden', 'Switzerland',

       'Thailand', 'Trinidad and Tobago', 'Turkey', 'Turkmenistan',

       'Ukraine', 'United Arab Emirates', 'United Kingdom',

       'United States', 'Uruguay', 'Uzbekistan',]
#function for each year preprocessing

def create_new_data(year):

    new_data = {(country):combine_data(reduced_suicide_data, year, country) for country in country_list}

    new_data_year = pd.DataFrame.from_dict(new_data, orient='index', dtype=None, columns={'suicide_nums'})

    row_zero = (new_data_year != 0).any(axis=1)

    new_data_ = new_data_year.loc[row_zero]

    to_string = [str(v) for v in new_data.values() if v >0]

    for idx, string in enumerate(to_string):

        to_string[idx] = string[0:1]

    return sorted(to_string)

    

#create string data

new_data_1987 = create_new_data(1987)

new_data_1988 = create_new_data(1988)

new_data_1989 = create_new_data(1989)

new_data_1990 = create_new_data(1990)

new_data_1991 = create_new_data(1991)

new_data_1992 = create_new_data(1992)

new_data_1993 = create_new_data(1993)

new_data_1994 = create_new_data(1994)

new_data_1995 = create_new_data(1995)

new_data_1996 = create_new_data(1996)

new_data_1997 = create_new_data(1997)

new_data_1998 = create_new_data(1998)

new_data_1999 = create_new_data(1999)

new_data_2000 = create_new_data(2000)

new_data_2001 = create_new_data(2001)

new_data_2002 = create_new_data(2002)

new_data_2003 = create_new_data(2003)

new_data_2004 = create_new_data(2004)

new_data_2005 = create_new_data(2005)

new_data_2006 = create_new_data(2006)

new_data_2007 = create_new_data(2007)

new_data_2008 = create_new_data(2008)

new_data_2009 = create_new_data(2009)

new_data_2010 = create_new_data(2010)

new_data_2011 = create_new_data(2011)

new_data_2012 = create_new_data(2012)

new_data_2013 = create_new_data(2013)

new_data_2014 = create_new_data(2014)

new_data_2015 = create_new_data(2015)

new_data_2016 = create_new_data(2016)
#convert to dict, work on creating funtion to return output as list

to_dict_1987 = {v: new_data_1987.count(v) for v in new_data_1987}

to_dict_1988 = {v: new_data_1988.count(v) for v in new_data_1988}

to_dict_1989 = {v: new_data_1989.count(v) for v in new_data_1989}

to_dict_1990 = {v: new_data_1990.count(v) for v in new_data_1990}

to_dict_1991 = {v: new_data_1991.count(v) for v in new_data_1991}

to_dict_1992 = {v: new_data_1992.count(v) for v in new_data_1992}

to_dict_1993 = {v: new_data_1993.count(v) for v in new_data_1993}

to_dict_1994 = {v: new_data_1994.count(v) for v in new_data_1994}

to_dict_1995 = {v: new_data_1995.count(v) for v in new_data_1995}

to_dict_1996 = {v: new_data_1996.count(v) for v in new_data_1996}

to_dict_1997 = {v: new_data_1997.count(v) for v in new_data_1997}

to_dict_1998 = {v: new_data_1998.count(v) for v in new_data_1998}

to_dict_1999 = {v: new_data_1999.count(v) for v in new_data_1999}

to_dict_2000 = {v: new_data_2000.count(v) for v in new_data_2000}

to_dict_2001 = {v: new_data_2001.count(v) for v in new_data_2001}

to_dict_2002 = {v: new_data_2002.count(v) for v in new_data_2002}

to_dict_2003 = {v: new_data_2003.count(v) for v in new_data_2003}

to_dict_2004 = {v: new_data_2004.count(v) for v in new_data_2004}

to_dict_2005 = {v: new_data_2005.count(v) for v in new_data_2005}

to_dict_2006 = {v: new_data_2006.count(v) for v in new_data_2006}

to_dict_2007 = {v: new_data_2007.count(v) for v in new_data_2007}

to_dict_2008 = {v: new_data_2008.count(v) for v in new_data_2008}

to_dict_2009 = {v: new_data_2009.count(v) for v in new_data_2009}

to_dict_2010 = {v: new_data_2010.count(v) for v in new_data_2010}

to_dict_2011 = {v: new_data_2011.count(v) for v in new_data_2011}

to_dict_2012 = {v: new_data_2012.count(v) for v in new_data_2012}

to_dict_2013 = {v: new_data_2013.count(v) for v in new_data_2013}

to_dict_2014 = {v: new_data_2014.count(v) for v in new_data_2014}

to_dict_2015 = {v: new_data_2015.count(v) for v in new_data_2015}

to_dict_2016 = {v: new_data_2016.count(v) for v in new_data_2016}
#function to change dict values into list of percentage overall

def get_percent(dict):

    temp_data = sum(dict.values())

    return [(v / temp_data) * 100 for v in dict.values()]



percent_1987 = get_percent(to_dict_1987)

percent_1988 = get_percent(to_dict_1988)

percent_1989 = get_percent(to_dict_1989)

percent_1990 = get_percent(to_dict_1990)

percent_1991 = get_percent(to_dict_1991)

percent_1992 = get_percent(to_dict_1992)

percent_1993 = get_percent(to_dict_1993)

percent_1994 = get_percent(to_dict_1994)

percent_1995 = get_percent(to_dict_1995)

percent_1996 = get_percent(to_dict_1996)

percent_1997 = get_percent(to_dict_1997)

percent_1998 = get_percent(to_dict_1998)

percent_1999 = get_percent(to_dict_1999)

percent_2000 = get_percent(to_dict_2000)

percent_2001 = get_percent(to_dict_2001)

percent_2002 = get_percent(to_dict_2002)

percent_2003 = get_percent(to_dict_2003)

percent_2004 = get_percent(to_dict_2004)

percent_2005 = get_percent(to_dict_2005)

percent_2006 = get_percent(to_dict_2006)

percent_2007 = get_percent(to_dict_2007)

percent_2008 = get_percent(to_dict_2008)

percent_2009 = get_percent(to_dict_2009)

percent_2010 = get_percent(to_dict_2010)

percent_2011 = get_percent(to_dict_2010)

percent_2012 = get_percent(to_dict_2010)

percent_2013 = get_percent(to_dict_2010)

percent_2014 = get_percent(to_dict_2010)

percent_2015 = get_percent(to_dict_2010)

percent_2016 = get_percent(to_dict_2010)



#add each year's percent occurence to chart

plt.plot(percent_1987)

plt.plot(percent_1988)

plt.plot(percent_1989)

plt.plot(percent_1990)

plt.plot(percent_1991)

plt.plot(percent_1992)

plt.plot(percent_1993)

plt.plot(percent_1994)

plt.plot(percent_1995)

plt.plot(percent_1996)

plt.plot(percent_1997)

plt.plot(percent_1998)

plt.plot(percent_1999)

plt.plot(percent_2000)

plt.plot(percent_2001)

plt.plot(percent_2002)

plt.plot(percent_2003)

plt.plot(percent_2004)

plt.plot(percent_2005)

plt.plot(percent_2006)

plt.plot(percent_2007)

plt.plot(percent_2008)

plt.plot(percent_2009)

plt.plot(percent_2010)

plt.plot(percent_2011)

plt.plot(percent_2012)

plt.plot(percent_2013)

plt.plot(percent_2014)

plt.plot(percent_2015)

plt.plot(percent_2016)



plt.title('Benford\'s Law compared to Suicide Data by Year')

#define Benford's Law and plot

benford = [(log10(1+1.0/i),str(i)) for i in range(1,10)]

x_val = [x[0] * 100 for x in benford]

plt.plot(x_val, label='Benford', linewidth=5, alpha=0.9)



#change x-axis to account for list index 0

plt.xticks(np.arange(len(benford)), np.arange(1, len(benford)+1))



#chart info and output file

plt.xlabel('First Digit in Suicide Numbers')

plt.ylabel('Occurence Percentage')

plt.legend()

plt.savefig('benford.jpg')