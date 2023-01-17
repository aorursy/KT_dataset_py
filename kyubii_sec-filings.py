# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas import option_context

import seaborn as sns



#pd.set_option('display.max_rows', 500)

#pd.set_option('display.max_columns', 500)

pd.set_option('display.max_colwidth', 400)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
listings = {}

for year in range(1994, 2021):

    for qtr in range(1, 5):

        if year == 2020 and qtr > 2: 

            break

        listings['q' + str(qtr) + '_' + str(year)] = str(year) + ".QTR" + str(qtr) + ".csv"
for var, csv_file in listings.items():

    print(var, csv_file)

    exec("%s = %s" % (var, str('pd.read_csv("/kaggle/input/sec-filings/' + csv_file + '")')))
annual_1994 = pd.concat([q1_1994, q2_1994, q3_1994, q4_1994])

annual_1994.dropna(inplace=True)

annual_1994.drop(['cik', 'report_url', 'accepted_date'], axis=1, inplace=True)

annual_1994 = annual_1994[annual_1994['form'] == '10-K']

annual_1994
concat_reports = {}

for year in range(1994, 2021):

    current_arr = []

    for var, quarters in listings.items():

        if var[3:] == str(year):

            current_arr.append(var)

    concat_reports['annual_' + str(year)] = current_arr
concat_reports
annual_1994 = pd.concat([q1_1994, q2_1994, q3_1994, q4_1994])

annual_1995 = pd.concat([q1_1995, q2_1995, q3_1995, q4_1995])

annual_1996 = pd.concat([q1_1996, q2_1996, q3_1996, q4_1996])

annual_1997 = pd.concat([q1_1997, q2_1997, q3_1997, q4_1997])

annual_1998 = pd.concat([q1_1998, q2_1998, q3_1998, q4_1998])

annual_1999 = pd.concat([q1_1999, q2_1999, q3_1999, q4_1999])

annual_2000 = pd.concat([q1_2000, q2_2000, q3_2000, q4_2000])

annual_2001 = pd.concat([q1_2001, q2_2001, q3_2001, q4_2001])

annual_2002 = pd.concat([q1_2002, q2_2002, q3_2002, q4_2002])

annual_2003 = pd.concat([q1_2003, q2_2003, q3_2003, q4_2003])

annual_2004 = pd.concat([q1_2004, q2_2004, q3_2004, q4_2004])

annual_2005 = pd.concat([q1_2005, q2_2005, q3_2005, q4_2005])

annual_2006 = pd.concat([q1_2006, q2_2006, q3_2006, q4_2006])

annual_2007 = pd.concat([q1_2007, q2_2007, q3_2007, q4_2007])

annual_2008 = pd.concat([q1_2008, q2_2008, q3_2008, q4_2008])

annual_2009 = pd.concat([q1_2009, q2_2009, q3_2009, q4_2009])

annual_2010 = pd.concat([q1_2010, q2_2010, q3_2010, q4_2010])

annual_2011 = pd.concat([q1_2011, q2_2011, q3_2011, q4_2011])

annual_2012 = pd.concat([q1_2012, q2_2012, q3_2012, q4_2012])

annual_2013 = pd.concat([q1_2013, q2_2013, q3_2013, q4_2013])

annual_2014 = pd.concat([q1_2014, q2_2014, q3_2014, q4_2014])

annual_2015 = pd.concat([q1_2015, q2_2015, q3_2015, q4_2015])

annual_2016 = pd.concat([q1_2016, q2_2016, q3_2016, q4_2016])

annual_2017 = pd.concat([q1_2017, q2_2017, q3_2017, q4_2017])

annual_2018 = pd.concat([q1_2018, q2_2018, q3_2018, q4_2018])

annual_2019 = pd.concat([q1_2019, q2_2019, q3_2019, q4_2019])

annual_2020 = pd.concat([q1_2020, q2_2020])
annuals = [

    annual_1994,

    annual_1995,

    annual_1996,

    annual_1997,

    annual_1998,

    annual_1999,

    annual_2000,

    annual_2001,

    annual_2002,

    annual_2003,

    annual_2004,

    annual_2005,

    annual_2006,

    annual_2007,

    annual_2008,

    annual_2009,

    annual_2010,

    annual_2011,

    annual_2012,

    annual_2013,

    annual_2014,

    annual_2015,

    annual_2016,

    annual_2017,

    annual_2018,

    annual_2019,

    annual_2020

]
for var, csv_file in concat_reports.items():

    exec("%s.dropna(inplace=True)" % (var))
for i, items in enumerate(annuals):

    print(i + 1994, ':', len(items))