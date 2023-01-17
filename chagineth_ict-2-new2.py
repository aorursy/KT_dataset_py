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
import pandas as pd

import matplotlib.pyplot as plt

uv2007 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2007.csv') 

uv2007.head(10)
uv2008 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2008.csv') 

uv2008.head(10)
uv2009 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2009.csv') 

uv2009.head(10)
uv2010 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2010.csv') 

uv2010.head(10)
uv2011 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2011.csv') 

uv2011.head(10)
uv2012 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2012.csv') 

uv2012.head(10)
uv2013 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2013.csv') 

uv2013.head(10)
uv2014 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2014.csv') 

uv2014.head(10)
uv2015 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2015.csv') 

uv2015.head(10)
uv2016 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2016.csv') 

uv2016.head(10)
uv2017 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2017.csv') 

uv2017.head(10)
uv2018 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2018.csv') 

uv2018.head(10)
uv2019 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2019.csv') 

uv2019.head(10)
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

uv2010 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2010.csv') 



Date_and_Time = uv2010[(uv2010["Date"]=="01/01/2010") & (uv2008["Time"]>='06:00') & (uv2008["Time"]<='18:00')]

print (Date_and_Time)



Date_and_Time.plot(figsize=(20,5),kind='line', x='Time',y='UV_Index', ylim =(0.0,17), color='green', sort_columns = True)

plt.title("Time", y=1.05);

plt.ylabel("UV_Index", labelpad=14)
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



pd.set_option('display.max_rows', None)

uv2010 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2010.csv')

# print(uv2010)



Date_and_Time = uv2010[(uv2010["Date"]=="01/01/2010") & (uv2010["Time"]>='06:00') & (uv2010["Time"]<='18:00')]

print (Date_and_Time)

Date_and_Time = uv2010[(uv2010["Date"]=="01/01/2010") & (uv2010["Time"]>='06:00') & (uv2010["Time"]<='18:00')]

# print (Date_and_Time)



Date_and_Time.plot(figsize=(20,5),kind='line', x='Time',y='UV_Index', ylim =(0.0,17), color='green', sort_columns = True)

plt.title("Time", y=1.05);

plt.ylabel("UV_Index", labelpad=14)
# Month Jnuary UV index



import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



# pd.set_option('display.max_rows', None)

uv2010 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2010.csv')



Year2010_6 = uv2010[(uv2010["Date"]=="01/01/2010") & (uv2010["Time"]=='07:00')]

Year2010_7 = uv2010[(uv2010["Date"]=="01/01/2010") & (uv2010["Time"]=='07:00')]

Year2010_8 = uv2010[(uv2010["Date"]=="01/01/2010") & (uv2010["Time"]=='08:00')]

Year2010_9 = uv2010[(uv2010["Date"]=="01/01/2010") & (uv2010["Time"]=='09:00')]

Year2010_10 = uv2010[(uv2010["Date"]=="01/01/2010") & (uv2010["Time"]=='10:00')]

Year2010_11 = uv2010[(uv2010["Date"]=="01/01/2010") & (uv2010["Time"]=='11:00')]

Year2010_12 = uv2010[(uv2010["Date"]=="01/01/2010") & (uv2010["Time"]=='12:00')]

Year2010_13 = uv2010[(uv2010["Date"]=="01/01/2010") & (uv2010["Time"]=='13:00')]

Year2010_14 = uv2010[(uv2010["Date"]=="01/01/2010") & (uv2010["Time"]=='14:00')]

Year2010_15 = uv2010[(uv2010["Date"]=="01/01/2010") & (uv2010["Time"]=='15:00')]

Year2010_16 = uv2010[(uv2010["Date"]=="01/01/2010") & (uv2010["Time"]=='16:00')]

Year2010_17 = uv2010[(uv2010["Date"]=="01/01/2010") & (uv2010["Time"]=='17:00')]

Year2010_18 = uv2010[(uv2010["Date"]=="01/01/2010") & (uv2010["Time"]=='18:00')]



print (Year2010_6)

# print (Year2010_6,Year2010_7,Year2010_8,Year2010_9,Year2010_10,Year2010_11,Year2010_12,Year2010_13,Year2010_14,

#        Year2010_15,Year2010_16,Year2010_17,Year2010_18)
# January Month UV-Indexes for past 10 Years 2010 - 2019



import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure





Y_2010 =  [0.000,0.31,1.50,3.68,5.47,06.23,08.73,08.72,07.19,5.39,2.94,1.14,0.17]

Y_2011 =  [0.000,0.38,1.82,4.27,6.59,08.36,10.81,07.83,07.05,5.10,2.90,1.11,0.17]

Y_2012 =  [0.000,0.41,1.78,4.53,8.18,11.81,11.19,11.07,10.07,6.95,3.99,1.42,0.29]

Y_2013 =  [0.002,0.31,1.50,3.92,7.05,09.42,10.24,10.40,08.94,6.25,3.45,1.23,0.16]

Y_2014 =  [0.040,0.48,1.91,4.31,6.66,08.39,09.61,10.24,08.26,6.56,3.44,1.38,0.29]

Y_2015 =  [0.017,0.42,1.76,4.21,7.14,09.49,11.13,10.14,08.01,5.90,3.12,1.24,0.21]

Y_2016 =  [0.015,0.41,1.79,3.96,6.82,08.29,10.36,10.97,09.11,5.88,3.45,1.22,0.21]

Y_2017 =  [0.017,0.43,1.90,4.30,7.40,08.92,11.22,10.26,07.39,5.80,3.14,1.19,0.21]

Y_2018 =  [0.038,0.47,1.84,4.14,6.34,08.66,09.74,10.16,08.50,6.08,3.19,1.22,0.27]

Y_2019 =  [0.044,0.45,1.70,3.76,5.89,06.88,07.45,08.32,06.55,5.18,2.95,1.25,0.28]





index = ["6.00","7.00","8.00","9.00","10.00","11.00","12.00","13.00","14.00","15.00","16.00","17.00","18.00",]



df = pd.DataFrame({'2010': Y_2010,'2011': Y_2011,'2012': Y_2012,'2013': Y_2013,'2014': Y_2014,

                   '2015': Y_2015,'2016': Y_2016,'2017': Y_2017,'2018': Y_2018, '2019': Y_2019}, index=index )



ax = df.plot.line(rot=90)

df.plot(figsize=(15,5))



# figure(num=1, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')



plt.title("January Month UV-Indexes for past 10 Years 2010 - 2019", y=1.05);

plt.ylabel("UV Index", labelpad=14)

plt.xlabel("Time", labelpad=14)

# lgd = legend({'2010','2011','2012','2013','2014','2015','2016','2017','2018','2019'},'Location','northwest','NumColumns',2)
# February Month UV-Indexes for past 10 Years 2010 - 2019



import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure





Y_2010 =  [0.00,0.19,1.19,3.39,6.52,08.27,10.25,09.42,8.20,5.29,3.18,1.08,0.16]

Y_2011 =  [0.00,0.19,1.33,3.18,6.45,08.24,08.88,08.23,6.95,4.92,2.77,0.91,0.14]

Y_2012 =  [0.00,0.24,1.65,4.03,7.42,11.34,10.80,11.41,9.62,6.80,3.85,1.41,0.21]

Y_2013 =  [0.00,0.16,1.25,3.81,6.48,09.06,09.91,09.92,8.66,5.73,3.31,1.08,0.11]

Y_2014 =  [0.00,0.30,1.49,3.53,6.21,07.86,08.31,09.41,7.02,5.51,3.09,1.07,0.19]

Y_2015 =  [0.00,0.26,1.50,3.96,6.93,09.14,09.96,10.25,8.64,5.60,3.34,1.11,0.19]

Y_2016 =  [0.00,0.24,1.50,3.94,6.90,10.41,10.68,10.66,9.41,5.84,3.23,1.14,0.16]

Y_2017 =  [0.00,0.24,1.48,4.06,6.65,09.62,11.75,11.34,8.18,6.83,3.56,1.20,0.16]

Y_2018 =  [0.01,0.29,1.31,3.39,5.94,07.87,09.52,09.09,7.86,5.77,2.94,1.09,0.19]

Y_2019 =  [0.01,0.26,1.27,3.67,6.18,07.66,09.50,08.62,7.70,5.19,2.95,1.07,0.21]





index = ["6.00","7.00","8.00","9.00","10.00","11.00","12.00","13.00","14.00","15.00","16.00","17.00","18.00",]



df = pd.DataFrame({'2010': Y_2010,'2011': Y_2011,'2012': Y_2012,'2013': Y_2013,'2014': Y_2014,

                   '2015': Y_2015,'2016': Y_2016,'2017': Y_2017,'2018': Y_2018, '2019': Y_2019}, index=index )



ax = df.plot.line(rot=90)

df.plot(figsize=(15,5))



plt.title("February Month UV-Indexes for past 10 Years 2010 - 2019", y=1.05);

plt.ylabel("UV Index", labelpad=14)

plt.xlabel("Time", labelpad=14)

# March Month UV-Indexes for past 10 Years 2010 - 2019



import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure





Y_2010 =  [0.00,0.12,1.11,3.21,5.75,07.92,08.21,08.58,6.99,4.43,2.04,0.61,0.04]

Y_2011 =  [0.00,0.09,0.84,2.56,4.80,06.87,07.45,07.78,6.07,4.15,1.89,0.50,0.03]

Y_2012 =  [0.00,0.13,1.24,3.79,7.26,10.43,13.42,11.22,7.72,5.74,3.30,1.09,0.11]

Y_2013 =  [0.00,0.09,1.13,3.35,6.20,07.99,08.03,08.80,6.96,5.02,2.47,0.69,0.01]

Y_2014 =  [0.00,0.21,1.25,3.48,6.37,07.61,09.56,07.92,7.22,5.25,2.48,0.84,0.09]

Y_2015 =  [0.00,0.19,1.27,3.49,6.08,08.33,09.75,09.82,8.18,5.76,2.72,0.81,0.08]

Y_2016 =  [0.00,0.13,1.02,2.95,5.42,07.47,07.84,09.36,6.67,4.64,2.16,0.64,0.04]

Y_2017 =  [0.00,0.14,1.08,3.32,5.55,06.79,08.50,08.52,7.35,4.32,1.94,0.61,0.04]

Y_2018 =  [0.01,0.19,1.15,3.15,5.82,07.79,08.37,07.84,7.05,4.21,2.15,0.71,0.09]

Y_2019 =  [0.01,0.19,1.17,2.98,5.93,07.49,08.57,08.27,6.88,4.44,2.46,0.71,0.09]





index = ["6.00","7.00","8.00","9.00","10.00","11.00","12.00","13.00","14.00","15.00","16.00","17.00","18.00",]



df = pd.DataFrame({'2010': Y_2010,'2011': Y_2011,'2012': Y_2012,'2013': Y_2013,'2014': Y_2014,

                   '2015': Y_2015,'2016': Y_2016,'2017': Y_2017,'2018': Y_2018, '2019': Y_2019}, index=index )



ax = df.plot.line(rot=90)

df.plot(figsize=(15,5))



plt.title("March Month UV-Indexes for past 10 Years 2010 - 2019", y=1.05);

plt.ylabel("UV Index", labelpad=14)

plt.xlabel("Time", labelpad=14)

# April Month UV-Indexes for past 10 Years 2010 - 2019



import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure





Y_2010 =  [0.00,0.06,0.82,2.38,4.08,5.91,6.81,6.55,4.84,3.07,1.34,0.27,0.00]

Y_2011 =  [0.00,0.07,0.86,2.53,4.81,6.74,7.81,7.21,5.16,3.07,1.28,0.27,0.00]

Y_2012 =  [0.00,0.10,0.91,2.70,4.95,6.83,7.37,6.61,5.35,3.42,1.47,0.32,0.00]

Y_2013 =  [0.00,0.04,0.74,2.29,4.36,6.73,6.86,5.98,5.05,3.24,1.27,0.24,0.00]

Y_2014 =  [0.00,0.13,0.95,2.79,4.90,6.54,6.74,6.13,5.54,3.49,1.51,0.36,0.01]

Y_2015 =  [0.00,0.12,0.99,2.99,5.44,7.10,7.82,6.93,5.93,3.94,1.60,0.35,0.01]

Y_2016 =  [0.00,0.10,0.93,2.74,5.62,6.76,7.82,6.70,6.00,3.68,1.59,0.33,0.00]

Y_2017 =  [0.00,0.12,0.99,2.98,5.31,7.79,7.49,6.68,5.79,3.73,1.57,0.31,0.03]

Y_2018 =  [0.01,0.15,0.99,2.91,4.94,7.15,7.41,6.29,4.86,3.02,1.56,0.38,0.02]

Y_2019 =  [0.01,0.15,0.98,2.77,5.16,6.80,7.62,7.08,5.05,3.59,1.57,0.39,0.02]





index = ["6.00","7.00","8.00","9.00","10.00","11.00","12.00","13.00","14.00","15.00","16.00","17.00","18.00",]



df = pd.DataFrame({'2010': Y_2010,'2011': Y_2011,'2012': Y_2012,'2013': Y_2013,'2014': Y_2014,

                   '2015': Y_2015,'2016': Y_2016,'2017': Y_2017,'2018': Y_2018, '2019': Y_2019}, index=index )



ax = df.plot.line(rot=90)

df.plot(figsize=(15,5))



plt.title("April Month UV-Indexes for past 10 Years 2010 - 2019", y=1.05);

plt.ylabel("UV Index", labelpad=14)

plt.xlabel("Time", labelpad=14)

# May Month UV-Indexes for past 10 Years 2010 - 2019



import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure





Y_2010 =  [0.00,0.02,0.57,2.03,3.90,5.38,5.73,5.44,4.09,2.49,0.87,0.11,0.00]

Y_2011 =  [0.00,0.03,0.57,1.95,3.90,5.44,5.96,5.57,4.09,2.51,0.87,0.11,0.00]

Y_2012 =  [0.00,0.01,0.47,1.65,3.17,4.65,5.10,4.70,3.67,2.02,0.79,0.09,0.00]

Y_2013 =  [0.00,0.06,0.60,2.02,3.90,5.35,6.12,5.69,4.06,2.28,0.98,0.16,0.00]

Y_2014 =  [0.00,0.07,0.67,2.02,4.01,5.32,5.91,5.15,3.79,2.15,0.89,0.17,0.00]

Y_2015 =  [0.00,0.06,0.73,2.29,4.15,6.14,6.51,6.26,4.77,2.72,1.04,0.18,0.00]

Y_2016 =  [0.00,0.04,0.60,2.03,3.70,4.94,5.43,5.69,4.22,2.72,0.95,0.14,0.00]

Y_2017 =  [0.00,0.07,0.67,2.15,4.05,5.24,5.16,4.84,4.02,2.18,0.81,0.11,0.00]

Y_2018 =  [0.01,0.08,0.67,1.98,3.87,5.31,5.79,4.60,3.63,2.28,0.98,0.19,0.01]

Y_2019 =  [0.01,0.08,0.61,1.96,3.58,4.80,5.61,5.39,3.79,2.23,0.89,0.19,0.01]





index = ["6.00","7.00","8.00","9.00","10.00","11.00","12.00","13.00","14.00","15.00","16.00","17.00","18.00",]



df = pd.DataFrame({'2010': Y_2010,'2011': Y_2011,'2012': Y_2012,'2013': Y_2013,'2014': Y_2014,

                   '2015': Y_2015,'2016': Y_2016,'2017': Y_2017,'2018': Y_2018, '2019': Y_2019}, index=index )



ax = df.plot.line(rot=90)

df.plot(figsize=(15,5))



plt.title("May Month UV-Indexes for past 10 Years 2010 - 2019", y=1.05);

plt.ylabel("UV Index", labelpad=14)

plt.xlabel("Time", labelpad=14)

# June Month UV-Indexes for past 10 Years 2010 - 2019



import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure





Y_2010 =  [0.00,0.00,0.42,1.57,3.27,4.89,5.44,4.95,3.72,1.98,0.74,0.08,0.00]

Y_2011 =  [0.00,0.00,0.38,1.47,3.10,4.52,4.92,4.56,3.43,1.88,0.71,0.08,0.00]

Y_2012 =  [0.00,0.00,0.30,1.22,2.74,4.23,4.61,4.14,3.09,1.80,0.65,0.07,0.00]

Y_2013 =  [0.00,0.04,0.52,1.83,3.73,5.37,6.40,5.52,4.34,2.54,0.97,0.15,0.01]

Y_2014 =  [0.00,0.03,0.42,1.36,2.73,4.01,4.59,4.18,3.08,1.82,0.73,0.14,0.00]

Y_2015 =  [0.00,0.02,0.46,1.60,3.17,4.49,5.07,4.54,3.22,1.89,0.80,0.13,0.00]

Y_2016 =  [0.00,0.01,0.42,1.59,3.39,4.85,5.35,4.78,3.82,2.34,0.85,0.11,0.00]

Y_2017 =  [0.01,0.05,0.48,1.56,3.10,4.47,4.85,4.48,3.46,1.86,0.68,0.11,0.00]

Y_2018 =  [0.01,0.04,0.43,1.48,2.97,4.04,4.86,4.32,3.53,2.03,0.81,0.16,0.01]

Y_2019 =  [0.01,0.05,0.47,1.59,3.18,4.71,5.29,4.68,3.60,2.12,0.87,0.17,0.01]





index = ["6.00","7.00","8.00","9.00","10.00","11.00","12.00","13.00","14.00","15.00","16.00","17.00","18.00",]



df = pd.DataFrame({'2010': Y_2010,'2011': Y_2011,'2012': Y_2012,'2013': Y_2013,'2014': Y_2014,

                   '2015': Y_2015,'2016': Y_2016,'2017': Y_2017,'2018': Y_2018, '2019': Y_2019}, index=index )



ax = df.plot.line(rot=90)

df.plot(figsize=(15,5))



plt.title("June Month UV-Indexes for past 10 Years 2010 - 2019", y=1.05);

plt.ylabel("UV Index", labelpad=14)

plt.xlabel("Time", labelpad=14)

# July Month UV-Indexes for past 10 Years 2010 - 2019



import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure





Y_2010 =  [0.00,0.00,0.38,1.58,3.14,4.85,5.62,5.14,4.04,2.43,1.03,0.15,0.00]

Y_2011 =  [0.00,0.00,0.36,1.48,3.25,4.59,5.17,4.93,3.92,2.24,0.92,0.14,0.00]

Y_2012 =  [0.00,0.00,0.29,1.30,2.82,4.04,5.26,5.22,3.73,2.27,0.92,0.13,0.00]

Y_2013 =  [0.01,0.03,0.44,1.61,3.45,5.07,6.16,6.10,5.06,2.94,1.15,0.20,0.01]

Y_2014 =  [0.00,0.03,0.49,1.72,3.33,5.12,5.69,5.66,4.25,2.62,1.08,0.23,0.00]

Y_2015 =  [0.00,0.01,0.46,1.74,3.55,5.24,6.11,5.85,4.34,2.73,1.14,0.22,0.00]

Y_2016 =  [0.00,0.01,0.42,1.64,3.48,5.12,6.15,5.90,4.19,2.71,1.03,0.18,0.00]

Y_2017 =  [0.01,0.05,0.44,1.48,2.90,4.28,4.83,4.67,3.82,2.35,1.06,0.23,0.02]

Y_2018 =  [0.01,0.04,0.44,1.58,3.02,4.68,5.45,5.30,4.10,2.44,1.00,0.23,0.01]

Y_2019 =  [0.01,0.05,0.47,1.68,3.39,4.82,5.76,4.92,4.06,2.59,1.11,0.25,0.01]





index = ["6.00","7.00","8.00","9.00","10.00","11.00","12.00","13.00","14.00","15.00","16.00","17.00","18.00",]



df = pd.DataFrame({'2010': Y_2010,'2011': Y_2011,'2012': Y_2012,'2013': Y_2013,'2014': Y_2014,

                   '2015': Y_2015,'2016': Y_2016,'2017': Y_2017,'2018': Y_2018, '2019': Y_2019}, index=index )



ax = df.plot.line(rot=90)

df.plot(figsize=(15,5))



plt.title("July Month UV-Indexes for past 10 Years 2010 - 2019", y=1.05);

plt.ylabel("UV Index", labelpad=14)

plt.xlabel("Time", labelpad=14)

# August Month UV-Indexes for past 10 Years 2010 - 2019



import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure





Y_2010 =  [0.00,0.02,0.55,1.98,4.06,5.53,6.72,6.44,5.15,3.26,1.23,0.24,0.00]

Y_2011 =  [0.00,0.02,0.59,1.97,3.84,5.28,6.54,6.30,5.05,2.89,1.16,0.21,0.00]

Y_2012 =  [0.00,0.01,0.55,1.98,4.19,6.08,7.03,6.54,5.65,3.32,1.32,0.24,0.00]

Y_2013 =  [0.01,0.11,0.87,2.58,5.01,7.03,8.07,7.82,6.29,3.89,1.72,0.42,0.02]

Y_2014 =  [0.00,0.08,0.75,2.35,4.47,6.26,7.19,6.73,5.34,3.42,1.44,0.33,0.01]

Y_2015 =  [0.00,0.06,0.69,2.34,4.76,6.90,8.28,8.04,6.55,3.72,1.60,0.32,0.01]

Y_2016 =  [0.00,0.05,0.69,2.45,4.40,6.29,7.61,7.57,5.51,3.49,1.46,0.29,0.00]

Y_2017 =  [0.01,0.09,0.68,2.05,3.82,5.45,6.40,6.32,5.00,3.10,1.34,0.34,0.02]

Y_2018 =  [0.01,0.09,0.73,2.19,4.16,6.11,7.18,6.78,5.42,3.43,1.50,0.36,0.02]

Y_2019 =  [0.01,0.10,0.77,2.35,4.43,6.49,7.67,6.86,5.37,3.20,1.51,0.38,0.02]





index = ["6.00","7.00","8.00","9.00","10.00","11.00","12.00","13.00","14.00","15.00","16.00","17.00","18.00",]



df = pd.DataFrame({'2010': Y_2010,'2011': Y_2011,'2012': Y_2012,'2013': Y_2013,'2014': Y_2014,

                   '2015': Y_2015,'2016': Y_2016,'2017': Y_2017,'2018': Y_2018, '2019': Y_2019}, index=index )



ax = df.plot.line(rot=90)

df.plot(figsize=(15,5))



plt.title("August Month UV-Indexes for past 10 Years 2010 - 2019", y=1.05);

plt.ylabel("UV Index", labelpad=14)

plt.xlabel("Time", labelpad=14)

# September Month UV-Indexes for past 10 Years 2010 - 2019



import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure





Y_2010 =  [0.00,0.15,1.21,2.89,5.50,7.61,8.45,8.19,6.40,3.97,1.74,0.36,0.00]

Y_2011 =  [0.00,0.15,1.05,2.93,5.51,7.16,8.37,7.80,6.33,3.63,1.43,0.29,0.00]

Y_2012 =  [0.00,0.14,1.12,3.03,5.59,8.09,9.23,8.36,6.60,4.18,1.68,0.33,0.00]

Y_2013 =  [0.01,0.32,1.47,3.64,6.56,8.24,9.16,8.92,7.16,4.04,2.01,0.53,0.04]

Y_2014 =  [0.00,0.26,1.34,3.32,6.17,7.81,8.76,8.49,6.78,4.39,1.84,0.45,0.02]

Y_2015 =  [0.00,0.20,1.30,3.36,6.10,8.20,9.45,8.98,6.84,4.20,1.68,0.38,0.01]

Y_2016 =  [0.00,0.20,1.29,3.30,5.58,7.77,8.87,8.42,6.63,4.44,1.86,0.38,0.00]

Y_2017 =  [0.01,0.25,1.18,2.93,5.27,7.03,8.17,7.86,6.01,3.64,1.61,0.42,0.03]

Y_2018 =  [0.01,0.26,1.26,3.09,5.59,7.72,8.69,7.82,6.40,3.94,1.74,0.45,0.03]

Y_2019 =  [0.01,0.30,1.47,3.67,6.49,8.32,9.51,9.18,7.29,4.47,1.97,0.52,0.03]





index = ["6.00","7.00","8.00","9.00","10.00","11.00","12.00","13.00","14.00","15.00","16.00","17.00","18.00",]



df = pd.DataFrame({'2010': Y_2010,'2011': Y_2011,'2012': Y_2012,'2013': Y_2013,'2014': Y_2014,

                   '2015': Y_2015,'2016': Y_2016,'2017': Y_2017,'2018': Y_2018, '2019': Y_2019}, index=index )



ax = df.plot.line(rot=90)

df.plot(figsize=(15,5))



plt.title("September Month UV-Indexes for past 10 Years 2010 - 2019", y=1.05);

plt.ylabel("UV Index", labelpad=14)

plt.xlabel("Time", labelpad=14)

# October Month UV-Indexes for past 10 Years 2010 - 2019



import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure





Y_2010 =  [0.01,0.45,1.88,4.26,6.77,08.72,08.96,08.36,7.00,4.20,1.70,0.42,0.00]

Y_2011 =  [0.00,0.41,1.51,3.48,6.12,07.95,09.21,08.22,6.70,4.15,1.60,0.36,0.00]

Y_2012 =  [0.00,0.39,1.80,4.13,6.77,08.33,10.36,08.76,7.00,4.45,1.92,0.43,0.00]

Y_2013 =  [0.06,0.67,2.10,4.63,7.42,09.38,11.33,10.53,8.02,5.34,2.41,0.66,0.06]

Y_2014 =  [0.04,0.56,1.97,4.32,7.13,09.12,10.15,09.12,7.52,4.60,2.00,0.52,0.03]

Y_2015 =  [0.02,0.48,1.86,4.24,6.59,09.13,10.27,09.57,7.38,4.58,1.99,0.46,0.02]

Y_2016 =  [0.02,0.55,2.08,4.58,7.53,10.74,11.30,10.93,8.24,5.01,2.17,0.50,0.01]

Y_2017 =  [0.05,0.49,1.55,3.44,6.17,07.92,08.46,07.64,6.03,3.74,1.70,0.46,0.04]

Y_2018 =  [0.05,0.50,1.74,3.86,5.74,08.15,09.25,09.25,6.67,4.03,1.83,0.51,0.04]

Y_2019 =  [0.05,0.58,1.97,4.15,6.46,08.76,10.02,10.13,7.76,4.67,2.03,0.56,0.05]





index = ["6.00","7.00","8.00","9.00","10.00","11.00","12.00","13.00","14.00","15.00","16.00","17.00","18.00",]



df = pd.DataFrame({'2010': Y_2010,'2011': Y_2011,'2012': Y_2012,'2013': Y_2013,'2014': Y_2014,

                   '2015': Y_2015,'2016': Y_2016,'2017': Y_2017,'2018': Y_2018, '2019': Y_2019}, index=index )



ax = df.plot.line(rot=90)

df.plot(figsize=(15,5))



plt.title("October Month UV-Indexes for past 10 Years 2010 - 2019", y=1.05);

plt.ylabel("UV Index", labelpad=14)

plt.xlabel("Time", labelpad=14)

# November Month UV-Indexes for past 10 Years 2010 - 2019



import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure





Y_2010 =  [0.02,0.58,2.15,4.12,6.19,07.50,07.09,07.43,4.83,3.34,1.62,0.43,0.01]

Y_2011 =  [0.03,0.66,2.08,4.28,7.59,09.31,09.47,09.80,7.99,5.35,2.38,0.64,0.02]

Y_2012 =  [0.02,0.60,2.20,4.82,7.27,10.05,11.73,11.11,9.18,5.73,2.46,0.63,0.01]

Y_2013 =  [0.11,0.81,2.48,5.21,8.35,10.33,10.28,10.21,7.57,5.23,2.27,0.72,0.09]

Y_2014 =  [0.08,0.72,2.18,4.74,7.27,08.41,09.96,09.96,7.72,4.93,2.35,0.69,0.07]

Y_2015 =  [0.05,0.63,2.01,4.98,7.53,10.91,11.88,10.57,8.71,5.22,2.45,0.67,0.05]

Y_2016 =  [0.06,0.73,2.41,5.03,8.29,10.24,11.54,11.14,8.72,5.54,2.58,0.68,0.04]

Y_2017 =  [0.11,0.78,2.26,4.23,7.48,08.10,08.71,08.64,6.93,4.28,2.03,0.65,0.09]

Y_2018 =  [0.11,0.81,2.24,4.92,7.49,09.75,10.29,10.08,7.73,4.89,2.31,0.74,0.10]

Y_2019 =  [0.11,0.81,2.40,4.83,8.01,10.20,11.63,11.14,8.69,5.44,2.59,0.81,0.11]





index = ["6.00","7.00","8.00","9.00","10.00","11.00","12.00","13.00","14.00","15.00","16.00","17.00","18.00",]



df = pd.DataFrame({'2010': Y_2010,'2011': Y_2011,'2012': Y_2012,'2013': Y_2013,'2014': Y_2014,

                   '2015': Y_2015,'2016': Y_2016,'2017': Y_2017,'2018': Y_2018, '2019': Y_2019}, index=index )



ax = df.plot.line(rot=90)

df.plot(figsize=(15,5))



plt.title("November Month UV-Indexes for past 10 Years 2010 - 2019", y=1.05);

plt.ylabel("UV Index", labelpad=14)

plt.xlabel("Time", labelpad=14)

# December Month UV-Indexes for past 10 Years 2010 - 2019



import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure





Y_2010 =  [0.02,0.50,1.93,4.21,6.26,07.69,08.70,07.96,5.98,4.34,2.34,0.67,0.09]

Y_2011 =  [0.01,0.50,2.03,4.52,7.19,08.94,09.61,09.82,8.32,5.30,2.63,0.93,0.13]

Y_2012 =  [0.01,0.56,2.07,4.71,7.74,08.35,09.54,10.59,8.83,5.81,2.99,1.02,0.10]

Y_2013 =  [0.11,0.80,2.52,5.23,8.41,10.14,12.25,11.12,9.35,6.25,3.27,1.15,0.21]

Y_2014 =  [0.07,0.65,1.97,4.53,7.28,09.48,10.72,10.94,8.95,5.57,2.91,1.02,0.16]

Y_2015 =  [0.05,0.60,2.10,4.81,7.42,09.57,11.17,11.63,8.83,5.92,3.00,0.93,0.12]

Y_2016 =  [0.05,0.65,2.14,4.44,6.91,09.83,10.69,10.41,8.43,6.27,3.11,1.03,0.14]

Y_2017 =  [0.10,0.77,2.37,4.70,7.59,09.99,11.74,11.47,9.20,6.09,3.15,1.19,0.23]

Y_2018 =  [0.09,0.64,2.04,4.15,6.09,08.17,07.93,08.21,7.21,5.01,2.63,1.00,0.18]

Y_2019 =  [0.10,0.72,2.10,4.70,7.33,09.88,10.64,10.85,8.63,6.10,3.03,1.08,0.20]





index = ["6.00","7.00","8.00","9.00","10.00","11.00","12.00","13.00","14.00","15.00","16.00","17.00","18.00",]



df = pd.DataFrame({'2010': Y_2010,'2011': Y_2011,'2012': Y_2012,'2013': Y_2013,'2014': Y_2014,

                   '2015': Y_2015,'2016': Y_2016,'2017': Y_2017,'2018': Y_2018, '2019': Y_2019}, index=index )



ax = df.plot.line(rot=90)

df.plot(figsize=(15,5))



plt.title("December Month UV-Indexes for past 10 Years 2010 - 2019", y=1.05);

plt.ylabel("UV Index", labelpad=14)

plt.xlabel("Time", labelpad=14)

# Highest UV occurrences in the past 10 years.



import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure



data = {'Year': [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019], 

        'count':[18.6, 16.76, 19.64, 17.79, 19.55, 17.41, 17.72, 18.96, 16.22, 17.72]}



# new_ds = pd.DataFrame.from_dict(data)

new_ds = pd.DataFrame(data)

print(new_ds)



# new_ds['Percentage Value'] = ((new_ds['count'] / new_ds['count'].sum()) * 100).round(2)

# print(new_ds)



new_ds.plot(kind='bar',ylim =(0,20), x='Year',y='count', color='green')

plt.title("Highest UV occurrences in the past 10 years", y=1.05);

plt.ylabel("UV Index", labelpad=14)
# Extreme UV Index times in the past 10 years. above 11



import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure



data = {'Year': [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019], 

        'Time':[11.56, 11.57, 11.23, 12.31,13.09, 12.21, 12.52, 12.09, 12.01, 12.23]}



new_ds = pd.DataFrame(data)

print(new_ds)



# new_ds['Percentage Value'] = ((new_ds['count'] / new_ds['count'].sum()) * 100).round(2)

# print(new_ds)

# datetick('x', 'HH:MM PM')



new_ds.plot(figsize=(8,5), kind='line',ylim =(9,18), x='Year',y='Time', color='purple', 

            xticks= np.arange(2010, 2020, 1), yticks= np.arange(9, 19, 1))

plt.title("Extreme UV occurrences times in the past 10 years", y=1.05);

plt.ylabel("Time", labelpad=14)
import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2019 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2019.csv')



filteredData1 = uv2019.Date == "01/01/2019"

print((uv2019[filteredData1])['UV_Index'].max())



filteredData2 = uv2019.Date == "02/01/2019"

print ((uv2019[filteredData2])['UV_Index'].max())



filteredData3 = uv2019.Date == "03/01/2019"

print ((uv2019[filteredData3])['UV_Index'].max())



filteredData4 = uv2019.Date == "04/01/2019"

print ((uv2019[filteredData4])['UV_Index'].max())



filteredData5 = uv2019.Date == "05/01/2019"

print ((uv2019[filteredData5])['UV_Index'].max())



filteredData6 = uv2019.Date == "06/01/2019"

print ((uv2019[filteredData6])['UV_Index'].max())



filteredData7 = uv2019.Date == "07/01/2019"

print ((uv2019[filteredData7])['UV_Index'].max())



filteredData8 = uv2019.Date == "08/01/2019"

print ((uv2019[filteredData8])['UV_Index'].max())



filteredData9 = uv2019.Date == "09/01/2019"

print ((uv2019[filteredData9])['UV_Index'].max())



filteredData10 = uv2019.Date == "10/01/2019"

print ((uv2019[filteredData10])['UV_Index'].max())



filteredData11 = uv2019.Date == "11/01/2019"

print ((uv2019[filteredData11])['UV_Index'].max())



filteredData12 = uv2019.Date == "12/01/2019"

print ((uv2019[filteredData12])['UV_Index'].max())



filteredData13 = uv2019.Date == "13/01/2019"

print ((uv2019[filteredData13])['UV_Index'].max())



filteredData14 = uv2019.Date == "14/01/2019"

print ((uv2019[filteredData14])['UV_Index'].max())



filteredData15 = uv2019.Date == "15/01/2019"

print ((uv2019[filteredData15])['UV_Index'].max())



filteredData16 = uv2019.Date == "16/01/2019"

print ((uv2019[filteredData16])['UV_Index'].max())



filteredData17 = uv2019.Date == "17/01/2019"

print ((uv2019[filteredData17])['UV_Index'].max())



filteredData18 = uv2019.Date == "18/01/2019"

print ((uv2019[filteredData18])['UV_Index'].max())



filteredData19 = uv2019.Date == "19/01/2019"

print ((uv2019[filteredData19])['UV_Index'].max())



filteredData20 = uv2019.Date == "20/01/2019"

print ((uv2019[filteredData20])['UV_Index'].max())



filteredData21 = uv2019.Date == "21/01/2019"

print ((uv2019[filteredData21])['UV_Index'].max())



filteredData22 = uv2019.Date == "22/01/2019"

print ((uv2019[filteredData22])['UV_Index'].max())



filteredData23 = uv2019.Date == "23/01/2019"

print ((uv2019[filteredData23])['UV_Index'].max())



filteredData24 = uv2019.Date == "24/01/2019"

print ((uv2019[filteredData24])['UV_Index'].max())



filteredData25 = uv2019.Date == "25/01/2019"

print ((uv2019[filteredData25])['UV_Index'].max())



filteredData26 = uv2019.Date == "26/01/2019"

print ((uv2019[filteredData26])['UV_Index'].max())



filteredData27 = uv2019.Date == "27/01/2019"

print ((uv2019[filteredData27])['UV_Index'].max())



filteredData28 = uv2019.Date == "28/01/2019"

print ((uv2019[filteredData28])['UV_Index'].max())



filteredData29 = uv2019.Date == "29/01/2019"

print ((uv2019[filteredData29])['UV_Index'].max())



filteredData30 = uv2019.Date == "30/01/2019"

print ((uv2019[filteredData30])['UV_Index'].max())



filteredData31 = uv2019.Date == "31/01/2019"

print ((uv2019[filteredData22])['UV_Index'].max())
import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2018 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2018.csv')



filteredData1 = uv2018.Date == "01/01/2018"

print((uv2018[filteredData1])['UV_Index'].max())



filteredData2 = uv2018.Date == "02/01/2018"

print ((uv2018[filteredData2])['UV_Index'].max())



filteredData3 = uv2018.Date == "03/01/2018"

print ((uv2018[filteredData3])['UV_Index'].max())



filteredData4 = uv2018.Date == "04/01/2018"

print ((uv2018[filteredData4])['UV_Index'].max())



filteredData5 = uv2018.Date == "05/01/2018"

print ((uv2018[filteredData5])['UV_Index'].max())



filteredData6 = uv2018.Date == "06/01/2018"

print ((uv2018[filteredData6])['UV_Index'].max())



filteredData7 = uv2018.Date == "07/01/2018"

print ((uv2018[filteredData7])['UV_Index'].max())



filteredData8 = uv2018.Date == "08/01/2018"

print ((uv2018[filteredData8])['UV_Index'].max())



filteredData9 = uv2018.Date == "09/01/2018"

print ((uv2018[filteredData9])['UV_Index'].max())



filteredData10 = uv2018.Date == "10/01/2018"

print ((uv2018[filteredData10])['UV_Index'].max())



filteredData11 = uv2018.Date == "11/01/2018"

print ((uv2018[filteredData11])['UV_Index'].max())



filteredData12 = uv2018.Date == "12/01/2018"

print ((uv2018[filteredData12])['UV_Index'].max())



filteredData13 = uv2018.Date == "13/01/2018"

print ((uv2018[filteredData13])['UV_Index'].max())



filteredData14 = uv2018.Date == "14/01/2018"

print ((uv2018[filteredData14])['UV_Index'].max())



filteredData15 = uv2018.Date == "15/01/2018"

print ((uv2018[filteredData15])['UV_Index'].max())



filteredData16 = uv2018.Date == "16/01/2018"

print ((uv2018[filteredData16])['UV_Index'].max())



filteredData17 = uv2018.Date == "17/01/2018"

print ((uv2018[filteredData17])['UV_Index'].max())



filteredData18 = uv2018.Date == "18/01/2018"

print ((uv2018[filteredData18])['UV_Index'].max())



filteredData19 = uv2018.Date == "19/01/2018"

print ((uv2018[filteredData19])['UV_Index'].max())



filteredData20 = uv2018.Date == "20/01/2019"

print ((uv2018[filteredData20])['UV_Index'].max())



filteredData21 = uv2018.Date == "21/01/2018"

print ((uv2018[filteredData21])['UV_Index'].max())



filteredData22 = uv2018.Date == "22/01/2018"

print ((uv2018[filteredData22])['UV_Index'].max())



filteredData23 = uv2018.Date == "23/01/2018"

print ((uv2018[filteredData23])['UV_Index'].max())



filteredData24 = uv2018.Date == "24/01/2018"

print ((uv2018[filteredData24])['UV_Index'].max())



filteredData25 = uv2018.Date == "25/01/2018"

print ((uv2018[filteredData25])['UV_Index'].max())



filteredData26 = uv2018.Date == "26/01/2018"

print ((uv2018[filteredData26])['UV_Index'].max())



filteredData27 = uv2018.Date == "27/01/2018"

print ((uv2018[filteredData27])['UV_Index'].max())



filteredData28 = uv2018.Date == "28/01/2018"

print ((uv2018[filteredData28])['UV_Index'].max())



filteredData29 = uv2018.Date == "29/01/2018"

print ((uv2018[filteredData29])['UV_Index'].max())



filteredData30 = uv2018.Date == "30/01/2018"

print ((uv2018[filteredData30])['UV_Index'].max())



filteredData31 = uv2018.Date == "31/01/2018"

print ((uv2018[filteredData22])['UV_Index'].max())
import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2017 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2017.csv')



filteredData1 = uv2017.Date == "01/01/2017"

print((uv2017[filteredData1])['UV_Index'].max())



filteredData2 = uv2017.Date == "02/01/2017"

print ((uv2017[filteredData2])['UV_Index'].max())



filteredData3 = uv2017.Date == "03/01/2017"

print ((uv2017[filteredData3])['UV_Index'].max())



filteredData4 = uv2017.Date == "04/01/2017"

print ((uv2017[filteredData4])['UV_Index'].max())



filteredData5 = uv2017.Date == "05/01/2017"

print ((uv2017[filteredData5])['UV_Index'].max())



filteredData6 = uv2017.Date == "06/01/2017"

print ((uv2017[filteredData6])['UV_Index'].max())



filteredData7 = uv2017.Date == "07/01/2017"

print ((uv2017[filteredData7])['UV_Index'].max())



filteredData8 = uv2017.Date == "08/01/2017"

print ((uv2017[filteredData8])['UV_Index'].max())



filteredData9 = uv2017.Date == "09/01/2017"

print ((uv2017[filteredData9])['UV_Index'].max())



filteredData10 = uv2017.Date == "10/01/2017"

print ((uv2017[filteredData10])['UV_Index'].max())



filteredData11 = uv2017.Date == "11/01/2017"

print ((uv2017[filteredData11])['UV_Index'].max())



filteredData12 = uv2017.Date == "12/01/2017"

print ((uv2017[filteredData12])['UV_Index'].max())



filteredData13 = uv2017.Date == "13/01/2017"

print ((uv2017[filteredData13])['UV_Index'].max())



filteredData14 = uv2017.Date == "14/01/2017"

print ((uv2017[filteredData14])['UV_Index'].max())



filteredData15 = uv2017.Date == "15/01/2017"

print ((uv2017[filteredData15])['UV_Index'].max())



filteredData16 = uv2017.Date == "16/01/2017"

print ((uv2017[filteredData16])['UV_Index'].max())



filteredData17 = uv2017.Date == "17/01/2017"

print ((uv2017[filteredData17])['UV_Index'].max())



filteredData18 = uv2017.Date == "18/01/2017"

print ((uv2017[filteredData18])['UV_Index'].max())



filteredData19 = uv2017.Date == "19/01/2017"

print ((uv2017[filteredData19])['UV_Index'].max())



filteredData20 = uv2017.Date == "20/01/2019"

print ((uv2017[filteredData20])['UV_Index'].max())



filteredData21 = uv2017.Date == "21/01/2017"

print ((uv2017[filteredData21])['UV_Index'].max())



filteredData22 = uv2017.Date == "22/01/2017"

print ((uv2017[filteredData22])['UV_Index'].max())



filteredData23 = uv2017.Date == "23/01/2017"

print ((uv2017[filteredData23])['UV_Index'].max())



filteredData24 = uv2017.Date == "24/01/2017"

print ((uv2017[filteredData24])['UV_Index'].max())



filteredData25 = uv2017.Date == "25/01/2017"

print ((uv2017[filteredData25])['UV_Index'].max())



filteredData26 = uv2017.Date == "26/01/2017"

print ((uv2017[filteredData26])['UV_Index'].max())



filteredData27 = uv2017.Date == "27/01/2017"

print ((uv2017[filteredData27])['UV_Index'].max())



filteredData28 = uv2017.Date == "28/01/2017"

print ((uv2017[filteredData28])['UV_Index'].max())



filteredData29 = uv2017.Date == "29/01/2017"

print ((uv2017[filteredData29])['UV_Index'].max())



filteredData30 = uv2017.Date == "30/01/2017"

print ((uv2017[filteredData30])['UV_Index'].max())



filteredData31 = uv2017.Date == "31/01/2017"

print ((uv2017[filteredData22])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2016 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2016.csv')



filteredData1 = uv2016.Date == "01/01/2016"

print((uv2016[filteredData1])['UV_Index'].max())



filteredData2 = uv2016.Date == "02/01/2016"

print ((uv2016[filteredData2])['UV_Index'].max())



filteredData3 = uv2016.Date == "03/01/2016"

print ((uv2016[filteredData3])['UV_Index'].max())



filteredData4 = uv2016.Date == "04/01/2016"

print ((uv2016[filteredData4])['UV_Index'].max())



filteredData5 = uv2016.Date == "05/01/2016"

print ((uv2016[filteredData5])['UV_Index'].max())



filteredData6 = uv2016.Date == "06/01/2016"

print ((uv2016[filteredData6])['UV_Index'].max())



filteredData7 = uv2016.Date == "07/01/2016"

print ((uv2016[filteredData7])['UV_Index'].max())



filteredData8 = uv2016.Date == "08/01/2016"

print ((uv2016[filteredData8])['UV_Index'].max())



filteredData9 = uv2016.Date == "09/01/2016"

print ((uv2016[filteredData9])['UV_Index'].max())



filteredData10 = uv2016.Date == "10/01/2016"

print ((uv2016[filteredData10])['UV_Index'].max())



filteredData11 = uv2016.Date == "11/01/2016"

print ((uv2016[filteredData11])['UV_Index'].max())



filteredData12 = uv2016.Date == "12/01/2016"

print ((uv2016[filteredData12])['UV_Index'].max())



filteredData13 = uv2016.Date == "13/01/2016"

print ((uv2016[filteredData13])['UV_Index'].max())



filteredData14 = uv2016.Date == "14/01/2016"

print ((uv2016[filteredData14])['UV_Index'].max())



filteredData15 = uv2016.Date == "15/01/2016"

print ((uv2016[filteredData15])['UV_Index'].max())



filteredData16 = uv2016.Date == "16/01/2016"

print ((uv2016[filteredData16])['UV_Index'].max())



filteredData17 = uv2016.Date == "17/01/2016"

print ((uv2016[filteredData17])['UV_Index'].max())



filteredData18 = uv2016.Date == "18/01/2016"

print ((uv2016[filteredData18])['UV_Index'].max())



filteredData19 = uv2016.Date == "19/01/2016"

print ((uv2016[filteredData19])['UV_Index'].max())



filteredData20 = uv2016.Date == "20/01/2019"

print ((uv2016[filteredData20])['UV_Index'].max())



filteredData21 = uv2016.Date == "21/01/2016"

print ((uv2016[filteredData21])['UV_Index'].max())



filteredData22 = uv2016.Date == "22/01/2016"

print ((uv2016[filteredData22])['UV_Index'].max())



filteredData23 = uv2016.Date == "23/01/2016"

print ((uv2016[filteredData23])['UV_Index'].max())



filteredData24 = uv2016.Date == "24/01/2016"

print ((uv2016[filteredData24])['UV_Index'].max())



filteredData25 = uv2016.Date == "25/01/2016"

print ((uv2016[filteredData25])['UV_Index'].max())



filteredData26 = uv2016.Date == "26/01/2016"

print ((uv2016[filteredData26])['UV_Index'].max())



filteredData27 = uv2016.Date == "27/01/2016"

print ((uv2016[filteredData27])['UV_Index'].max())



filteredData28 = uv2016.Date == "28/01/2016"

print ((uv2016[filteredData28])['UV_Index'].max())



filteredData29 = uv2016.Date == "29/01/2016"

print ((uv2016[filteredData29])['UV_Index'].max())



filteredData30 = uv2016.Date == "30/01/2016"

print ((uv2016[filteredData30])['UV_Index'].max())



filteredData31 = uv2016.Date == "31/01/2016"

print ((uv2016[filteredData22])['UV_Index'].max())



import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2015 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2015.csv')



filteredData1 = uv2015.Date == "01/01/2015"

print((uv2015[filteredData1])['UV_Index'].max())



filteredData2 = uv2015.Date == "02/01/2015"

print ((uv2015[filteredData2])['UV_Index'].max())



filteredData3 = uv2015.Date == "03/01/2015"

print ((uv2015[filteredData3])['UV_Index'].max())



filteredData4 = uv2015.Date == "04/01/2015"

print ((uv2015[filteredData4])['UV_Index'].max())



filteredData5 = uv2015.Date == "05/01/2015"

print ((uv2015[filteredData5])['UV_Index'].max())



filteredData6 = uv2015.Date == "06/01/2015"

print ((uv2015[filteredData6])['UV_Index'].max())



filteredData7 = uv2015.Date == "07/01/2015"

print ((uv2015[filteredData7])['UV_Index'].max())



filteredData8 = uv2015.Date == "08/01/2015"

print ((uv2015[filteredData8])['UV_Index'].max())



filteredData9 = uv2015.Date == "09/01/2015"

print ((uv2015[filteredData9])['UV_Index'].max())



filteredData10 = uv2015.Date == "10/01/2015"

print ((uv2015[filteredData10])['UV_Index'].max())



filteredData11 = uv2015.Date == "11/01/2015"

print ((uv2015[filteredData11])['UV_Index'].max())



filteredData12 = uv2015.Date == "12/01/2015"

print ((uv2015[filteredData12])['UV_Index'].max())



filteredData13 = uv2015.Date == "13/01/2015"

print ((uv2015[filteredData13])['UV_Index'].max())



filteredData14 = uv2015.Date == "14/01/2015"

print ((uv2015[filteredData14])['UV_Index'].max())



filteredData15 = uv2015.Date == "15/01/2015"

print ((uv2015[filteredData15])['UV_Index'].max())



filteredData16 = uv2015.Date == "16/01/2015"

print ((uv2015[filteredData16])['UV_Index'].max())



filteredData17 = uv2015.Date == "17/01/2015"

print ((uv2015[filteredData17])['UV_Index'].max())



filteredData18 = uv2015.Date == "18/01/2015"

print ((uv2015[filteredData18])['UV_Index'].max())



filteredData19 = uv2015.Date == "19/01/2015"

print ((uv2015[filteredData19])['UV_Index'].max())



filteredData20 = uv2015.Date == "20/01/2019"

print ((uv2015[filteredData20])['UV_Index'].max())



filteredData21 = uv2015.Date == "21/01/2015"

print ((uv2015[filteredData21])['UV_Index'].max())



filteredData22 = uv2015.Date == "22/01/2015"

print ((uv2015[filteredData22])['UV_Index'].max())



filteredData23 = uv2015.Date == "23/01/2015"

print ((uv2015[filteredData23])['UV_Index'].max())



filteredData24 = uv2015.Date == "24/01/2015"

print ((uv2015[filteredData24])['UV_Index'].max())



filteredData25 = uv2015.Date == "25/01/2015"

print ((uv2015[filteredData25])['UV_Index'].max())



filteredData26 = uv2015.Date == "26/01/2015"

print ((uv2015[filteredData26])['UV_Index'].max())



filteredData27 = uv2015.Date == "27/01/2015"

print ((uv2015[filteredData27])['UV_Index'].max())



filteredData28 = uv2015.Date == "28/01/2015"

print ((uv2015[filteredData28])['UV_Index'].max())



filteredData29 = uv2015.Date == "29/01/2015"

print ((uv2015[filteredData29])['UV_Index'].max())



filteredData30 = uv2015.Date == "30/01/2015"

print ((uv2015[filteredData30])['UV_Index'].max())



filteredData31 = uv2015.Date == "31/01/2015"

print ((uv2015[filteredData22])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2014 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2014.csv')



filteredData1 = uv2014.Date == "01/01/2014"

print((uv2014[filteredData1])['UV_Index'].max())



filteredData2 = uv2014.Date == "02/01/2014"

print ((uv2014[filteredData2])['UV_Index'].max())



filteredData3 = uv2014.Date == "03/01/2014"

print ((uv2014[filteredData3])['UV_Index'].max())



filteredData4 = uv2014.Date == "04/01/2014"

print ((uv2014[filteredData4])['UV_Index'].max())



filteredData5 = uv2014.Date == "05/01/2014"

print ((uv2014[filteredData5])['UV_Index'].max())



filteredData6 = uv2014.Date == "06/01/2014"

print ((uv2014[filteredData6])['UV_Index'].max())



filteredData7 = uv2014.Date == "07/01/2014"

print ((uv2014[filteredData7])['UV_Index'].max())



filteredData8 = uv2014.Date == "08/01/2014"

print ((uv2014[filteredData8])['UV_Index'].max())



filteredData9 = uv2014.Date == "09/01/2014"

print ((uv2014[filteredData9])['UV_Index'].max())



filteredData10 = uv2014.Date == "10/01/2014"

print ((uv2014[filteredData10])['UV_Index'].max())



filteredData11 = uv2014.Date == "11/01/2014"

print ((uv2014[filteredData11])['UV_Index'].max())



filteredData12 = uv2014.Date == "12/01/2014"

print ((uv2014[filteredData12])['UV_Index'].max())



filteredData13 = uv2014.Date == "13/01/2014"

print ((uv2014[filteredData13])['UV_Index'].max())



filteredData14 = uv2014.Date == "14/01/2014"

print ((uv2014[filteredData14])['UV_Index'].max())



filteredData15 = uv2014.Date == "15/01/2014"

print ((uv2014[filteredData15])['UV_Index'].max())



filteredData16 = uv2014.Date == "16/01/2014"

print ((uv2014[filteredData16])['UV_Index'].max())



filteredData17 = uv2014.Date == "17/01/2014"

print ((uv2014[filteredData17])['UV_Index'].max())



filteredData18 = uv2014.Date == "18/01/2014"

print ((uv2014[filteredData18])['UV_Index'].max())



filteredData19 = uv2014.Date == "19/01/2014"

print ((uv2014[filteredData19])['UV_Index'].max())



filteredData20 = uv2014.Date == "20/01/2019"

print ((uv2014[filteredData20])['UV_Index'].max())



filteredData21 = uv2014.Date == "21/01/2014"

print ((uv2014[filteredData21])['UV_Index'].max())



filteredData22 = uv2014.Date == "22/01/2014"

print ((uv2014[filteredData22])['UV_Index'].max())



filteredData23 = uv2014.Date == "23/01/2014"

print ((uv2014[filteredData23])['UV_Index'].max())



filteredData24 = uv2014.Date == "24/01/2014"

print ((uv2014[filteredData24])['UV_Index'].max())



filteredData25 = uv2014.Date == "25/01/2014"

print ((uv2014[filteredData25])['UV_Index'].max())



filteredData26 = uv2014.Date == "26/01/2014"

print ((uv2014[filteredData26])['UV_Index'].max())



filteredData27 = uv2014.Date == "27/01/2014"

print ((uv2014[filteredData27])['UV_Index'].max())



filteredData28 = uv2014.Date == "28/01/2014"

print ((uv2014[filteredData28])['UV_Index'].max())



filteredData29 = uv2014.Date == "29/01/2014"

print ((uv2014[filteredData29])['UV_Index'].max())



filteredData30 = uv2014.Date == "30/01/2014"

print ((uv2014[filteredData30])['UV_Index'].max())



filteredData31 = uv2014.Date == "31/01/2014"

print ((uv2014[filteredData22])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2013 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2013.csv')



filteredData1 = uv2013.Date == "01/01/2013"

print((uv2013[filteredData1])['UV_Index'].max())



filteredData2 = uv2013.Date == "02/01/2013"

print ((uv2013[filteredData2])['UV_Index'].max())



filteredData3 = uv2013.Date == "03/01/2013"

print ((uv2013[filteredData3])['UV_Index'].max())



filteredData4 = uv2013.Date == "04/01/2013"

print ((uv2013[filteredData4])['UV_Index'].max())



filteredData5 = uv2013.Date == "05/01/2013"

print ((uv2013[filteredData5])['UV_Index'].max())



filteredData6 = uv2013.Date == "06/01/2013"

print ((uv2013[filteredData6])['UV_Index'].max())



filteredData7 = uv2013.Date == "07/01/2013"

print ((uv2013[filteredData7])['UV_Index'].max())



filteredData8 = uv2013.Date == "08/01/2013"

print ((uv2013[filteredData8])['UV_Index'].max())



filteredData9 = uv2013.Date == "09/01/2013"

print ((uv2013[filteredData9])['UV_Index'].max())



filteredData10 = uv2013.Date == "10/01/2013"

print ((uv2013[filteredData10])['UV_Index'].max())



filteredData11 = uv2013.Date == "11/01/2013"

print ((uv2013[filteredData11])['UV_Index'].max())



filteredData12 = uv2013.Date == "12/01/2013"

print ((uv2013[filteredData12])['UV_Index'].max())



filteredData13 = uv2013.Date == "13/01/2013"

print ((uv2013[filteredData13])['UV_Index'].max())



filteredData14 = uv2013.Date == "14/01/2013"

print ((uv2013[filteredData14])['UV_Index'].max())



filteredData15 = uv2013.Date == "15/01/2013"

print ((uv2013[filteredData15])['UV_Index'].max())



filteredData16 = uv2013.Date == "16/01/2013"

print ((uv2013[filteredData16])['UV_Index'].max())



filteredData17 = uv2013.Date == "17/01/2013"

print ((uv2013[filteredData17])['UV_Index'].max())



filteredData18 = uv2013.Date == "18/01/2013"

print ((uv2013[filteredData18])['UV_Index'].max())



filteredData19 = uv2013.Date == "19/01/2013"

print ((uv2013[filteredData19])['UV_Index'].max())



filteredData20 = uv2013.Date == "20/01/2019"

print ((uv2013[filteredData20])['UV_Index'].max())



filteredData21 = uv2013.Date == "21/01/2013"

print ((uv2013[filteredData21])['UV_Index'].max())



filteredData22 = uv2013.Date == "22/01/2013"

print ((uv2013[filteredData22])['UV_Index'].max())



filteredData23 = uv2013.Date == "23/01/2013"

print ((uv2013[filteredData23])['UV_Index'].max())



filteredData24 = uv2013.Date == "24/01/2013"

print ((uv2013[filteredData24])['UV_Index'].max())



filteredData25 = uv2013.Date == "25/01/2013"

print ((uv2013[filteredData25])['UV_Index'].max())



filteredData26 = uv2013.Date == "26/01/2013"

print ((uv2013[filteredData26])['UV_Index'].max())



filteredData27 = uv2013.Date == "27/01/2013"

print ((uv2013[filteredData27])['UV_Index'].max())



filteredData28 = uv2013.Date == "28/01/2013"

print ((uv2013[filteredData28])['UV_Index'].max())



filteredData29 = uv2013.Date == "29/01/2013"

print ((uv2013[filteredData29])['UV_Index'].max())



filteredData30 = uv2013.Date == "30/01/2013"

print ((uv2013[filteredData30])['UV_Index'].max())



filteredData31 = uv2013.Date == "31/01/2013"

print ((uv2013[filteredData22])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2012 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2012.csv')



filteredData1 = uv2012.Date == "01/01/2012"

print((uv2012[filteredData1])['UV_Index'].max())



filteredData2 = uv2012.Date == "02/01/2012"

print ((uv2012[filteredData2])['UV_Index'].max())



filteredData3 = uv2012.Date == "03/01/2012"

print ((uv2012[filteredData3])['UV_Index'].max())



filteredData4 = uv2012.Date == "04/01/2012"

print ((uv2012[filteredData4])['UV_Index'].max())



filteredData5 = uv2012.Date == "05/01/2012"

print ((uv2012[filteredData5])['UV_Index'].max())



filteredData6 = uv2012.Date == "06/01/2012"

print ((uv2012[filteredData6])['UV_Index'].max())



filteredData7 = uv2012.Date == "07/01/2012"

print ((uv2012[filteredData7])['UV_Index'].max())



filteredData8 = uv2012.Date == "08/01/2012"

print ((uv2012[filteredData8])['UV_Index'].max())



filteredData9 = uv2012.Date == "09/01/2012"

print ((uv2012[filteredData9])['UV_Index'].max())



filteredData10 = uv2012.Date == "10/01/2012"

print ((uv2012[filteredData10])['UV_Index'].max())



filteredData11 = uv2012.Date == "11/01/2012"

print ((uv2012[filteredData11])['UV_Index'].max())



filteredData12 = uv2012.Date == "12/01/2012"

print ((uv2012[filteredData12])['UV_Index'].max())



filteredData13 = uv2012.Date == "13/01/2012"

print ((uv2012[filteredData13])['UV_Index'].max())



filteredData14 = uv2012.Date == "14/01/2012"

print ((uv2012[filteredData14])['UV_Index'].max())



filteredData15 = uv2012.Date == "15/01/2012"

print ((uv2012[filteredData15])['UV_Index'].max())



filteredData16 = uv2012.Date == "16/01/2012"

print ((uv2012[filteredData16])['UV_Index'].max())



filteredData17 = uv2012.Date == "17/01/2012"

print ((uv2012[filteredData17])['UV_Index'].max())



filteredData18 = uv2012.Date == "18/01/2012"

print ((uv2012[filteredData18])['UV_Index'].max())



filteredData19 = uv2012.Date == "19/01/2012"

print ((uv2012[filteredData19])['UV_Index'].max())



filteredData20 = uv2012.Date == "20/01/2019"

print ((uv2012[filteredData20])['UV_Index'].max())



filteredData21 = uv2012.Date == "21/01/2012"

print ((uv2012[filteredData21])['UV_Index'].max())



filteredData22 = uv2012.Date == "22/01/2012"

print ((uv2012[filteredData22])['UV_Index'].max())



filteredData23 = uv2012.Date == "23/01/2012"

print ((uv2012[filteredData23])['UV_Index'].max())



filteredData24 = uv2012.Date == "24/01/2012"

print ((uv2012[filteredData24])['UV_Index'].max())



filteredData25 = uv2012.Date == "25/01/2012"

print ((uv2012[filteredData25])['UV_Index'].max())



filteredData26 = uv2012.Date == "26/01/2012"

print ((uv2012[filteredData26])['UV_Index'].max())



filteredData27 = uv2012.Date == "27/01/2012"

print ((uv2012[filteredData27])['UV_Index'].max())



filteredData28 = uv2012.Date == "28/01/2012"

print ((uv2012[filteredData28])['UV_Index'].max())



filteredData29 = uv2012.Date == "29/01/2012"

print ((uv2012[filteredData29])['UV_Index'].max())



filteredData30 = uv2012.Date == "30/01/2012"

print ((uv2012[filteredData30])['UV_Index'].max())



filteredData31 = uv2012.Date == "31/01/2012"

print ((uv2012[filteredData22])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2011 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2011.csv')



filteredData1 = uv2011.Date == "01/01/2011"

print((uv2011[filteredData1])['UV_Index'].max())



filteredData2 = uv2011.Date == "02/01/2011"

print ((uv2011[filteredData2])['UV_Index'].max())



filteredData3 = uv2011.Date == "03/01/2011"

print ((uv2011[filteredData3])['UV_Index'].max())



filteredData4 = uv2011.Date == "04/01/2011"

print ((uv2011[filteredData4])['UV_Index'].max())



filteredData5 = uv2011.Date == "05/01/2011"

print ((uv2011[filteredData5])['UV_Index'].max())



filteredData6 = uv2011.Date == "06/01/2011"

print ((uv2011[filteredData6])['UV_Index'].max())



filteredData7 = uv2011.Date == "07/01/2011"

print ((uv2011[filteredData7])['UV_Index'].max())



filteredData8 = uv2011.Date == "08/01/2011"

print ((uv2011[filteredData8])['UV_Index'].max())



filteredData9 = uv2011.Date == "09/01/2011"

print ((uv2011[filteredData9])['UV_Index'].max())



filteredData10 = uv2011.Date == "10/01/2011"

print ((uv2011[filteredData10])['UV_Index'].max())



filteredData11 = uv2011.Date == "11/01/2011"

print ((uv2011[filteredData11])['UV_Index'].max())



filteredData12 = uv2011.Date == "12/01/2011"

print ((uv2011[filteredData12])['UV_Index'].max())



filteredData13 = uv2011.Date == "13/01/2011"

print ((uv2011[filteredData13])['UV_Index'].max())



filteredData14 = uv2011.Date == "14/01/2011"

print ((uv2011[filteredData14])['UV_Index'].max())



filteredData15 = uv2011.Date == "15/01/2011"

print ((uv2011[filteredData15])['UV_Index'].max())



filteredData16 = uv2011.Date == "16/01/2011"

print ((uv2011[filteredData16])['UV_Index'].max())



filteredData17 = uv2011.Date == "17/01/2011"

print ((uv2011[filteredData17])['UV_Index'].max())



filteredData18 = uv2011.Date == "18/01/2011"

print ((uv2011[filteredData18])['UV_Index'].max())



filteredData19 = uv2011.Date == "19/01/2011"

print ((uv2011[filteredData19])['UV_Index'].max())



filteredData20 = uv2011.Date == "20/01/2019"

print ((uv2011[filteredData20])['UV_Index'].max())



filteredData21 = uv2011.Date == "21/01/2011"

print ((uv2011[filteredData21])['UV_Index'].max())



filteredData22 = uv2011.Date == "22/01/2011"

print ((uv2011[filteredData22])['UV_Index'].max())



filteredData23 = uv2011.Date == "23/01/2011"

print ((uv2011[filteredData23])['UV_Index'].max())



filteredData24 = uv2011.Date == "24/01/2011"

print ((uv2011[filteredData24])['UV_Index'].max())



filteredData25 = uv2011.Date == "25/01/2011"

print ((uv2011[filteredData25])['UV_Index'].max())



filteredData26 = uv2011.Date == "26/01/2011"

print ((uv2011[filteredData26])['UV_Index'].max())



filteredData27 = uv2011.Date == "27/01/2011"

print ((uv2011[filteredData27])['UV_Index'].max())



filteredData28 = uv2011.Date == "28/01/2011"

print ((uv2011[filteredData28])['UV_Index'].max())



filteredData29 = uv2011.Date == "29/01/2011"

print ((uv2011[filteredData29])['UV_Index'].max())



filteredData30 = uv2011.Date == "30/01/2011"

print ((uv2011[filteredData30])['UV_Index'].max())



filteredData31 = uv2011.Date == "31/01/2011"

print ((uv2011[filteredData22])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2010 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2010.csv')



filteredData1 = uv2010.Date == "01/01/2010"

print((uv2010[filteredData1])['UV_Index'].max())



filteredData2 = uv2010.Date == "02/01/2010"

print ((uv2010[filteredData2])['UV_Index'].max())



filteredData3 = uv2010.Date == "03/01/2010"

print ((uv2010[filteredData3])['UV_Index'].max())



filteredData4 = uv2010.Date == "04/01/2010"

print ((uv2010[filteredData4])['UV_Index'].max())



filteredData5 = uv2010.Date == "05/01/2010"

print ((uv2010[filteredData5])['UV_Index'].max())



filteredData6 = uv2010.Date == "06/01/2010"

print ((uv2010[filteredData6])['UV_Index'].max())



filteredData7 = uv2010.Date == "07/01/2010"

print ((uv2010[filteredData7])['UV_Index'].max())



filteredData8 = uv2010.Date == "08/01/2010"

print ((uv2010[filteredData8])['UV_Index'].max())



filteredData9 = uv2010.Date == "09/01/2010"

print ((uv2010[filteredData9])['UV_Index'].max())



filteredData10 = uv2010.Date == "10/01/2010"

print ((uv2010[filteredData10])['UV_Index'].max())



filteredData11 = uv2010.Date == "11/01/2010"

print ((uv2010[filteredData11])['UV_Index'].max())



filteredData12 = uv2010.Date == "12/01/2010"

print ((uv2010[filteredData12])['UV_Index'].max())



filteredData13 = uv2010.Date == "13/01/2010"

print ((uv2010[filteredData13])['UV_Index'].max())



filteredData14 = uv2010.Date == "14/01/2010"

print ((uv2010[filteredData14])['UV_Index'].max())



filteredData15 = uv2010.Date == "15/01/2010"

print ((uv2010[filteredData15])['UV_Index'].max())



filteredData16 = uv2010.Date == "16/01/2010"

print ((uv2010[filteredData16])['UV_Index'].max())



filteredData17 = uv2010.Date == "17/01/2010"

print ((uv2010[filteredData17])['UV_Index'].max())



filteredData18 = uv2010.Date == "18/01/2010"

print ((uv2010[filteredData18])['UV_Index'].max())



filteredData19 = uv2010.Date == "19/01/2010"

print ((uv2010[filteredData19])['UV_Index'].max())



filteredData20 = uv2010.Date == "20/01/2019"

print ((uv2010[filteredData20])['UV_Index'].max())



filteredData21 = uv2010.Date == "21/01/2010"

print ((uv2010[filteredData21])['UV_Index'].max())



filteredData22 = uv2010.Date == "22/01/2010"

print ((uv2010[filteredData22])['UV_Index'].max())



filteredData23 = uv2010.Date == "23/01/2010"

print ((uv2010[filteredData23])['UV_Index'].max())



filteredData24 = uv2010.Date == "24/01/2010"

print ((uv2010[filteredData24])['UV_Index'].max())



filteredData25 = uv2010.Date == "25/01/2010"

print ((uv2010[filteredData25])['UV_Index'].max())



filteredData26 = uv2010.Date == "26/01/2010"

print ((uv2010[filteredData26])['UV_Index'].max())



filteredData27 = uv2010.Date == "27/01/2010"

print ((uv2010[filteredData27])['UV_Index'].max())



filteredData28 = uv2010.Date == "28/01/2010"

print ((uv2010[filteredData28])['UV_Index'].max())



filteredData29 = uv2010.Date == "29/01/2010"

print ((uv2010[filteredData29])['UV_Index'].max())



filteredData30 = uv2010.Date == "30/01/2010"

print ((uv2010[filteredData30])['UV_Index'].max())



filteredData31 = uv2010.Date == "31/01/2010"

print ((uv2010[filteredData22])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2010 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2010.csv')





filteredData1 = uv2010.Date == "01/02/2010"

print((uv2010[filteredData1])['UV_Index'].max())



filteredData2 = uv2010.Date == "02/02/2010"

print ((uv2010[filteredData2])['UV_Index'].max())



filteredData3 = uv2010.Date == "03/02/2010"

print ((uv2010[filteredData3])['UV_Index'].max())



filteredData4 = uv2010.Date == "04/02/2010"

print ((uv2010[filteredData4])['UV_Index'].max())



filteredData5 = uv2010.Date == "05/02/2010"

print ((uv2010[filteredData5])['UV_Index'].max())



filteredData6 = uv2010.Date == "06/02/2010"

print ((uv2010[filteredData6])['UV_Index'].max())



filteredData7 = uv2010.Date == "07/02/2010"

print ((uv2010[filteredData7])['UV_Index'].max())



filteredData8 = uv2010.Date == "08/02/2010"

print ((uv2010[filteredData8])['UV_Index'].max())



filteredData9 = uv2010.Date == "09/02/2010"

print ((uv2010[filteredData9])['UV_Index'].max())



filteredData10 = uv2010.Date == "10/02/2010"

print ((uv2010[filteredData10])['UV_Index'].max())



filteredData11 = uv2010.Date == "11/02/2010"

print ((uv2010[filteredData11])['UV_Index'].max())



filteredData12 = uv2010.Date == "12/02/2010"

print ((uv2010[filteredData12])['UV_Index'].max())



filteredData13 = uv2010.Date == "13/02/2010"

print ((uv2010[filteredData13])['UV_Index'].max())



filteredData14 = uv2010.Date == "14/02/2010"

print ((uv2010[filteredData14])['UV_Index'].max())



filteredData15 = uv2010.Date == "15/02/2010"

print ((uv2010[filteredData15])['UV_Index'].max())



filteredData16 = uv2010.Date == "16/02/2010"

print ((uv2010[filteredData16])['UV_Index'].max())



filteredData17 = uv2010.Date == "17/02/2010"

print ((uv2010[filteredData17])['UV_Index'].max())



filteredData18 = uv2010.Date == "18/02/2010"

print ((uv2010[filteredData18])['UV_Index'].max())



filteredData19 = uv2010.Date == "19/02/2010"

print ((uv2010[filteredData19])['UV_Index'].max())



filteredData20 = uv2010.Date == "20/02/2029"

print ((uv2010[filteredData20])['UV_Index'].max())



filteredData21 = uv2010.Date == "21/02/2010"

print ((uv2010[filteredData21])['UV_Index'].max())



filteredData22 = uv2010.Date == "22/02/2010"

print ((uv2010[filteredData22])['UV_Index'].max())



filteredData23 = uv2010.Date == "23/02/2010"

print ((uv2010[filteredData23])['UV_Index'].max())



filteredData24 = uv2010.Date == "24/02/2010"

print ((uv2010[filteredData24])['UV_Index'].max())



filteredData25 = uv2010.Date == "25/02/2010"

print ((uv2010[filteredData25])['UV_Index'].max())



filteredData26 = uv2010.Date == "26/02/2010"

print ((uv2010[filteredData26])['UV_Index'].max())



filteredData27 = uv2010.Date == "27/02/2010"

print ((uv2010[filteredData27])['UV_Index'].max())



filteredData28 = uv2010.Date == "28/02/2010"

print ((uv2010[filteredData28])['UV_Index'].max())



filteredData29 = uv2010.Date == "29/02/2010"

print ((uv2010[filteredData29])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2011 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2011.csv')



filteredData1 = uv2011.Date == "01/02/2011"

print((uv2011[filteredData1])['UV_Index'].max())



filteredData2 = uv2011.Date == "02/02/2011"

print ((uv2011[filteredData2])['UV_Index'].max())



filteredData3 = uv2011.Date == "03/02/2011"

print ((uv2011[filteredData3])['UV_Index'].max())



filteredData4 = uv2011.Date == "04/02/2011"

print ((uv2011[filteredData4])['UV_Index'].max())



filteredData5 = uv2011.Date == "05/02/2011"

print ((uv2011[filteredData5])['UV_Index'].max())



filteredData6 = uv2011.Date == "06/02/2011"

print ((uv2011[filteredData6])['UV_Index'].max())



filteredData7 = uv2011.Date == "07/02/2011"

print ((uv2011[filteredData7])['UV_Index'].max())



filteredData8 = uv2011.Date == "08/02/2011"

print ((uv2011[filteredData8])['UV_Index'].max())



filteredData9 = uv2011.Date == "09/02/2011"

print ((uv2011[filteredData9])['UV_Index'].max())



filteredData10 = uv2011.Date == "10/02/2011"

print ((uv2011[filteredData10])['UV_Index'].max())



filteredData11 = uv2011.Date == "11/02/2011"

print ((uv2011[filteredData11])['UV_Index'].max())



filteredData12 = uv2011.Date == "12/02/2011"

print ((uv2011[filteredData12])['UV_Index'].max())



filteredData13 = uv2011.Date == "13/02/2011"

print ((uv2011[filteredData13])['UV_Index'].max())



filteredData14 = uv2011.Date == "14/02/2011"

print ((uv2011[filteredData14])['UV_Index'].max())



filteredData15 = uv2011.Date == "15/02/2011"

print ((uv2011[filteredData15])['UV_Index'].max())



filteredData16 = uv2011.Date == "16/02/2011"

print ((uv2011[filteredData16])['UV_Index'].max())



filteredData17 = uv2011.Date == "17/02/2011"

print ((uv2011[filteredData17])['UV_Index'].max())



filteredData18 = uv2011.Date == "18/02/2011"

print ((uv2011[filteredData18])['UV_Index'].max())



filteredData19 = uv2011.Date == "19/02/2011"

print ((uv2011[filteredData19])['UV_Index'].max())



filteredData20 = uv2011.Date == "20/02/2029"

print ((uv2011[filteredData20])['UV_Index'].max())



filteredData21 = uv2011.Date == "21/02/2011"

print ((uv2011[filteredData21])['UV_Index'].max())



filteredData22 = uv2011.Date == "22/02/2011"

print ((uv2011[filteredData22])['UV_Index'].max())



filteredData23 = uv2011.Date == "23/02/2011"

print ((uv2011[filteredData23])['UV_Index'].max())



filteredData24 = uv2011.Date == "24/02/2011"

print ((uv2011[filteredData24])['UV_Index'].max())



filteredData25 = uv2011.Date == "25/02/2011"

print ((uv2011[filteredData25])['UV_Index'].max())



filteredData26 = uv2011.Date == "26/02/2011"

print ((uv2011[filteredData26])['UV_Index'].max())



filteredData27 = uv2011.Date == "27/02/2011"

print ((uv2011[filteredData27])['UV_Index'].max())



filteredData28 = uv2011.Date == "28/02/2011"

print ((uv2011[filteredData28])['UV_Index'].max())



filteredData29 = uv2011.Date == "29/02/2011"

print ((uv2011[filteredData29])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2012 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2012.csv')



filteredData1 = uv2012.Date == "01/02/2012"

print((uv2012[filteredData1])['UV_Index'].max())



filteredData2 = uv2012.Date == "02/02/2012"

print ((uv2012[filteredData2])['UV_Index'].max())



filteredData3 = uv2012.Date == "03/02/2012"

print ((uv2012[filteredData3])['UV_Index'].max())



filteredData4 = uv2012.Date == "04/02/2012"

print ((uv2012[filteredData4])['UV_Index'].max())



filteredData5 = uv2012.Date == "05/02/2012"

print ((uv2012[filteredData5])['UV_Index'].max())



filteredData6 = uv2012.Date == "06/02/2012"

print ((uv2012[filteredData6])['UV_Index'].max())



filteredData7 = uv2012.Date == "07/02/2012"

print ((uv2012[filteredData7])['UV_Index'].max())



filteredData8 = uv2012.Date == "08/02/2012"

print ((uv2012[filteredData8])['UV_Index'].max())



filteredData9 = uv2012.Date == "09/02/2012"

print ((uv2012[filteredData9])['UV_Index'].max())



filteredData10 = uv2012.Date == "10/02/2012"

print ((uv2012[filteredData10])['UV_Index'].max())



filteredData11 = uv2012.Date == "11/02/2012"

print ((uv2012[filteredData11])['UV_Index'].max())



filteredData12 = uv2012.Date == "12/02/2012"

print ((uv2012[filteredData12])['UV_Index'].max())



filteredData13 = uv2012.Date == "13/02/2012"

print ((uv2012[filteredData13])['UV_Index'].max())



filteredData14 = uv2012.Date == "14/02/2012"

print ((uv2012[filteredData14])['UV_Index'].max())



filteredData15 = uv2012.Date == "15/02/2012"

print ((uv2012[filteredData15])['UV_Index'].max())



filteredData16 = uv2012.Date == "16/02/2012"

print ((uv2012[filteredData16])['UV_Index'].max())



filteredData17 = uv2012.Date == "17/02/2012"

print ((uv2012[filteredData17])['UV_Index'].max())



filteredData18 = uv2012.Date == "18/02/2012"

print ((uv2012[filteredData18])['UV_Index'].max())



filteredData19 = uv2012.Date == "19/02/2012"

print ((uv2012[filteredData19])['UV_Index'].max())



filteredData20 = uv2012.Date == "20/02/2029"

print ((uv2012[filteredData20])['UV_Index'].max())



filteredData21 = uv2012.Date == "21/02/2012"

print ((uv2012[filteredData21])['UV_Index'].max())



filteredData22 = uv2012.Date == "22/02/2012"

print ((uv2012[filteredData22])['UV_Index'].max())



filteredData23 = uv2012.Date == "23/02/2012"

print ((uv2012[filteredData23])['UV_Index'].max())



filteredData24 = uv2012.Date == "24/02/2012"

print ((uv2012[filteredData24])['UV_Index'].max())



filteredData25 = uv2012.Date == "25/02/2012"

print ((uv2012[filteredData25])['UV_Index'].max())



filteredData26 = uv2012.Date == "26/02/2012"

print ((uv2012[filteredData26])['UV_Index'].max())



filteredData27 = uv2012.Date == "27/02/2012"

print ((uv2012[filteredData27])['UV_Index'].max())



filteredData28 = uv2012.Date == "28/02/2012"

print ((uv2012[filteredData28])['UV_Index'].max())



filteredData29 = uv2012.Date == "29/02/2012"

print ((uv2012[filteredData29])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2013 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2013.csv')



filteredData1 = uv2013.Date == "01/02/2013"

print((uv2013[filteredData1])['UV_Index'].max())



filteredData2 = uv2013.Date == "02/02/2013"

print ((uv2013[filteredData2])['UV_Index'].max())



filteredData3 = uv2013.Date == "03/02/2013"

print ((uv2013[filteredData3])['UV_Index'].max())



filteredData4 = uv2013.Date == "04/02/2013"

print ((uv2013[filteredData4])['UV_Index'].max())



filteredData5 = uv2013.Date == "05/02/2013"

print ((uv2013[filteredData5])['UV_Index'].max())



filteredData6 = uv2013.Date == "06/02/2013"

print ((uv2013[filteredData6])['UV_Index'].max())



filteredData7 = uv2013.Date == "07/02/2013"

print ((uv2013[filteredData7])['UV_Index'].max())



filteredData8 = uv2013.Date == "08/02/2013"

print ((uv2013[filteredData8])['UV_Index'].max())



filteredData9 = uv2013.Date == "09/02/2013"

print ((uv2013[filteredData9])['UV_Index'].max())



filteredData10 = uv2013.Date == "10/02/2013"

print ((uv2013[filteredData10])['UV_Index'].max())



filteredData11 = uv2013.Date == "11/02/2013"

print ((uv2013[filteredData11])['UV_Index'].max())



filteredData12 = uv2013.Date == "12/02/2013"

print ((uv2013[filteredData12])['UV_Index'].max())



filteredData13 = uv2013.Date == "13/02/2013"

print ((uv2013[filteredData13])['UV_Index'].max())



filteredData14 = uv2013.Date == "14/02/2013"

print ((uv2013[filteredData14])['UV_Index'].max())



filteredData15 = uv2013.Date == "15/02/2013"

print ((uv2013[filteredData15])['UV_Index'].max())



filteredData16 = uv2013.Date == "16/02/2013"

print ((uv2013[filteredData16])['UV_Index'].max())



filteredData17 = uv2013.Date == "17/02/2013"

print ((uv2013[filteredData17])['UV_Index'].max())



filteredData18 = uv2013.Date == "18/02/2013"

print ((uv2013[filteredData18])['UV_Index'].max())



filteredData19 = uv2013.Date == "19/02/2013"

print ((uv2013[filteredData19])['UV_Index'].max())



filteredData20 = uv2013.Date == "20/02/2029"

print ((uv2013[filteredData20])['UV_Index'].max())



filteredData21 = uv2013.Date == "21/02/2013"

print ((uv2013[filteredData21])['UV_Index'].max())



filteredData22 = uv2013.Date == "22/02/2013"

print ((uv2013[filteredData22])['UV_Index'].max())



filteredData23 = uv2013.Date == "23/02/2013"

print ((uv2013[filteredData23])['UV_Index'].max())



filteredData24 = uv2013.Date == "24/02/2013"

print ((uv2013[filteredData24])['UV_Index'].max())



filteredData25 = uv2013.Date == "25/02/2013"

print ((uv2013[filteredData25])['UV_Index'].max())



filteredData26 = uv2013.Date == "26/02/2013"

print ((uv2013[filteredData26])['UV_Index'].max())



filteredData27 = uv2013.Date == "27/02/2013"

print ((uv2013[filteredData27])['UV_Index'].max())



filteredData28 = uv2013.Date == "28/02/2013"

print ((uv2013[filteredData28])['UV_Index'].max())



filteredData29 = uv2013.Date == "29/02/2013"

print ((uv2013[filteredData29])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2014 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2014.csv')



filteredData1 = uv2014.Date == "01/02/2014"

print((uv2014[filteredData1])['UV_Index'].max())



filteredData2 = uv2014.Date == "02/02/2014"

print ((uv2014[filteredData2])['UV_Index'].max())



filteredData3 = uv2014.Date == "03/02/2014"

print ((uv2014[filteredData3])['UV_Index'].max())



filteredData4 = uv2014.Date == "04/02/2014"

print ((uv2014[filteredData4])['UV_Index'].max())



filteredData5 = uv2014.Date == "05/02/2014"

print ((uv2014[filteredData5])['UV_Index'].max())



filteredData6 = uv2014.Date == "06/02/2014"

print ((uv2014[filteredData6])['UV_Index'].max())



filteredData7 = uv2014.Date == "07/02/2014"

print ((uv2014[filteredData7])['UV_Index'].max())



filteredData8 = uv2014.Date == "08/02/2014"

print ((uv2014[filteredData8])['UV_Index'].max())



filteredData9 = uv2014.Date == "09/02/2014"

print ((uv2014[filteredData9])['UV_Index'].max())



filteredData10 = uv2014.Date == "10/02/2014"

print ((uv2014[filteredData10])['UV_Index'].max())



filteredData11 = uv2014.Date == "11/02/2014"

print ((uv2014[filteredData11])['UV_Index'].max())



filteredData12 = uv2014.Date == "12/02/2014"

print ((uv2014[filteredData12])['UV_Index'].max())



filteredData13 = uv2014.Date == "13/02/2014"

print ((uv2014[filteredData13])['UV_Index'].max())



filteredData14 = uv2014.Date == "14/02/2014"

print ((uv2014[filteredData14])['UV_Index'].max())



filteredData15 = uv2014.Date == "15/02/2014"

print ((uv2014[filteredData15])['UV_Index'].max())



filteredData16 = uv2014.Date == "16/02/2014"

print ((uv2014[filteredData16])['UV_Index'].max())



filteredData17 = uv2014.Date == "17/02/2014"

print ((uv2014[filteredData17])['UV_Index'].max())



filteredData18 = uv2014.Date == "18/02/2014"

print ((uv2014[filteredData18])['UV_Index'].max())



filteredData19 = uv2014.Date == "19/02/2014"

print ((uv2014[filteredData19])['UV_Index'].max())



filteredData20 = uv2014.Date == "20/02/2029"

print ((uv2014[filteredData20])['UV_Index'].max())



filteredData21 = uv2014.Date == "21/02/2014"

print ((uv2014[filteredData21])['UV_Index'].max())



filteredData22 = uv2014.Date == "22/02/2014"

print ((uv2014[filteredData22])['UV_Index'].max())



filteredData23 = uv2014.Date == "23/02/2014"

print ((uv2014[filteredData23])['UV_Index'].max())



filteredData24 = uv2014.Date == "24/02/2014"

print ((uv2014[filteredData24])['UV_Index'].max())



filteredData25 = uv2014.Date == "25/02/2014"

print ((uv2014[filteredData25])['UV_Index'].max())



filteredData26 = uv2014.Date == "26/02/2014"

print ((uv2014[filteredData26])['UV_Index'].max())



filteredData27 = uv2014.Date == "27/02/2014"

print ((uv2014[filteredData27])['UV_Index'].max())



filteredData28 = uv2014.Date == "28/02/2014"

print ((uv2014[filteredData28])['UV_Index'].max())



filteredData29 = uv2014.Date == "29/02/2014"

print ((uv2014[filteredData29])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2015 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2015.csv')



filteredData1 = uv2015.Date == "01/02/2015"

print((uv2015[filteredData1])['UV_Index'].max())



filteredData2 = uv2015.Date == "02/02/2015"

print ((uv2015[filteredData2])['UV_Index'].max())



filteredData3 = uv2015.Date == "03/02/2015"

print ((uv2015[filteredData3])['UV_Index'].max())



filteredData4 = uv2015.Date == "04/02/2015"

print ((uv2015[filteredData4])['UV_Index'].max())



filteredData5 = uv2015.Date == "05/02/2015"

print ((uv2015[filteredData5])['UV_Index'].max())



filteredData6 = uv2015.Date == "06/02/2015"

print ((uv2015[filteredData6])['UV_Index'].max())



filteredData7 = uv2015.Date == "07/02/2015"

print ((uv2015[filteredData7])['UV_Index'].max())



filteredData8 = uv2015.Date == "08/02/2015"

print ((uv2015[filteredData8])['UV_Index'].max())



filteredData9 = uv2015.Date == "09/02/2015"

print ((uv2015[filteredData9])['UV_Index'].max())



filteredData10 = uv2015.Date == "10/02/2015"

print ((uv2015[filteredData10])['UV_Index'].max())



filteredData11 = uv2015.Date == "11/02/2015"

print ((uv2015[filteredData11])['UV_Index'].max())



filteredData12 = uv2015.Date == "12/02/2015"

print ((uv2015[filteredData12])['UV_Index'].max())



filteredData13 = uv2015.Date == "13/02/2015"

print ((uv2015[filteredData13])['UV_Index'].max())



filteredData14 = uv2015.Date == "14/02/2015"

print ((uv2015[filteredData14])['UV_Index'].max())



filteredData15 = uv2015.Date == "15/02/2015"

print ((uv2015[filteredData15])['UV_Index'].max())



filteredData16 = uv2015.Date == "16/02/2015"

print ((uv2015[filteredData16])['UV_Index'].max())



filteredData17 = uv2015.Date == "17/02/2015"

print ((uv2015[filteredData17])['UV_Index'].max())



filteredData18 = uv2015.Date == "18/02/2015"

print ((uv2015[filteredData18])['UV_Index'].max())



filteredData19 = uv2015.Date == "19/02/2015"

print ((uv2015[filteredData19])['UV_Index'].max())



filteredData20 = uv2015.Date == "20/02/2029"

print ((uv2015[filteredData20])['UV_Index'].max())



filteredData21 = uv2015.Date == "21/02/2015"

print ((uv2015[filteredData21])['UV_Index'].max())



filteredData22 = uv2015.Date == "22/02/2015"

print ((uv2015[filteredData22])['UV_Index'].max())



filteredData23 = uv2015.Date == "23/02/2015"

print ((uv2015[filteredData23])['UV_Index'].max())



filteredData24 = uv2015.Date == "24/02/2015"

print ((uv2015[filteredData24])['UV_Index'].max())



filteredData25 = uv2015.Date == "25/02/2015"

print ((uv2015[filteredData25])['UV_Index'].max())



filteredData26 = uv2015.Date == "26/02/2015"

print ((uv2015[filteredData26])['UV_Index'].max())



filteredData27 = uv2015.Date == "27/02/2015"

print ((uv2015[filteredData27])['UV_Index'].max())



filteredData28 = uv2015.Date == "28/02/2015"

print ((uv2015[filteredData28])['UV_Index'].max())



filteredData29 = uv2015.Date == "29/02/2015"

print ((uv2015[filteredData29])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2016 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2016.csv')



filteredData1 = uv2016.Date == "01/02/2016"

print((uv2016[filteredData1])['UV_Index'].max())



filteredData2 = uv2016.Date == "02/02/2016"

print ((uv2016[filteredData2])['UV_Index'].max())



filteredData3 = uv2016.Date == "03/02/2016"

print ((uv2016[filteredData3])['UV_Index'].max())



filteredData4 = uv2016.Date == "04/02/2016"

print ((uv2016[filteredData4])['UV_Index'].max())



filteredData5 = uv2016.Date == "05/02/2016"

print ((uv2016[filteredData5])['UV_Index'].max())



filteredData6 = uv2016.Date == "06/02/2016"

print ((uv2016[filteredData6])['UV_Index'].max())



filteredData7 = uv2016.Date == "07/02/2016"

print ((uv2016[filteredData7])['UV_Index'].max())



filteredData8 = uv2016.Date == "08/02/2016"

print ((uv2016[filteredData8])['UV_Index'].max())



filteredData9 = uv2016.Date == "09/02/2016"

print ((uv2016[filteredData9])['UV_Index'].max())



filteredData10 = uv2016.Date == "10/02/2016"

print ((uv2016[filteredData10])['UV_Index'].max())



filteredData11 = uv2016.Date == "11/02/2016"

print ((uv2016[filteredData11])['UV_Index'].max())



filteredData12 = uv2016.Date == "12/02/2016"

print ((uv2016[filteredData12])['UV_Index'].max())



filteredData13 = uv2016.Date == "13/02/2016"

print ((uv2016[filteredData13])['UV_Index'].max())



filteredData14 = uv2016.Date == "14/02/2016"

print ((uv2016[filteredData14])['UV_Index'].max())



filteredData15 = uv2016.Date == "15/02/2016"

print ((uv2016[filteredData15])['UV_Index'].max())



filteredData16 = uv2016.Date == "16/02/2016"

print ((uv2016[filteredData16])['UV_Index'].max())



filteredData17 = uv2016.Date == "17/02/2016"

print ((uv2016[filteredData17])['UV_Index'].max())



filteredData18 = uv2016.Date == "18/02/2016"

print ((uv2016[filteredData18])['UV_Index'].max())



filteredData19 = uv2016.Date == "19/02/2016"

print ((uv2016[filteredData19])['UV_Index'].max())



filteredData20 = uv2016.Date == "20/02/2029"

print ((uv2016[filteredData20])['UV_Index'].max())



filteredData21 = uv2016.Date == "21/02/2016"

print ((uv2016[filteredData21])['UV_Index'].max())



filteredData22 = uv2016.Date == "22/02/2016"

print ((uv2016[filteredData22])['UV_Index'].max())



filteredData23 = uv2016.Date == "23/02/2016"

print ((uv2016[filteredData23])['UV_Index'].max())



filteredData24 = uv2016.Date == "24/02/2016"

print ((uv2016[filteredData24])['UV_Index'].max())



filteredData25 = uv2016.Date == "25/02/2016"

print ((uv2016[filteredData25])['UV_Index'].max())



filteredData26 = uv2016.Date == "26/02/2016"

print ((uv2016[filteredData26])['UV_Index'].max())



filteredData27 = uv2016.Date == "27/02/2016"

print ((uv2016[filteredData27])['UV_Index'].max())



filteredData28 = uv2016.Date == "28/02/2016"

print ((uv2016[filteredData28])['UV_Index'].max())



filteredData29 = uv2016.Date == "29/02/2016"

print ((uv2016[filteredData29])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2017 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2017.csv')



filteredData1 = uv2017.Date == "01/02/2017"

print((uv2017[filteredData1])['UV_Index'].max())



filteredData2 = uv2017.Date == "02/02/2017"

print ((uv2017[filteredData2])['UV_Index'].max())



filteredData3 = uv2017.Date == "03/02/2017"

print ((uv2017[filteredData3])['UV_Index'].max())



filteredData4 = uv2017.Date == "04/02/2017"

print ((uv2017[filteredData4])['UV_Index'].max())



filteredData5 = uv2017.Date == "05/02/2017"

print ((uv2017[filteredData5])['UV_Index'].max())



filteredData6 = uv2017.Date == "06/02/2017"

print ((uv2017[filteredData6])['UV_Index'].max())



filteredData7 = uv2017.Date == "07/02/2017"

print ((uv2017[filteredData7])['UV_Index'].max())



filteredData8 = uv2017.Date == "08/02/2017"

print ((uv2017[filteredData8])['UV_Index'].max())



filteredData9 = uv2017.Date == "09/02/2017"

print ((uv2017[filteredData9])['UV_Index'].max())



filteredData10 = uv2017.Date == "10/02/2017"

print ((uv2017[filteredData10])['UV_Index'].max())



filteredData11 = uv2017.Date == "11/02/2017"

print ((uv2017[filteredData11])['UV_Index'].max())



filteredData12 = uv2017.Date == "12/02/2017"

print ((uv2017[filteredData12])['UV_Index'].max())



filteredData13 = uv2017.Date == "13/02/2017"

print ((uv2017[filteredData13])['UV_Index'].max())



filteredData14 = uv2017.Date == "14/02/2017"

print ((uv2017[filteredData14])['UV_Index'].max())



filteredData15 = uv2017.Date == "15/02/2017"

print ((uv2017[filteredData15])['UV_Index'].max())



filteredData16 = uv2017.Date == "16/02/2017"

print ((uv2017[filteredData16])['UV_Index'].max())



filteredData17 = uv2017.Date == "17/02/2017"

print ((uv2017[filteredData17])['UV_Index'].max())



filteredData18 = uv2017.Date == "18/02/2017"

print ((uv2017[filteredData18])['UV_Index'].max())



filteredData19 = uv2017.Date == "19/02/2017"

print ((uv2017[filteredData19])['UV_Index'].max())



filteredData20 = uv2017.Date == "20/02/2029"

print ((uv2017[filteredData20])['UV_Index'].max())



filteredData21 = uv2017.Date == "21/02/2017"

print ((uv2017[filteredData21])['UV_Index'].max())



filteredData22 = uv2017.Date == "22/02/2017"

print ((uv2017[filteredData22])['UV_Index'].max())



filteredData23 = uv2017.Date == "23/02/2017"

print ((uv2017[filteredData23])['UV_Index'].max())



filteredData24 = uv2017.Date == "24/02/2017"

print ((uv2017[filteredData24])['UV_Index'].max())



filteredData25 = uv2017.Date == "25/02/2017"

print ((uv2017[filteredData25])['UV_Index'].max())



filteredData26 = uv2017.Date == "26/02/2017"

print ((uv2017[filteredData26])['UV_Index'].max())



filteredData27 = uv2017.Date == "27/02/2017"

print ((uv2017[filteredData27])['UV_Index'].max())



filteredData28 = uv2017.Date == "28/02/2017"

print ((uv2017[filteredData28])['UV_Index'].max())



filteredData29 = uv2017.Date == "29/02/2017"

print ((uv2017[filteredData29])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2018 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2018.csv')



filteredData1 = uv2018.Date == "01/02/2018"

print((uv2018[filteredData1])['UV_Index'].max())



filteredData2 = uv2018.Date == "02/02/2018"

print ((uv2018[filteredData2])['UV_Index'].max())



filteredData3 = uv2018.Date == "03/02/2018"

print ((uv2018[filteredData3])['UV_Index'].max())



filteredData4 = uv2018.Date == "04/02/2018"

print ((uv2018[filteredData4])['UV_Index'].max())



filteredData5 = uv2018.Date == "05/02/2018"

print ((uv2018[filteredData5])['UV_Index'].max())



filteredData6 = uv2018.Date == "06/02/2018"

print ((uv2018[filteredData6])['UV_Index'].max())



filteredData7 = uv2018.Date == "07/02/2018"

print ((uv2018[filteredData7])['UV_Index'].max())



filteredData8 = uv2018.Date == "08/02/2018"

print ((uv2018[filteredData8])['UV_Index'].max())



filteredData9 = uv2018.Date == "09/02/2018"

print ((uv2018[filteredData9])['UV_Index'].max())



filteredData10 = uv2018.Date == "10/02/2018"

print ((uv2018[filteredData10])['UV_Index'].max())



filteredData11 = uv2018.Date == "11/02/2018"

print ((uv2018[filteredData11])['UV_Index'].max())



filteredData12 = uv2018.Date == "12/02/2018"

print ((uv2018[filteredData12])['UV_Index'].max())



filteredData13 = uv2018.Date == "13/02/2018"

print ((uv2018[filteredData13])['UV_Index'].max())



filteredData14 = uv2018.Date == "14/02/2018"

print ((uv2018[filteredData14])['UV_Index'].max())



filteredData15 = uv2018.Date == "15/02/2018"

print ((uv2018[filteredData15])['UV_Index'].max())



filteredData16 = uv2018.Date == "16/02/2018"

print ((uv2018[filteredData16])['UV_Index'].max())



filteredData17 = uv2018.Date == "17/02/2018"

print ((uv2018[filteredData17])['UV_Index'].max())



filteredData18 = uv2018.Date == "18/02/2018"

print ((uv2018[filteredData18])['UV_Index'].max())



filteredData19 = uv2018.Date == "19/02/2018"

print ((uv2018[filteredData19])['UV_Index'].max())



filteredData20 = uv2018.Date == "20/02/2029"

print ((uv2018[filteredData20])['UV_Index'].max())



filteredData21 = uv2018.Date == "21/02/2018"

print ((uv2018[filteredData21])['UV_Index'].max())



filteredData22 = uv2018.Date == "22/02/2018"

print ((uv2018[filteredData22])['UV_Index'].max())



filteredData23 = uv2018.Date == "23/02/2018"

print ((uv2018[filteredData23])['UV_Index'].max())



filteredData24 = uv2018.Date == "24/02/2018"

print ((uv2018[filteredData24])['UV_Index'].max())



filteredData25 = uv2018.Date == "25/02/2018"

print ((uv2018[filteredData25])['UV_Index'].max())



filteredData26 = uv2018.Date == "26/02/2018"

print ((uv2018[filteredData26])['UV_Index'].max())



filteredData27 = uv2018.Date == "27/02/2018"

print ((uv2018[filteredData27])['UV_Index'].max())



filteredData28 = uv2018.Date == "28/02/2018"

print ((uv2018[filteredData28])['UV_Index'].max())



filteredData29 = uv2018.Date == "29/02/2018"

print ((uv2018[filteredData29])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2019 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2019.csv')



filteredData1 = uv2019.Date == "01/02/2019"

print((uv2019[filteredData1])['UV_Index'].max())



filteredData2 = uv2019.Date == "02/02/2019"

print ((uv2019[filteredData2])['UV_Index'].max())



filteredData3 = uv2019.Date == "03/02/2019"

print ((uv2019[filteredData3])['UV_Index'].max())



filteredData4 = uv2019.Date == "04/02/2019"

print ((uv2019[filteredData4])['UV_Index'].max())



filteredData5 = uv2019.Date == "05/02/2019"

print ((uv2019[filteredData5])['UV_Index'].max())



filteredData6 = uv2019.Date == "06/02/2019"

print ((uv2019[filteredData6])['UV_Index'].max())



filteredData7 = uv2019.Date == "07/02/2019"

print ((uv2019[filteredData7])['UV_Index'].max())



filteredData8 = uv2019.Date == "08/02/2019"

print ((uv2019[filteredData8])['UV_Index'].max())



filteredData9 = uv2019.Date == "09/02/2019"

print ((uv2019[filteredData9])['UV_Index'].max())



filteredData10 = uv2019.Date == "10/02/2019"

print ((uv2019[filteredData10])['UV_Index'].max())



filteredData11 = uv2019.Date == "11/02/2019"

print ((uv2019[filteredData11])['UV_Index'].max())



filteredData12 = uv2019.Date == "12/02/2019"

print ((uv2019[filteredData12])['UV_Index'].max())



filteredData13 = uv2019.Date == "13/02/2019"

print ((uv2019[filteredData13])['UV_Index'].max())



filteredData14 = uv2019.Date == "14/02/2019"

print ((uv2019[filteredData14])['UV_Index'].max())



filteredData15 = uv2019.Date == "15/02/2019"

print ((uv2019[filteredData15])['UV_Index'].max())



filteredData16 = uv2019.Date == "16/02/2019"

print ((uv2019[filteredData16])['UV_Index'].max())



filteredData17 = uv2019.Date == "17/02/2019"

print ((uv2019[filteredData17])['UV_Index'].max())



filteredData18 = uv2019.Date == "18/02/2019"

print ((uv2019[filteredData18])['UV_Index'].max())



filteredData19 = uv2019.Date == "19/02/2019"

print ((uv2019[filteredData19])['UV_Index'].max())



filteredData20 = uv2019.Date == "20/02/2029"

print ((uv2019[filteredData20])['UV_Index'].max())



filteredData21 = uv2019.Date == "21/02/2019"

print ((uv2019[filteredData21])['UV_Index'].max())



filteredData22 = uv2019.Date == "22/02/2019"

print ((uv2019[filteredData22])['UV_Index'].max())



filteredData23 = uv2019.Date == "23/02/2019"

print ((uv2019[filteredData23])['UV_Index'].max())



filteredData24 = uv2019.Date == "24/02/2019"

print ((uv2019[filteredData24])['UV_Index'].max())



filteredData25 = uv2019.Date == "25/02/2019"

print ((uv2019[filteredData25])['UV_Index'].max())



filteredData26 = uv2019.Date == "26/02/2019"

print ((uv2019[filteredData26])['UV_Index'].max())



filteredData27 = uv2019.Date == "27/02/2019"

print ((uv2019[filteredData27])['UV_Index'].max())



filteredData28 = uv2019.Date == "28/02/2019"

print ((uv2019[filteredData28])['UV_Index'].max())



filteredData29 = uv2019.Date == "29/02/2019"

print ((uv2019[filteredData29])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2019 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2019.csv')



filteredData1 = uv2019.Date == "01/03/2019"

print((uv2019[filteredData1])['UV_Index'].max())



filteredData2 = uv2019.Date == "02/03/2019"

print ((uv2019[filteredData2])['UV_Index'].max())



filteredData3 = uv2019.Date == "03/03/2019"

print ((uv2019[filteredData3])['UV_Index'].max())



filteredData4 = uv2019.Date == "04/03/2019"

print ((uv2019[filteredData4])['UV_Index'].max())



filteredData5 = uv2019.Date == "05/03/2019"

print ((uv2019[filteredData5])['UV_Index'].max())



filteredData6 = uv2019.Date == "06/03/2019"

print ((uv2019[filteredData6])['UV_Index'].max())



filteredData7 = uv2019.Date == "07/03/2019"

print ((uv2019[filteredData7])['UV_Index'].max())



filteredData8 = uv2019.Date == "08/03/2019"

print ((uv2019[filteredData8])['UV_Index'].max())



filteredData9 = uv2019.Date == "09/03/2019"

print ((uv2019[filteredData9])['UV_Index'].max())



filteredData10 = uv2019.Date == "10/03/2019"

print ((uv2019[filteredData10])['UV_Index'].max())



filteredData11 = uv2019.Date == "11/03/2019"

print ((uv2019[filteredData11])['UV_Index'].max())



filteredData12 = uv2019.Date == "12/03/2019"

print ((uv2019[filteredData12])['UV_Index'].max())



filteredData13 = uv2019.Date == "13/03/2019"

print ((uv2019[filteredData13])['UV_Index'].max())



filteredData14 = uv2019.Date == "14/03/2019"

print ((uv2019[filteredData14])['UV_Index'].max())



filteredData15 = uv2019.Date == "15/03/2019"

print ((uv2019[filteredData15])['UV_Index'].max())



filteredData16 = uv2019.Date == "16/03/2019"

print ((uv2019[filteredData16])['UV_Index'].max())



filteredData17 = uv2019.Date == "17/03/2019"

print ((uv2019[filteredData17])['UV_Index'].max())



filteredData18 = uv2019.Date == "18/03/2019"

print ((uv2019[filteredData18])['UV_Index'].max())



filteredData19 = uv2019.Date == "19/03/2019"

print ((uv2019[filteredData19])['UV_Index'].max())



filteredData20 = uv2019.Date == "20/03/2039"

print ((uv2019[filteredData20])['UV_Index'].max())



filteredData21 = uv2019.Date == "21/03/2019"

print ((uv2019[filteredData21])['UV_Index'].max())



filteredData22 = uv2019.Date == "22/03/2019"

print ((uv2019[filteredData22])['UV_Index'].max())



filteredData23 = uv2019.Date == "23/03/2019"

print ((uv2019[filteredData23])['UV_Index'].max())



filteredData24 = uv2019.Date == "24/03/2019"

print ((uv2019[filteredData24])['UV_Index'].max())



filteredData25 = uv2019.Date == "25/03/2019"

print ((uv2019[filteredData25])['UV_Index'].max())



filteredData26 = uv2019.Date == "26/03/2019"

print ((uv2019[filteredData26])['UV_Index'].max())



filteredData27 = uv2019.Date == "27/03/2019"

print ((uv2019[filteredData27])['UV_Index'].max())



filteredData28 = uv2019.Date == "28/03/2019"

print ((uv2019[filteredData28])['UV_Index'].max())



filteredData29 = uv2019.Date == "29/03/2019"

print ((uv2019[filteredData29])['UV_Index'].max())



filteredData30 = uv2019.Date == "30/03/2019"

print ((uv2019[filteredData30])['UV_Index'].max())



filteredData31 = uv2019.Date == "31/03/2019"

print ((uv2019[filteredData22])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2018 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2018.csv')



filteredData1 = uv2018.Date == "01/03/2018"

print((uv2018[filteredData1])['UV_Index'].max())



filteredData2 = uv2018.Date == "02/03/2018"

print ((uv2018[filteredData2])['UV_Index'].max())



filteredData3 = uv2018.Date == "03/03/2018"

print ((uv2018[filteredData3])['UV_Index'].max())



filteredData4 = uv2018.Date == "04/03/2018"

print ((uv2018[filteredData4])['UV_Index'].max())



filteredData5 = uv2018.Date == "05/03/2018"

print ((uv2018[filteredData5])['UV_Index'].max())



filteredData6 = uv2018.Date == "06/03/2018"

print ((uv2018[filteredData6])['UV_Index'].max())



filteredData7 = uv2018.Date == "07/03/2018"

print ((uv2018[filteredData7])['UV_Index'].max())



filteredData8 = uv2018.Date == "08/03/2018"

print ((uv2018[filteredData8])['UV_Index'].max())



filteredData9 = uv2018.Date == "09/03/2018"

print ((uv2018[filteredData9])['UV_Index'].max())



filteredData10 = uv2018.Date == "10/03/2018"

print ((uv2018[filteredData10])['UV_Index'].max())



filteredData11 = uv2018.Date == "11/03/2018"

print ((uv2018[filteredData11])['UV_Index'].max())



filteredData12 = uv2018.Date == "12/03/2018"

print ((uv2018[filteredData12])['UV_Index'].max())



filteredData13 = uv2018.Date == "13/03/2018"

print ((uv2018[filteredData13])['UV_Index'].max())



filteredData14 = uv2018.Date == "14/03/2018"

print ((uv2018[filteredData14])['UV_Index'].max())



filteredData15 = uv2018.Date == "15/03/2018"

print ((uv2018[filteredData15])['UV_Index'].max())



filteredData16 = uv2018.Date == "16/03/2018"

print ((uv2018[filteredData16])['UV_Index'].max())



filteredData17 = uv2018.Date == "17/03/2018"

print ((uv2018[filteredData17])['UV_Index'].max())



filteredData18 = uv2018.Date == "18/03/2018"

print ((uv2018[filteredData18])['UV_Index'].max())



filteredData19 = uv2018.Date == "19/03/2018"

print ((uv2018[filteredData19])['UV_Index'].max())



filteredData20 = uv2018.Date == "20/03/2039"

print ((uv2018[filteredData20])['UV_Index'].max())



filteredData21 = uv2018.Date == "21/03/2018"

print ((uv2018[filteredData21])['UV_Index'].max())



filteredData22 = uv2018.Date == "22/03/2018"

print ((uv2018[filteredData22])['UV_Index'].max())



filteredData23 = uv2018.Date == "23/03/2018"

print ((uv2018[filteredData23])['UV_Index'].max())



filteredData24 = uv2018.Date == "24/03/2018"

print ((uv2018[filteredData24])['UV_Index'].max())



filteredData25 = uv2018.Date == "25/03/2018"

print ((uv2018[filteredData25])['UV_Index'].max())



filteredData26 = uv2018.Date == "26/03/2018"

print ((uv2018[filteredData26])['UV_Index'].max())



filteredData27 = uv2018.Date == "27/03/2018"

print ((uv2018[filteredData27])['UV_Index'].max())



filteredData28 = uv2018.Date == "28/03/2018"

print ((uv2018[filteredData28])['UV_Index'].max())



filteredData29 = uv2018.Date == "29/03/2018"

print ((uv2018[filteredData29])['UV_Index'].max())



filteredData30 = uv2018.Date == "30/03/2018"

print ((uv2018[filteredData30])['UV_Index'].max())



filteredData31 = uv2018.Date == "31/03/2018"

print ((uv2018[filteredData22])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2017 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2017.csv')



filteredData1 = uv2017.Date == "01/03/2017"

print((uv2017[filteredData1])['UV_Index'].max())



filteredData2 = uv2017.Date == "02/03/2017"

print ((uv2017[filteredData2])['UV_Index'].max())



filteredData3 = uv2017.Date == "03/03/2017"

print ((uv2017[filteredData3])['UV_Index'].max())



filteredData4 = uv2017.Date == "04/03/2017"

print ((uv2017[filteredData4])['UV_Index'].max())



filteredData5 = uv2017.Date == "05/03/2017"

print ((uv2017[filteredData5])['UV_Index'].max())



filteredData6 = uv2017.Date == "06/03/2017"

print ((uv2017[filteredData6])['UV_Index'].max())



filteredData7 = uv2017.Date == "07/03/2017"

print ((uv2017[filteredData7])['UV_Index'].max())



filteredData8 = uv2017.Date == "08/03/2017"

print ((uv2017[filteredData8])['UV_Index'].max())



filteredData9 = uv2017.Date == "09/03/2017"

print ((uv2017[filteredData9])['UV_Index'].max())



filteredData10 = uv2017.Date == "10/03/2017"

print ((uv2017[filteredData10])['UV_Index'].max())



filteredData11 = uv2017.Date == "11/03/2017"

print ((uv2017[filteredData11])['UV_Index'].max())



filteredData12 = uv2017.Date == "12/03/2017"

print ((uv2017[filteredData12])['UV_Index'].max())



filteredData13 = uv2017.Date == "13/03/2017"

print ((uv2017[filteredData13])['UV_Index'].max())



filteredData14 = uv2017.Date == "14/03/2017"

print ((uv2017[filteredData14])['UV_Index'].max())



filteredData15 = uv2017.Date == "15/03/2017"

print ((uv2017[filteredData15])['UV_Index'].max())



filteredData16 = uv2017.Date == "16/03/2017"

print ((uv2017[filteredData16])['UV_Index'].max())



filteredData17 = uv2017.Date == "17/03/2017"

print ((uv2017[filteredData17])['UV_Index'].max())



filteredData18 = uv2017.Date == "18/03/2017"

print ((uv2017[filteredData18])['UV_Index'].max())



filteredData19 = uv2017.Date == "19/03/2017"

print ((uv2017[filteredData19])['UV_Index'].max())



filteredData20 = uv2017.Date == "20/03/2039"

print ((uv2017[filteredData20])['UV_Index'].max())



filteredData21 = uv2017.Date == "21/03/2017"

print ((uv2017[filteredData21])['UV_Index'].max())



filteredData22 = uv2017.Date == "22/03/2017"

print ((uv2017[filteredData22])['UV_Index'].max())



filteredData23 = uv2017.Date == "23/03/2017"

print ((uv2017[filteredData23])['UV_Index'].max())



filteredData24 = uv2017.Date == "24/03/2017"

print ((uv2017[filteredData24])['UV_Index'].max())



filteredData25 = uv2017.Date == "25/03/2017"

print ((uv2017[filteredData25])['UV_Index'].max())



filteredData26 = uv2017.Date == "26/03/2017"

print ((uv2017[filteredData26])['UV_Index'].max())



filteredData27 = uv2017.Date == "27/03/2017"

print ((uv2017[filteredData27])['UV_Index'].max())



filteredData28 = uv2017.Date == "28/03/2017"

print ((uv2017[filteredData28])['UV_Index'].max())



filteredData29 = uv2017.Date == "29/03/2017"

print ((uv2017[filteredData29])['UV_Index'].max())



filteredData30 = uv2017.Date == "30/03/2017"

print ((uv2017[filteredData30])['UV_Index'].max())



filteredData31 = uv2017.Date == "31/03/2017"

print ((uv2017[filteredData22])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2016 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2016.csv')



filteredData1 = uv2016.Date == "01/03/2016"

print((uv2016[filteredData1])['UV_Index'].max())



filteredData2 = uv2016.Date == "02/03/2016"

print ((uv2016[filteredData2])['UV_Index'].max())



filteredData3 = uv2016.Date == "03/03/2016"

print ((uv2016[filteredData3])['UV_Index'].max())



filteredData4 = uv2016.Date == "04/03/2016"

print ((uv2016[filteredData4])['UV_Index'].max())



filteredData5 = uv2016.Date == "05/03/2016"

print ((uv2016[filteredData5])['UV_Index'].max())



filteredData6 = uv2016.Date == "06/03/2016"

print ((uv2016[filteredData6])['UV_Index'].max())



filteredData7 = uv2016.Date == "07/03/2016"

print ((uv2016[filteredData7])['UV_Index'].max())



filteredData8 = uv2016.Date == "08/03/2016"

print ((uv2016[filteredData8])['UV_Index'].max())



filteredData9 = uv2016.Date == "09/03/2016"

print ((uv2016[filteredData9])['UV_Index'].max())



filteredData10 = uv2016.Date == "10/03/2016"

print ((uv2016[filteredData10])['UV_Index'].max())



filteredData11 = uv2016.Date == "11/03/2016"

print ((uv2016[filteredData11])['UV_Index'].max())



filteredData12 = uv2016.Date == "12/03/2016"

print ((uv2016[filteredData12])['UV_Index'].max())



filteredData13 = uv2016.Date == "13/03/2016"

print ((uv2016[filteredData13])['UV_Index'].max())



filteredData14 = uv2016.Date == "14/03/2016"

print ((uv2016[filteredData14])['UV_Index'].max())



filteredData15 = uv2016.Date == "15/03/2016"

print ((uv2016[filteredData15])['UV_Index'].max())



filteredData16 = uv2016.Date == "16/03/2016"

print ((uv2016[filteredData16])['UV_Index'].max())



filteredData17 = uv2016.Date == "17/03/2016"

print ((uv2016[filteredData17])['UV_Index'].max())



filteredData18 = uv2016.Date == "18/03/2016"

print ((uv2016[filteredData18])['UV_Index'].max())



filteredData19 = uv2016.Date == "19/03/2016"

print ((uv2016[filteredData19])['UV_Index'].max())



filteredData20 = uv2016.Date == "20/03/2039"

print ((uv2016[filteredData20])['UV_Index'].max())



filteredData21 = uv2016.Date == "21/03/2016"

print ((uv2016[filteredData21])['UV_Index'].max())



filteredData22 = uv2016.Date == "22/03/2016"

print ((uv2016[filteredData22])['UV_Index'].max())



filteredData23 = uv2016.Date == "23/03/2016"

print ((uv2016[filteredData23])['UV_Index'].max())



filteredData24 = uv2016.Date == "24/03/2016"

print ((uv2016[filteredData24])['UV_Index'].max())



filteredData25 = uv2016.Date == "25/03/2016"

print ((uv2016[filteredData25])['UV_Index'].max())



filteredData26 = uv2016.Date == "26/03/2016"

print ((uv2016[filteredData26])['UV_Index'].max())



filteredData27 = uv2016.Date == "27/03/2016"

print ((uv2016[filteredData27])['UV_Index'].max())



filteredData28 = uv2016.Date == "28/03/2016"

print ((uv2016[filteredData28])['UV_Index'].max())



filteredData29 = uv2016.Date == "29/03/2016"

print ((uv2016[filteredData29])['UV_Index'].max())



filteredData30 = uv2016.Date == "30/03/2016"

print ((uv2016[filteredData30])['UV_Index'].max())



filteredData31 = uv2016.Date == "31/03/2016"

print ((uv2016[filteredData22])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2015 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2015.csv')



filteredData1 = uv2015.Date == "01/03/2015"

print((uv2015[filteredData1])['UV_Index'].max())



filteredData2 = uv2015.Date == "02/03/2015"

print ((uv2015[filteredData2])['UV_Index'].max())



filteredData3 = uv2015.Date == "03/03/2015"

print ((uv2015[filteredData3])['UV_Index'].max())



filteredData4 = uv2015.Date == "04/03/2015"

print ((uv2015[filteredData4])['UV_Index'].max())



filteredData5 = uv2015.Date == "05/03/2015"

print ((uv2015[filteredData5])['UV_Index'].max())



filteredData6 = uv2015.Date == "06/03/2015"

print ((uv2015[filteredData6])['UV_Index'].max())



filteredData7 = uv2015.Date == "07/03/2015"

print ((uv2015[filteredData7])['UV_Index'].max())



filteredData8 = uv2015.Date == "08/03/2015"

print ((uv2015[filteredData8])['UV_Index'].max())



filteredData9 = uv2015.Date == "09/03/2015"

print ((uv2015[filteredData9])['UV_Index'].max())



filteredData10 = uv2015.Date == "10/03/2015"

print ((uv2015[filteredData10])['UV_Index'].max())



filteredData11 = uv2015.Date == "11/03/2015"

print ((uv2015[filteredData11])['UV_Index'].max())



filteredData12 = uv2015.Date == "12/03/2015"

print ((uv2015[filteredData12])['UV_Index'].max())



filteredData13 = uv2015.Date == "13/03/2015"

print ((uv2015[filteredData13])['UV_Index'].max())



filteredData14 = uv2015.Date == "14/03/2015"

print ((uv2015[filteredData14])['UV_Index'].max())



filteredData15 = uv2015.Date == "15/03/2015"

print ((uv2015[filteredData15])['UV_Index'].max())



filteredData16 = uv2015.Date == "16/03/2015"

print ((uv2015[filteredData16])['UV_Index'].max())



filteredData17 = uv2015.Date == "17/03/2015"

print ((uv2015[filteredData17])['UV_Index'].max())



filteredData18 = uv2015.Date == "18/03/2015"

print ((uv2015[filteredData18])['UV_Index'].max())



filteredData19 = uv2015.Date == "19/03/2015"

print ((uv2015[filteredData19])['UV_Index'].max())



filteredData20 = uv2015.Date == "20/03/2039"

print ((uv2015[filteredData20])['UV_Index'].max())



filteredData21 = uv2015.Date == "21/03/2015"

print ((uv2015[filteredData21])['UV_Index'].max())



filteredData22 = uv2015.Date == "22/03/2015"

print ((uv2015[filteredData22])['UV_Index'].max())



filteredData23 = uv2015.Date == "23/03/2015"

print ((uv2015[filteredData23])['UV_Index'].max())



filteredData24 = uv2015.Date == "24/03/2015"

print ((uv2015[filteredData24])['UV_Index'].max())



filteredData25 = uv2015.Date == "25/03/2015"

print ((uv2015[filteredData25])['UV_Index'].max())



filteredData26 = uv2015.Date == "26/03/2015"

print ((uv2015[filteredData26])['UV_Index'].max())



filteredData27 = uv2015.Date == "27/03/2015"

print ((uv2015[filteredData27])['UV_Index'].max())



filteredData28 = uv2015.Date == "28/03/2015"

print ((uv2015[filteredData28])['UV_Index'].max())



filteredData29 = uv2015.Date == "29/03/2015"

print ((uv2015[filteredData29])['UV_Index'].max())



filteredData30 = uv2015.Date == "30/03/2015"

print ((uv2015[filteredData30])['UV_Index'].max())



filteredData31 = uv2015.Date == "31/03/2015"

print ((uv2015[filteredData22])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2014 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2014.csv')



filteredData1 = uv2014.Date == "01/03/2014"

print((uv2014[filteredData1])['UV_Index'].max())



filteredData2 = uv2014.Date == "02/03/2014"

print ((uv2014[filteredData2])['UV_Index'].max())



filteredData3 = uv2014.Date == "03/03/2014"

print ((uv2014[filteredData3])['UV_Index'].max())



filteredData4 = uv2014.Date == "04/03/2014"

print ((uv2014[filteredData4])['UV_Index'].max())



filteredData5 = uv2014.Date == "05/03/2014"

print ((uv2014[filteredData5])['UV_Index'].max())



filteredData6 = uv2014.Date == "06/03/2014"

print ((uv2014[filteredData6])['UV_Index'].max())



filteredData7 = uv2014.Date == "07/03/2014"

print ((uv2014[filteredData7])['UV_Index'].max())



filteredData8 = uv2014.Date == "08/03/2014"

print ((uv2014[filteredData8])['UV_Index'].max())



filteredData9 = uv2014.Date == "09/03/2014"

print ((uv2014[filteredData9])['UV_Index'].max())



filteredData10 = uv2014.Date == "10/03/2014"

print ((uv2014[filteredData10])['UV_Index'].max())



filteredData11 = uv2014.Date == "11/03/2014"

print ((uv2014[filteredData11])['UV_Index'].max())



filteredData12 = uv2014.Date == "12/03/2014"

print ((uv2014[filteredData12])['UV_Index'].max())



filteredData13 = uv2014.Date == "13/03/2014"

print ((uv2014[filteredData13])['UV_Index'].max())



filteredData14 = uv2014.Date == "14/03/2014"

print ((uv2014[filteredData14])['UV_Index'].max())



filteredData15 = uv2014.Date == "15/03/2014"

print ((uv2014[filteredData15])['UV_Index'].max())



filteredData16 = uv2014.Date == "16/03/2014"

print ((uv2014[filteredData16])['UV_Index'].max())



filteredData17 = uv2014.Date == "17/03/2014"

print ((uv2014[filteredData17])['UV_Index'].max())



filteredData18 = uv2014.Date == "18/03/2014"

print ((uv2014[filteredData18])['UV_Index'].max())



filteredData19 = uv2014.Date == "19/03/2014"

print ((uv2014[filteredData19])['UV_Index'].max())



filteredData20 = uv2014.Date == "20/03/2039"

print ((uv2014[filteredData20])['UV_Index'].max())



filteredData21 = uv2014.Date == "21/03/2014"

print ((uv2014[filteredData21])['UV_Index'].max())



filteredData22 = uv2014.Date == "22/03/2014"

print ((uv2014[filteredData22])['UV_Index'].max())



filteredData23 = uv2014.Date == "23/03/2014"

print ((uv2014[filteredData23])['UV_Index'].max())



filteredData24 = uv2014.Date == "24/03/2014"

print ((uv2014[filteredData24])['UV_Index'].max())



filteredData25 = uv2014.Date == "25/03/2014"

print ((uv2014[filteredData25])['UV_Index'].max())



filteredData26 = uv2014.Date == "26/03/2014"

print ((uv2014[filteredData26])['UV_Index'].max())



filteredData27 = uv2014.Date == "27/03/2014"

print ((uv2014[filteredData27])['UV_Index'].max())



filteredData28 = uv2014.Date == "28/03/2014"

print ((uv2014[filteredData28])['UV_Index'].max())



filteredData29 = uv2014.Date == "29/03/2014"

print ((uv2014[filteredData29])['UV_Index'].max())



filteredData30 = uv2014.Date == "30/03/2014"

print ((uv2014[filteredData30])['UV_Index'].max())



filteredData31 = uv2014.Date == "31/03/2014"

print ((uv2014[filteredData22])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2013 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2013.csv')



filteredData1 = uv2013.Date == "01/03/2013"

print((uv2013[filteredData1])['UV_Index'].max())



filteredData2 = uv2013.Date == "02/03/2013"

print ((uv2013[filteredData2])['UV_Index'].max())



filteredData3 = uv2013.Date == "03/03/2013"

print ((uv2013[filteredData3])['UV_Index'].max())



filteredData4 = uv2013.Date == "04/03/2013"

print ((uv2013[filteredData4])['UV_Index'].max())



filteredData5 = uv2013.Date == "05/03/2013"

print ((uv2013[filteredData5])['UV_Index'].max())



filteredData6 = uv2013.Date == "06/03/2013"

print ((uv2013[filteredData6])['UV_Index'].max())



filteredData7 = uv2013.Date == "07/03/2013"

print ((uv2013[filteredData7])['UV_Index'].max())



filteredData8 = uv2013.Date == "08/03/2013"

print ((uv2013[filteredData8])['UV_Index'].max())



filteredData9 = uv2013.Date == "09/03/2013"

print ((uv2013[filteredData9])['UV_Index'].max())



filteredData10 = uv2013.Date == "10/03/2013"

print ((uv2013[filteredData10])['UV_Index'].max())



filteredData11 = uv2013.Date == "11/03/2013"

print ((uv2013[filteredData11])['UV_Index'].max())



filteredData12 = uv2013.Date == "12/03/2013"

print ((uv2013[filteredData12])['UV_Index'].max())



filteredData13 = uv2013.Date == "13/03/2013"

print ((uv2013[filteredData13])['UV_Index'].max())



filteredData14 = uv2013.Date == "14/03/2013"

print ((uv2013[filteredData14])['UV_Index'].max())



filteredData15 = uv2013.Date == "15/03/2013"

print ((uv2013[filteredData15])['UV_Index'].max())



filteredData16 = uv2013.Date == "16/03/2013"

print ((uv2013[filteredData16])['UV_Index'].max())



filteredData17 = uv2013.Date == "17/03/2013"

print ((uv2013[filteredData17])['UV_Index'].max())



filteredData18 = uv2013.Date == "18/03/2013"

print ((uv2013[filteredData18])['UV_Index'].max())



filteredData19 = uv2013.Date == "19/03/2013"

print ((uv2013[filteredData19])['UV_Index'].max())



filteredData20 = uv2013.Date == "20/03/2039"

print ((uv2013[filteredData20])['UV_Index'].max())



filteredData21 = uv2013.Date == "21/03/2013"

print ((uv2013[filteredData21])['UV_Index'].max())



filteredData22 = uv2013.Date == "22/03/2013"

print ((uv2013[filteredData22])['UV_Index'].max())



filteredData23 = uv2013.Date == "23/03/2013"

print ((uv2013[filteredData23])['UV_Index'].max())



filteredData24 = uv2013.Date == "24/03/2013"

print ((uv2013[filteredData24])['UV_Index'].max())



filteredData25 = uv2013.Date == "25/03/2013"

print ((uv2013[filteredData25])['UV_Index'].max())



filteredData26 = uv2013.Date == "26/03/2013"

print ((uv2013[filteredData26])['UV_Index'].max())



filteredData27 = uv2013.Date == "27/03/2013"

print ((uv2013[filteredData27])['UV_Index'].max())



filteredData28 = uv2013.Date == "28/03/2013"

print ((uv2013[filteredData28])['UV_Index'].max())



filteredData29 = uv2013.Date == "29/03/2013"

print ((uv2013[filteredData29])['UV_Index'].max())



filteredData30 = uv2013.Date == "30/03/2013"

print ((uv2013[filteredData30])['UV_Index'].max())



filteredData31 = uv2013.Date == "31/03/2013"

print ((uv2013[filteredData22])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2012 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2012.csv')



filteredData1 = uv2012.Date == "01/03/2012"

print((uv2012[filteredData1])['UV_Index'].max())



filteredData2 = uv2012.Date == "02/03/2012"

print ((uv2012[filteredData2])['UV_Index'].max())



filteredData3 = uv2012.Date == "03/03/2012"

print ((uv2012[filteredData3])['UV_Index'].max())



filteredData4 = uv2012.Date == "04/03/2012"

print ((uv2012[filteredData4])['UV_Index'].max())



filteredData5 = uv2012.Date == "05/03/2012"

print ((uv2012[filteredData5])['UV_Index'].max())



filteredData6 = uv2012.Date == "06/03/2012"

print ((uv2012[filteredData6])['UV_Index'].max())



filteredData7 = uv2012.Date == "07/03/2012"

print ((uv2012[filteredData7])['UV_Index'].max())



filteredData8 = uv2012.Date == "08/03/2012"

print ((uv2012[filteredData8])['UV_Index'].max())



filteredData9 = uv2012.Date == "09/03/2012"

print ((uv2012[filteredData9])['UV_Index'].max())



filteredData10 = uv2012.Date == "10/03/2012"

print ((uv2012[filteredData10])['UV_Index'].max())



filteredData11 = uv2012.Date == "11/03/2012"

print ((uv2012[filteredData11])['UV_Index'].max())



filteredData12 = uv2012.Date == "12/03/2012"

print ((uv2012[filteredData12])['UV_Index'].max())



filteredData13 = uv2012.Date == "13/03/2012"

print ((uv2012[filteredData13])['UV_Index'].max())



filteredData14 = uv2012.Date == "14/03/2012"

print ((uv2012[filteredData14])['UV_Index'].max())



filteredData15 = uv2012.Date == "15/03/2012"

print ((uv2012[filteredData15])['UV_Index'].max())



filteredData16 = uv2012.Date == "16/03/2012"

print ((uv2012[filteredData16])['UV_Index'].max())



filteredData17 = uv2012.Date == "17/03/2012"

print ((uv2012[filteredData17])['UV_Index'].max())



filteredData18 = uv2012.Date == "18/03/2012"

print ((uv2012[filteredData18])['UV_Index'].max())



filteredData19 = uv2012.Date == "19/03/2012"

print ((uv2012[filteredData19])['UV_Index'].max())



filteredData20 = uv2012.Date == "20/03/2039"

print ((uv2012[filteredData20])['UV_Index'].max())



filteredData21 = uv2012.Date == "21/03/2012"

print ((uv2012[filteredData21])['UV_Index'].max())



filteredData22 = uv2012.Date == "22/03/2012"

print ((uv2012[filteredData22])['UV_Index'].max())



filteredData23 = uv2012.Date == "23/03/2012"

print ((uv2012[filteredData23])['UV_Index'].max())



filteredData24 = uv2012.Date == "24/03/2012"

print ((uv2012[filteredData24])['UV_Index'].max())



filteredData25 = uv2012.Date == "25/03/2012"

print ((uv2012[filteredData25])['UV_Index'].max())



filteredData26 = uv2012.Date == "26/03/2012"

print ((uv2012[filteredData26])['UV_Index'].max())



filteredData27 = uv2012.Date == "27/03/2012"

print ((uv2012[filteredData27])['UV_Index'].max())



filteredData28 = uv2012.Date == "28/03/2012"

print ((uv2012[filteredData28])['UV_Index'].max())



filteredData29 = uv2012.Date == "29/03/2012"

print ((uv2012[filteredData29])['UV_Index'].max())



filteredData30 = uv2012.Date == "30/03/2012"

print ((uv2012[filteredData30])['UV_Index'].max())



filteredData31 = uv2012.Date == "31/03/2012"

print ((uv2012[filteredData22])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2011 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2011.csv')



filteredData1 = uv2011.Date == "01/03/2011"

print((uv2011[filteredData1])['UV_Index'].max())



filteredData2 = uv2011.Date == "02/03/2011"

print ((uv2011[filteredData2])['UV_Index'].max())



filteredData3 = uv2011.Date == "03/03/2011"

print ((uv2011[filteredData3])['UV_Index'].max())



filteredData4 = uv2011.Date == "04/03/2011"

print ((uv2011[filteredData4])['UV_Index'].max())



filteredData5 = uv2011.Date == "05/03/2011"

print ((uv2011[filteredData5])['UV_Index'].max())



filteredData6 = uv2011.Date == "06/03/2011"

print ((uv2011[filteredData6])['UV_Index'].max())



filteredData7 = uv2011.Date == "07/03/2011"

print ((uv2011[filteredData7])['UV_Index'].max())



filteredData8 = uv2011.Date == "08/03/2011"

print ((uv2011[filteredData8])['UV_Index'].max())



filteredData9 = uv2011.Date == "09/03/2011"

print ((uv2011[filteredData9])['UV_Index'].max())



filteredData10 = uv2011.Date == "10/03/2011"

print ((uv2011[filteredData10])['UV_Index'].max())



filteredData11 = uv2011.Date == "11/03/2011"

print ((uv2011[filteredData11])['UV_Index'].max())



filteredData12 = uv2011.Date == "12/03/2011"

print ((uv2011[filteredData12])['UV_Index'].max())



filteredData13 = uv2011.Date == "13/03/2011"

print ((uv2011[filteredData13])['UV_Index'].max())



filteredData14 = uv2011.Date == "14/03/2011"

print ((uv2011[filteredData14])['UV_Index'].max())



filteredData15 = uv2011.Date == "15/03/2011"

print ((uv2011[filteredData15])['UV_Index'].max())



filteredData16 = uv2011.Date == "16/03/2011"

print ((uv2011[filteredData16])['UV_Index'].max())



filteredData17 = uv2011.Date == "17/03/2011"

print ((uv2011[filteredData17])['UV_Index'].max())



filteredData18 = uv2011.Date == "18/03/2011"

print ((uv2011[filteredData18])['UV_Index'].max())



filteredData19 = uv2011.Date == "19/03/2011"

print ((uv2011[filteredData19])['UV_Index'].max())



filteredData20 = uv2011.Date == "20/03/2039"

print ((uv2011[filteredData20])['UV_Index'].max())



filteredData21 = uv2011.Date == "21/03/2011"

print ((uv2011[filteredData21])['UV_Index'].max())



filteredData22 = uv2011.Date == "22/03/2011"

print ((uv2011[filteredData22])['UV_Index'].max())



filteredData23 = uv2011.Date == "23/03/2011"

print ((uv2011[filteredData23])['UV_Index'].max())



filteredData24 = uv2011.Date == "24/03/2011"

print ((uv2011[filteredData24])['UV_Index'].max())



filteredData25 = uv2011.Date == "25/03/2011"

print ((uv2011[filteredData25])['UV_Index'].max())



filteredData26 = uv2011.Date == "26/03/2011"

print ((uv2011[filteredData26])['UV_Index'].max())



filteredData27 = uv2011.Date == "27/03/2011"

print ((uv2011[filteredData27])['UV_Index'].max())



filteredData28 = uv2011.Date == "28/03/2011"

print ((uv2011[filteredData28])['UV_Index'].max())



filteredData29 = uv2011.Date == "29/03/2011"

print ((uv2011[filteredData29])['UV_Index'].max())



filteredData30 = uv2011.Date == "30/03/2011"

print ((uv2011[filteredData30])['UV_Index'].max())



filteredData31 = uv2011.Date == "31/03/2011"

print ((uv2011[filteredData22])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2010 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2010.csv')



filteredData1 = uv2010.Date == "01/03/2010"

print((uv2010[filteredData1])['UV_Index'].max())



filteredData2 = uv2010.Date == "02/03/2010"

print ((uv2010[filteredData2])['UV_Index'].max())



filteredData3 = uv2010.Date == "03/03/2010"

print ((uv2010[filteredData3])['UV_Index'].max())



filteredData4 = uv2010.Date == "04/03/2010"

print ((uv2010[filteredData4])['UV_Index'].max())



filteredData5 = uv2010.Date == "05/03/2010"

print ((uv2010[filteredData5])['UV_Index'].max())



filteredData6 = uv2010.Date == "06/03/2010"

print ((uv2010[filteredData6])['UV_Index'].max())



filteredData7 = uv2010.Date == "07/03/2010"

print ((uv2010[filteredData7])['UV_Index'].max())



filteredData8 = uv2010.Date == "08/03/2010"

print ((uv2010[filteredData8])['UV_Index'].max())



filteredData9 = uv2010.Date == "09/03/2010"

print ((uv2010[filteredData9])['UV_Index'].max())



filteredData10 = uv2010.Date == "10/03/2010"

print ((uv2010[filteredData10])['UV_Index'].max())



filteredData11 = uv2010.Date == "11/03/2010"

print ((uv2010[filteredData11])['UV_Index'].max())



filteredData12 = uv2010.Date == "12/03/2010"

print ((uv2010[filteredData12])['UV_Index'].max())



filteredData13 = uv2010.Date == "13/03/2010"

print ((uv2010[filteredData13])['UV_Index'].max())



filteredData14 = uv2010.Date == "14/03/2010"

print ((uv2010[filteredData14])['UV_Index'].max())



filteredData15 = uv2010.Date == "15/03/2010"

print ((uv2010[filteredData15])['UV_Index'].max())



filteredData16 = uv2010.Date == "16/03/2010"

print ((uv2010[filteredData16])['UV_Index'].max())



filteredData17 = uv2010.Date == "17/03/2010"

print ((uv2010[filteredData17])['UV_Index'].max())



filteredData18 = uv2010.Date == "18/03/2010"

print ((uv2010[filteredData18])['UV_Index'].max())



filteredData19 = uv2010.Date == "19/03/2010"

print ((uv2010[filteredData19])['UV_Index'].max())



filteredData20 = uv2010.Date == "20/03/2039"

print ((uv2010[filteredData20])['UV_Index'].max())



filteredData21 = uv2010.Date == "21/03/2010"

print ((uv2010[filteredData21])['UV_Index'].max())



filteredData22 = uv2010.Date == "22/03/2010"

print ((uv2010[filteredData22])['UV_Index'].max())



filteredData23 = uv2010.Date == "23/03/2010"

print ((uv2010[filteredData23])['UV_Index'].max())



filteredData24 = uv2010.Date == "24/03/2010"

print ((uv2010[filteredData24])['UV_Index'].max())



filteredData25 = uv2010.Date == "25/03/2010"

print ((uv2010[filteredData25])['UV_Index'].max())



filteredData26 = uv2010.Date == "26/03/2010"

print ((uv2010[filteredData26])['UV_Index'].max())



filteredData27 = uv2010.Date == "27/03/2010"

print ((uv2010[filteredData27])['UV_Index'].max())



filteredData28 = uv2010.Date == "28/03/2010"

print ((uv2010[filteredData28])['UV_Index'].max())



filteredData29 = uv2010.Date == "29/03/2010"

print ((uv2010[filteredData29])['UV_Index'].max())



filteredData30 = uv2010.Date == "30/03/2010"

print ((uv2010[filteredData30])['UV_Index'].max())



filteredData31 = uv2010.Date == "31/03/2010"

print ((uv2010[filteredData22])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2010 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2010.csv')



filteredData1 = uv2010.Date == "01/04/2010"

print((uv2010[filteredData1])['UV_Index'].max())



filteredData2 = uv2010.Date == "02/04/2010"

print ((uv2010[filteredData2])['UV_Index'].max())



filteredData3 = uv2010.Date == "03/04/2010"

print ((uv2010[filteredData3])['UV_Index'].max())



filteredData4 = uv2010.Date == "04/04/2010"

print ((uv2010[filteredData4])['UV_Index'].max())



filteredData5 = uv2010.Date == "05/04/2010"

print ((uv2010[filteredData5])['UV_Index'].max())



filteredData6 = uv2010.Date == "06/04/2010"

print ((uv2010[filteredData6])['UV_Index'].max())



filteredData7 = uv2010.Date == "07/04/2010"

print ((uv2010[filteredData7])['UV_Index'].max())



filteredData8 = uv2010.Date == "08/04/2010"

print ((uv2010[filteredData8])['UV_Index'].max())



filteredData9 = uv2010.Date == "09/04/2010"

print ((uv2010[filteredData9])['UV_Index'].max())



filteredData10 = uv2010.Date == "10/04/2010"

print ((uv2010[filteredData10])['UV_Index'].max())



filteredData11 = uv2010.Date == "11/04/2010"

print ((uv2010[filteredData11])['UV_Index'].max())



filteredData12 = uv2010.Date == "12/04/2010"

print ((uv2010[filteredData12])['UV_Index'].max())



filteredData13 = uv2010.Date == "13/04/2010"

print ((uv2010[filteredData13])['UV_Index'].max())



filteredData14 = uv2010.Date == "14/04/2010"

print ((uv2010[filteredData14])['UV_Index'].max())



filteredData15 = uv2010.Date == "15/04/2010"

print ((uv2010[filteredData15])['UV_Index'].max())



filteredData16 = uv2010.Date == "16/04/2010"

print ((uv2010[filteredData16])['UV_Index'].max())



filteredData17 = uv2010.Date == "17/04/2010"

print ((uv2010[filteredData17])['UV_Index'].max())



filteredData18 = uv2010.Date == "18/04/2010"

print ((uv2010[filteredData18])['UV_Index'].max())



filteredData19 = uv2010.Date == "19/04/2010"

print ((uv2010[filteredData19])['UV_Index'].max())



filteredData20 = uv2010.Date == "20/04/2049"

print ((uv2010[filteredData20])['UV_Index'].max())



filteredData21 = uv2010.Date == "21/04/2010"

print ((uv2010[filteredData21])['UV_Index'].max())



filteredData22 = uv2010.Date == "22/04/2010"

print ((uv2010[filteredData22])['UV_Index'].max())



filteredData23 = uv2010.Date == "23/04/2010"

print ((uv2010[filteredData23])['UV_Index'].max())



filteredData24 = uv2010.Date == "24/04/2010"

print ((uv2010[filteredData24])['UV_Index'].max())



filteredData25 = uv2010.Date == "25/04/2010"

print ((uv2010[filteredData25])['UV_Index'].max())



filteredData26 = uv2010.Date == "26/04/2010"

print ((uv2010[filteredData26])['UV_Index'].max())



filteredData27 = uv2010.Date == "27/04/2010"

print ((uv2010[filteredData27])['UV_Index'].max())



filteredData28 = uv2010.Date == "28/04/2010"

print ((uv2010[filteredData28])['UV_Index'].max())



filteredData29 = uv2010.Date == "29/04/2010"

print ((uv2010[filteredData29])['UV_Index'].max())



filteredData30 = uv2010.Date == "30/04/2010"

print ((uv2010[filteredData30])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2011 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2011.csv')



filteredData1 = uv2011.Date == "01/04/2011"

print((uv2011[filteredData1])['UV_Index'].max())



filteredData2 = uv2011.Date == "02/04/2011"

print ((uv2011[filteredData2])['UV_Index'].max())



filteredData3 = uv2011.Date == "03/04/2011"

print ((uv2011[filteredData3])['UV_Index'].max())



filteredData4 = uv2011.Date == "04/04/2011"

print ((uv2011[filteredData4])['UV_Index'].max())



filteredData5 = uv2011.Date == "05/04/2011"

print ((uv2011[filteredData5])['UV_Index'].max())



filteredData6 = uv2011.Date == "06/04/2011"

print ((uv2011[filteredData6])['UV_Index'].max())



filteredData7 = uv2011.Date == "07/04/2011"

print ((uv2011[filteredData7])['UV_Index'].max())



filteredData8 = uv2011.Date == "08/04/2011"

print ((uv2011[filteredData8])['UV_Index'].max())



filteredData9 = uv2011.Date == "09/04/2011"

print ((uv2011[filteredData9])['UV_Index'].max())



filteredData10 = uv2011.Date == "10/04/2011"

print ((uv2011[filteredData10])['UV_Index'].max())



filteredData11 = uv2011.Date == "11/04/2011"

print ((uv2011[filteredData11])['UV_Index'].max())



filteredData12 = uv2011.Date == "12/04/2011"

print ((uv2011[filteredData12])['UV_Index'].max())



filteredData13 = uv2011.Date == "13/04/2011"

print ((uv2011[filteredData13])['UV_Index'].max())



filteredData14 = uv2011.Date == "14/04/2011"

print ((uv2011[filteredData14])['UV_Index'].max())



filteredData15 = uv2011.Date == "15/04/2011"

print ((uv2011[filteredData15])['UV_Index'].max())



filteredData16 = uv2011.Date == "16/04/2011"

print ((uv2011[filteredData16])['UV_Index'].max())



filteredData17 = uv2011.Date == "17/04/2011"

print ((uv2011[filteredData17])['UV_Index'].max())



filteredData18 = uv2011.Date == "18/04/2011"

print ((uv2011[filteredData18])['UV_Index'].max())



filteredData19 = uv2011.Date == "19/04/2011"

print ((uv2011[filteredData19])['UV_Index'].max())



filteredData20 = uv2011.Date == "20/04/2049"

print ((uv2011[filteredData20])['UV_Index'].max())



filteredData21 = uv2011.Date == "21/04/2011"

print ((uv2011[filteredData21])['UV_Index'].max())



filteredData22 = uv2011.Date == "22/04/2011"

print ((uv2011[filteredData22])['UV_Index'].max())



filteredData23 = uv2011.Date == "23/04/2011"

print ((uv2011[filteredData23])['UV_Index'].max())



filteredData24 = uv2011.Date == "24/04/2011"

print ((uv2011[filteredData24])['UV_Index'].max())



filteredData25 = uv2011.Date == "25/04/2011"

print ((uv2011[filteredData25])['UV_Index'].max())



filteredData26 = uv2011.Date == "26/04/2011"

print ((uv2011[filteredData26])['UV_Index'].max())



filteredData27 = uv2011.Date == "27/04/2011"

print ((uv2011[filteredData27])['UV_Index'].max())



filteredData28 = uv2011.Date == "28/04/2011"

print ((uv2011[filteredData28])['UV_Index'].max())



filteredData29 = uv2011.Date == "29/04/2011"

print ((uv2011[filteredData29])['UV_Index'].max())



filteredData30 = uv2011.Date == "30/04/2011"

print ((uv2011[filteredData30])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2012 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2012.csv')



filteredData1 = uv2012.Date == "01/04/2012"

print((uv2012[filteredData1])['UV_Index'].max())



filteredData2 = uv2012.Date == "02/04/2012"

print ((uv2012[filteredData2])['UV_Index'].max())



filteredData3 = uv2012.Date == "03/04/2012"

print ((uv2012[filteredData3])['UV_Index'].max())



filteredData4 = uv2012.Date == "04/04/2012"

print ((uv2012[filteredData4])['UV_Index'].max())



filteredData5 = uv2012.Date == "05/04/2012"

print ((uv2012[filteredData5])['UV_Index'].max())



filteredData6 = uv2012.Date == "06/04/2012"

print ((uv2012[filteredData6])['UV_Index'].max())



filteredData7 = uv2012.Date == "07/04/2012"

print ((uv2012[filteredData7])['UV_Index'].max())



filteredData8 = uv2012.Date == "08/04/2012"

print ((uv2012[filteredData8])['UV_Index'].max())



filteredData9 = uv2012.Date == "09/04/2012"

print ((uv2012[filteredData9])['UV_Index'].max())



filteredData10 = uv2012.Date == "10/04/2012"

print ((uv2012[filteredData10])['UV_Index'].max())



filteredData11 = uv2012.Date == "11/04/2012"

print ((uv2012[filteredData11])['UV_Index'].max())



filteredData12 = uv2012.Date == "12/04/2012"

print ((uv2012[filteredData12])['UV_Index'].max())



filteredData13 = uv2012.Date == "13/04/2012"

print ((uv2012[filteredData13])['UV_Index'].max())



filteredData14 = uv2012.Date == "14/04/2012"

print ((uv2012[filteredData14])['UV_Index'].max())



filteredData15 = uv2012.Date == "15/04/2012"

print ((uv2012[filteredData15])['UV_Index'].max())



filteredData16 = uv2012.Date == "16/04/2012"

print ((uv2012[filteredData16])['UV_Index'].max())



filteredData17 = uv2012.Date == "17/04/2012"

print ((uv2012[filteredData17])['UV_Index'].max())



filteredData18 = uv2012.Date == "18/04/2012"

print ((uv2012[filteredData18])['UV_Index'].max())



filteredData19 = uv2012.Date == "19/04/2012"

print ((uv2012[filteredData19])['UV_Index'].max())



filteredData20 = uv2012.Date == "20/04/2049"

print ((uv2012[filteredData20])['UV_Index'].max())



filteredData21 = uv2012.Date == "21/04/2012"

print ((uv2012[filteredData21])['UV_Index'].max())



filteredData22 = uv2012.Date == "22/04/2012"

print ((uv2012[filteredData22])['UV_Index'].max())



filteredData23 = uv2012.Date == "23/04/2012"

print ((uv2012[filteredData23])['UV_Index'].max())



filteredData24 = uv2012.Date == "24/04/2012"

print ((uv2012[filteredData24])['UV_Index'].max())



filteredData25 = uv2012.Date == "25/04/2012"

print ((uv2012[filteredData25])['UV_Index'].max())



filteredData26 = uv2012.Date == "26/04/2012"

print ((uv2012[filteredData26])['UV_Index'].max())



filteredData27 = uv2012.Date == "27/04/2012"

print ((uv2012[filteredData27])['UV_Index'].max())



filteredData28 = uv2012.Date == "28/04/2012"

print ((uv2012[filteredData28])['UV_Index'].max())



filteredData29 = uv2012.Date == "29/04/2012"

print ((uv2012[filteredData29])['UV_Index'].max())



filteredData30 = uv2012.Date == "30/04/2012"

print ((uv2012[filteredData30])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2013 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2013.csv')



filteredData1 = uv2013.Date == "01/04/2013"

print((uv2013[filteredData1])['UV_Index'].max())



filteredData2 = uv2013.Date == "02/04/2013"

print ((uv2013[filteredData2])['UV_Index'].max())



filteredData3 = uv2013.Date == "03/04/2013"

print ((uv2013[filteredData3])['UV_Index'].max())



filteredData4 = uv2013.Date == "04/04/2013"

print ((uv2013[filteredData4])['UV_Index'].max())



filteredData5 = uv2013.Date == "05/04/2013"

print ((uv2013[filteredData5])['UV_Index'].max())



filteredData6 = uv2013.Date == "06/04/2013"

print ((uv2013[filteredData6])['UV_Index'].max())



filteredData7 = uv2013.Date == "07/04/2013"

print ((uv2013[filteredData7])['UV_Index'].max())



filteredData8 = uv2013.Date == "08/04/2013"

print ((uv2013[filteredData8])['UV_Index'].max())



filteredData9 = uv2013.Date == "09/04/2013"

print ((uv2013[filteredData9])['UV_Index'].max())



filteredData10 = uv2013.Date == "10/04/2013"

print ((uv2013[filteredData10])['UV_Index'].max())



filteredData11 = uv2013.Date == "11/04/2013"

print ((uv2013[filteredData11])['UV_Index'].max())



filteredData12 = uv2013.Date == "12/04/2013"

print ((uv2013[filteredData12])['UV_Index'].max())



filteredData13 = uv2013.Date == "13/04/2013"

print ((uv2013[filteredData13])['UV_Index'].max())



filteredData14 = uv2013.Date == "14/04/2013"

print ((uv2013[filteredData14])['UV_Index'].max())



filteredData15 = uv2013.Date == "15/04/2013"

print ((uv2013[filteredData15])['UV_Index'].max())



filteredData16 = uv2013.Date == "16/04/2013"

print ((uv2013[filteredData16])['UV_Index'].max())



filteredData17 = uv2013.Date == "17/04/2013"

print ((uv2013[filteredData17])['UV_Index'].max())



filteredData18 = uv2013.Date == "18/04/2013"

print ((uv2013[filteredData18])['UV_Index'].max())



filteredData19 = uv2013.Date == "19/04/2013"

print ((uv2013[filteredData19])['UV_Index'].max())



filteredData20 = uv2013.Date == "20/04/2049"

print ((uv2013[filteredData20])['UV_Index'].max())



filteredData21 = uv2013.Date == "21/04/2013"

print ((uv2013[filteredData21])['UV_Index'].max())



filteredData22 = uv2013.Date == "22/04/2013"

print ((uv2013[filteredData22])['UV_Index'].max())



filteredData23 = uv2013.Date == "23/04/2013"

print ((uv2013[filteredData23])['UV_Index'].max())



filteredData24 = uv2013.Date == "24/04/2013"

print ((uv2013[filteredData24])['UV_Index'].max())



filteredData25 = uv2013.Date == "25/04/2013"

print ((uv2013[filteredData25])['UV_Index'].max())



filteredData26 = uv2013.Date == "26/04/2013"

print ((uv2013[filteredData26])['UV_Index'].max())



filteredData27 = uv2013.Date == "27/04/2013"

print ((uv2013[filteredData27])['UV_Index'].max())



filteredData28 = uv2013.Date == "28/04/2013"

print ((uv2013[filteredData28])['UV_Index'].max())



filteredData29 = uv2013.Date == "29/04/2013"

print ((uv2013[filteredData29])['UV_Index'].max())



filteredData30 = uv2013.Date == "30/04/2013"

print ((uv2013[filteredData30])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2014 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2014.csv')



filteredData1 = uv2014.Date == "01/04/2014"

print((uv2014[filteredData1])['UV_Index'].max())



filteredData2 = uv2014.Date == "02/04/2014"

print ((uv2014[filteredData2])['UV_Index'].max())



filteredData3 = uv2014.Date == "03/04/2014"

print ((uv2014[filteredData3])['UV_Index'].max())



filteredData4 = uv2014.Date == "04/04/2014"

print ((uv2014[filteredData4])['UV_Index'].max())



filteredData5 = uv2014.Date == "05/04/2014"

print ((uv2014[filteredData5])['UV_Index'].max())



filteredData6 = uv2014.Date == "06/04/2014"

print ((uv2014[filteredData6])['UV_Index'].max())



filteredData7 = uv2014.Date == "07/04/2014"

print ((uv2014[filteredData7])['UV_Index'].max())



filteredData8 = uv2014.Date == "08/04/2014"

print ((uv2014[filteredData8])['UV_Index'].max())



filteredData9 = uv2014.Date == "09/04/2014"

print ((uv2014[filteredData9])['UV_Index'].max())



filteredData10 = uv2014.Date == "10/04/2014"

print ((uv2014[filteredData10])['UV_Index'].max())



filteredData11 = uv2014.Date == "11/04/2014"

print ((uv2014[filteredData11])['UV_Index'].max())



filteredData12 = uv2014.Date == "12/04/2014"

print ((uv2014[filteredData12])['UV_Index'].max())



filteredData13 = uv2014.Date == "13/04/2014"

print ((uv2014[filteredData13])['UV_Index'].max())



filteredData14 = uv2014.Date == "14/04/2014"

print ((uv2014[filteredData14])['UV_Index'].max())



filteredData15 = uv2014.Date == "15/04/2014"

print ((uv2014[filteredData15])['UV_Index'].max())



filteredData16 = uv2014.Date == "16/04/2014"

print ((uv2014[filteredData16])['UV_Index'].max())



filteredData17 = uv2014.Date == "17/04/2014"

print ((uv2014[filteredData17])['UV_Index'].max())



filteredData18 = uv2014.Date == "18/04/2014"

print ((uv2014[filteredData18])['UV_Index'].max())



filteredData19 = uv2014.Date == "19/04/2014"

print ((uv2014[filteredData19])['UV_Index'].max())



filteredData20 = uv2014.Date == "20/04/2049"

print ((uv2014[filteredData20])['UV_Index'].max())



filteredData21 = uv2014.Date == "21/04/2014"

print ((uv2014[filteredData21])['UV_Index'].max())



filteredData22 = uv2014.Date == "22/04/2014"

print ((uv2014[filteredData22])['UV_Index'].max())



filteredData23 = uv2014.Date == "23/04/2014"

print ((uv2014[filteredData23])['UV_Index'].max())



filteredData24 = uv2014.Date == "24/04/2014"

print ((uv2014[filteredData24])['UV_Index'].max())



filteredData25 = uv2014.Date == "25/04/2014"

print ((uv2014[filteredData25])['UV_Index'].max())



filteredData26 = uv2014.Date == "26/04/2014"

print ((uv2014[filteredData26])['UV_Index'].max())



filteredData27 = uv2014.Date == "27/04/2014"

print ((uv2014[filteredData27])['UV_Index'].max())



filteredData28 = uv2014.Date == "28/04/2014"

print ((uv2014[filteredData28])['UV_Index'].max())



filteredData29 = uv2014.Date == "29/04/2014"

print ((uv2014[filteredData29])['UV_Index'].max())



filteredData30 = uv2014.Date == "30/04/2014"

print ((uv2014[filteredData30])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2015 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2015.csv')



filteredData1 = uv2015.Date == "01/04/2015"

print((uv2015[filteredData1])['UV_Index'].max())



filteredData2 = uv2015.Date == "02/04/2015"

print ((uv2015[filteredData2])['UV_Index'].max())



filteredData3 = uv2015.Date == "03/04/2015"

print ((uv2015[filteredData3])['UV_Index'].max())



filteredData4 = uv2015.Date == "04/04/2015"

print ((uv2015[filteredData4])['UV_Index'].max())



filteredData5 = uv2015.Date == "05/04/2015"

print ((uv2015[filteredData5])['UV_Index'].max())



filteredData6 = uv2015.Date == "06/04/2015"

print ((uv2015[filteredData6])['UV_Index'].max())



filteredData7 = uv2015.Date == "07/04/2015"

print ((uv2015[filteredData7])['UV_Index'].max())



filteredData8 = uv2015.Date == "08/04/2015"

print ((uv2015[filteredData8])['UV_Index'].max())



filteredData9 = uv2015.Date == "09/04/2015"

print ((uv2015[filteredData9])['UV_Index'].max())



filteredData10 = uv2015.Date == "10/04/2015"

print ((uv2015[filteredData10])['UV_Index'].max())



filteredData11 = uv2015.Date == "11/04/2015"

print ((uv2015[filteredData11])['UV_Index'].max())



filteredData12 = uv2015.Date == "12/04/2015"

print ((uv2015[filteredData12])['UV_Index'].max())



filteredData13 = uv2015.Date == "13/04/2015"

print ((uv2015[filteredData13])['UV_Index'].max())



filteredData14 = uv2015.Date == "14/04/2015"

print ((uv2015[filteredData14])['UV_Index'].max())



filteredData15 = uv2015.Date == "15/04/2015"

print ((uv2015[filteredData15])['UV_Index'].max())



filteredData16 = uv2015.Date == "16/04/2015"

print ((uv2015[filteredData16])['UV_Index'].max())



filteredData17 = uv2015.Date == "17/04/2015"

print ((uv2015[filteredData17])['UV_Index'].max())



filteredData18 = uv2015.Date == "18/04/2015"

print ((uv2015[filteredData18])['UV_Index'].max())



filteredData19 = uv2015.Date == "19/04/2015"

print ((uv2015[filteredData19])['UV_Index'].max())



filteredData20 = uv2015.Date == "20/04/2049"

print ((uv2015[filteredData20])['UV_Index'].max())



filteredData21 = uv2015.Date == "21/04/2015"

print ((uv2015[filteredData21])['UV_Index'].max())



filteredData22 = uv2015.Date == "22/04/2015"

print ((uv2015[filteredData22])['UV_Index'].max())



filteredData23 = uv2015.Date == "23/04/2015"

print ((uv2015[filteredData23])['UV_Index'].max())



filteredData24 = uv2015.Date == "24/04/2015"

print ((uv2015[filteredData24])['UV_Index'].max())



filteredData25 = uv2015.Date == "25/04/2015"

print ((uv2015[filteredData25])['UV_Index'].max())



filteredData26 = uv2015.Date == "26/04/2015"

print ((uv2015[filteredData26])['UV_Index'].max())



filteredData27 = uv2015.Date == "27/04/2015"

print ((uv2015[filteredData27])['UV_Index'].max())



filteredData28 = uv2015.Date == "28/04/2015"

print ((uv2015[filteredData28])['UV_Index'].max())



filteredData29 = uv2015.Date == "29/04/2015"

print ((uv2015[filteredData29])['UV_Index'].max())



filteredData30 = uv2015.Date == "30/04/2015"

print ((uv2015[filteredData30])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2016 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2016.csv')



filteredData1 = uv2016.Date == "01/04/2016"

print((uv2016[filteredData1])['UV_Index'].max())



filteredData2 = uv2016.Date == "02/04/2016"

print ((uv2016[filteredData2])['UV_Index'].max())



filteredData3 = uv2016.Date == "03/04/2016"

print ((uv2016[filteredData3])['UV_Index'].max())



filteredData4 = uv2016.Date == "04/04/2016"

print ((uv2016[filteredData4])['UV_Index'].max())



filteredData5 = uv2016.Date == "05/04/2016"

print ((uv2016[filteredData5])['UV_Index'].max())



filteredData6 = uv2016.Date == "06/04/2016"

print ((uv2016[filteredData6])['UV_Index'].max())



filteredData7 = uv2016.Date == "07/04/2016"

print ((uv2016[filteredData7])['UV_Index'].max())



filteredData8 = uv2016.Date == "08/04/2016"

print ((uv2016[filteredData8])['UV_Index'].max())



filteredData9 = uv2016.Date == "09/04/2016"

print ((uv2016[filteredData9])['UV_Index'].max())



filteredData10 = uv2016.Date == "10/04/2016"

print ((uv2016[filteredData10])['UV_Index'].max())



filteredData11 = uv2016.Date == "11/04/2016"

print ((uv2016[filteredData11])['UV_Index'].max())



filteredData12 = uv2016.Date == "12/04/2016"

print ((uv2016[filteredData12])['UV_Index'].max())



filteredData13 = uv2016.Date == "13/04/2016"

print ((uv2016[filteredData13])['UV_Index'].max())



filteredData14 = uv2016.Date == "14/04/2016"

print ((uv2016[filteredData14])['UV_Index'].max())



filteredData15 = uv2016.Date == "15/04/2016"

print ((uv2016[filteredData15])['UV_Index'].max())



filteredData16 = uv2016.Date == "16/04/2016"

print ((uv2016[filteredData16])['UV_Index'].max())



filteredData17 = uv2016.Date == "17/04/2016"

print ((uv2016[filteredData17])['UV_Index'].max())



filteredData18 = uv2016.Date == "18/04/2016"

print ((uv2016[filteredData18])['UV_Index'].max())



filteredData19 = uv2016.Date == "19/04/2016"

print ((uv2016[filteredData19])['UV_Index'].max())



filteredData20 = uv2016.Date == "20/04/2049"

print ((uv2016[filteredData20])['UV_Index'].max())



filteredData21 = uv2016.Date == "21/04/2016"

print ((uv2016[filteredData21])['UV_Index'].max())



filteredData22 = uv2016.Date == "22/04/2016"

print ((uv2016[filteredData22])['UV_Index'].max())



filteredData23 = uv2016.Date == "23/04/2016"

print ((uv2016[filteredData23])['UV_Index'].max())



filteredData24 = uv2016.Date == "24/04/2016"

print ((uv2016[filteredData24])['UV_Index'].max())



filteredData25 = uv2016.Date == "25/04/2016"

print ((uv2016[filteredData25])['UV_Index'].max())



filteredData26 = uv2016.Date == "26/04/2016"

print ((uv2016[filteredData26])['UV_Index'].max())



filteredData27 = uv2016.Date == "27/04/2016"

print ((uv2016[filteredData27])['UV_Index'].max())



filteredData28 = uv2016.Date == "28/04/2016"

print ((uv2016[filteredData28])['UV_Index'].max())



filteredData29 = uv2016.Date == "29/04/2016"

print ((uv2016[filteredData29])['UV_Index'].max())



filteredData30 = uv2016.Date == "30/04/2016"

print ((uv2016[filteredData30])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2017 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2017.csv')



filteredData1 = uv2017.Date == "01/04/2017"

print((uv2017[filteredData1])['UV_Index'].max())



filteredData2 = uv2017.Date == "02/04/2017"

print ((uv2017[filteredData2])['UV_Index'].max())



filteredData3 = uv2017.Date == "03/04/2017"

print ((uv2017[filteredData3])['UV_Index'].max())



filteredData4 = uv2017.Date == "04/04/2017"

print ((uv2017[filteredData4])['UV_Index'].max())



filteredData5 = uv2017.Date == "05/04/2017"

print ((uv2017[filteredData5])['UV_Index'].max())



filteredData6 = uv2017.Date == "06/04/2017"

print ((uv2017[filteredData6])['UV_Index'].max())



filteredData7 = uv2017.Date == "07/04/2017"

print ((uv2017[filteredData7])['UV_Index'].max())



filteredData8 = uv2017.Date == "08/04/2017"

print ((uv2017[filteredData8])['UV_Index'].max())



filteredData9 = uv2017.Date == "09/04/2017"

print ((uv2017[filteredData9])['UV_Index'].max())



filteredData10 = uv2017.Date == "10/04/2017"

print ((uv2017[filteredData10])['UV_Index'].max())



filteredData11 = uv2017.Date == "11/04/2017"

print ((uv2017[filteredData11])['UV_Index'].max())



filteredData12 = uv2017.Date == "12/04/2017"

print ((uv2017[filteredData12])['UV_Index'].max())



filteredData13 = uv2017.Date == "13/04/2017"

print ((uv2017[filteredData13])['UV_Index'].max())



filteredData14 = uv2017.Date == "14/04/2017"

print ((uv2017[filteredData14])['UV_Index'].max())



filteredData15 = uv2017.Date == "15/04/2017"

print ((uv2017[filteredData15])['UV_Index'].max())



filteredData16 = uv2017.Date == "16/04/2017"

print ((uv2017[filteredData16])['UV_Index'].max())



filteredData17 = uv2017.Date == "17/04/2017"

print ((uv2017[filteredData17])['UV_Index'].max())



filteredData18 = uv2017.Date == "18/04/2017"

print ((uv2017[filteredData18])['UV_Index'].max())



filteredData19 = uv2017.Date == "19/04/2017"

print ((uv2017[filteredData19])['UV_Index'].max())



filteredData20 = uv2017.Date == "20/04/2049"

print ((uv2017[filteredData20])['UV_Index'].max())



filteredData21 = uv2017.Date == "21/04/2017"

print ((uv2017[filteredData21])['UV_Index'].max())



filteredData22 = uv2017.Date == "22/04/2017"

print ((uv2017[filteredData22])['UV_Index'].max())



filteredData23 = uv2017.Date == "23/04/2017"

print ((uv2017[filteredData23])['UV_Index'].max())



filteredData24 = uv2017.Date == "24/04/2017"

print ((uv2017[filteredData24])['UV_Index'].max())



filteredData25 = uv2017.Date == "25/04/2017"

print ((uv2017[filteredData25])['UV_Index'].max())



filteredData26 = uv2017.Date == "26/04/2017"

print ((uv2017[filteredData26])['UV_Index'].max())



filteredData27 = uv2017.Date == "27/04/2017"

print ((uv2017[filteredData27])['UV_Index'].max())



filteredData28 = uv2017.Date == "28/04/2017"

print ((uv2017[filteredData28])['UV_Index'].max())



filteredData29 = uv2017.Date == "29/04/2017"

print ((uv2017[filteredData29])['UV_Index'].max())



filteredData30 = uv2017.Date == "30/04/2017"

print ((uv2017[filteredData30])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2018 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2018.csv')



filteredData1 = uv2018.Date == "01/04/2018"

print((uv2018[filteredData1])['UV_Index'].max())



filteredData2 = uv2018.Date == "02/04/2018"

print ((uv2018[filteredData2])['UV_Index'].max())



filteredData3 = uv2018.Date == "03/04/2018"

print ((uv2018[filteredData3])['UV_Index'].max())



filteredData4 = uv2018.Date == "04/04/2018"

print ((uv2018[filteredData4])['UV_Index'].max())



filteredData5 = uv2018.Date == "05/04/2018"

print ((uv2018[filteredData5])['UV_Index'].max())



filteredData6 = uv2018.Date == "06/04/2018"

print ((uv2018[filteredData6])['UV_Index'].max())



filteredData7 = uv2018.Date == "07/04/2018"

print ((uv2018[filteredData7])['UV_Index'].max())



filteredData8 = uv2018.Date == "08/04/2018"

print ((uv2018[filteredData8])['UV_Index'].max())



filteredData9 = uv2018.Date == "09/04/2018"

print ((uv2018[filteredData9])['UV_Index'].max())



filteredData10 = uv2018.Date == "10/04/2018"

print ((uv2018[filteredData10])['UV_Index'].max())



filteredData11 = uv2018.Date == "11/04/2018"

print ((uv2018[filteredData11])['UV_Index'].max())



filteredData12 = uv2018.Date == "12/04/2018"

print ((uv2018[filteredData12])['UV_Index'].max())



filteredData13 = uv2018.Date == "13/04/2018"

print ((uv2018[filteredData13])['UV_Index'].max())



filteredData14 = uv2018.Date == "14/04/2018"

print ((uv2018[filteredData14])['UV_Index'].max())



filteredData15 = uv2018.Date == "15/04/2018"

print ((uv2018[filteredData15])['UV_Index'].max())



filteredData16 = uv2018.Date == "16/04/2018"

print ((uv2018[filteredData16])['UV_Index'].max())



filteredData17 = uv2018.Date == "17/04/2018"

print ((uv2018[filteredData17])['UV_Index'].max())



filteredData18 = uv2018.Date == "18/04/2018"

print ((uv2018[filteredData18])['UV_Index'].max())



filteredData19 = uv2018.Date == "19/04/2018"

print ((uv2018[filteredData19])['UV_Index'].max())



filteredData20 = uv2018.Date == "20/04/2049"

print ((uv2018[filteredData20])['UV_Index'].max())



filteredData21 = uv2018.Date == "21/04/2018"

print ((uv2018[filteredData21])['UV_Index'].max())



filteredData22 = uv2018.Date == "22/04/2018"

print ((uv2018[filteredData22])['UV_Index'].max())



filteredData23 = uv2018.Date == "23/04/2018"

print ((uv2018[filteredData23])['UV_Index'].max())



filteredData24 = uv2018.Date == "24/04/2018"

print ((uv2018[filteredData24])['UV_Index'].max())



filteredData25 = uv2018.Date == "25/04/2018"

print ((uv2018[filteredData25])['UV_Index'].max())



filteredData26 = uv2018.Date == "26/04/2018"

print ((uv2018[filteredData26])['UV_Index'].max())



filteredData27 = uv2018.Date == "27/04/2018"

print ((uv2018[filteredData27])['UV_Index'].max())



filteredData28 = uv2018.Date == "28/04/2018"

print ((uv2018[filteredData28])['UV_Index'].max())



filteredData29 = uv2018.Date == "29/04/2018"

print ((uv2018[filteredData29])['UV_Index'].max())



filteredData30 = uv2018.Date == "30/04/2018"

print ((uv2018[filteredData30])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2019 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2019.csv')



filteredData1 = uv2019.Date == "01/04/2019"

print((uv2019[filteredData1])['UV_Index'].max())



filteredData2 = uv2019.Date == "02/04/2019"

print ((uv2019[filteredData2])['UV_Index'].max())



filteredData3 = uv2019.Date == "03/04/2019"

print ((uv2019[filteredData3])['UV_Index'].max())



filteredData4 = uv2019.Date == "04/04/2019"

print ((uv2019[filteredData4])['UV_Index'].max())



filteredData5 = uv2019.Date == "05/04/2019"

print ((uv2019[filteredData5])['UV_Index'].max())



filteredData6 = uv2019.Date == "06/04/2019"

print ((uv2019[filteredData6])['UV_Index'].max())



filteredData7 = uv2019.Date == "07/04/2019"

print ((uv2019[filteredData7])['UV_Index'].max())



filteredData8 = uv2019.Date == "08/04/2019"

print ((uv2019[filteredData8])['UV_Index'].max())



filteredData9 = uv2019.Date == "09/04/2019"

print ((uv2019[filteredData9])['UV_Index'].max())



filteredData10 = uv2019.Date == "10/04/2019"

print ((uv2019[filteredData10])['UV_Index'].max())



filteredData11 = uv2019.Date == "11/04/2019"

print ((uv2019[filteredData11])['UV_Index'].max())



filteredData12 = uv2019.Date == "12/04/2019"

print ((uv2019[filteredData12])['UV_Index'].max())



filteredData13 = uv2019.Date == "13/04/2019"

print ((uv2019[filteredData13])['UV_Index'].max())



filteredData14 = uv2019.Date == "14/04/2019"

print ((uv2019[filteredData14])['UV_Index'].max())



filteredData15 = uv2019.Date == "15/04/2019"

print ((uv2019[filteredData15])['UV_Index'].max())



filteredData16 = uv2019.Date == "16/04/2019"

print ((uv2019[filteredData16])['UV_Index'].max())



filteredData17 = uv2019.Date == "17/04/2019"

print ((uv2019[filteredData17])['UV_Index'].max())



filteredData18 = uv2019.Date == "18/04/2019"

print ((uv2019[filteredData18])['UV_Index'].max())



filteredData19 = uv2019.Date == "19/04/2019"

print ((uv2019[filteredData19])['UV_Index'].max())



filteredData20 = uv2019.Date == "20/04/2049"

print ((uv2019[filteredData20])['UV_Index'].max())



filteredData21 = uv2019.Date == "21/04/2019"

print ((uv2019[filteredData21])['UV_Index'].max())



filteredData22 = uv2019.Date == "22/04/2019"

print ((uv2019[filteredData22])['UV_Index'].max())



filteredData23 = uv2019.Date == "23/04/2019"

print ((uv2019[filteredData23])['UV_Index'].max())



filteredData24 = uv2019.Date == "24/04/2019"

print ((uv2019[filteredData24])['UV_Index'].max())



filteredData25 = uv2019.Date == "25/04/2019"

print ((uv2019[filteredData25])['UV_Index'].max())



filteredData26 = uv2019.Date == "26/04/2019"

print ((uv2019[filteredData26])['UV_Index'].max())



filteredData27 = uv2019.Date == "27/04/2019"

print ((uv2019[filteredData27])['UV_Index'].max())



filteredData28 = uv2019.Date == "28/04/2019"

print ((uv2019[filteredData28])['UV_Index'].max())



filteredData29 = uv2019.Date == "29/04/2019"

print ((uv2019[filteredData29])['UV_Index'].max())



filteredData30 = uv2019.Date == "30/04/2019"

print ((uv2019[filteredData30])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2019 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2019.csv')



filteredData1 = uv2019.Date == "01/05/2019"

print((uv2019[filteredData1])['UV_Index'].max())



filteredData2 = uv2019.Date == "02/05/2019"

print ((uv2019[filteredData2])['UV_Index'].max())



filteredData3 = uv2019.Date == "03/05/2019"

print ((uv2019[filteredData3])['UV_Index'].max())



filteredData4 = uv2019.Date == "04/05/2019"

print ((uv2019[filteredData4])['UV_Index'].max())



filteredData5 = uv2019.Date == "05/05/2019"

print ((uv2019[filteredData5])['UV_Index'].max())



filteredData6 = uv2019.Date == "06/05/2019"

print ((uv2019[filteredData6])['UV_Index'].max())



filteredData7 = uv2019.Date == "07/05/2019"

print ((uv2019[filteredData7])['UV_Index'].max())



filteredData8 = uv2019.Date == "08/05/2019"

print ((uv2019[filteredData8])['UV_Index'].max())



filteredData9 = uv2019.Date == "09/05/2019"

print ((uv2019[filteredData9])['UV_Index'].max())



filteredData10 = uv2019.Date == "10/05/2019"

print ((uv2019[filteredData10])['UV_Index'].max())



filteredData11 = uv2019.Date == "11/05/2019"

print ((uv2019[filteredData11])['UV_Index'].max())



filteredData12 = uv2019.Date == "12/05/2019"

print ((uv2019[filteredData12])['UV_Index'].max())



filteredData13 = uv2019.Date == "13/05/2019"

print ((uv2019[filteredData13])['UV_Index'].max())



filteredData14 = uv2019.Date == "14/05/2019"

print ((uv2019[filteredData14])['UV_Index'].max())



filteredData15 = uv2019.Date == "15/05/2019"

print ((uv2019[filteredData15])['UV_Index'].max())



filteredData16 = uv2019.Date == "16/05/2019"

print ((uv2019[filteredData16])['UV_Index'].max())



filteredData17 = uv2019.Date == "17/05/2019"

print ((uv2019[filteredData17])['UV_Index'].max())



filteredData18 = uv2019.Date == "18/05/2019"

print ((uv2019[filteredData18])['UV_Index'].max())



filteredData19 = uv2019.Date == "19/05/2019"

print ((uv2019[filteredData19])['UV_Index'].max())



filteredData20 = uv2019.Date == "20/05/2059"

print ((uv2019[filteredData20])['UV_Index'].max())



filteredData21 = uv2019.Date == "21/05/2019"

print ((uv2019[filteredData21])['UV_Index'].max())



filteredData22 = uv2019.Date == "22/05/2019"

print ((uv2019[filteredData22])['UV_Index'].max())



filteredData23 = uv2019.Date == "23/05/2019"

print ((uv2019[filteredData23])['UV_Index'].max())



filteredData24 = uv2019.Date == "24/05/2019"

print ((uv2019[filteredData24])['UV_Index'].max())



filteredData25 = uv2019.Date == "25/05/2019"

print ((uv2019[filteredData25])['UV_Index'].max())



filteredData26 = uv2019.Date == "26/05/2019"

print ((uv2019[filteredData26])['UV_Index'].max())



filteredData27 = uv2019.Date == "27/05/2019"

print ((uv2019[filteredData27])['UV_Index'].max())



filteredData28 = uv2019.Date == "28/05/2019"

print ((uv2019[filteredData28])['UV_Index'].max())



filteredData29 = uv2019.Date == "29/05/2019"

print ((uv2019[filteredData29])['UV_Index'].max())



filteredData30 = uv2019.Date == "30/05/2019"

print ((uv2019[filteredData30])['UV_Index'].max())



filteredData31 = uv2019.Date == "31/05/2019" 

print ((uv2019[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2018 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2018.csv')



filteredData1 = uv2018.Date == "01/05/2018"

print((uv2018[filteredData1])['UV_Index'].max())



filteredData2 = uv2018.Date == "02/05/2018"

print ((uv2018[filteredData2])['UV_Index'].max())



filteredData3 = uv2018.Date == "03/05/2018"

print ((uv2018[filteredData3])['UV_Index'].max())



filteredData4 = uv2018.Date == "04/05/2018"

print ((uv2018[filteredData4])['UV_Index'].max())



filteredData5 = uv2018.Date == "05/05/2018"

print ((uv2018[filteredData5])['UV_Index'].max())



filteredData6 = uv2018.Date == "06/05/2018"

print ((uv2018[filteredData6])['UV_Index'].max())



filteredData7 = uv2018.Date == "07/05/2018"

print ((uv2018[filteredData7])['UV_Index'].max())



filteredData8 = uv2018.Date == "08/05/2018"

print ((uv2018[filteredData8])['UV_Index'].max())



filteredData9 = uv2018.Date == "09/05/2018"

print ((uv2018[filteredData9])['UV_Index'].max())



filteredData10 = uv2018.Date == "10/05/2018"

print ((uv2018[filteredData10])['UV_Index'].max())



filteredData11 = uv2018.Date == "11/05/2018"

print ((uv2018[filteredData11])['UV_Index'].max())



filteredData12 = uv2018.Date == "12/05/2018"

print ((uv2018[filteredData12])['UV_Index'].max())



filteredData13 = uv2018.Date == "13/05/2018"

print ((uv2018[filteredData13])['UV_Index'].max())



filteredData14 = uv2018.Date == "14/05/2018"

print ((uv2018[filteredData14])['UV_Index'].max())



filteredData15 = uv2018.Date == "15/05/2018"

print ((uv2018[filteredData15])['UV_Index'].max())



filteredData16 = uv2018.Date == "16/05/2018"

print ((uv2018[filteredData16])['UV_Index'].max())



filteredData17 = uv2018.Date == "17/05/2018"

print ((uv2018[filteredData17])['UV_Index'].max())



filteredData18 = uv2018.Date == "18/05/2018"

print ((uv2018[filteredData18])['UV_Index'].max())



filteredData19 = uv2018.Date == "19/05/2018"

print ((uv2018[filteredData19])['UV_Index'].max())



filteredData20 = uv2018.Date == "20/05/2059"

print ((uv2018[filteredData20])['UV_Index'].max())



filteredData21 = uv2018.Date == "21/05/2018"

print ((uv2018[filteredData21])['UV_Index'].max())



filteredData22 = uv2018.Date == "22/05/2018"

print ((uv2018[filteredData22])['UV_Index'].max())



filteredData23 = uv2018.Date == "23/05/2018"

print ((uv2018[filteredData23])['UV_Index'].max())



filteredData24 = uv2018.Date == "24/05/2018"

print ((uv2018[filteredData24])['UV_Index'].max())



filteredData25 = uv2018.Date == "25/05/2018"

print ((uv2018[filteredData25])['UV_Index'].max())



filteredData26 = uv2018.Date == "26/05/2018"

print ((uv2018[filteredData26])['UV_Index'].max())



filteredData27 = uv2018.Date == "27/05/2018"

print ((uv2018[filteredData27])['UV_Index'].max())



filteredData28 = uv2018.Date == "28/05/2018"

print ((uv2018[filteredData28])['UV_Index'].max())



filteredData29 = uv2018.Date == "29/05/2018"

print ((uv2018[filteredData29])['UV_Index'].max())



filteredData30 = uv2018.Date == "30/05/2018"

print ((uv2018[filteredData30])['UV_Index'].max())



filteredData31 = uv2018.Date == "31/05/2018"

print ((uv2018[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2017 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2017.csv')



filteredData1 = uv2017.Date == "01/05/2017"

print((uv2017[filteredData1])['UV_Index'].max())



filteredData2 = uv2017.Date == "02/05/2017"

print ((uv2017[filteredData2])['UV_Index'].max())



filteredData3 = uv2017.Date == "03/05/2017"

print ((uv2017[filteredData3])['UV_Index'].max())



filteredData4 = uv2017.Date == "04/05/2017"

print ((uv2017[filteredData4])['UV_Index'].max())



filteredData5 = uv2017.Date == "05/05/2017"

print ((uv2017[filteredData5])['UV_Index'].max())



filteredData6 = uv2017.Date == "06/05/2017"

print ((uv2017[filteredData6])['UV_Index'].max())



filteredData7 = uv2017.Date == "07/05/2017"

print ((uv2017[filteredData7])['UV_Index'].max())



filteredData8 = uv2017.Date == "08/05/2017"

print ((uv2017[filteredData8])['UV_Index'].max())



filteredData9 = uv2017.Date == "09/05/2017"

print ((uv2017[filteredData9])['UV_Index'].max())



filteredData10 = uv2017.Date == "10/05/2017"

print ((uv2017[filteredData10])['UV_Index'].max())



filteredData11 = uv2017.Date == "11/05/2017"

print ((uv2017[filteredData11])['UV_Index'].max())



filteredData12 = uv2017.Date == "12/05/2017"

print ((uv2017[filteredData12])['UV_Index'].max())



filteredData13 = uv2017.Date == "13/05/2017"

print ((uv2017[filteredData13])['UV_Index'].max())



filteredData14 = uv2017.Date == "14/05/2017"

print ((uv2017[filteredData14])['UV_Index'].max())



filteredData15 = uv2017.Date == "15/05/2017"

print ((uv2017[filteredData15])['UV_Index'].max())



filteredData16 = uv2017.Date == "16/05/2017"

print ((uv2017[filteredData16])['UV_Index'].max())



filteredData17 = uv2017.Date == "17/05/2017"

print ((uv2017[filteredData17])['UV_Index'].max())



filteredData18 = uv2017.Date == "18/05/2017"

print ((uv2017[filteredData18])['UV_Index'].max())



filteredData19 = uv2017.Date == "19/05/2017"

print ((uv2017[filteredData19])['UV_Index'].max())



filteredData20 = uv2017.Date == "20/05/2059"

print ((uv2017[filteredData20])['UV_Index'].max())



filteredData21 = uv2017.Date == "21/05/2017"

print ((uv2017[filteredData21])['UV_Index'].max())



filteredData22 = uv2017.Date == "22/05/2017"

print ((uv2017[filteredData22])['UV_Index'].max())



filteredData23 = uv2017.Date == "23/05/2017"

print ((uv2017[filteredData23])['UV_Index'].max())



filteredData24 = uv2017.Date == "24/05/2017"

print ((uv2017[filteredData24])['UV_Index'].max())



filteredData25 = uv2017.Date == "25/05/2017"

print ((uv2017[filteredData25])['UV_Index'].max())



filteredData26 = uv2017.Date == "26/05/2017"

print ((uv2017[filteredData26])['UV_Index'].max())



filteredData27 = uv2017.Date == "27/05/2017"

print ((uv2017[filteredData27])['UV_Index'].max())



filteredData28 = uv2017.Date == "28/05/2017"

print ((uv2017[filteredData28])['UV_Index'].max())



filteredData29 = uv2017.Date == "29/05/2017"

print ((uv2017[filteredData29])['UV_Index'].max())



filteredData30 = uv2017.Date == "30/05/2017"

print ((uv2017[filteredData30])['UV_Index'].max())



filteredData31 = uv2017.Date == "31/05/2017"

print ((uv2017[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2016 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2016.csv')



filteredData1 = uv2016.Date == "01/05/2016"

print((uv2016[filteredData1])['UV_Index'].max())



filteredData2 = uv2016.Date == "02/05/2016"

print ((uv2016[filteredData2])['UV_Index'].max())



filteredData3 = uv2016.Date == "03/05/2016"

print ((uv2016[filteredData3])['UV_Index'].max())



filteredData4 = uv2016.Date == "04/05/2016"

print ((uv2016[filteredData4])['UV_Index'].max())



filteredData5 = uv2016.Date == "05/05/2016"

print ((uv2016[filteredData5])['UV_Index'].max())



filteredData6 = uv2016.Date == "06/05/2016"

print ((uv2016[filteredData6])['UV_Index'].max())



filteredData7 = uv2016.Date == "07/05/2016"

print ((uv2016[filteredData7])['UV_Index'].max())



filteredData8 = uv2016.Date == "08/05/2016"

print ((uv2016[filteredData8])['UV_Index'].max())



filteredData9 = uv2016.Date == "09/05/2016"

print ((uv2016[filteredData9])['UV_Index'].max())



filteredData10 = uv2016.Date == "10/05/2016"

print ((uv2016[filteredData10])['UV_Index'].max())



filteredData11 = uv2016.Date == "11/05/2016"

print ((uv2016[filteredData11])['UV_Index'].max())



filteredData12 = uv2016.Date == "12/05/2016"

print ((uv2016[filteredData12])['UV_Index'].max())



filteredData13 = uv2016.Date == "13/05/2016"

print ((uv2016[filteredData13])['UV_Index'].max())



filteredData14 = uv2016.Date == "14/05/2016"

print ((uv2016[filteredData14])['UV_Index'].max())



filteredData15 = uv2016.Date == "15/05/2016"

print ((uv2016[filteredData15])['UV_Index'].max())



filteredData16 = uv2016.Date == "16/05/2016"

print ((uv2016[filteredData16])['UV_Index'].max())



filteredData17 = uv2016.Date == "17/05/2016"

print ((uv2016[filteredData17])['UV_Index'].max())



filteredData18 = uv2016.Date == "18/05/2016"

print ((uv2016[filteredData18])['UV_Index'].max())



filteredData19 = uv2016.Date == "19/05/2016"

print ((uv2016[filteredData19])['UV_Index'].max())



filteredData20 = uv2016.Date == "20/05/2059"

print ((uv2016[filteredData20])['UV_Index'].max())



filteredData21 = uv2016.Date == "21/05/2016"

print ((uv2016[filteredData21])['UV_Index'].max())



filteredData22 = uv2016.Date == "22/05/2016"

print ((uv2016[filteredData22])['UV_Index'].max())



filteredData23 = uv2016.Date == "23/05/2016"

print ((uv2016[filteredData23])['UV_Index'].max())



filteredData24 = uv2016.Date == "24/05/2016"

print ((uv2016[filteredData24])['UV_Index'].max())



filteredData25 = uv2016.Date == "25/05/2016"

print ((uv2016[filteredData25])['UV_Index'].max())



filteredData26 = uv2016.Date == "26/05/2016"

print ((uv2016[filteredData26])['UV_Index'].max())



filteredData27 = uv2016.Date == "27/05/2016"

print ((uv2016[filteredData27])['UV_Index'].max())



filteredData28 = uv2016.Date == "28/05/2016"

print ((uv2016[filteredData28])['UV_Index'].max())



filteredData29 = uv2016.Date == "29/05/2016"

print ((uv2016[filteredData29])['UV_Index'].max())



filteredData30 = uv2016.Date == "30/05/2016"

print ((uv2016[filteredData30])['UV_Index'].max())



filteredData31 = uv2016.Date == "31/05/2016"

print ((uv2016[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2015 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2015.csv')



filteredData1 = uv2015.Date == "01/05/2015"

print((uv2015[filteredData1])['UV_Index'].max())



filteredData2 = uv2015.Date == "02/05/2015"

print ((uv2015[filteredData2])['UV_Index'].max())



filteredData3 = uv2015.Date == "03/05/2015"

print ((uv2015[filteredData3])['UV_Index'].max())



filteredData4 = uv2015.Date == "04/05/2015"

print ((uv2015[filteredData4])['UV_Index'].max())



filteredData5 = uv2015.Date == "05/05/2015"

print ((uv2015[filteredData5])['UV_Index'].max())



filteredData6 = uv2015.Date == "06/05/2015"

print ((uv2015[filteredData6])['UV_Index'].max())



filteredData7 = uv2015.Date == "07/05/2015"

print ((uv2015[filteredData7])['UV_Index'].max())



filteredData8 = uv2015.Date == "08/05/2015"

print ((uv2015[filteredData8])['UV_Index'].max())



filteredData9 = uv2015.Date == "09/05/2015"

print ((uv2015[filteredData9])['UV_Index'].max())



filteredData10 = uv2015.Date == "10/05/2015"

print ((uv2015[filteredData10])['UV_Index'].max())



filteredData11 = uv2015.Date == "11/05/2015"

print ((uv2015[filteredData11])['UV_Index'].max())



filteredData12 = uv2015.Date == "12/05/2015"

print ((uv2015[filteredData12])['UV_Index'].max())



filteredData13 = uv2015.Date == "13/05/2015"

print ((uv2015[filteredData13])['UV_Index'].max())



filteredData14 = uv2015.Date == "14/05/2015"

print ((uv2015[filteredData14])['UV_Index'].max())



filteredData15 = uv2015.Date == "15/05/2015"

print ((uv2015[filteredData15])['UV_Index'].max())



filteredData16 = uv2015.Date == "16/05/2015"

print ((uv2015[filteredData16])['UV_Index'].max())



filteredData17 = uv2015.Date == "17/05/2015"

print ((uv2015[filteredData17])['UV_Index'].max())



filteredData18 = uv2015.Date == "18/05/2015"

print ((uv2015[filteredData18])['UV_Index'].max())



filteredData19 = uv2015.Date == "19/05/2015"

print ((uv2015[filteredData19])['UV_Index'].max())



filteredData20 = uv2015.Date == "20/05/2059"

print ((uv2015[filteredData20])['UV_Index'].max())



filteredData21 = uv2015.Date == "21/05/2015"

print ((uv2015[filteredData21])['UV_Index'].max())



filteredData22 = uv2015.Date == "22/05/2015"

print ((uv2015[filteredData22])['UV_Index'].max())



filteredData23 = uv2015.Date == "23/05/2015"

print ((uv2015[filteredData23])['UV_Index'].max())



filteredData24 = uv2015.Date == "24/05/2015"

print ((uv2015[filteredData24])['UV_Index'].max())



filteredData25 = uv2015.Date == "25/05/2015"

print ((uv2015[filteredData25])['UV_Index'].max())



filteredData26 = uv2015.Date == "26/05/2015"

print ((uv2015[filteredData26])['UV_Index'].max())



filteredData27 = uv2015.Date == "27/05/2015"

print ((uv2015[filteredData27])['UV_Index'].max())



filteredData28 = uv2015.Date == "28/05/2015"

print ((uv2015[filteredData28])['UV_Index'].max())



filteredData29 = uv2015.Date == "29/05/2015"

print ((uv2015[filteredData29])['UV_Index'].max())



filteredData30 = uv2015.Date == "30/05/2015"

print ((uv2015[filteredData30])['UV_Index'].max())



filteredData31 = uv2015.Date == "31/05/2015"

print ((uv2015[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2014 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2014.csv')



filteredData1 = uv2014.Date == "01/05/2014"

print((uv2014[filteredData1])['UV_Index'].max())



filteredData2 = uv2014.Date == "02/05/2014"

print ((uv2014[filteredData2])['UV_Index'].max())



filteredData3 = uv2014.Date == "03/05/2014"

print ((uv2014[filteredData3])['UV_Index'].max())



filteredData4 = uv2014.Date == "04/05/2014"

print ((uv2014[filteredData4])['UV_Index'].max())



filteredData5 = uv2014.Date == "05/05/2014"

print ((uv2014[filteredData5])['UV_Index'].max())



filteredData6 = uv2014.Date == "06/05/2014"

print ((uv2014[filteredData6])['UV_Index'].max())



filteredData7 = uv2014.Date == "07/05/2014"

print ((uv2014[filteredData7])['UV_Index'].max())



filteredData8 = uv2014.Date == "08/05/2014"

print ((uv2014[filteredData8])['UV_Index'].max())



filteredData9 = uv2014.Date == "09/05/2014"

print ((uv2014[filteredData9])['UV_Index'].max())



filteredData10 = uv2014.Date == "10/05/2014"

print ((uv2014[filteredData10])['UV_Index'].max())



filteredData11 = uv2014.Date == "11/05/2014"

print ((uv2014[filteredData11])['UV_Index'].max())



filteredData12 = uv2014.Date == "12/05/2014"

print ((uv2014[filteredData12])['UV_Index'].max())



filteredData13 = uv2014.Date == "13/05/2014"

print ((uv2014[filteredData13])['UV_Index'].max())



filteredData14 = uv2014.Date == "14/05/2014"

print ((uv2014[filteredData14])['UV_Index'].max())



filteredData15 = uv2014.Date == "15/05/2014"

print ((uv2014[filteredData15])['UV_Index'].max())



filteredData16 = uv2014.Date == "16/05/2014"

print ((uv2014[filteredData16])['UV_Index'].max())



filteredData17 = uv2014.Date == "17/05/2014"

print ((uv2014[filteredData17])['UV_Index'].max())



filteredData18 = uv2014.Date == "18/05/2014"

print ((uv2014[filteredData18])['UV_Index'].max())



filteredData19 = uv2014.Date == "19/05/2014"

print ((uv2014[filteredData19])['UV_Index'].max())



filteredData20 = uv2014.Date == "20/05/2059"

print ((uv2014[filteredData20])['UV_Index'].max())



filteredData21 = uv2014.Date == "21/05/2014"

print ((uv2014[filteredData21])['UV_Index'].max())



filteredData22 = uv2014.Date == "22/05/2014"

print ((uv2014[filteredData22])['UV_Index'].max())



filteredData23 = uv2014.Date == "23/05/2014"

print ((uv2014[filteredData23])['UV_Index'].max())



filteredData24 = uv2014.Date == "24/05/2014"

print ((uv2014[filteredData24])['UV_Index'].max())



filteredData25 = uv2014.Date == "25/05/2014"

print ((uv2014[filteredData25])['UV_Index'].max())



filteredData26 = uv2014.Date == "26/05/2014"

print ((uv2014[filteredData26])['UV_Index'].max())



filteredData27 = uv2014.Date == "27/05/2014"

print ((uv2014[filteredData27])['UV_Index'].max())



filteredData28 = uv2014.Date == "28/05/2014"

print ((uv2014[filteredData28])['UV_Index'].max())



filteredData29 = uv2014.Date == "29/05/2014"

print ((uv2014[filteredData29])['UV_Index'].max())



filteredData30 = uv2014.Date == "30/05/2014"

print ((uv2014[filteredData30])['UV_Index'].max())



filteredData31 = uv2014.Date == "31/05/2014"

print ((uv2014[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2013 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2013.csv')



filteredData1 = uv2013.Date == "01/05/2013"

print((uv2013[filteredData1])['UV_Index'].max())



filteredData2 = uv2013.Date == "02/05/2013"

print ((uv2013[filteredData2])['UV_Index'].max())



filteredData3 = uv2013.Date == "03/05/2013"

print ((uv2013[filteredData3])['UV_Index'].max())



filteredData4 = uv2013.Date == "04/05/2013"

print ((uv2013[filteredData4])['UV_Index'].max())



filteredData5 = uv2013.Date == "05/05/2013"

print ((uv2013[filteredData5])['UV_Index'].max())



filteredData6 = uv2013.Date == "06/05/2013"

print ((uv2013[filteredData6])['UV_Index'].max())



filteredData7 = uv2013.Date == "07/05/2013"

print ((uv2013[filteredData7])['UV_Index'].max())



filteredData8 = uv2013.Date == "08/05/2013"

print ((uv2013[filteredData8])['UV_Index'].max())



filteredData9 = uv2013.Date == "09/05/2013"

print ((uv2013[filteredData9])['UV_Index'].max())



filteredData10 = uv2013.Date == "10/05/2013"

print ((uv2013[filteredData10])['UV_Index'].max())



filteredData11 = uv2013.Date == "11/05/2013"

print ((uv2013[filteredData11])['UV_Index'].max())



filteredData12 = uv2013.Date == "12/05/2013"

print ((uv2013[filteredData12])['UV_Index'].max())



filteredData13 = uv2013.Date == "13/05/2013"

print ((uv2013[filteredData13])['UV_Index'].max())



filteredData14 = uv2013.Date == "14/05/2013"

print ((uv2013[filteredData14])['UV_Index'].max())



filteredData15 = uv2013.Date == "15/05/2013"

print ((uv2013[filteredData15])['UV_Index'].max())



filteredData16 = uv2013.Date == "16/05/2013"

print ((uv2013[filteredData16])['UV_Index'].max())



filteredData17 = uv2013.Date == "17/05/2013"

print ((uv2013[filteredData17])['UV_Index'].max())



filteredData18 = uv2013.Date == "18/05/2013"

print ((uv2013[filteredData18])['UV_Index'].max())



filteredData19 = uv2013.Date == "19/05/2013"

print ((uv2013[filteredData19])['UV_Index'].max())



filteredData20 = uv2013.Date == "20/05/2059"

print ((uv2013[filteredData20])['UV_Index'].max())



filteredData21 = uv2013.Date == "21/05/2013"

print ((uv2013[filteredData21])['UV_Index'].max())



filteredData22 = uv2013.Date == "22/05/2013"

print ((uv2013[filteredData22])['UV_Index'].max())



filteredData23 = uv2013.Date == "23/05/2013"

print ((uv2013[filteredData23])['UV_Index'].max())



filteredData24 = uv2013.Date == "24/05/2013"

print ((uv2013[filteredData24])['UV_Index'].max())



filteredData25 = uv2013.Date == "25/05/2013"

print ((uv2013[filteredData25])['UV_Index'].max())



filteredData26 = uv2013.Date == "26/05/2013"

print ((uv2013[filteredData26])['UV_Index'].max())



filteredData27 = uv2013.Date == "27/05/2013"

print ((uv2013[filteredData27])['UV_Index'].max())



filteredData28 = uv2013.Date == "28/05/2013"

print ((uv2013[filteredData28])['UV_Index'].max())



filteredData29 = uv2013.Date == "29/05/2013"

print ((uv2013[filteredData29])['UV_Index'].max())



filteredData30 = uv2013.Date == "30/05/2013"

print ((uv2013[filteredData30])['UV_Index'].max())



filteredData31 = uv2013.Date == "31/05/2013"

print ((uv2013[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2012 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2012.csv')



filteredData1 = uv2012.Date == "01/05/2012"

print((uv2012[filteredData1])['UV_Index'].max())



filteredData2 = uv2012.Date == "02/05/2012"

print ((uv2012[filteredData2])['UV_Index'].max())



filteredData3 = uv2012.Date == "03/05/2012"

print ((uv2012[filteredData3])['UV_Index'].max())



filteredData4 = uv2012.Date == "04/05/2012"

print ((uv2012[filteredData4])['UV_Index'].max())



filteredData5 = uv2012.Date == "05/05/2012"

print ((uv2012[filteredData5])['UV_Index'].max())



filteredData6 = uv2012.Date == "06/05/2012"

print ((uv2012[filteredData6])['UV_Index'].max())



filteredData7 = uv2012.Date == "07/05/2012"

print ((uv2012[filteredData7])['UV_Index'].max())



filteredData8 = uv2012.Date == "08/05/2012"

print ((uv2012[filteredData8])['UV_Index'].max())



filteredData9 = uv2012.Date == "09/05/2012"

print ((uv2012[filteredData9])['UV_Index'].max())



filteredData10 = uv2012.Date == "10/05/2012"

print ((uv2012[filteredData10])['UV_Index'].max())



filteredData11 = uv2012.Date == "11/05/2012"

print ((uv2012[filteredData11])['UV_Index'].max())



filteredData12 = uv2012.Date == "12/05/2012"

print ((uv2012[filteredData12])['UV_Index'].max())



filteredData13 = uv2012.Date == "13/05/2012"

print ((uv2012[filteredData13])['UV_Index'].max())



filteredData14 = uv2012.Date == "14/05/2012"

print ((uv2012[filteredData14])['UV_Index'].max())



filteredData15 = uv2012.Date == "15/05/2012"

print ((uv2012[filteredData15])['UV_Index'].max())



filteredData16 = uv2012.Date == "16/05/2012"

print ((uv2012[filteredData16])['UV_Index'].max())



filteredData17 = uv2012.Date == "17/05/2012"

print ((uv2012[filteredData17])['UV_Index'].max())



filteredData18 = uv2012.Date == "18/05/2012"

print ((uv2012[filteredData18])['UV_Index'].max())



filteredData19 = uv2012.Date == "19/05/2012"

print ((uv2012[filteredData19])['UV_Index'].max())



filteredData20 = uv2012.Date == "20/05/2059"

print ((uv2012[filteredData20])['UV_Index'].max())



filteredData21 = uv2012.Date == "21/05/2012"

print ((uv2012[filteredData21])['UV_Index'].max())



filteredData22 = uv2012.Date == "22/05/2012"

print ((uv2012[filteredData22])['UV_Index'].max())



filteredData23 = uv2012.Date == "23/05/2012"

print ((uv2012[filteredData23])['UV_Index'].max())



filteredData24 = uv2012.Date == "24/05/2012"

print ((uv2012[filteredData24])['UV_Index'].max())



filteredData25 = uv2012.Date == "25/05/2012"

print ((uv2012[filteredData25])['UV_Index'].max())



filteredData26 = uv2012.Date == "26/05/2012"

print ((uv2012[filteredData26])['UV_Index'].max())



filteredData27 = uv2012.Date == "27/05/2012"

print ((uv2012[filteredData27])['UV_Index'].max())



filteredData28 = uv2012.Date == "28/05/2012"

print ((uv2012[filteredData28])['UV_Index'].max())



filteredData29 = uv2012.Date == "29/05/2012"

print ((uv2012[filteredData29])['UV_Index'].max())



filteredData30 = uv2012.Date == "30/05/2012"

print ((uv2012[filteredData30])['UV_Index'].max())



filteredData31 = uv2012.Date == "31/05/2012"

print ((uv2012[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2011 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2011.csv')



filteredData1 = uv2011.Date == "01/05/2011"

print((uv2011[filteredData1])['UV_Index'].max())



filteredData2 = uv2011.Date == "02/05/2011"

print ((uv2011[filteredData2])['UV_Index'].max())



filteredData3 = uv2011.Date == "03/05/2011"

print ((uv2011[filteredData3])['UV_Index'].max())



filteredData4 = uv2011.Date == "04/05/2011"

print ((uv2011[filteredData4])['UV_Index'].max())



filteredData5 = uv2011.Date == "05/05/2011"

print ((uv2011[filteredData5])['UV_Index'].max())



filteredData6 = uv2011.Date == "06/05/2011"

print ((uv2011[filteredData6])['UV_Index'].max())



filteredData7 = uv2011.Date == "07/05/2011"

print ((uv2011[filteredData7])['UV_Index'].max())



filteredData8 = uv2011.Date == "08/05/2011"

print ((uv2011[filteredData8])['UV_Index'].max())



filteredData9 = uv2011.Date == "09/05/2011"

print ((uv2011[filteredData9])['UV_Index'].max())



filteredData10 = uv2011.Date == "10/05/2011"

print ((uv2011[filteredData10])['UV_Index'].max())



filteredData11 = uv2011.Date == "11/05/2011"

print ((uv2011[filteredData11])['UV_Index'].max())



filteredData12 = uv2011.Date == "12/05/2011"

print ((uv2011[filteredData12])['UV_Index'].max())



filteredData13 = uv2011.Date == "13/05/2011"

print ((uv2011[filteredData13])['UV_Index'].max())



filteredData14 = uv2011.Date == "14/05/2011"

print ((uv2011[filteredData14])['UV_Index'].max())



filteredData15 = uv2011.Date == "15/05/2011"

print ((uv2011[filteredData15])['UV_Index'].max())



filteredData16 = uv2011.Date == "16/05/2011"

print ((uv2011[filteredData16])['UV_Index'].max())



filteredData17 = uv2011.Date == "17/05/2011"

print ((uv2011[filteredData17])['UV_Index'].max())



filteredData18 = uv2011.Date == "18/05/2011"

print ((uv2011[filteredData18])['UV_Index'].max())



filteredData19 = uv2011.Date == "19/05/2011"

print ((uv2011[filteredData19])['UV_Index'].max())



filteredData20 = uv2011.Date == "20/05/2059"

print ((uv2011[filteredData20])['UV_Index'].max())



filteredData21 = uv2011.Date == "21/05/2011"

print ((uv2011[filteredData21])['UV_Index'].max())



filteredData22 = uv2011.Date == "22/05/2011"

print ((uv2011[filteredData22])['UV_Index'].max())



filteredData23 = uv2011.Date == "23/05/2011"

print ((uv2011[filteredData23])['UV_Index'].max())



filteredData24 = uv2011.Date == "24/05/2011"

print ((uv2011[filteredData24])['UV_Index'].max())



filteredData25 = uv2011.Date == "25/05/2011"

print ((uv2011[filteredData25])['UV_Index'].max())



filteredData26 = uv2011.Date == "26/05/2011"

print ((uv2011[filteredData26])['UV_Index'].max())



filteredData27 = uv2011.Date == "27/05/2011"

print ((uv2011[filteredData27])['UV_Index'].max())



filteredData28 = uv2011.Date == "28/05/2011"

print ((uv2011[filteredData28])['UV_Index'].max())



filteredData29 = uv2011.Date == "29/05/2011"

print ((uv2011[filteredData29])['UV_Index'].max())



filteredData30 = uv2011.Date == "30/05/2011"

print ((uv2011[filteredData30])['UV_Index'].max())



filteredData31 = uv2011.Date == "31/05/2011"

print ((uv2011[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2010 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2010.csv')



filteredData1 = uv2010.Date == "01/05/2010"

print((uv2010[filteredData1])['UV_Index'].max())



filteredData2 = uv2010.Date == "02/05/2010"

print ((uv2010[filteredData2])['UV_Index'].max())



filteredData3 = uv2010.Date == "03/05/2010"

print ((uv2010[filteredData3])['UV_Index'].max())



filteredData4 = uv2010.Date == "04/05/2010"

print ((uv2010[filteredData4])['UV_Index'].max())



filteredData5 = uv2010.Date == "05/05/2010"

print ((uv2010[filteredData5])['UV_Index'].max())



filteredData6 = uv2010.Date == "06/05/2010"

print ((uv2010[filteredData6])['UV_Index'].max())



filteredData7 = uv2010.Date == "07/05/2010"

print ((uv2010[filteredData7])['UV_Index'].max())



filteredData8 = uv2010.Date == "08/05/2010"

print ((uv2010[filteredData8])['UV_Index'].max())



filteredData9 = uv2010.Date == "09/05/2010"

print ((uv2010[filteredData9])['UV_Index'].max())



filteredData10 = uv2010.Date == "10/05/2010"

print ((uv2010[filteredData10])['UV_Index'].max())



filteredData11 = uv2010.Date == "11/05/2010"

print ((uv2010[filteredData11])['UV_Index'].max())



filteredData12 = uv2010.Date == "12/05/2010"

print ((uv2010[filteredData12])['UV_Index'].max())



filteredData13 = uv2010.Date == "13/05/2010"

print ((uv2010[filteredData13])['UV_Index'].max())



filteredData14 = uv2010.Date == "14/05/2010"

print ((uv2010[filteredData14])['UV_Index'].max())



filteredData15 = uv2010.Date == "15/05/2010"

print ((uv2010[filteredData15])['UV_Index'].max())



filteredData16 = uv2010.Date == "16/05/2010"

print ((uv2010[filteredData16])['UV_Index'].max())



filteredData17 = uv2010.Date == "17/05/2010"

print ((uv2010[filteredData17])['UV_Index'].max())



filteredData18 = uv2010.Date == "18/05/2010"

print ((uv2010[filteredData18])['UV_Index'].max())



filteredData19 = uv2010.Date == "19/05/2010"

print ((uv2010[filteredData19])['UV_Index'].max())



filteredData20 = uv2010.Date == "20/05/2059"

print ((uv2010[filteredData20])['UV_Index'].max())



filteredData21 = uv2010.Date == "21/05/2010"

print ((uv2010[filteredData21])['UV_Index'].max())



filteredData22 = uv2010.Date == "22/05/2010"

print ((uv2010[filteredData22])['UV_Index'].max())



filteredData23 = uv2010.Date == "23/05/2010"

print ((uv2010[filteredData23])['UV_Index'].max())



filteredData24 = uv2010.Date == "24/05/2010"

print ((uv2010[filteredData24])['UV_Index'].max())



filteredData25 = uv2010.Date == "25/05/2010"

print ((uv2010[filteredData25])['UV_Index'].max())



filteredData26 = uv2010.Date == "26/05/2010"

print ((uv2010[filteredData26])['UV_Index'].max())



filteredData27 = uv2010.Date == "27/05/2010"

print ((uv2010[filteredData27])['UV_Index'].max())



filteredData28 = uv2010.Date == "28/05/2010"

print ((uv2010[filteredData28])['UV_Index'].max())



filteredData29 = uv2010.Date == "29/05/2010"

print ((uv2010[filteredData29])['UV_Index'].max())



filteredData30 = uv2010.Date == "30/05/2010"

print ((uv2010[filteredData30])['UV_Index'].max())



filteredData31 = uv2010.Date == "31/05/2010"

print ((uv2010[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2010 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2010.csv')



filteredData1 = uv2010.Date == "01/06/2010"

print((uv2010[filteredData1])['UV_Index'].max())



filteredData2 = uv2010.Date == "02/06/2010"

print ((uv2010[filteredData2])['UV_Index'].max())



filteredData3 = uv2010.Date == "03/06/2010"

print ((uv2010[filteredData3])['UV_Index'].max())



filteredData4 = uv2010.Date == "04/06/2010"

print ((uv2010[filteredData4])['UV_Index'].max())



filteredData5 = uv2010.Date == "05/06/2010"

print ((uv2010[filteredData5])['UV_Index'].max())



filteredData6 = uv2010.Date == "06/06/2010"

print ((uv2010[filteredData6])['UV_Index'].max())



filteredData7 = uv2010.Date == "07/06/2010"

print ((uv2010[filteredData7])['UV_Index'].max())



filteredData8 = uv2010.Date == "08/06/2010"

print ((uv2010[filteredData8])['UV_Index'].max())



filteredData9 = uv2010.Date == "09/06/2010"

print ((uv2010[filteredData9])['UV_Index'].max())



filteredData10 = uv2010.Date == "10/06/2010"

print ((uv2010[filteredData10])['UV_Index'].max())



filteredData11 = uv2010.Date == "11/06/2010"

print ((uv2010[filteredData11])['UV_Index'].max())



filteredData12 = uv2010.Date == "12/06/2010"

print ((uv2010[filteredData12])['UV_Index'].max())



filteredData13 = uv2010.Date == "13/06/2010"

print ((uv2010[filteredData13])['UV_Index'].max())



filteredData14 = uv2010.Date == "14/06/2010"

print ((uv2010[filteredData14])['UV_Index'].max())



filteredData15 = uv2010.Date == "15/06/2010"

print ((uv2010[filteredData15])['UV_Index'].max())



filteredData16 = uv2010.Date == "16/06/2010"

print ((uv2010[filteredData16])['UV_Index'].max())



filteredData17 = uv2010.Date == "17/06/2010"

print ((uv2010[filteredData17])['UV_Index'].max())



filteredData18 = uv2010.Date == "18/06/2010"

print ((uv2010[filteredData18])['UV_Index'].max())



filteredData19 = uv2010.Date == "19/06/2010"

print ((uv2010[filteredData19])['UV_Index'].max())



filteredData20 = uv2010.Date == "20/06/2069"

print ((uv2010[filteredData20])['UV_Index'].max())



filteredData21 = uv2010.Date == "21/06/2010"

print ((uv2010[filteredData21])['UV_Index'].max())



filteredData22 = uv2010.Date == "22/06/2010"

print ((uv2010[filteredData22])['UV_Index'].max())



filteredData23 = uv2010.Date == "23/06/2010"

print ((uv2010[filteredData23])['UV_Index'].max())



filteredData24 = uv2010.Date == "24/06/2010"

print ((uv2010[filteredData24])['UV_Index'].max())



filteredData25 = uv2010.Date == "25/06/2010"

print ((uv2010[filteredData25])['UV_Index'].max())



filteredData26 = uv2010.Date == "26/06/2010"

print ((uv2010[filteredData26])['UV_Index'].max())



filteredData27 = uv2010.Date == "27/06/2010"

print ((uv2010[filteredData27])['UV_Index'].max())



filteredData28 = uv2010.Date == "28/06/2010"

print ((uv2010[filteredData28])['UV_Index'].max())



filteredData29 = uv2010.Date == "29/06/2010"

print ((uv2010[filteredData29])['UV_Index'].max())



filteredData30 = uv2010.Date == "30/06/2010"

print ((uv2010[filteredData30])['UV_Index'].max())



filteredData31 = uv2010.Date == "31/06/2010"

print ((uv2010[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2011 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2011.csv')



filteredData1 = uv2011.Date == "01/06/2011"

print((uv2011[filteredData1])['UV_Index'].max())



filteredData2 = uv2011.Date == "02/06/2011"

print ((uv2011[filteredData2])['UV_Index'].max())



filteredData3 = uv2011.Date == "03/06/2011"

print ((uv2011[filteredData3])['UV_Index'].max())



filteredData4 = uv2011.Date == "04/06/2011"

print ((uv2011[filteredData4])['UV_Index'].max())



filteredData5 = uv2011.Date == "05/06/2011"

print ((uv2011[filteredData5])['UV_Index'].max())



filteredData6 = uv2011.Date == "06/06/2011"

print ((uv2011[filteredData6])['UV_Index'].max())



filteredData7 = uv2011.Date == "07/06/2011"

print ((uv2011[filteredData7])['UV_Index'].max())



filteredData8 = uv2011.Date == "08/06/2011"

print ((uv2011[filteredData8])['UV_Index'].max())



filteredData9 = uv2011.Date == "09/06/2011"

print ((uv2011[filteredData9])['UV_Index'].max())



filteredData10 = uv2011.Date == "10/06/2011"

print ((uv2011[filteredData10])['UV_Index'].max())



filteredData11 = uv2011.Date == "11/06/2011"

print ((uv2011[filteredData11])['UV_Index'].max())



filteredData12 = uv2011.Date == "12/06/2011"

print ((uv2011[filteredData12])['UV_Index'].max())



filteredData13 = uv2011.Date == "13/06/2011"

print ((uv2011[filteredData13])['UV_Index'].max())



filteredData14 = uv2011.Date == "14/06/2011"

print ((uv2011[filteredData14])['UV_Index'].max())



filteredData15 = uv2011.Date == "15/06/2011"

print ((uv2011[filteredData15])['UV_Index'].max())



filteredData16 = uv2011.Date == "16/06/2011"

print ((uv2011[filteredData16])['UV_Index'].max())



filteredData17 = uv2011.Date == "17/06/2011"

print ((uv2011[filteredData17])['UV_Index'].max())



filteredData18 = uv2011.Date == "18/06/2011"

print ((uv2011[filteredData18])['UV_Index'].max())



filteredData19 = uv2011.Date == "19/06/2011"

print ((uv2011[filteredData19])['UV_Index'].max())



filteredData20 = uv2011.Date == "20/06/2069"

print ((uv2011[filteredData20])['UV_Index'].max())



filteredData21 = uv2011.Date == "21/06/2011"

print ((uv2011[filteredData21])['UV_Index'].max())



filteredData22 = uv2011.Date == "22/06/2011"

print ((uv2011[filteredData22])['UV_Index'].max())



filteredData23 = uv2011.Date == "23/06/2011"

print ((uv2011[filteredData23])['UV_Index'].max())



filteredData24 = uv2011.Date == "24/06/2011"

print ((uv2011[filteredData24])['UV_Index'].max())



filteredData25 = uv2011.Date == "25/06/2011"

print ((uv2011[filteredData25])['UV_Index'].max())



filteredData26 = uv2011.Date == "26/06/2011"

print ((uv2011[filteredData26])['UV_Index'].max())



filteredData27 = uv2011.Date == "27/06/2011"

print ((uv2011[filteredData27])['UV_Index'].max())



filteredData28 = uv2011.Date == "28/06/2011"

print ((uv2011[filteredData28])['UV_Index'].max())



filteredData29 = uv2011.Date == "29/06/2011"

print ((uv2011[filteredData29])['UV_Index'].max())



filteredData30 = uv2011.Date == "30/06/2011"

print ((uv2011[filteredData30])['UV_Index'].max())



filteredData31 = uv2011.Date == "31/06/2011"

print ((uv2011[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2012 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2012.csv')



filteredData1 = uv2012.Date == "01/06/2012"

print((uv2012[filteredData1])['UV_Index'].max())



filteredData2 = uv2012.Date == "02/06/2012"

print ((uv2012[filteredData2])['UV_Index'].max())



filteredData3 = uv2012.Date == "03/06/2012"

print ((uv2012[filteredData3])['UV_Index'].max())



filteredData4 = uv2012.Date == "04/06/2012"

print ((uv2012[filteredData4])['UV_Index'].max())



filteredData5 = uv2012.Date == "05/06/2012"

print ((uv2012[filteredData5])['UV_Index'].max())



filteredData6 = uv2012.Date == "06/06/2012"

print ((uv2012[filteredData6])['UV_Index'].max())



filteredData7 = uv2012.Date == "07/06/2012"

print ((uv2012[filteredData7])['UV_Index'].max())



filteredData8 = uv2012.Date == "08/06/2012"

print ((uv2012[filteredData8])['UV_Index'].max())



filteredData9 = uv2012.Date == "09/06/2012"

print ((uv2012[filteredData9])['UV_Index'].max())



filteredData10 = uv2012.Date == "10/06/2012"

print ((uv2012[filteredData10])['UV_Index'].max())



filteredData11 = uv2012.Date == "11/06/2012"

print ((uv2012[filteredData11])['UV_Index'].max())



filteredData12 = uv2012.Date == "12/06/2012"

print ((uv2012[filteredData12])['UV_Index'].max())



filteredData13 = uv2012.Date == "13/06/2012"

print ((uv2012[filteredData13])['UV_Index'].max())



filteredData14 = uv2012.Date == "14/06/2012"

print ((uv2012[filteredData14])['UV_Index'].max())



filteredData15 = uv2012.Date == "15/06/2012"

print ((uv2012[filteredData15])['UV_Index'].max())



filteredData16 = uv2012.Date == "16/06/2012"

print ((uv2012[filteredData16])['UV_Index'].max())



filteredData17 = uv2012.Date == "17/06/2012"

print ((uv2012[filteredData17])['UV_Index'].max())



filteredData18 = uv2012.Date == "18/06/2012"

print ((uv2012[filteredData18])['UV_Index'].max())



filteredData19 = uv2012.Date == "19/06/2012"

print ((uv2012[filteredData19])['UV_Index'].max())



filteredData20 = uv2012.Date == "20/06/2069"

print ((uv2012[filteredData20])['UV_Index'].max())



filteredData21 = uv2012.Date == "21/06/2012"

print ((uv2012[filteredData21])['UV_Index'].max())



filteredData22 = uv2012.Date == "22/06/2012"

print ((uv2012[filteredData22])['UV_Index'].max())



filteredData23 = uv2012.Date == "23/06/2012"

print ((uv2012[filteredData23])['UV_Index'].max())



filteredData24 = uv2012.Date == "24/06/2012"

print ((uv2012[filteredData24])['UV_Index'].max())



filteredData25 = uv2012.Date == "25/06/2012"

print ((uv2012[filteredData25])['UV_Index'].max())



filteredData26 = uv2012.Date == "26/06/2012"

print ((uv2012[filteredData26])['UV_Index'].max())



filteredData27 = uv2012.Date == "27/06/2012"

print ((uv2012[filteredData27])['UV_Index'].max())



filteredData28 = uv2012.Date == "28/06/2012"

print ((uv2012[filteredData28])['UV_Index'].max())



filteredData29 = uv2012.Date == "29/06/2012"

print ((uv2012[filteredData29])['UV_Index'].max())



filteredData30 = uv2012.Date == "30/06/2012"

print ((uv2012[filteredData30])['UV_Index'].max())



filteredData31 = uv2012.Date == "31/06/2012"

print ((uv2012[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2013 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2013.csv')



filteredData1 = uv2013.Date == "01/06/2013"

print((uv2013[filteredData1])['UV_Index'].max())



filteredData2 = uv2013.Date == "02/06/2013"

print ((uv2013[filteredData2])['UV_Index'].max())



filteredData3 = uv2013.Date == "03/06/2013"

print ((uv2013[filteredData3])['UV_Index'].max())



filteredData4 = uv2013.Date == "04/06/2013"

print ((uv2013[filteredData4])['UV_Index'].max())



filteredData5 = uv2013.Date == "05/06/2013"

print ((uv2013[filteredData5])['UV_Index'].max())



filteredData6 = uv2013.Date == "06/06/2013"

print ((uv2013[filteredData6])['UV_Index'].max())



filteredData7 = uv2013.Date == "07/06/2013"

print ((uv2013[filteredData7])['UV_Index'].max())



filteredData8 = uv2013.Date == "08/06/2013"

print ((uv2013[filteredData8])['UV_Index'].max())



filteredData9 = uv2013.Date == "09/06/2013"

print ((uv2013[filteredData9])['UV_Index'].max())



filteredData10 = uv2013.Date == "10/06/2013"

print ((uv2013[filteredData10])['UV_Index'].max())



filteredData11 = uv2013.Date == "11/06/2013"

print ((uv2013[filteredData11])['UV_Index'].max())



filteredData12 = uv2013.Date == "12/06/2013"

print ((uv2013[filteredData12])['UV_Index'].max())



filteredData13 = uv2013.Date == "13/06/2013"

print ((uv2013[filteredData13])['UV_Index'].max())



filteredData14 = uv2013.Date == "14/06/2013"

print ((uv2013[filteredData14])['UV_Index'].max())



filteredData15 = uv2013.Date == "15/06/2013"

print ((uv2013[filteredData15])['UV_Index'].max())



filteredData16 = uv2013.Date == "16/06/2013"

print ((uv2013[filteredData16])['UV_Index'].max())



filteredData17 = uv2013.Date == "17/06/2013"

print ((uv2013[filteredData17])['UV_Index'].max())



filteredData18 = uv2013.Date == "18/06/2013"

print ((uv2013[filteredData18])['UV_Index'].max())



filteredData19 = uv2013.Date == "19/06/2013"

print ((uv2013[filteredData19])['UV_Index'].max())



filteredData20 = uv2013.Date == "20/06/2069"

print ((uv2013[filteredData20])['UV_Index'].max())



filteredData21 = uv2013.Date == "21/06/2013"

print ((uv2013[filteredData21])['UV_Index'].max())



filteredData22 = uv2013.Date == "22/06/2013"

print ((uv2013[filteredData22])['UV_Index'].max())



filteredData23 = uv2013.Date == "23/06/2013"

print ((uv2013[filteredData23])['UV_Index'].max())



filteredData24 = uv2013.Date == "24/06/2013"

print ((uv2013[filteredData24])['UV_Index'].max())



filteredData25 = uv2013.Date == "25/06/2013"

print ((uv2013[filteredData25])['UV_Index'].max())



filteredData26 = uv2013.Date == "26/06/2013"

print ((uv2013[filteredData26])['UV_Index'].max())



filteredData27 = uv2013.Date == "27/06/2013"

print ((uv2013[filteredData27])['UV_Index'].max())



filteredData28 = uv2013.Date == "28/06/2013"

print ((uv2013[filteredData28])['UV_Index'].max())



filteredData29 = uv2013.Date == "29/06/2013"

print ((uv2013[filteredData29])['UV_Index'].max())



filteredData30 = uv2013.Date == "30/06/2013"

print ((uv2013[filteredData30])['UV_Index'].max())



filteredData31 = uv2013.Date == "31/06/2013"

print ((uv2013[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2014 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2014.csv')



filteredData1 = uv2014.Date == "01/06/2014"

print((uv2014[filteredData1])['UV_Index'].max())



filteredData2 = uv2014.Date == "02/06/2014"

print ((uv2014[filteredData2])['UV_Index'].max())



filteredData3 = uv2014.Date == "03/06/2014"

print ((uv2014[filteredData3])['UV_Index'].max())



filteredData4 = uv2014.Date == "04/06/2014"

print ((uv2014[filteredData4])['UV_Index'].max())



filteredData5 = uv2014.Date == "05/06/2014"

print ((uv2014[filteredData5])['UV_Index'].max())



filteredData6 = uv2014.Date == "06/06/2014"

print ((uv2014[filteredData6])['UV_Index'].max())



filteredData7 = uv2014.Date == "07/06/2014"

print ((uv2014[filteredData7])['UV_Index'].max())



filteredData8 = uv2014.Date == "08/06/2014"

print ((uv2014[filteredData8])['UV_Index'].max())



filteredData9 = uv2014.Date == "09/06/2014"

print ((uv2014[filteredData9])['UV_Index'].max())



filteredData10 = uv2014.Date == "10/06/2014"

print ((uv2014[filteredData10])['UV_Index'].max())



filteredData11 = uv2014.Date == "11/06/2014"

print ((uv2014[filteredData11])['UV_Index'].max())



filteredData12 = uv2014.Date == "12/06/2014"

print ((uv2014[filteredData12])['UV_Index'].max())



filteredData13 = uv2014.Date == "13/06/2014"

print ((uv2014[filteredData13])['UV_Index'].max())



filteredData14 = uv2014.Date == "14/06/2014"

print ((uv2014[filteredData14])['UV_Index'].max())



filteredData15 = uv2014.Date == "15/06/2014"

print ((uv2014[filteredData15])['UV_Index'].max())



filteredData16 = uv2014.Date == "16/06/2014"

print ((uv2014[filteredData16])['UV_Index'].max())



filteredData17 = uv2014.Date == "17/06/2014"

print ((uv2014[filteredData17])['UV_Index'].max())



filteredData18 = uv2014.Date == "18/06/2014"

print ((uv2014[filteredData18])['UV_Index'].max())



filteredData19 = uv2014.Date == "19/06/2014"

print ((uv2014[filteredData19])['UV_Index'].max())



filteredData20 = uv2014.Date == "20/06/2069"

print ((uv2014[filteredData20])['UV_Index'].max())



filteredData21 = uv2014.Date == "21/06/2014"

print ((uv2014[filteredData21])['UV_Index'].max())



filteredData22 = uv2014.Date == "22/06/2014"

print ((uv2014[filteredData22])['UV_Index'].max())



filteredData23 = uv2014.Date == "23/06/2014"

print ((uv2014[filteredData23])['UV_Index'].max())



filteredData24 = uv2014.Date == "24/06/2014"

print ((uv2014[filteredData24])['UV_Index'].max())



filteredData25 = uv2014.Date == "25/06/2014"

print ((uv2014[filteredData25])['UV_Index'].max())



filteredData26 = uv2014.Date == "26/06/2014"

print ((uv2014[filteredData26])['UV_Index'].max())



filteredData27 = uv2014.Date == "27/06/2014"

print ((uv2014[filteredData27])['UV_Index'].max())



filteredData28 = uv2014.Date == "28/06/2014"

print ((uv2014[filteredData28])['UV_Index'].max())



filteredData29 = uv2014.Date == "29/06/2014"

print ((uv2014[filteredData29])['UV_Index'].max())



filteredData30 = uv2014.Date == "30/06/2014"

print ((uv2014[filteredData30])['UV_Index'].max())



filteredData31 = uv2014.Date == "31/06/2014"

print ((uv2014[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2015 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2015.csv')



filteredData1 = uv2015.Date == "01/06/2015"

print((uv2015[filteredData1])['UV_Index'].max())



filteredData2 = uv2015.Date == "02/06/2015"

print ((uv2015[filteredData2])['UV_Index'].max())



filteredData3 = uv2015.Date == "03/06/2015"

print ((uv2015[filteredData3])['UV_Index'].max())



filteredData4 = uv2015.Date == "04/06/2015"

print ((uv2015[filteredData4])['UV_Index'].max())



filteredData5 = uv2015.Date == "05/06/2015"

print ((uv2015[filteredData5])['UV_Index'].max())



filteredData6 = uv2015.Date == "06/06/2015"

print ((uv2015[filteredData6])['UV_Index'].max())



filteredData7 = uv2015.Date == "07/06/2015"

print ((uv2015[filteredData7])['UV_Index'].max())



filteredData8 = uv2015.Date == "08/06/2015"

print ((uv2015[filteredData8])['UV_Index'].max())



filteredData9 = uv2015.Date == "09/06/2015"

print ((uv2015[filteredData9])['UV_Index'].max())



filteredData10 = uv2015.Date == "10/06/2015"

print ((uv2015[filteredData10])['UV_Index'].max())



filteredData11 = uv2015.Date == "11/06/2015"

print ((uv2015[filteredData11])['UV_Index'].max())



filteredData12 = uv2015.Date == "12/06/2015"

print ((uv2015[filteredData12])['UV_Index'].max())



filteredData13 = uv2015.Date == "13/06/2015"

print ((uv2015[filteredData13])['UV_Index'].max())



filteredData14 = uv2015.Date == "14/06/2015"

print ((uv2015[filteredData14])['UV_Index'].max())



filteredData15 = uv2015.Date == "15/06/2015"

print ((uv2015[filteredData15])['UV_Index'].max())



filteredData16 = uv2015.Date == "16/06/2015"

print ((uv2015[filteredData16])['UV_Index'].max())



filteredData17 = uv2015.Date == "17/06/2015"

print ((uv2015[filteredData17])['UV_Index'].max())



filteredData18 = uv2015.Date == "18/06/2015"

print ((uv2015[filteredData18])['UV_Index'].max())



filteredData19 = uv2015.Date == "19/06/2015"

print ((uv2015[filteredData19])['UV_Index'].max())



filteredData20 = uv2015.Date == "20/06/2069"

print ((uv2015[filteredData20])['UV_Index'].max())



filteredData21 = uv2015.Date == "21/06/2015"

print ((uv2015[filteredData21])['UV_Index'].max())



filteredData22 = uv2015.Date == "22/06/2015"

print ((uv2015[filteredData22])['UV_Index'].max())



filteredData23 = uv2015.Date == "23/06/2015"

print ((uv2015[filteredData23])['UV_Index'].max())



filteredData24 = uv2015.Date == "24/06/2015"

print ((uv2015[filteredData24])['UV_Index'].max())



filteredData25 = uv2015.Date == "25/06/2015"

print ((uv2015[filteredData25])['UV_Index'].max())



filteredData26 = uv2015.Date == "26/06/2015"

print ((uv2015[filteredData26])['UV_Index'].max())



filteredData27 = uv2015.Date == "27/06/2015"

print ((uv2015[filteredData27])['UV_Index'].max())



filteredData28 = uv2015.Date == "28/06/2015"

print ((uv2015[filteredData28])['UV_Index'].max())



filteredData29 = uv2015.Date == "29/06/2015"

print ((uv2015[filteredData29])['UV_Index'].max())



filteredData30 = uv2015.Date == "30/06/2015"

print ((uv2015[filteredData30])['UV_Index'].max())



filteredData31 = uv2015.Date == "31/06/2015"

print ((uv2015[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2016 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2016.csv')



filteredData1 = uv2016.Date == "01/06/2016"

print((uv2016[filteredData1])['UV_Index'].max())



filteredData2 = uv2016.Date == "02/06/2016"

print ((uv2016[filteredData2])['UV_Index'].max())



filteredData3 = uv2016.Date == "03/06/2016"

print ((uv2016[filteredData3])['UV_Index'].max())



filteredData4 = uv2016.Date == "04/06/2016"

print ((uv2016[filteredData4])['UV_Index'].max())



filteredData5 = uv2016.Date == "05/06/2016"

print ((uv2016[filteredData5])['UV_Index'].max())



filteredData6 = uv2016.Date == "06/06/2016"

print ((uv2016[filteredData6])['UV_Index'].max())



filteredData7 = uv2016.Date == "07/06/2016"

print ((uv2016[filteredData7])['UV_Index'].max())



filteredData8 = uv2016.Date == "08/06/2016"

print ((uv2016[filteredData8])['UV_Index'].max())



filteredData9 = uv2016.Date == "09/06/2016"

print ((uv2016[filteredData9])['UV_Index'].max())



filteredData10 = uv2016.Date == "10/06/2016"

print ((uv2016[filteredData10])['UV_Index'].max())



filteredData11 = uv2016.Date == "11/06/2016"

print ((uv2016[filteredData11])['UV_Index'].max())



filteredData12 = uv2016.Date == "12/06/2016"

print ((uv2016[filteredData12])['UV_Index'].max())



filteredData13 = uv2016.Date == "13/06/2016"

print ((uv2016[filteredData13])['UV_Index'].max())



filteredData14 = uv2016.Date == "14/06/2016"

print ((uv2016[filteredData14])['UV_Index'].max())



filteredData15 = uv2016.Date == "15/06/2016"

print ((uv2016[filteredData15])['UV_Index'].max())



filteredData16 = uv2016.Date == "16/06/2016"

print ((uv2016[filteredData16])['UV_Index'].max())



filteredData17 = uv2016.Date == "17/06/2016"

print ((uv2016[filteredData17])['UV_Index'].max())



filteredData18 = uv2016.Date == "18/06/2016"

print ((uv2016[filteredData18])['UV_Index'].max())



filteredData19 = uv2016.Date == "19/06/2016"

print ((uv2016[filteredData19])['UV_Index'].max())



filteredData20 = uv2016.Date == "20/06/2069"

print ((uv2016[filteredData20])['UV_Index'].max())



filteredData21 = uv2016.Date == "21/06/2016"

print ((uv2016[filteredData21])['UV_Index'].max())



filteredData22 = uv2016.Date == "22/06/2016"

print ((uv2016[filteredData22])['UV_Index'].max())



filteredData23 = uv2016.Date == "23/06/2016"

print ((uv2016[filteredData23])['UV_Index'].max())



filteredData24 = uv2016.Date == "24/06/2016"

print ((uv2016[filteredData24])['UV_Index'].max())



filteredData25 = uv2016.Date == "25/06/2016"

print ((uv2016[filteredData25])['UV_Index'].max())



filteredData26 = uv2016.Date == "26/06/2016"

print ((uv2016[filteredData26])['UV_Index'].max())



filteredData27 = uv2016.Date == "27/06/2016"

print ((uv2016[filteredData27])['UV_Index'].max())



filteredData28 = uv2016.Date == "28/06/2016"

print ((uv2016[filteredData28])['UV_Index'].max())



filteredData29 = uv2016.Date == "29/06/2016"

print ((uv2016[filteredData29])['UV_Index'].max())



filteredData30 = uv2016.Date == "30/06/2016"

print ((uv2016[filteredData30])['UV_Index'].max())



filteredData31 = uv2016.Date == "31/06/2016"

print ((uv2016[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2017 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2017.csv')



filteredData1 = uv2017.Date == "01/06/2017"

print((uv2017[filteredData1])['UV_Index'].max())



filteredData2 = uv2017.Date == "02/06/2017"

print ((uv2017[filteredData2])['UV_Index'].max())



filteredData3 = uv2017.Date == "03/06/2017"

print ((uv2017[filteredData3])['UV_Index'].max())



filteredData4 = uv2017.Date == "04/06/2017"

print ((uv2017[filteredData4])['UV_Index'].max())



filteredData5 = uv2017.Date == "05/06/2017"

print ((uv2017[filteredData5])['UV_Index'].max())



filteredData6 = uv2017.Date == "06/06/2017"

print ((uv2017[filteredData6])['UV_Index'].max())



filteredData7 = uv2017.Date == "07/06/2017"

print ((uv2017[filteredData7])['UV_Index'].max())



filteredData8 = uv2017.Date == "08/06/2017"

print ((uv2017[filteredData8])['UV_Index'].max())



filteredData9 = uv2017.Date == "09/06/2017"

print ((uv2017[filteredData9])['UV_Index'].max())



filteredData10 = uv2017.Date == "10/06/2017"

print ((uv2017[filteredData10])['UV_Index'].max())



filteredData11 = uv2017.Date == "11/06/2017"

print ((uv2017[filteredData11])['UV_Index'].max())



filteredData12 = uv2017.Date == "12/06/2017"

print ((uv2017[filteredData12])['UV_Index'].max())



filteredData13 = uv2017.Date == "13/06/2017"

print ((uv2017[filteredData13])['UV_Index'].max())



filteredData14 = uv2017.Date == "14/06/2017"

print ((uv2017[filteredData14])['UV_Index'].max())



filteredData15 = uv2017.Date == "15/06/2017"

print ((uv2017[filteredData15])['UV_Index'].max())



filteredData16 = uv2017.Date == "16/06/2017"

print ((uv2017[filteredData16])['UV_Index'].max())



filteredData17 = uv2017.Date == "17/06/2017"

print ((uv2017[filteredData17])['UV_Index'].max())



filteredData18 = uv2017.Date == "18/06/2017"

print ((uv2017[filteredData18])['UV_Index'].max())



filteredData19 = uv2017.Date == "19/06/2017"

print ((uv2017[filteredData19])['UV_Index'].max())



filteredData20 = uv2017.Date == "20/06/2069"

print ((uv2017[filteredData20])['UV_Index'].max())



filteredData21 = uv2017.Date == "21/06/2017"

print ((uv2017[filteredData21])['UV_Index'].max())



filteredData22 = uv2017.Date == "22/06/2017"

print ((uv2017[filteredData22])['UV_Index'].max())



filteredData23 = uv2017.Date == "23/06/2017"

print ((uv2017[filteredData23])['UV_Index'].max())



filteredData24 = uv2017.Date == "24/06/2017"

print ((uv2017[filteredData24])['UV_Index'].max())



filteredData25 = uv2017.Date == "25/06/2017"

print ((uv2017[filteredData25])['UV_Index'].max())



filteredData26 = uv2017.Date == "26/06/2017"

print ((uv2017[filteredData26])['UV_Index'].max())



filteredData27 = uv2017.Date == "27/06/2017"

print ((uv2017[filteredData27])['UV_Index'].max())



filteredData28 = uv2017.Date == "28/06/2017"

print ((uv2017[filteredData28])['UV_Index'].max())



filteredData29 = uv2017.Date == "29/06/2017"

print ((uv2017[filteredData29])['UV_Index'].max())



filteredData30 = uv2017.Date == "30/06/2017"

print ((uv2017[filteredData30])['UV_Index'].max())



filteredData31 = uv2017.Date == "31/06/2017"

print ((uv2017[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2018 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2018.csv')



filteredData1 = uv2018.Date == "01/06/2018"

print((uv2018[filteredData1])['UV_Index'].max())



filteredData2 = uv2018.Date == "02/06/2018"

print ((uv2018[filteredData2])['UV_Index'].max())



filteredData3 = uv2018.Date == "03/06/2018"

print ((uv2018[filteredData3])['UV_Index'].max())



filteredData4 = uv2018.Date == "04/06/2018"

print ((uv2018[filteredData4])['UV_Index'].max())



filteredData5 = uv2018.Date == "05/06/2018"

print ((uv2018[filteredData5])['UV_Index'].max())



filteredData6 = uv2018.Date == "06/06/2018"

print ((uv2018[filteredData6])['UV_Index'].max())



filteredData7 = uv2018.Date == "07/06/2018"

print ((uv2018[filteredData7])['UV_Index'].max())



filteredData8 = uv2018.Date == "08/06/2018"

print ((uv2018[filteredData8])['UV_Index'].max())



filteredData9 = uv2018.Date == "09/06/2018"

print ((uv2018[filteredData9])['UV_Index'].max())



filteredData10 = uv2018.Date == "10/06/2018"

print ((uv2018[filteredData10])['UV_Index'].max())



filteredData11 = uv2018.Date == "11/06/2018"

print ((uv2018[filteredData11])['UV_Index'].max())



filteredData12 = uv2018.Date == "12/06/2018"

print ((uv2018[filteredData12])['UV_Index'].max())



filteredData13 = uv2018.Date == "13/06/2018"

print ((uv2018[filteredData13])['UV_Index'].max())



filteredData14 = uv2018.Date == "14/06/2018"

print ((uv2018[filteredData14])['UV_Index'].max())



filteredData15 = uv2018.Date == "15/06/2018"

print ((uv2018[filteredData15])['UV_Index'].max())



filteredData16 = uv2018.Date == "16/06/2018"

print ((uv2018[filteredData16])['UV_Index'].max())



filteredData17 = uv2018.Date == "17/06/2018"

print ((uv2018[filteredData17])['UV_Index'].max())



filteredData18 = uv2018.Date == "18/06/2018"

print ((uv2018[filteredData18])['UV_Index'].max())



filteredData19 = uv2018.Date == "19/06/2018"

print ((uv2018[filteredData19])['UV_Index'].max())



filteredData20 = uv2018.Date == "20/06/2069"

print ((uv2018[filteredData20])['UV_Index'].max())



filteredData21 = uv2018.Date == "21/06/2018"

print ((uv2018[filteredData21])['UV_Index'].max())



filteredData22 = uv2018.Date == "22/06/2018"

print ((uv2018[filteredData22])['UV_Index'].max())



filteredData23 = uv2018.Date == "23/06/2018"

print ((uv2018[filteredData23])['UV_Index'].max())



filteredData24 = uv2018.Date == "24/06/2018"

print ((uv2018[filteredData24])['UV_Index'].max())



filteredData25 = uv2018.Date == "25/06/2018"

print ((uv2018[filteredData25])['UV_Index'].max())



filteredData26 = uv2018.Date == "26/06/2018"

print ((uv2018[filteredData26])['UV_Index'].max())



filteredData27 = uv2018.Date == "27/06/2018"

print ((uv2018[filteredData27])['UV_Index'].max())



filteredData28 = uv2018.Date == "28/06/2018"

print ((uv2018[filteredData28])['UV_Index'].max())



filteredData29 = uv2018.Date == "29/06/2018"

print ((uv2018[filteredData29])['UV_Index'].max())



filteredData30 = uv2018.Date == "30/06/2018"

print ((uv2018[filteredData30])['UV_Index'].max())



filteredData31 = uv2018.Date == "31/06/2018"

print ((uv2018[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2019 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2019.csv')



filteredData1 = uv2019.Date == "01/06/2019"

print((uv2019[filteredData1])['UV_Index'].max())



filteredData2 = uv2019.Date == "02/06/2019"

print ((uv2019[filteredData2])['UV_Index'].max())



filteredData3 = uv2019.Date == "03/06/2019"

print ((uv2019[filteredData3])['UV_Index'].max())



filteredData4 = uv2019.Date == "04/06/2019"

print ((uv2019[filteredData4])['UV_Index'].max())



filteredData5 = uv2019.Date == "05/06/2019"

print ((uv2019[filteredData5])['UV_Index'].max())



filteredData6 = uv2019.Date == "06/06/2019"

print ((uv2019[filteredData6])['UV_Index'].max())



filteredData7 = uv2019.Date == "07/06/2019"

print ((uv2019[filteredData7])['UV_Index'].max())



filteredData8 = uv2019.Date == "08/06/2019"

print ((uv2019[filteredData8])['UV_Index'].max())



filteredData9 = uv2019.Date == "09/06/2019"

print ((uv2019[filteredData9])['UV_Index'].max())



filteredData10 = uv2019.Date == "10/06/2019"

print ((uv2019[filteredData10])['UV_Index'].max())



filteredData11 = uv2019.Date == "11/06/2019"

print ((uv2019[filteredData11])['UV_Index'].max())



filteredData12 = uv2019.Date == "12/06/2019"

print ((uv2019[filteredData12])['UV_Index'].max())



filteredData13 = uv2019.Date == "13/06/2019"

print ((uv2019[filteredData13])['UV_Index'].max())



filteredData14 = uv2019.Date == "14/06/2019"

print ((uv2019[filteredData14])['UV_Index'].max())



filteredData15 = uv2019.Date == "15/06/2019"

print ((uv2019[filteredData15])['UV_Index'].max())



filteredData16 = uv2019.Date == "16/06/2019"

print ((uv2019[filteredData16])['UV_Index'].max())



filteredData17 = uv2019.Date == "17/06/2019"

print ((uv2019[filteredData17])['UV_Index'].max())



filteredData18 = uv2019.Date == "18/06/2019"

print ((uv2019[filteredData18])['UV_Index'].max())



filteredData19 = uv2019.Date == "19/06/2019"

print ((uv2019[filteredData19])['UV_Index'].max())



filteredData20 = uv2019.Date == "20/06/2069"

print ((uv2019[filteredData20])['UV_Index'].max())



filteredData21 = uv2019.Date == "21/06/2019"

print ((uv2019[filteredData21])['UV_Index'].max())



filteredData22 = uv2019.Date == "22/06/2019"

print ((uv2019[filteredData22])['UV_Index'].max())



filteredData23 = uv2019.Date == "23/06/2019"

print ((uv2019[filteredData23])['UV_Index'].max())



filteredData24 = uv2019.Date == "24/06/2019"

print ((uv2019[filteredData24])['UV_Index'].max())



filteredData25 = uv2019.Date == "25/06/2019"

print ((uv2019[filteredData25])['UV_Index'].max())



filteredData26 = uv2019.Date == "26/06/2019"

print ((uv2019[filteredData26])['UV_Index'].max())



filteredData27 = uv2019.Date == "27/06/2019"

print ((uv2019[filteredData27])['UV_Index'].max())



filteredData28 = uv2019.Date == "28/06/2019"

print ((uv2019[filteredData28])['UV_Index'].max())



filteredData29 = uv2019.Date == "29/06/2019"

print ((uv2019[filteredData29])['UV_Index'].max())



filteredData30 = uv2019.Date == "30/06/2019"

print ((uv2019[filteredData30])['UV_Index'].max())



filteredData31 = uv2019.Date == "31/06/2019"

print ((uv2019[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2019 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2019.csv')



filteredData1 = uv2019.Date == "01/07/2019"

print((uv2019[filteredData1])['UV_Index'].max())



filteredData2 = uv2019.Date == "02/07/2019"

print ((uv2019[filteredData2])['UV_Index'].max())



filteredData3 = uv2019.Date == "03/07/2019"

print ((uv2019[filteredData3])['UV_Index'].max())



filteredData4 = uv2019.Date == "04/07/2019"

print ((uv2019[filteredData4])['UV_Index'].max())



filteredData5 = uv2019.Date == "05/07/2019"

print ((uv2019[filteredData5])['UV_Index'].max())



filteredData6 = uv2019.Date == "06/07/2019"

print ((uv2019[filteredData6])['UV_Index'].max())



filteredData7 = uv2019.Date == "07/07/2019"

print ((uv2019[filteredData7])['UV_Index'].max())



filteredData8 = uv2019.Date == "08/07/2019"

print ((uv2019[filteredData8])['UV_Index'].max())



filteredData9 = uv2019.Date == "09/07/2019"

print ((uv2019[filteredData9])['UV_Index'].max())



filteredData10 = uv2019.Date == "10/07/2019"

print ((uv2019[filteredData10])['UV_Index'].max())



filteredData11 = uv2019.Date == "11/07/2019"

print ((uv2019[filteredData11])['UV_Index'].max())



filteredData12 = uv2019.Date == "12/07/2019"

print ((uv2019[filteredData12])['UV_Index'].max())



filteredData13 = uv2019.Date == "13/07/2019"

print ((uv2019[filteredData13])['UV_Index'].max())



filteredData14 = uv2019.Date == "14/07/2019"

print ((uv2019[filteredData14])['UV_Index'].max())



filteredData15 = uv2019.Date == "15/07/2019"

print ((uv2019[filteredData15])['UV_Index'].max())



filteredData16 = uv2019.Date == "16/07/2019"

print ((uv2019[filteredData16])['UV_Index'].max())



filteredData17 = uv2019.Date == "17/07/2019"

print ((uv2019[filteredData17])['UV_Index'].max())



filteredData18 = uv2019.Date == "18/07/2019"

print ((uv2019[filteredData18])['UV_Index'].max())



filteredData19 = uv2019.Date == "19/07/2019"

print ((uv2019[filteredData19])['UV_Index'].max())



filteredData20 = uv2019.Date == "20/07/2079"

print ((uv2019[filteredData20])['UV_Index'].max())



filteredData21 = uv2019.Date == "21/07/2019"

print ((uv2019[filteredData21])['UV_Index'].max())



filteredData22 = uv2019.Date == "22/07/2019"

print ((uv2019[filteredData22])['UV_Index'].max())



filteredData23 = uv2019.Date == "23/07/2019"

print ((uv2019[filteredData23])['UV_Index'].max())



filteredData24 = uv2019.Date == "24/07/2019"

print ((uv2019[filteredData24])['UV_Index'].max())



filteredData25 = uv2019.Date == "25/07/2019"

print ((uv2019[filteredData25])['UV_Index'].max())



filteredData26 = uv2019.Date == "26/07/2019"

print ((uv2019[filteredData26])['UV_Index'].max())



filteredData27 = uv2019.Date == "27/07/2019"

print ((uv2019[filteredData27])['UV_Index'].max())



filteredData28 = uv2019.Date == "28/07/2019"

print ((uv2019[filteredData28])['UV_Index'].max())



filteredData29 = uv2019.Date == "29/07/2019"

print ((uv2019[filteredData29])['UV_Index'].max())



filteredData30 = uv2019.Date == "30/07/2019"

print ((uv2019[filteredData30])['UV_Index'].max())



filteredData31 = uv2019.Date == "31/07/2019"

print ((uv2019[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2018 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2018.csv')



filteredData1 = uv2018.Date == "01/07/2018"

print((uv2018[filteredData1])['UV_Index'].max())



filteredData2 = uv2018.Date == "02/07/2018"

print ((uv2018[filteredData2])['UV_Index'].max())



filteredData3 = uv2018.Date == "03/07/2018"

print ((uv2018[filteredData3])['UV_Index'].max())



filteredData4 = uv2018.Date == "04/07/2018"

print ((uv2018[filteredData4])['UV_Index'].max())



filteredData5 = uv2018.Date == "05/07/2018"

print ((uv2018[filteredData5])['UV_Index'].max())



filteredData6 = uv2018.Date == "06/07/2018"

print ((uv2018[filteredData6])['UV_Index'].max())



filteredData7 = uv2018.Date == "07/07/2018"

print ((uv2018[filteredData7])['UV_Index'].max())



filteredData8 = uv2018.Date == "08/07/2018"

print ((uv2018[filteredData8])['UV_Index'].max())



filteredData9 = uv2018.Date == "09/07/2018"

print ((uv2018[filteredData9])['UV_Index'].max())



filteredData10 = uv2018.Date == "10/07/2018"

print ((uv2018[filteredData10])['UV_Index'].max())



filteredData11 = uv2018.Date == "11/07/2018"

print ((uv2018[filteredData11])['UV_Index'].max())



filteredData12 = uv2018.Date == "12/07/2018"

print ((uv2018[filteredData12])['UV_Index'].max())



filteredData13 = uv2018.Date == "13/07/2018"

print ((uv2018[filteredData13])['UV_Index'].max())



filteredData14 = uv2018.Date == "14/07/2018"

print ((uv2018[filteredData14])['UV_Index'].max())



filteredData15 = uv2018.Date == "15/07/2018"

print ((uv2018[filteredData15])['UV_Index'].max())



filteredData16 = uv2018.Date == "16/07/2018"

print ((uv2018[filteredData16])['UV_Index'].max())



filteredData17 = uv2018.Date == "17/07/2018"

print ((uv2018[filteredData17])['UV_Index'].max())



filteredData18 = uv2018.Date == "18/07/2018"

print ((uv2018[filteredData18])['UV_Index'].max())



filteredData19 = uv2018.Date == "19/07/2018"

print ((uv2018[filteredData19])['UV_Index'].max())



filteredData20 = uv2018.Date == "20/07/2079"

print ((uv2018[filteredData20])['UV_Index'].max())



filteredData21 = uv2018.Date == "21/07/2018"

print ((uv2018[filteredData21])['UV_Index'].max())



filteredData22 = uv2018.Date == "22/07/2018"

print ((uv2018[filteredData22])['UV_Index'].max())



filteredData23 = uv2018.Date == "23/07/2018"

print ((uv2018[filteredData23])['UV_Index'].max())



filteredData24 = uv2018.Date == "24/07/2018"

print ((uv2018[filteredData24])['UV_Index'].max())



filteredData25 = uv2018.Date == "25/07/2018"

print ((uv2018[filteredData25])['UV_Index'].max())



filteredData26 = uv2018.Date == "26/07/2018"

print ((uv2018[filteredData26])['UV_Index'].max())



filteredData27 = uv2018.Date == "27/07/2018"

print ((uv2018[filteredData27])['UV_Index'].max())



filteredData28 = uv2018.Date == "28/07/2018"

print ((uv2018[filteredData28])['UV_Index'].max())



filteredData29 = uv2018.Date == "29/07/2018"

print ((uv2018[filteredData29])['UV_Index'].max())



filteredData30 = uv2018.Date == "30/07/2018"

print ((uv2018[filteredData30])['UV_Index'].max())



filteredData31 = uv2018.Date == "31/07/2018"

print ((uv2018[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2017 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2017.csv')



filteredData1 = uv2017.Date == "01/07/2017"

print((uv2017[filteredData1])['UV_Index'].max())



filteredData2 = uv2017.Date == "02/07/2017"

print ((uv2017[filteredData2])['UV_Index'].max())



filteredData3 = uv2017.Date == "03/07/2017"

print ((uv2017[filteredData3])['UV_Index'].max())



filteredData4 = uv2017.Date == "04/07/2017"

print ((uv2017[filteredData4])['UV_Index'].max())



filteredData5 = uv2017.Date == "05/07/2017"

print ((uv2017[filteredData5])['UV_Index'].max())



filteredData6 = uv2017.Date == "06/07/2017"

print ((uv2017[filteredData6])['UV_Index'].max())



filteredData7 = uv2017.Date == "07/07/2017"

print ((uv2017[filteredData7])['UV_Index'].max())



filteredData8 = uv2017.Date == "08/07/2017"

print ((uv2017[filteredData8])['UV_Index'].max())



filteredData9 = uv2017.Date == "09/07/2017"

print ((uv2017[filteredData9])['UV_Index'].max())



filteredData10 = uv2017.Date == "10/07/2017"

print ((uv2017[filteredData10])['UV_Index'].max())



filteredData11 = uv2017.Date == "11/07/2017"

print ((uv2017[filteredData11])['UV_Index'].max())



filteredData12 = uv2017.Date == "12/07/2017"

print ((uv2017[filteredData12])['UV_Index'].max())



filteredData13 = uv2017.Date == "13/07/2017"

print ((uv2017[filteredData13])['UV_Index'].max())



filteredData14 = uv2017.Date == "14/07/2017"

print ((uv2017[filteredData14])['UV_Index'].max())



filteredData15 = uv2017.Date == "15/07/2017"

print ((uv2017[filteredData15])['UV_Index'].max())



filteredData16 = uv2017.Date == "16/07/2017"

print ((uv2017[filteredData16])['UV_Index'].max())



filteredData17 = uv2017.Date == "17/07/2017"

print ((uv2017[filteredData17])['UV_Index'].max())



filteredData18 = uv2017.Date == "18/07/2017"

print ((uv2017[filteredData18])['UV_Index'].max())



filteredData19 = uv2017.Date == "19/07/2017"

print ((uv2017[filteredData19])['UV_Index'].max())



filteredData20 = uv2017.Date == "20/07/2079"

print ((uv2017[filteredData20])['UV_Index'].max())



filteredData21 = uv2017.Date == "21/07/2017"

print ((uv2017[filteredData21])['UV_Index'].max())



filteredData22 = uv2017.Date == "22/07/2017"

print ((uv2017[filteredData22])['UV_Index'].max())



filteredData23 = uv2017.Date == "23/07/2017"

print ((uv2017[filteredData23])['UV_Index'].max())



filteredData24 = uv2017.Date == "24/07/2017"

print ((uv2017[filteredData24])['UV_Index'].max())



filteredData25 = uv2017.Date == "25/07/2017"

print ((uv2017[filteredData25])['UV_Index'].max())



filteredData26 = uv2017.Date == "26/07/2017"

print ((uv2017[filteredData26])['UV_Index'].max())



filteredData27 = uv2017.Date == "27/07/2017"

print ((uv2017[filteredData27])['UV_Index'].max())



filteredData28 = uv2017.Date == "28/07/2017"

print ((uv2017[filteredData28])['UV_Index'].max())



filteredData29 = uv2017.Date == "29/07/2017"

print ((uv2017[filteredData29])['UV_Index'].max())



filteredData30 = uv2017.Date == "30/07/2017"

print ((uv2017[filteredData30])['UV_Index'].max())



filteredData31 = uv2017.Date == "31/07/2017"

print ((uv2017[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2016 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2016.csv')



filteredData1 = uv2016.Date == "01/07/2016"

print((uv2016[filteredData1])['UV_Index'].max())



filteredData2 = uv2016.Date == "02/07/2016"

print ((uv2016[filteredData2])['UV_Index'].max())



filteredData3 = uv2016.Date == "03/07/2016"

print ((uv2016[filteredData3])['UV_Index'].max())



filteredData4 = uv2016.Date == "04/07/2016"

print ((uv2016[filteredData4])['UV_Index'].max())



filteredData5 = uv2016.Date == "05/07/2016"

print ((uv2016[filteredData5])['UV_Index'].max())



filteredData6 = uv2016.Date == "06/07/2016"

print ((uv2016[filteredData6])['UV_Index'].max())



filteredData7 = uv2016.Date == "07/07/2016"

print ((uv2016[filteredData7])['UV_Index'].max())



filteredData8 = uv2016.Date == "08/07/2016"

print ((uv2016[filteredData8])['UV_Index'].max())



filteredData9 = uv2016.Date == "09/07/2016"

print ((uv2016[filteredData9])['UV_Index'].max())



filteredData10 = uv2016.Date == "10/07/2016"

print ((uv2016[filteredData10])['UV_Index'].max())



filteredData11 = uv2016.Date == "11/07/2016"

print ((uv2016[filteredData11])['UV_Index'].max())



filteredData12 = uv2016.Date == "12/07/2016"

print ((uv2016[filteredData12])['UV_Index'].max())



filteredData13 = uv2016.Date == "13/07/2016"

print ((uv2016[filteredData13])['UV_Index'].max())



filteredData14 = uv2016.Date == "14/07/2016"

print ((uv2016[filteredData14])['UV_Index'].max())



filteredData15 = uv2016.Date == "15/07/2016"

print ((uv2016[filteredData15])['UV_Index'].max())



filteredData16 = uv2016.Date == "16/07/2016"

print ((uv2016[filteredData16])['UV_Index'].max())



filteredData17 = uv2016.Date == "17/07/2016"

print ((uv2016[filteredData17])['UV_Index'].max())



filteredData18 = uv2016.Date == "18/07/2016"

print ((uv2016[filteredData18])['UV_Index'].max())



filteredData19 = uv2016.Date == "19/07/2016"

print ((uv2016[filteredData19])['UV_Index'].max())



filteredData20 = uv2016.Date == "20/07/2079"

print ((uv2016[filteredData20])['UV_Index'].max())



filteredData21 = uv2016.Date == "21/07/2016"

print ((uv2016[filteredData21])['UV_Index'].max())



filteredData22 = uv2016.Date == "22/07/2016"

print ((uv2016[filteredData22])['UV_Index'].max())



filteredData23 = uv2016.Date == "23/07/2016"

print ((uv2016[filteredData23])['UV_Index'].max())



filteredData24 = uv2016.Date == "24/07/2016"

print ((uv2016[filteredData24])['UV_Index'].max())



filteredData25 = uv2016.Date == "25/07/2016"

print ((uv2016[filteredData25])['UV_Index'].max())



filteredData26 = uv2016.Date == "26/07/2016"

print ((uv2016[filteredData26])['UV_Index'].max())



filteredData27 = uv2016.Date == "27/07/2016"

print ((uv2016[filteredData27])['UV_Index'].max())



filteredData28 = uv2016.Date == "28/07/2016"

print ((uv2016[filteredData28])['UV_Index'].max())



filteredData29 = uv2016.Date == "29/07/2016"

print ((uv2016[filteredData29])['UV_Index'].max())



filteredData30 = uv2016.Date == "30/07/2016"

print ((uv2016[filteredData30])['UV_Index'].max())



filteredData31 = uv2016.Date == "31/07/2016"

print ((uv2016[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2015 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2015.csv')



filteredData1 = uv2015.Date == "01/07/2015"

print((uv2015[filteredData1])['UV_Index'].max())



filteredData2 = uv2015.Date == "02/07/2015"

print ((uv2015[filteredData2])['UV_Index'].max())



filteredData3 = uv2015.Date == "03/07/2015"

print ((uv2015[filteredData3])['UV_Index'].max())



filteredData4 = uv2015.Date == "04/07/2015"

print ((uv2015[filteredData4])['UV_Index'].max())



filteredData5 = uv2015.Date == "05/07/2015"

print ((uv2015[filteredData5])['UV_Index'].max())



filteredData6 = uv2015.Date == "06/07/2015"

print ((uv2015[filteredData6])['UV_Index'].max())



filteredData7 = uv2015.Date == "07/07/2015"

print ((uv2015[filteredData7])['UV_Index'].max())



filteredData8 = uv2015.Date == "08/07/2015"

print ((uv2015[filteredData8])['UV_Index'].max())



filteredData9 = uv2015.Date == "09/07/2015"

print ((uv2015[filteredData9])['UV_Index'].max())



filteredData10 = uv2015.Date == "10/07/2015"

print ((uv2015[filteredData10])['UV_Index'].max())



filteredData11 = uv2015.Date == "11/07/2015"

print ((uv2015[filteredData11])['UV_Index'].max())



filteredData12 = uv2015.Date == "12/07/2015"

print ((uv2015[filteredData12])['UV_Index'].max())



filteredData13 = uv2015.Date == "13/07/2015"

print ((uv2015[filteredData13])['UV_Index'].max())



filteredData14 = uv2015.Date == "14/07/2015"

print ((uv2015[filteredData14])['UV_Index'].max())



filteredData15 = uv2015.Date == "15/07/2015"

print ((uv2015[filteredData15])['UV_Index'].max())



filteredData16 = uv2015.Date == "16/07/2015"

print ((uv2015[filteredData16])['UV_Index'].max())



filteredData17 = uv2015.Date == "17/07/2015"

print ((uv2015[filteredData17])['UV_Index'].max())



filteredData18 = uv2015.Date == "18/07/2015"

print ((uv2015[filteredData18])['UV_Index'].max())



filteredData19 = uv2015.Date == "19/07/2015"

print ((uv2015[filteredData19])['UV_Index'].max())



filteredData20 = uv2015.Date == "20/07/2079"

print ((uv2015[filteredData20])['UV_Index'].max())



filteredData21 = uv2015.Date == "21/07/2015"

print ((uv2015[filteredData21])['UV_Index'].max())



filteredData22 = uv2015.Date == "22/07/2015"

print ((uv2015[filteredData22])['UV_Index'].max())



filteredData23 = uv2015.Date == "23/07/2015"

print ((uv2015[filteredData23])['UV_Index'].max())



filteredData24 = uv2015.Date == "24/07/2015"

print ((uv2015[filteredData24])['UV_Index'].max())



filteredData25 = uv2015.Date == "25/07/2015"

print ((uv2015[filteredData25])['UV_Index'].max())



filteredData26 = uv2015.Date == "26/07/2015"

print ((uv2015[filteredData26])['UV_Index'].max())



filteredData27 = uv2015.Date == "27/07/2015"

print ((uv2015[filteredData27])['UV_Index'].max())



filteredData28 = uv2015.Date == "28/07/2015"

print ((uv2015[filteredData28])['UV_Index'].max())



filteredData29 = uv2015.Date == "29/07/2015"

print ((uv2015[filteredData29])['UV_Index'].max())



filteredData30 = uv2015.Date == "30/07/2015"

print ((uv2015[filteredData30])['UV_Index'].max())



filteredData31 = uv2015.Date == "31/07/2015"

print ((uv2015[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2014 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2014.csv')



filteredData1 = uv2014.Date == "01/07/2014"

print((uv2014[filteredData1])['UV_Index'].max())



filteredData2 = uv2014.Date == "02/07/2014"

print ((uv2014[filteredData2])['UV_Index'].max())



filteredData3 = uv2014.Date == "03/07/2014"

print ((uv2014[filteredData3])['UV_Index'].max())



filteredData4 = uv2014.Date == "04/07/2014"

print ((uv2014[filteredData4])['UV_Index'].max())



filteredData5 = uv2014.Date == "05/07/2014"

print ((uv2014[filteredData5])['UV_Index'].max())



filteredData6 = uv2014.Date == "06/07/2014"

print ((uv2014[filteredData6])['UV_Index'].max())



filteredData7 = uv2014.Date == "07/07/2014"

print ((uv2014[filteredData7])['UV_Index'].max())



filteredData8 = uv2014.Date == "08/07/2014"

print ((uv2014[filteredData8])['UV_Index'].max())



filteredData9 = uv2014.Date == "09/07/2014"

print ((uv2014[filteredData9])['UV_Index'].max())



filteredData10 = uv2014.Date == "10/07/2014"

print ((uv2014[filteredData10])['UV_Index'].max())



filteredData11 = uv2014.Date == "11/07/2014"

print ((uv2014[filteredData11])['UV_Index'].max())



filteredData12 = uv2014.Date == "12/07/2014"

print ((uv2014[filteredData12])['UV_Index'].max())



filteredData13 = uv2014.Date == "13/07/2014"

print ((uv2014[filteredData13])['UV_Index'].max())



filteredData14 = uv2014.Date == "14/07/2014"

print ((uv2014[filteredData14])['UV_Index'].max())



filteredData15 = uv2014.Date == "15/07/2014"

print ((uv2014[filteredData15])['UV_Index'].max())



filteredData16 = uv2014.Date == "16/07/2014"

print ((uv2014[filteredData16])['UV_Index'].max())



filteredData17 = uv2014.Date == "17/07/2014"

print ((uv2014[filteredData17])['UV_Index'].max())



filteredData18 = uv2014.Date == "18/07/2014"

print ((uv2014[filteredData18])['UV_Index'].max())



filteredData19 = uv2014.Date == "19/07/2014"

print ((uv2014[filteredData19])['UV_Index'].max())



filteredData20 = uv2014.Date == "20/07/2079"

print ((uv2014[filteredData20])['UV_Index'].max())



filteredData21 = uv2014.Date == "21/07/2014"

print ((uv2014[filteredData21])['UV_Index'].max())



filteredData22 = uv2014.Date == "22/07/2014"

print ((uv2014[filteredData22])['UV_Index'].max())



filteredData23 = uv2014.Date == "23/07/2014"

print ((uv2014[filteredData23])['UV_Index'].max())



filteredData24 = uv2014.Date == "24/07/2014"

print ((uv2014[filteredData24])['UV_Index'].max())



filteredData25 = uv2014.Date == "25/07/2014"

print ((uv2014[filteredData25])['UV_Index'].max())



filteredData26 = uv2014.Date == "26/07/2014"

print ((uv2014[filteredData26])['UV_Index'].max())



filteredData27 = uv2014.Date == "27/07/2014"

print ((uv2014[filteredData27])['UV_Index'].max())



filteredData28 = uv2014.Date == "28/07/2014"

print ((uv2014[filteredData28])['UV_Index'].max())



filteredData29 = uv2014.Date == "29/07/2014"

print ((uv2014[filteredData29])['UV_Index'].max())



filteredData30 = uv2014.Date == "30/07/2014"

print ((uv2014[filteredData30])['UV_Index'].max())



filteredData31 = uv2014.Date == "31/07/2014"

print ((uv2014[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2013 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2013.csv')



filteredData1 = uv2013.Date == "01/07/2013"

print((uv2013[filteredData1])['UV_Index'].max())



filteredData2 = uv2013.Date == "02/07/2013"

print ((uv2013[filteredData2])['UV_Index'].max())



filteredData3 = uv2013.Date == "03/07/2013"

print ((uv2013[filteredData3])['UV_Index'].max())



filteredData4 = uv2013.Date == "04/07/2013"

print ((uv2013[filteredData4])['UV_Index'].max())



filteredData5 = uv2013.Date == "05/07/2013"

print ((uv2013[filteredData5])['UV_Index'].max())



filteredData6 = uv2013.Date == "06/07/2013"

print ((uv2013[filteredData6])['UV_Index'].max())



filteredData7 = uv2013.Date == "07/07/2013"

print ((uv2013[filteredData7])['UV_Index'].max())



filteredData8 = uv2013.Date == "08/07/2013"

print ((uv2013[filteredData8])['UV_Index'].max())



filteredData9 = uv2013.Date == "09/07/2013"

print ((uv2013[filteredData9])['UV_Index'].max())



filteredData10 = uv2013.Date == "10/07/2013"

print ((uv2013[filteredData10])['UV_Index'].max())



filteredData11 = uv2013.Date == "11/07/2013"

print ((uv2013[filteredData11])['UV_Index'].max())



filteredData12 = uv2013.Date == "12/07/2013"

print ((uv2013[filteredData12])['UV_Index'].max())



filteredData13 = uv2013.Date == "13/07/2013"

print ((uv2013[filteredData13])['UV_Index'].max())



filteredData14 = uv2013.Date == "14/07/2013"

print ((uv2013[filteredData14])['UV_Index'].max())



filteredData15 = uv2013.Date == "15/07/2013"

print ((uv2013[filteredData15])['UV_Index'].max())



filteredData16 = uv2013.Date == "16/07/2013"

print ((uv2013[filteredData16])['UV_Index'].max())



filteredData17 = uv2013.Date == "17/07/2013"

print ((uv2013[filteredData17])['UV_Index'].max())



filteredData18 = uv2013.Date == "18/07/2013"

print ((uv2013[filteredData18])['UV_Index'].max())



filteredData19 = uv2013.Date == "19/07/2013"

print ((uv2013[filteredData19])['UV_Index'].max())



filteredData20 = uv2013.Date == "20/07/2079"

print ((uv2013[filteredData20])['UV_Index'].max())



filteredData21 = uv2013.Date == "21/07/2013"

print ((uv2013[filteredData21])['UV_Index'].max())



filteredData22 = uv2013.Date == "22/07/2013"

print ((uv2013[filteredData22])['UV_Index'].max())



filteredData23 = uv2013.Date == "23/07/2013"

print ((uv2013[filteredData23])['UV_Index'].max())



filteredData24 = uv2013.Date == "24/07/2013"

print ((uv2013[filteredData24])['UV_Index'].max())



filteredData25 = uv2013.Date == "25/07/2013"

print ((uv2013[filteredData25])['UV_Index'].max())



filteredData26 = uv2013.Date == "26/07/2013"

print ((uv2013[filteredData26])['UV_Index'].max())



filteredData27 = uv2013.Date == "27/07/2013"

print ((uv2013[filteredData27])['UV_Index'].max())



filteredData28 = uv2013.Date == "28/07/2013"

print ((uv2013[filteredData28])['UV_Index'].max())



filteredData29 = uv2013.Date == "29/07/2013"

print ((uv2013[filteredData29])['UV_Index'].max())



filteredData30 = uv2013.Date == "30/07/2013"

print ((uv2013[filteredData30])['UV_Index'].max())



filteredData31 = uv2013.Date == "31/07/2013"

print ((uv2013[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2012 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2012.csv')



filteredData1 = uv2012.Date == "01/07/2012"

print((uv2012[filteredData1])['UV_Index'].max())



filteredData2 = uv2012.Date == "02/07/2012"

print ((uv2012[filteredData2])['UV_Index'].max())



filteredData3 = uv2012.Date == "03/07/2012"

print ((uv2012[filteredData3])['UV_Index'].max())



filteredData4 = uv2012.Date == "04/07/2012"

print ((uv2012[filteredData4])['UV_Index'].max())



filteredData5 = uv2012.Date == "05/07/2012"

print ((uv2012[filteredData5])['UV_Index'].max())



filteredData6 = uv2012.Date == "06/07/2012"

print ((uv2012[filteredData6])['UV_Index'].max())



filteredData7 = uv2012.Date == "07/07/2012"

print ((uv2012[filteredData7])['UV_Index'].max())



filteredData8 = uv2012.Date == "08/07/2012"

print ((uv2012[filteredData8])['UV_Index'].max())



filteredData9 = uv2012.Date == "09/07/2012"

print ((uv2012[filteredData9])['UV_Index'].max())



filteredData10 = uv2012.Date == "10/07/2012"

print ((uv2012[filteredData10])['UV_Index'].max())



filteredData11 = uv2012.Date == "11/07/2012"

print ((uv2012[filteredData11])['UV_Index'].max())



filteredData12 = uv2012.Date == "12/07/2012"

print ((uv2012[filteredData12])['UV_Index'].max())



filteredData13 = uv2012.Date == "13/07/2012"

print ((uv2012[filteredData13])['UV_Index'].max())



filteredData14 = uv2012.Date == "14/07/2012"

print ((uv2012[filteredData14])['UV_Index'].max())



filteredData15 = uv2012.Date == "15/07/2012"

print ((uv2012[filteredData15])['UV_Index'].max())



filteredData16 = uv2012.Date == "16/07/2012"

print ((uv2012[filteredData16])['UV_Index'].max())



filteredData17 = uv2012.Date == "17/07/2012"

print ((uv2012[filteredData17])['UV_Index'].max())



filteredData18 = uv2012.Date == "18/07/2012"

print ((uv2012[filteredData18])['UV_Index'].max())



filteredData19 = uv2012.Date == "19/07/2012"

print ((uv2012[filteredData19])['UV_Index'].max())



filteredData20 = uv2012.Date == "20/07/2079"

print ((uv2012[filteredData20])['UV_Index'].max())



filteredData21 = uv2012.Date == "21/07/2012"

print ((uv2012[filteredData21])['UV_Index'].max())



filteredData22 = uv2012.Date == "22/07/2012"

print ((uv2012[filteredData22])['UV_Index'].max())



filteredData23 = uv2012.Date == "23/07/2012"

print ((uv2012[filteredData23])['UV_Index'].max())



filteredData24 = uv2012.Date == "24/07/2012"

print ((uv2012[filteredData24])['UV_Index'].max())



filteredData25 = uv2012.Date == "25/07/2012"

print ((uv2012[filteredData25])['UV_Index'].max())



filteredData26 = uv2012.Date == "26/07/2012"

print ((uv2012[filteredData26])['UV_Index'].max())



filteredData27 = uv2012.Date == "27/07/2012"

print ((uv2012[filteredData27])['UV_Index'].max())



filteredData28 = uv2012.Date == "28/07/2012"

print ((uv2012[filteredData28])['UV_Index'].max())



filteredData29 = uv2012.Date == "29/07/2012"

print ((uv2012[filteredData29])['UV_Index'].max())



filteredData30 = uv2012.Date == "30/07/2012"

print ((uv2012[filteredData30])['UV_Index'].max())



filteredData31 = uv2012.Date == "31/07/2012"

print ((uv2012[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2011 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2011.csv')



filteredData1 = uv2011.Date == "01/07/2011"

print((uv2011[filteredData1])['UV_Index'].max())



filteredData2 = uv2011.Date == "02/07/2011"

print ((uv2011[filteredData2])['UV_Index'].max())



filteredData3 = uv2011.Date == "03/07/2011"

print ((uv2011[filteredData3])['UV_Index'].max())



filteredData4 = uv2011.Date == "04/07/2011"

print ((uv2011[filteredData4])['UV_Index'].max())



filteredData5 = uv2011.Date == "05/07/2011"

print ((uv2011[filteredData5])['UV_Index'].max())



filteredData6 = uv2011.Date == "06/07/2011"

print ((uv2011[filteredData6])['UV_Index'].max())



filteredData7 = uv2011.Date == "07/07/2011"

print ((uv2011[filteredData7])['UV_Index'].max())



filteredData8 = uv2011.Date == "08/07/2011"

print ((uv2011[filteredData8])['UV_Index'].max())



filteredData9 = uv2011.Date == "09/07/2011"

print ((uv2011[filteredData9])['UV_Index'].max())



filteredData10 = uv2011.Date == "10/07/2011"

print ((uv2011[filteredData10])['UV_Index'].max())



filteredData11 = uv2011.Date == "11/07/2011"

print ((uv2011[filteredData11])['UV_Index'].max())



filteredData12 = uv2011.Date == "12/07/2011"

print ((uv2011[filteredData12])['UV_Index'].max())



filteredData13 = uv2011.Date == "13/07/2011"

print ((uv2011[filteredData13])['UV_Index'].max())



filteredData14 = uv2011.Date == "14/07/2011"

print ((uv2011[filteredData14])['UV_Index'].max())



filteredData15 = uv2011.Date == "15/07/2011"

print ((uv2011[filteredData15])['UV_Index'].max())



filteredData16 = uv2011.Date == "16/07/2011"

print ((uv2011[filteredData16])['UV_Index'].max())



filteredData17 = uv2011.Date == "17/07/2011"

print ((uv2011[filteredData17])['UV_Index'].max())



filteredData18 = uv2011.Date == "18/07/2011"

print ((uv2011[filteredData18])['UV_Index'].max())



filteredData19 = uv2011.Date == "19/07/2011"

print ((uv2011[filteredData19])['UV_Index'].max())



filteredData20 = uv2011.Date == "20/07/2079"

print ((uv2011[filteredData20])['UV_Index'].max())



filteredData21 = uv2011.Date == "21/07/2011"

print ((uv2011[filteredData21])['UV_Index'].max())



filteredData22 = uv2011.Date == "22/07/2011"

print ((uv2011[filteredData22])['UV_Index'].max())



filteredData23 = uv2011.Date == "23/07/2011"

print ((uv2011[filteredData23])['UV_Index'].max())



filteredData24 = uv2011.Date == "24/07/2011"

print ((uv2011[filteredData24])['UV_Index'].max())



filteredData25 = uv2011.Date == "25/07/2011"

print ((uv2011[filteredData25])['UV_Index'].max())



filteredData26 = uv2011.Date == "26/07/2011"

print ((uv2011[filteredData26])['UV_Index'].max())



filteredData27 = uv2011.Date == "27/07/2011"

print ((uv2011[filteredData27])['UV_Index'].max())



filteredData28 = uv2011.Date == "28/07/2011"

print ((uv2011[filteredData28])['UV_Index'].max())



filteredData29 = uv2011.Date == "29/07/2011"

print ((uv2011[filteredData29])['UV_Index'].max())



filteredData30 = uv2011.Date == "30/07/2011"

print ((uv2011[filteredData30])['UV_Index'].max())



filteredData31 = uv2011.Date == "31/07/2011"

print ((uv2011[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2010 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2010.csv')



filteredData1 = uv2010.Date == "01/07/2010"

print((uv2010[filteredData1])['UV_Index'].max())



filteredData2 = uv2010.Date == "02/07/2010"

print ((uv2010[filteredData2])['UV_Index'].max())



filteredData3 = uv2010.Date == "03/07/2010"

print ((uv2010[filteredData3])['UV_Index'].max())



filteredData4 = uv2010.Date == "04/07/2010"

print ((uv2010[filteredData4])['UV_Index'].max())



filteredData5 = uv2010.Date == "05/07/2010"

print ((uv2010[filteredData5])['UV_Index'].max())



filteredData6 = uv2010.Date == "06/07/2010"

print ((uv2010[filteredData6])['UV_Index'].max())



filteredData7 = uv2010.Date == "07/07/2010"

print ((uv2010[filteredData7])['UV_Index'].max())



filteredData8 = uv2010.Date == "08/07/2010"

print ((uv2010[filteredData8])['UV_Index'].max())



filteredData9 = uv2010.Date == "09/07/2010"

print ((uv2010[filteredData9])['UV_Index'].max())



filteredData10 = uv2010.Date == "10/07/2010"

print ((uv2010[filteredData10])['UV_Index'].max())



filteredData11 = uv2010.Date == "11/07/2010"

print ((uv2010[filteredData11])['UV_Index'].max())



filteredData12 = uv2010.Date == "12/07/2010"

print ((uv2010[filteredData12])['UV_Index'].max())



filteredData13 = uv2010.Date == "13/07/2010"

print ((uv2010[filteredData13])['UV_Index'].max())



filteredData14 = uv2010.Date == "14/07/2010"

print ((uv2010[filteredData14])['UV_Index'].max())



filteredData15 = uv2010.Date == "15/07/2010"

print ((uv2010[filteredData15])['UV_Index'].max())



filteredData16 = uv2010.Date == "16/07/2010"

print ((uv2010[filteredData16])['UV_Index'].max())



filteredData17 = uv2010.Date == "17/07/2010"

print ((uv2010[filteredData17])['UV_Index'].max())



filteredData18 = uv2010.Date == "18/07/2010"

print ((uv2010[filteredData18])['UV_Index'].max())



filteredData19 = uv2010.Date == "19/07/2010"

print ((uv2010[filteredData19])['UV_Index'].max())



filteredData20 = uv2010.Date == "20/07/2079"

print ((uv2010[filteredData20])['UV_Index'].max())



filteredData21 = uv2010.Date == "21/07/2010"

print ((uv2010[filteredData21])['UV_Index'].max())



filteredData22 = uv2010.Date == "22/07/2010"

print ((uv2010[filteredData22])['UV_Index'].max())



filteredData23 = uv2010.Date == "23/07/2010"

print ((uv2010[filteredData23])['UV_Index'].max())



filteredData24 = uv2010.Date == "24/07/2010"

print ((uv2010[filteredData24])['UV_Index'].max())



filteredData25 = uv2010.Date == "25/07/2010"

print ((uv2010[filteredData25])['UV_Index'].max())



filteredData26 = uv2010.Date == "26/07/2010"

print ((uv2010[filteredData26])['UV_Index'].max())



filteredData27 = uv2010.Date == "27/07/2010"

print ((uv2010[filteredData27])['UV_Index'].max())



filteredData28 = uv2010.Date == "28/07/2010"

print ((uv2010[filteredData28])['UV_Index'].max())



filteredData29 = uv2010.Date == "29/07/2010"

print ((uv2010[filteredData29])['UV_Index'].max())



filteredData30 = uv2010.Date == "30/07/2010"

print ((uv2010[filteredData30])['UV_Index'].max())



filteredData31 = uv2010.Date == "31/07/2010"

print ((uv2010[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2010 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2010.csv')



filteredData1 = uv2010.Date == "01/08/2010"

print((uv2010[filteredData1])['UV_Index'].max())



filteredData2 = uv2010.Date == "02/08/2010"

print ((uv2010[filteredData2])['UV_Index'].max())



filteredData3 = uv2010.Date == "03/08/2010"

print ((uv2010[filteredData3])['UV_Index'].max())



filteredData4 = uv2010.Date == "04/08/2010"

print ((uv2010[filteredData4])['UV_Index'].max())



filteredData5 = uv2010.Date == "05/08/2010"

print ((uv2010[filteredData5])['UV_Index'].max())



filteredData6 = uv2010.Date == "06/08/2010"

print ((uv2010[filteredData6])['UV_Index'].max())



filteredData7 = uv2010.Date == "07/08/2010"

print ((uv2010[filteredData7])['UV_Index'].max())



filteredData8 = uv2010.Date == "08/08/2010"

print ((uv2010[filteredData8])['UV_Index'].max())



filteredData9 = uv2010.Date == "09/08/2010"

print ((uv2010[filteredData9])['UV_Index'].max())



filteredData10 = uv2010.Date == "10/08/2010"

print ((uv2010[filteredData10])['UV_Index'].max())



filteredData11 = uv2010.Date == "11/08/2010"

print ((uv2010[filteredData11])['UV_Index'].max())



filteredData12 = uv2010.Date == "12/08/2010"

print ((uv2010[filteredData12])['UV_Index'].max())



filteredData13 = uv2010.Date == "13/08/2010"

print ((uv2010[filteredData13])['UV_Index'].max())



filteredData14 = uv2010.Date == "14/08/2010"

print ((uv2010[filteredData14])['UV_Index'].max())



filteredData15 = uv2010.Date == "15/08/2010"

print ((uv2010[filteredData15])['UV_Index'].max())



filteredData16 = uv2010.Date == "16/08/2010"

print ((uv2010[filteredData16])['UV_Index'].max())



filteredData17 = uv2010.Date == "17/08/2010"

print ((uv2010[filteredData17])['UV_Index'].max())



filteredData18 = uv2010.Date == "18/08/2010"

print ((uv2010[filteredData18])['UV_Index'].max())



filteredData19 = uv2010.Date == "19/08/2010"

print ((uv2010[filteredData19])['UV_Index'].max())



filteredData20 = uv2010.Date == "20/08/2089"

print ((uv2010[filteredData20])['UV_Index'].max())



filteredData21 = uv2010.Date == "21/08/2010"

print ((uv2010[filteredData21])['UV_Index'].max())



filteredData22 = uv2010.Date == "22/08/2010"

print ((uv2010[filteredData22])['UV_Index'].max())



filteredData23 = uv2010.Date == "23/08/2010"

print ((uv2010[filteredData23])['UV_Index'].max())



filteredData24 = uv2010.Date == "24/08/2010"

print ((uv2010[filteredData24])['UV_Index'].max())



filteredData25 = uv2010.Date == "25/08/2010"

print ((uv2010[filteredData25])['UV_Index'].max())



filteredData26 = uv2010.Date == "26/08/2010"

print ((uv2010[filteredData26])['UV_Index'].max())



filteredData27 = uv2010.Date == "27/08/2010"

print ((uv2010[filteredData27])['UV_Index'].max())



filteredData28 = uv2010.Date == "28/08/2010"

print ((uv2010[filteredData28])['UV_Index'].max())



filteredData29 = uv2010.Date == "29/08/2010"

print ((uv2010[filteredData29])['UV_Index'].max())



filteredData30 = uv2010.Date == "30/08/2010"

print ((uv2010[filteredData30])['UV_Index'].max())



filteredData31 = uv2010.Date == "31/08/2010"

print ((uv2010[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2011 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2011.csv')



filteredData1 = uv2011.Date == "01/08/2011"

print((uv2011[filteredData1])['UV_Index'].max())



filteredData2 = uv2011.Date == "02/08/2011"

print ((uv2011[filteredData2])['UV_Index'].max())



filteredData3 = uv2011.Date == "03/08/2011"

print ((uv2011[filteredData3])['UV_Index'].max())



filteredData4 = uv2011.Date == "04/08/2011"

print ((uv2011[filteredData4])['UV_Index'].max())



filteredData5 = uv2011.Date == "05/08/2011"

print ((uv2011[filteredData5])['UV_Index'].max())



filteredData6 = uv2011.Date == "06/08/2011"

print ((uv2011[filteredData6])['UV_Index'].max())



filteredData7 = uv2011.Date == "07/08/2011"

print ((uv2011[filteredData7])['UV_Index'].max())



filteredData8 = uv2011.Date == "08/08/2011"

print ((uv2011[filteredData8])['UV_Index'].max())



filteredData9 = uv2011.Date == "09/08/2011"

print ((uv2011[filteredData9])['UV_Index'].max())



filteredData10 = uv2011.Date == "10/08/2011"

print ((uv2011[filteredData10])['UV_Index'].max())



filteredData11 = uv2011.Date == "11/08/2011"

print ((uv2011[filteredData11])['UV_Index'].max())



filteredData12 = uv2011.Date == "12/08/2011"

print ((uv2011[filteredData12])['UV_Index'].max())



filteredData13 = uv2011.Date == "13/08/2011"

print ((uv2011[filteredData13])['UV_Index'].max())



filteredData14 = uv2011.Date == "14/08/2011"

print ((uv2011[filteredData14])['UV_Index'].max())



filteredData15 = uv2011.Date == "15/08/2011"

print ((uv2011[filteredData15])['UV_Index'].max())



filteredData16 = uv2011.Date == "16/08/2011"

print ((uv2011[filteredData16])['UV_Index'].max())



filteredData17 = uv2011.Date == "17/08/2011"

print ((uv2011[filteredData17])['UV_Index'].max())



filteredData18 = uv2011.Date == "18/08/2011"

print ((uv2011[filteredData18])['UV_Index'].max())



filteredData19 = uv2011.Date == "19/08/2011"

print ((uv2011[filteredData19])['UV_Index'].max())



filteredData20 = uv2011.Date == "20/08/2089"

print ((uv2011[filteredData20])['UV_Index'].max())



filteredData21 = uv2011.Date == "21/08/2011"

print ((uv2011[filteredData21])['UV_Index'].max())



filteredData22 = uv2011.Date == "22/08/2011"

print ((uv2011[filteredData22])['UV_Index'].max())



filteredData23 = uv2011.Date == "23/08/2011"

print ((uv2011[filteredData23])['UV_Index'].max())



filteredData24 = uv2011.Date == "24/08/2011"

print ((uv2011[filteredData24])['UV_Index'].max())



filteredData25 = uv2011.Date == "25/08/2011"

print ((uv2011[filteredData25])['UV_Index'].max())



filteredData26 = uv2011.Date == "26/08/2011"

print ((uv2011[filteredData26])['UV_Index'].max())



filteredData27 = uv2011.Date == "27/08/2011"

print ((uv2011[filteredData27])['UV_Index'].max())



filteredData28 = uv2011.Date == "28/08/2011"

print ((uv2011[filteredData28])['UV_Index'].max())



filteredData29 = uv2011.Date == "29/08/2011"

print ((uv2011[filteredData29])['UV_Index'].max())



filteredData30 = uv2011.Date == "30/08/2011"

print ((uv2011[filteredData30])['UV_Index'].max())



filteredData31 = uv2011.Date == "31/08/2011"

print ((uv2011[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2012 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2012.csv')



filteredData1 = uv2012.Date == "01/08/2012"

print((uv2012[filteredData1])['UV_Index'].max())



filteredData2 = uv2012.Date == "02/08/2012"

print ((uv2012[filteredData2])['UV_Index'].max())



filteredData3 = uv2012.Date == "03/08/2012"

print ((uv2012[filteredData3])['UV_Index'].max())



filteredData4 = uv2012.Date == "04/08/2012"

print ((uv2012[filteredData4])['UV_Index'].max())



filteredData5 = uv2012.Date == "05/08/2012"

print ((uv2012[filteredData5])['UV_Index'].max())



filteredData6 = uv2012.Date == "06/08/2012"

print ((uv2012[filteredData6])['UV_Index'].max())



filteredData7 = uv2012.Date == "07/08/2012"

print ((uv2012[filteredData7])['UV_Index'].max())



filteredData8 = uv2012.Date == "08/08/2012"

print ((uv2012[filteredData8])['UV_Index'].max())



filteredData9 = uv2012.Date == "09/08/2012"

print ((uv2012[filteredData9])['UV_Index'].max())



filteredData10 = uv2012.Date == "10/08/2012"

print ((uv2012[filteredData10])['UV_Index'].max())



filteredData11 = uv2012.Date == "11/08/2012"

print ((uv2012[filteredData11])['UV_Index'].max())



filteredData12 = uv2012.Date == "12/08/2012"

print ((uv2012[filteredData12])['UV_Index'].max())



filteredData13 = uv2012.Date == "13/08/2012"

print ((uv2012[filteredData13])['UV_Index'].max())



filteredData14 = uv2012.Date == "14/08/2012"

print ((uv2012[filteredData14])['UV_Index'].max())



filteredData15 = uv2012.Date == "15/08/2012"

print ((uv2012[filteredData15])['UV_Index'].max())



filteredData16 = uv2012.Date == "16/08/2012"

print ((uv2012[filteredData16])['UV_Index'].max())



filteredData17 = uv2012.Date == "17/08/2012"

print ((uv2012[filteredData17])['UV_Index'].max())



filteredData18 = uv2012.Date == "18/08/2012"

print ((uv2012[filteredData18])['UV_Index'].max())



filteredData19 = uv2012.Date == "19/08/2012"

print ((uv2012[filteredData19])['UV_Index'].max())



filteredData20 = uv2012.Date == "20/08/2089"

print ((uv2012[filteredData20])['UV_Index'].max())



filteredData21 = uv2012.Date == "21/08/2012"

print ((uv2012[filteredData21])['UV_Index'].max())



filteredData22 = uv2012.Date == "22/08/2012"

print ((uv2012[filteredData22])['UV_Index'].max())



filteredData23 = uv2012.Date == "23/08/2012"

print ((uv2012[filteredData23])['UV_Index'].max())



filteredData24 = uv2012.Date == "24/08/2012"

print ((uv2012[filteredData24])['UV_Index'].max())



filteredData25 = uv2012.Date == "25/08/2012"

print ((uv2012[filteredData25])['UV_Index'].max())



filteredData26 = uv2012.Date == "26/08/2012"

print ((uv2012[filteredData26])['UV_Index'].max())



filteredData27 = uv2012.Date == "27/08/2012"

print ((uv2012[filteredData27])['UV_Index'].max())



filteredData28 = uv2012.Date == "28/08/2012"

print ((uv2012[filteredData28])['UV_Index'].max())



filteredData29 = uv2012.Date == "29/08/2012"

print ((uv2012[filteredData29])['UV_Index'].max())



filteredData30 = uv2012.Date == "30/08/2012"

print ((uv2012[filteredData30])['UV_Index'].max())



filteredData31 = uv2012.Date == "31/08/2012"

print ((uv2012[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2013 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2013.csv')



filteredData1 = uv2013.Date == "01/08/2013"

print((uv2013[filteredData1])['UV_Index'].max())



filteredData2 = uv2013.Date == "02/08/2013"

print ((uv2013[filteredData2])['UV_Index'].max())



filteredData3 = uv2013.Date == "03/08/2013"

print ((uv2013[filteredData3])['UV_Index'].max())



filteredData4 = uv2013.Date == "04/08/2013"

print ((uv2013[filteredData4])['UV_Index'].max())



filteredData5 = uv2013.Date == "05/08/2013"

print ((uv2013[filteredData5])['UV_Index'].max())



filteredData6 = uv2013.Date == "06/08/2013"

print ((uv2013[filteredData6])['UV_Index'].max())



filteredData7 = uv2013.Date == "07/08/2013"

print ((uv2013[filteredData7])['UV_Index'].max())



filteredData8 = uv2013.Date == "08/08/2013"

print ((uv2013[filteredData8])['UV_Index'].max())



filteredData9 = uv2013.Date == "09/08/2013"

print ((uv2013[filteredData9])['UV_Index'].max())



filteredData10 = uv2013.Date == "10/08/2013"

print ((uv2013[filteredData10])['UV_Index'].max())



filteredData11 = uv2013.Date == "11/08/2013"

print ((uv2013[filteredData11])['UV_Index'].max())



filteredData12 = uv2013.Date == "12/08/2013"

print ((uv2013[filteredData12])['UV_Index'].max())



filteredData13 = uv2013.Date == "13/08/2013"

print ((uv2013[filteredData13])['UV_Index'].max())



filteredData14 = uv2013.Date == "14/08/2013"

print ((uv2013[filteredData14])['UV_Index'].max())



filteredData15 = uv2013.Date == "15/08/2013"

print ((uv2013[filteredData15])['UV_Index'].max())



filteredData16 = uv2013.Date == "16/08/2013"

print ((uv2013[filteredData16])['UV_Index'].max())



filteredData17 = uv2013.Date == "17/08/2013"

print ((uv2013[filteredData17])['UV_Index'].max())



filteredData18 = uv2013.Date == "18/08/2013"

print ((uv2013[filteredData18])['UV_Index'].max())



filteredData19 = uv2013.Date == "19/08/2013"

print ((uv2013[filteredData19])['UV_Index'].max())



filteredData20 = uv2013.Date == "20/08/2089"

print ((uv2013[filteredData20])['UV_Index'].max())



filteredData21 = uv2013.Date == "21/08/2013"

print ((uv2013[filteredData21])['UV_Index'].max())



filteredData22 = uv2013.Date == "22/08/2013"

print ((uv2013[filteredData22])['UV_Index'].max())



filteredData23 = uv2013.Date == "23/08/2013"

print ((uv2013[filteredData23])['UV_Index'].max())



filteredData24 = uv2013.Date == "24/08/2013"

print ((uv2013[filteredData24])['UV_Index'].max())



filteredData25 = uv2013.Date == "25/08/2013"

print ((uv2013[filteredData25])['UV_Index'].max())



filteredData26 = uv2013.Date == "26/08/2013"

print ((uv2013[filteredData26])['UV_Index'].max())



filteredData27 = uv2013.Date == "27/08/2013"

print ((uv2013[filteredData27])['UV_Index'].max())



filteredData28 = uv2013.Date == "28/08/2013"

print ((uv2013[filteredData28])['UV_Index'].max())



filteredData29 = uv2013.Date == "29/08/2013"

print ((uv2013[filteredData29])['UV_Index'].max())



filteredData30 = uv2013.Date == "30/08/2013"

print ((uv2013[filteredData30])['UV_Index'].max())



filteredData31 = uv2013.Date == "31/08/2013"

print ((uv2013[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2014 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2014.csv')



filteredData1 = uv2014.Date == "01/08/2014"

print((uv2014[filteredData1])['UV_Index'].max())



filteredData2 = uv2014.Date == "02/08/2014"

print ((uv2014[filteredData2])['UV_Index'].max())



filteredData3 = uv2014.Date == "03/08/2014"

print ((uv2014[filteredData3])['UV_Index'].max())



filteredData4 = uv2014.Date == "04/08/2014"

print ((uv2014[filteredData4])['UV_Index'].max())



filteredData5 = uv2014.Date == "05/08/2014"

print ((uv2014[filteredData5])['UV_Index'].max())



filteredData6 = uv2014.Date == "06/08/2014"

print ((uv2014[filteredData6])['UV_Index'].max())



filteredData7 = uv2014.Date == "07/08/2014"

print ((uv2014[filteredData7])['UV_Index'].max())



filteredData8 = uv2014.Date == "08/08/2014"

print ((uv2014[filteredData8])['UV_Index'].max())



filteredData9 = uv2014.Date == "09/08/2014"

print ((uv2014[filteredData9])['UV_Index'].max())



filteredData10 = uv2014.Date == "10/08/2014"

print ((uv2014[filteredData10])['UV_Index'].max())



filteredData11 = uv2014.Date == "11/08/2014"

print ((uv2014[filteredData11])['UV_Index'].max())



filteredData12 = uv2014.Date == "12/08/2014"

print ((uv2014[filteredData12])['UV_Index'].max())



filteredData13 = uv2014.Date == "13/08/2014"

print ((uv2014[filteredData13])['UV_Index'].max())



filteredData14 = uv2014.Date == "14/08/2014"

print ((uv2014[filteredData14])['UV_Index'].max())



filteredData15 = uv2014.Date == "15/08/2014"

print ((uv2014[filteredData15])['UV_Index'].max())



filteredData16 = uv2014.Date == "16/08/2014"

print ((uv2014[filteredData16])['UV_Index'].max())



filteredData17 = uv2014.Date == "17/08/2014"

print ((uv2014[filteredData17])['UV_Index'].max())



filteredData18 = uv2014.Date == "18/08/2014"

print ((uv2014[filteredData18])['UV_Index'].max())



filteredData19 = uv2014.Date == "19/08/2014"

print ((uv2014[filteredData19])['UV_Index'].max())



filteredData20 = uv2014.Date == "20/08/2089"

print ((uv2014[filteredData20])['UV_Index'].max())



filteredData21 = uv2014.Date == "21/08/2014"

print ((uv2014[filteredData21])['UV_Index'].max())



filteredData22 = uv2014.Date == "22/08/2014"

print ((uv2014[filteredData22])['UV_Index'].max())



filteredData23 = uv2014.Date == "23/08/2014"

print ((uv2014[filteredData23])['UV_Index'].max())



filteredData24 = uv2014.Date == "24/08/2014"

print ((uv2014[filteredData24])['UV_Index'].max())



filteredData25 = uv2014.Date == "25/08/2014"

print ((uv2014[filteredData25])['UV_Index'].max())



filteredData26 = uv2014.Date == "26/08/2014"

print ((uv2014[filteredData26])['UV_Index'].max())



filteredData27 = uv2014.Date == "27/08/2014"

print ((uv2014[filteredData27])['UV_Index'].max())



filteredData28 = uv2014.Date == "28/08/2014"

print ((uv2014[filteredData28])['UV_Index'].max())



filteredData29 = uv2014.Date == "29/08/2014"

print ((uv2014[filteredData29])['UV_Index'].max())



filteredData30 = uv2014.Date == "30/08/2014"

print ((uv2014[filteredData30])['UV_Index'].max())



filteredData31 = uv2014.Date == "31/08/2014"

print ((uv2014[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2015 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2015.csv')



filteredData1 = uv2015.Date == "01/08/2015"

print((uv2015[filteredData1])['UV_Index'].max())



filteredData2 = uv2015.Date == "02/08/2015"

print ((uv2015[filteredData2])['UV_Index'].max())



filteredData3 = uv2015.Date == "03/08/2015"

print ((uv2015[filteredData3])['UV_Index'].max())



filteredData4 = uv2015.Date == "04/08/2015"

print ((uv2015[filteredData4])['UV_Index'].max())



filteredData5 = uv2015.Date == "05/08/2015"

print ((uv2015[filteredData5])['UV_Index'].max())



filteredData6 = uv2015.Date == "06/08/2015"

print ((uv2015[filteredData6])['UV_Index'].max())



filteredData7 = uv2015.Date == "07/08/2015"

print ((uv2015[filteredData7])['UV_Index'].max())



filteredData8 = uv2015.Date == "08/08/2015"

print ((uv2015[filteredData8])['UV_Index'].max())



filteredData9 = uv2015.Date == "09/08/2015"

print ((uv2015[filteredData9])['UV_Index'].max())



filteredData10 = uv2015.Date == "10/08/2015"

print ((uv2015[filteredData10])['UV_Index'].max())



filteredData11 = uv2015.Date == "11/08/2015"

print ((uv2015[filteredData11])['UV_Index'].max())



filteredData12 = uv2015.Date == "12/08/2015"

print ((uv2015[filteredData12])['UV_Index'].max())



filteredData13 = uv2015.Date == "13/08/2015"

print ((uv2015[filteredData13])['UV_Index'].max())



filteredData14 = uv2015.Date == "14/08/2015"

print ((uv2015[filteredData14])['UV_Index'].max())



filteredData15 = uv2015.Date == "15/08/2015"

print ((uv2015[filteredData15])['UV_Index'].max())



filteredData16 = uv2015.Date == "16/08/2015"

print ((uv2015[filteredData16])['UV_Index'].max())



filteredData17 = uv2015.Date == "17/08/2015"

print ((uv2015[filteredData17])['UV_Index'].max())



filteredData18 = uv2015.Date == "18/08/2015"

print ((uv2015[filteredData18])['UV_Index'].max())



filteredData19 = uv2015.Date == "19/08/2015"

print ((uv2015[filteredData19])['UV_Index'].max())



filteredData20 = uv2015.Date == "20/08/2089"

print ((uv2015[filteredData20])['UV_Index'].max())



filteredData21 = uv2015.Date == "21/08/2015"

print ((uv2015[filteredData21])['UV_Index'].max())



filteredData22 = uv2015.Date == "22/08/2015"

print ((uv2015[filteredData22])['UV_Index'].max())



filteredData23 = uv2015.Date == "23/08/2015"

print ((uv2015[filteredData23])['UV_Index'].max())



filteredData24 = uv2015.Date == "24/08/2015"

print ((uv2015[filteredData24])['UV_Index'].max())



filteredData25 = uv2015.Date == "25/08/2015"

print ((uv2015[filteredData25])['UV_Index'].max())



filteredData26 = uv2015.Date == "26/08/2015"

print ((uv2015[filteredData26])['UV_Index'].max())



filteredData27 = uv2015.Date == "27/08/2015"

print ((uv2015[filteredData27])['UV_Index'].max())



filteredData28 = uv2015.Date == "28/08/2015"

print ((uv2015[filteredData28])['UV_Index'].max())



filteredData29 = uv2015.Date == "29/08/2015"

print ((uv2015[filteredData29])['UV_Index'].max())



filteredData30 = uv2015.Date == "30/08/2015"

print ((uv2015[filteredData30])['UV_Index'].max())



filteredData31 = uv2015.Date == "31/08/2015"

print ((uv2015[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2016 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2016.csv')



filteredData1 = uv2016.Date == "01/08/2016"

print((uv2016[filteredData1])['UV_Index'].max())



filteredData2 = uv2016.Date == "02/08/2016"

print ((uv2016[filteredData2])['UV_Index'].max())



filteredData3 = uv2016.Date == "03/08/2016"

print ((uv2016[filteredData3])['UV_Index'].max())



filteredData4 = uv2016.Date == "04/08/2016"

print ((uv2016[filteredData4])['UV_Index'].max())



filteredData5 = uv2016.Date == "05/08/2016"

print ((uv2016[filteredData5])['UV_Index'].max())



filteredData6 = uv2016.Date == "06/08/2016"

print ((uv2016[filteredData6])['UV_Index'].max())



filteredData7 = uv2016.Date == "07/08/2016"

print ((uv2016[filteredData7])['UV_Index'].max())



filteredData8 = uv2016.Date == "08/08/2016"

print ((uv2016[filteredData8])['UV_Index'].max())



filteredData9 = uv2016.Date == "09/08/2016"

print ((uv2016[filteredData9])['UV_Index'].max())



filteredData10 = uv2016.Date == "10/08/2016"

print ((uv2016[filteredData10])['UV_Index'].max())



filteredData11 = uv2016.Date == "11/08/2016"

print ((uv2016[filteredData11])['UV_Index'].max())



filteredData12 = uv2016.Date == "12/08/2016"

print ((uv2016[filteredData12])['UV_Index'].max())



filteredData13 = uv2016.Date == "13/08/2016"

print ((uv2016[filteredData13])['UV_Index'].max())



filteredData14 = uv2016.Date == "14/08/2016"

print ((uv2016[filteredData14])['UV_Index'].max())



filteredData15 = uv2016.Date == "15/08/2016"

print ((uv2016[filteredData15])['UV_Index'].max())



filteredData16 = uv2016.Date == "16/08/2016"

print ((uv2016[filteredData16])['UV_Index'].max())



filteredData17 = uv2016.Date == "17/08/2016"

print ((uv2016[filteredData17])['UV_Index'].max())



filteredData18 = uv2016.Date == "18/08/2016"

print ((uv2016[filteredData18])['UV_Index'].max())



filteredData19 = uv2016.Date == "19/08/2016"

print ((uv2016[filteredData19])['UV_Index'].max())



filteredData20 = uv2016.Date == "20/08/2089"

print ((uv2016[filteredData20])['UV_Index'].max())



filteredData21 = uv2016.Date == "21/08/2016"

print ((uv2016[filteredData21])['UV_Index'].max())



filteredData22 = uv2016.Date == "22/08/2016"

print ((uv2016[filteredData22])['UV_Index'].max())



filteredData23 = uv2016.Date == "23/08/2016"

print ((uv2016[filteredData23])['UV_Index'].max())



filteredData24 = uv2016.Date == "24/08/2016"

print ((uv2016[filteredData24])['UV_Index'].max())



filteredData25 = uv2016.Date == "25/08/2016"

print ((uv2016[filteredData25])['UV_Index'].max())



filteredData26 = uv2016.Date == "26/08/2016"

print ((uv2016[filteredData26])['UV_Index'].max())



filteredData27 = uv2016.Date == "27/08/2016"

print ((uv2016[filteredData27])['UV_Index'].max())



filteredData28 = uv2016.Date == "28/08/2016"

print ((uv2016[filteredData28])['UV_Index'].max())



filteredData29 = uv2016.Date == "29/08/2016"

print ((uv2016[filteredData29])['UV_Index'].max())



filteredData30 = uv2016.Date == "30/08/2016"

print ((uv2016[filteredData30])['UV_Index'].max())



filteredData31 = uv2016.Date == "31/08/2016"

print ((uv2016[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2017 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2017.csv')



filteredData1 = uv2017.Date == "01/08/2017"

print((uv2017[filteredData1])['UV_Index'].max())



filteredData2 = uv2017.Date == "02/08/2017"

print ((uv2017[filteredData2])['UV_Index'].max())



filteredData3 = uv2017.Date == "03/08/2017"

print ((uv2017[filteredData3])['UV_Index'].max())



filteredData4 = uv2017.Date == "04/08/2017"

print ((uv2017[filteredData4])['UV_Index'].max())



filteredData5 = uv2017.Date == "05/08/2017"

print ((uv2017[filteredData5])['UV_Index'].max())



filteredData6 = uv2017.Date == "06/08/2017"

print ((uv2017[filteredData6])['UV_Index'].max())



filteredData7 = uv2017.Date == "07/08/2017"

print ((uv2017[filteredData7])['UV_Index'].max())



filteredData8 = uv2017.Date == "08/08/2017"

print ((uv2017[filteredData8])['UV_Index'].max())



filteredData9 = uv2017.Date == "09/08/2017"

print ((uv2017[filteredData9])['UV_Index'].max())



filteredData10 = uv2017.Date == "10/08/2017"

print ((uv2017[filteredData10])['UV_Index'].max())



filteredData11 = uv2017.Date == "11/08/2017"

print ((uv2017[filteredData11])['UV_Index'].max())



filteredData12 = uv2017.Date == "12/08/2017"

print ((uv2017[filteredData12])['UV_Index'].max())



filteredData13 = uv2017.Date == "13/08/2017"

print ((uv2017[filteredData13])['UV_Index'].max())



filteredData14 = uv2017.Date == "14/08/2017"

print ((uv2017[filteredData14])['UV_Index'].max())



filteredData15 = uv2017.Date == "15/08/2017"

print ((uv2017[filteredData15])['UV_Index'].max())



filteredData16 = uv2017.Date == "16/08/2017"

print ((uv2017[filteredData16])['UV_Index'].max())



filteredData17 = uv2017.Date == "17/08/2017"

print ((uv2017[filteredData17])['UV_Index'].max())



filteredData18 = uv2017.Date == "18/08/2017"

print ((uv2017[filteredData18])['UV_Index'].max())



filteredData19 = uv2017.Date == "19/08/2017"

print ((uv2017[filteredData19])['UV_Index'].max())



filteredData20 = uv2017.Date == "20/08/2089"

print ((uv2017[filteredData20])['UV_Index'].max())



filteredData21 = uv2017.Date == "21/08/2017"

print ((uv2017[filteredData21])['UV_Index'].max())



filteredData22 = uv2017.Date == "22/08/2017"

print ((uv2017[filteredData22])['UV_Index'].max())



filteredData23 = uv2017.Date == "23/08/2017"

print ((uv2017[filteredData23])['UV_Index'].max())



filteredData24 = uv2017.Date == "24/08/2017"

print ((uv2017[filteredData24])['UV_Index'].max())



filteredData25 = uv2017.Date == "25/08/2017"

print ((uv2017[filteredData25])['UV_Index'].max())



filteredData26 = uv2017.Date == "26/08/2017"

print ((uv2017[filteredData26])['UV_Index'].max())



filteredData27 = uv2017.Date == "27/08/2017"

print ((uv2017[filteredData27])['UV_Index'].max())



filteredData28 = uv2017.Date == "28/08/2017"

print ((uv2017[filteredData28])['UV_Index'].max())



filteredData29 = uv2017.Date == "29/08/2017"

print ((uv2017[filteredData29])['UV_Index'].max())



filteredData30 = uv2017.Date == "30/08/2017"

print ((uv2017[filteredData30])['UV_Index'].max())



filteredData31 = uv2017.Date == "31/08/2017"

print ((uv2017[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2018 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2018.csv')



filteredData1 = uv2018.Date == "01/08/2018"

print((uv2018[filteredData1])['UV_Index'].max())



filteredData2 = uv2018.Date == "02/08/2018"

print ((uv2018[filteredData2])['UV_Index'].max())



filteredData3 = uv2018.Date == "03/08/2018"

print ((uv2018[filteredData3])['UV_Index'].max())



filteredData4 = uv2018.Date == "04/08/2018"

print ((uv2018[filteredData4])['UV_Index'].max())



filteredData5 = uv2018.Date == "05/08/2018"

print ((uv2018[filteredData5])['UV_Index'].max())



filteredData6 = uv2018.Date == "06/08/2018"

print ((uv2018[filteredData6])['UV_Index'].max())



filteredData7 = uv2018.Date == "07/08/2018"

print ((uv2018[filteredData7])['UV_Index'].max())



filteredData8 = uv2018.Date == "08/08/2018"

print ((uv2018[filteredData8])['UV_Index'].max())



filteredData9 = uv2018.Date == "09/08/2018"

print ((uv2018[filteredData9])['UV_Index'].max())



filteredData10 = uv2018.Date == "10/08/2018"

print ((uv2018[filteredData10])['UV_Index'].max())



filteredData11 = uv2018.Date == "11/08/2018"

print ((uv2018[filteredData11])['UV_Index'].max())



filteredData12 = uv2018.Date == "12/08/2018"

print ((uv2018[filteredData12])['UV_Index'].max())



filteredData13 = uv2018.Date == "13/08/2018"

print ((uv2018[filteredData13])['UV_Index'].max())



filteredData14 = uv2018.Date == "14/08/2018"

print ((uv2018[filteredData14])['UV_Index'].max())



filteredData15 = uv2018.Date == "15/08/2018"

print ((uv2018[filteredData15])['UV_Index'].max())



filteredData16 = uv2018.Date == "16/08/2018"

print ((uv2018[filteredData16])['UV_Index'].max())



filteredData17 = uv2018.Date == "17/08/2018"

print ((uv2018[filteredData17])['UV_Index'].max())



filteredData18 = uv2018.Date == "18/08/2018"

print ((uv2018[filteredData18])['UV_Index'].max())



filteredData19 = uv2018.Date == "19/08/2018"

print ((uv2018[filteredData19])['UV_Index'].max())



filteredData20 = uv2018.Date == "20/08/2089"

print ((uv2018[filteredData20])['UV_Index'].max())



filteredData21 = uv2018.Date == "21/08/2018"

print ((uv2018[filteredData21])['UV_Index'].max())



filteredData22 = uv2018.Date == "22/08/2018"

print ((uv2018[filteredData22])['UV_Index'].max())



filteredData23 = uv2018.Date == "23/08/2018"

print ((uv2018[filteredData23])['UV_Index'].max())



filteredData24 = uv2018.Date == "24/08/2018"

print ((uv2018[filteredData24])['UV_Index'].max())



filteredData25 = uv2018.Date == "25/08/2018"

print ((uv2018[filteredData25])['UV_Index'].max())



filteredData26 = uv2018.Date == "26/08/2018"

print ((uv2018[filteredData26])['UV_Index'].max())



filteredData27 = uv2018.Date == "27/08/2018"

print ((uv2018[filteredData27])['UV_Index'].max())



filteredData28 = uv2018.Date == "28/08/2018"

print ((uv2018[filteredData28])['UV_Index'].max())



filteredData29 = uv2018.Date == "29/08/2018"

print ((uv2018[filteredData29])['UV_Index'].max())



filteredData30 = uv2018.Date == "30/08/2018"

print ((uv2018[filteredData30])['UV_Index'].max())



filteredData31 = uv2018.Date == "31/08/2018"

print ((uv2018[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2019 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2019.csv')



filteredData1 = uv2019.Date == "01/08/2019"

print((uv2019[filteredData1])['UV_Index'].max())



filteredData2 = uv2019.Date == "02/08/2019"

print ((uv2019[filteredData2])['UV_Index'].max())



filteredData3 = uv2019.Date == "03/08/2019"

print ((uv2019[filteredData3])['UV_Index'].max())



filteredData4 = uv2019.Date == "04/08/2019"

print ((uv2019[filteredData4])['UV_Index'].max())



filteredData5 = uv2019.Date == "05/08/2019"

print ((uv2019[filteredData5])['UV_Index'].max())



filteredData6 = uv2019.Date == "06/08/2019"

print ((uv2019[filteredData6])['UV_Index'].max())



filteredData7 = uv2019.Date == "07/08/2019"

print ((uv2019[filteredData7])['UV_Index'].max())



filteredData8 = uv2019.Date == "08/08/2019"

print ((uv2019[filteredData8])['UV_Index'].max())



filteredData9 = uv2019.Date == "09/08/2019"

print ((uv2019[filteredData9])['UV_Index'].max())



filteredData10 = uv2019.Date == "10/08/2019"

print ((uv2019[filteredData10])['UV_Index'].max())



filteredData11 = uv2019.Date == "11/08/2019"

print ((uv2019[filteredData11])['UV_Index'].max())



filteredData12 = uv2019.Date == "12/08/2019"

print ((uv2019[filteredData12])['UV_Index'].max())



filteredData13 = uv2019.Date == "13/08/2019"

print ((uv2019[filteredData13])['UV_Index'].max())



filteredData14 = uv2019.Date == "14/08/2019"

print ((uv2019[filteredData14])['UV_Index'].max())



filteredData15 = uv2019.Date == "15/08/2019"

print ((uv2019[filteredData15])['UV_Index'].max())



filteredData16 = uv2019.Date == "16/08/2019"

print ((uv2019[filteredData16])['UV_Index'].max())



filteredData17 = uv2019.Date == "17/08/2019"

print ((uv2019[filteredData17])['UV_Index'].max())



filteredData18 = uv2019.Date == "18/08/2019"

print ((uv2019[filteredData18])['UV_Index'].max())



filteredData19 = uv2019.Date == "19/08/2019"

print ((uv2019[filteredData19])['UV_Index'].max())



filteredData20 = uv2019.Date == "20/08/2089"

print ((uv2019[filteredData20])['UV_Index'].max())



filteredData21 = uv2019.Date == "21/08/2019"

print ((uv2019[filteredData21])['UV_Index'].max())



filteredData22 = uv2019.Date == "22/08/2019"

print ((uv2019[filteredData22])['UV_Index'].max())



filteredData23 = uv2019.Date == "23/08/2019"

print ((uv2019[filteredData23])['UV_Index'].max())



filteredData24 = uv2019.Date == "24/08/2019"

print ((uv2019[filteredData24])['UV_Index'].max())



filteredData25 = uv2019.Date == "25/08/2019"

print ((uv2019[filteredData25])['UV_Index'].max())



filteredData26 = uv2019.Date == "26/08/2019"

print ((uv2019[filteredData26])['UV_Index'].max())



filteredData27 = uv2019.Date == "27/08/2019"

print ((uv2019[filteredData27])['UV_Index'].max())



filteredData28 = uv2019.Date == "28/08/2019"

print ((uv2019[filteredData28])['UV_Index'].max())



filteredData29 = uv2019.Date == "29/08/2019"

print ((uv2019[filteredData29])['UV_Index'].max())



filteredData30 = uv2019.Date == "30/08/2019"

print ((uv2019[filteredData30])['UV_Index'].max())



filteredData31 = uv2019.Date == "31/08/2019"

print ((uv2019[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2019 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2019.csv')



filteredData1 = uv2019.Date == "01/09/2019"

print((uv2019[filteredData1])['UV_Index'].max())



filteredData2 = uv2019.Date == "02/09/2019"

print ((uv2019[filteredData2])['UV_Index'].max())



filteredData3 = uv2019.Date == "03/09/2019"

print ((uv2019[filteredData3])['UV_Index'].max())



filteredData4 = uv2019.Date == "04/09/2019"

print ((uv2019[filteredData4])['UV_Index'].max())



filteredData5 = uv2019.Date == "05/09/2019"

print ((uv2019[filteredData5])['UV_Index'].max())



filteredData6 = uv2019.Date == "06/09/2019"

print ((uv2019[filteredData6])['UV_Index'].max())



filteredData7 = uv2019.Date == "07/09/2019"

print ((uv2019[filteredData7])['UV_Index'].max())



filteredData8 = uv2019.Date == "08/09/2019"

print ((uv2019[filteredData8])['UV_Index'].max())



filteredData9 = uv2019.Date == "09/09/2019"

print ((uv2019[filteredData9])['UV_Index'].max())



filteredData10 = uv2019.Date == "10/09/2019"

print ((uv2019[filteredData10])['UV_Index'].max())



filteredData11 = uv2019.Date == "11/09/2019"

print ((uv2019[filteredData11])['UV_Index'].max())



filteredData12 = uv2019.Date == "12/09/2019"

print ((uv2019[filteredData12])['UV_Index'].max())



filteredData13 = uv2019.Date == "13/09/2019"

print ((uv2019[filteredData13])['UV_Index'].max())



filteredData14 = uv2019.Date == "14/09/2019"

print ((uv2019[filteredData14])['UV_Index'].max())



filteredData15 = uv2019.Date == "15/09/2019"

print ((uv2019[filteredData15])['UV_Index'].max())



filteredData16 = uv2019.Date == "16/09/2019"

print ((uv2019[filteredData16])['UV_Index'].max())



filteredData17 = uv2019.Date == "17/09/2019"

print ((uv2019[filteredData17])['UV_Index'].max())



filteredData18 = uv2019.Date == "18/09/2019"

print ((uv2019[filteredData18])['UV_Index'].max())



filteredData19 = uv2019.Date == "19/09/2019"

print ((uv2019[filteredData19])['UV_Index'].max())



filteredData20 = uv2019.Date == "20/09/2099"

print ((uv2019[filteredData20])['UV_Index'].max())



filteredData21 = uv2019.Date == "21/09/2019"

print ((uv2019[filteredData21])['UV_Index'].max())



filteredData22 = uv2019.Date == "22/09/2019"

print ((uv2019[filteredData22])['UV_Index'].max())



filteredData23 = uv2019.Date == "23/09/2019"

print ((uv2019[filteredData23])['UV_Index'].max())



filteredData24 = uv2019.Date == "24/09/2019"

print ((uv2019[filteredData24])['UV_Index'].max())



filteredData25 = uv2019.Date == "25/09/2019"

print ((uv2019[filteredData25])['UV_Index'].max())



filteredData26 = uv2019.Date == "26/09/2019"

print ((uv2019[filteredData26])['UV_Index'].max())



filteredData27 = uv2019.Date == "27/09/2019"

print ((uv2019[filteredData27])['UV_Index'].max())



filteredData28 = uv2019.Date == "28/09/2019"

print ((uv2019[filteredData28])['UV_Index'].max())



filteredData29 = uv2019.Date == "29/09/2019"

print ((uv2019[filteredData29])['UV_Index'].max())



filteredData30 = uv2019.Date == "30/09/2019"

print ((uv2019[filteredData30])['UV_Index'].max())



filteredData31 = uv2019.Date == "31/09/2019"

print ((uv2019[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2018 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2018.csv')



filteredData1 = uv2018.Date == "01/09/2018"

print((uv2018[filteredData1])['UV_Index'].max())



filteredData2 = uv2018.Date == "02/09/2018"

print ((uv2018[filteredData2])['UV_Index'].max())



filteredData3 = uv2018.Date == "03/09/2018"

print ((uv2018[filteredData3])['UV_Index'].max())



filteredData4 = uv2018.Date == "04/09/2018"

print ((uv2018[filteredData4])['UV_Index'].max())



filteredData5 = uv2018.Date == "05/09/2018"

print ((uv2018[filteredData5])['UV_Index'].max())



filteredData6 = uv2018.Date == "06/09/2018"

print ((uv2018[filteredData6])['UV_Index'].max())



filteredData7 = uv2018.Date == "07/09/2018"

print ((uv2018[filteredData7])['UV_Index'].max())



filteredData8 = uv2018.Date == "08/09/2018"

print ((uv2018[filteredData8])['UV_Index'].max())



filteredData9 = uv2018.Date == "09/09/2018"

print ((uv2018[filteredData9])['UV_Index'].max())



filteredData10 = uv2018.Date == "10/09/2018"

print ((uv2018[filteredData10])['UV_Index'].max())



filteredData11 = uv2018.Date == "11/09/2018"

print ((uv2018[filteredData11])['UV_Index'].max())



filteredData12 = uv2018.Date == "12/09/2018"

print ((uv2018[filteredData12])['UV_Index'].max())



filteredData13 = uv2018.Date == "13/09/2018"

print ((uv2018[filteredData13])['UV_Index'].max())



filteredData14 = uv2018.Date == "14/09/2018"

print ((uv2018[filteredData14])['UV_Index'].max())



filteredData15 = uv2018.Date == "15/09/2018"

print ((uv2018[filteredData15])['UV_Index'].max())



filteredData16 = uv2018.Date == "16/09/2018"

print ((uv2018[filteredData16])['UV_Index'].max())



filteredData17 = uv2018.Date == "17/09/2018"

print ((uv2018[filteredData17])['UV_Index'].max())



filteredData18 = uv2018.Date == "18/09/2018"

print ((uv2018[filteredData18])['UV_Index'].max())



filteredData19 = uv2018.Date == "19/09/2018"

print ((uv2018[filteredData19])['UV_Index'].max())



filteredData20 = uv2018.Date == "20/09/2099"

print ((uv2018[filteredData20])['UV_Index'].max())



filteredData21 = uv2018.Date == "21/09/2018"

print ((uv2018[filteredData21])['UV_Index'].max())



filteredData22 = uv2018.Date == "22/09/2018"

print ((uv2018[filteredData22])['UV_Index'].max())



filteredData23 = uv2018.Date == "23/09/2018"

print ((uv2018[filteredData23])['UV_Index'].max())



filteredData24 = uv2018.Date == "24/09/2018"

print ((uv2018[filteredData24])['UV_Index'].max())



filteredData25 = uv2018.Date == "25/09/2018"

print ((uv2018[filteredData25])['UV_Index'].max())



filteredData26 = uv2018.Date == "26/09/2018"

print ((uv2018[filteredData26])['UV_Index'].max())



filteredData27 = uv2018.Date == "27/09/2018"

print ((uv2018[filteredData27])['UV_Index'].max())



filteredData28 = uv2018.Date == "28/09/2018"

print ((uv2018[filteredData28])['UV_Index'].max())



filteredData29 = uv2018.Date == "29/09/2018"

print ((uv2018[filteredData29])['UV_Index'].max())



filteredData30 = uv2018.Date == "30/09/2018"

print ((uv2018[filteredData30])['UV_Index'].max())



filteredData31 = uv2018.Date == "31/09/2018"

print ((uv2018[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2017 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2017.csv')



filteredData1 = uv2017.Date == "01/09/2017"

print((uv2017[filteredData1])['UV_Index'].max())



filteredData2 = uv2017.Date == "02/09/2017"

print ((uv2017[filteredData2])['UV_Index'].max())



filteredData3 = uv2017.Date == "03/09/2017"

print ((uv2017[filteredData3])['UV_Index'].max())



filteredData4 = uv2017.Date == "04/09/2017"

print ((uv2017[filteredData4])['UV_Index'].max())



filteredData5 = uv2017.Date == "05/09/2017"

print ((uv2017[filteredData5])['UV_Index'].max())



filteredData6 = uv2017.Date == "06/09/2017"

print ((uv2017[filteredData6])['UV_Index'].max())



filteredData7 = uv2017.Date == "07/09/2017"

print ((uv2017[filteredData7])['UV_Index'].max())



filteredData8 = uv2017.Date == "08/09/2017"

print ((uv2017[filteredData8])['UV_Index'].max())



filteredData9 = uv2017.Date == "09/09/2017"

print ((uv2017[filteredData9])['UV_Index'].max())



filteredData10 = uv2017.Date == "10/09/2017"

print ((uv2017[filteredData10])['UV_Index'].max())



filteredData11 = uv2017.Date == "11/09/2017"

print ((uv2017[filteredData11])['UV_Index'].max())



filteredData12 = uv2017.Date == "12/09/2017"

print ((uv2017[filteredData12])['UV_Index'].max())



filteredData13 = uv2017.Date == "13/09/2017"

print ((uv2017[filteredData13])['UV_Index'].max())



filteredData14 = uv2017.Date == "14/09/2017"

print ((uv2017[filteredData14])['UV_Index'].max())



filteredData15 = uv2017.Date == "15/09/2017"

print ((uv2017[filteredData15])['UV_Index'].max())



filteredData16 = uv2017.Date == "16/09/2017"

print ((uv2017[filteredData16])['UV_Index'].max())



filteredData17 = uv2017.Date == "17/09/2017"

print ((uv2017[filteredData17])['UV_Index'].max())



filteredData18 = uv2017.Date == "18/09/2017"

print ((uv2017[filteredData18])['UV_Index'].max())



filteredData19 = uv2017.Date == "19/09/2017"

print ((uv2017[filteredData19])['UV_Index'].max())



filteredData20 = uv2017.Date == "20/09/2099"

print ((uv2017[filteredData20])['UV_Index'].max())



filteredData21 = uv2017.Date == "21/09/2017"

print ((uv2017[filteredData21])['UV_Index'].max())



filteredData22 = uv2017.Date == "22/09/2017"

print ((uv2017[filteredData22])['UV_Index'].max())



filteredData23 = uv2017.Date == "23/09/2017"

print ((uv2017[filteredData23])['UV_Index'].max())



filteredData24 = uv2017.Date == "24/09/2017"

print ((uv2017[filteredData24])['UV_Index'].max())



filteredData25 = uv2017.Date == "25/09/2017"

print ((uv2017[filteredData25])['UV_Index'].max())



filteredData26 = uv2017.Date == "26/09/2017"

print ((uv2017[filteredData26])['UV_Index'].max())



filteredData27 = uv2017.Date == "27/09/2017"

print ((uv2017[filteredData27])['UV_Index'].max())



filteredData28 = uv2017.Date == "28/09/2017"

print ((uv2017[filteredData28])['UV_Index'].max())



filteredData29 = uv2017.Date == "29/09/2017"

print ((uv2017[filteredData29])['UV_Index'].max())



filteredData30 = uv2017.Date == "30/09/2017"

print ((uv2017[filteredData30])['UV_Index'].max())



filteredData31 = uv2017.Date == "31/09/2017"

print ((uv2017[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2016 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2016.csv')



filteredData1 = uv2016.Date == "01/09/2016"

print((uv2016[filteredData1])['UV_Index'].max())



filteredData2 = uv2016.Date == "02/09/2016"

print ((uv2016[filteredData2])['UV_Index'].max())



filteredData3 = uv2016.Date == "03/09/2016"

print ((uv2016[filteredData3])['UV_Index'].max())



filteredData4 = uv2016.Date == "04/09/2016"

print ((uv2016[filteredData4])['UV_Index'].max())



filteredData5 = uv2016.Date == "05/09/2016"

print ((uv2016[filteredData5])['UV_Index'].max())



filteredData6 = uv2016.Date == "06/09/2016"

print ((uv2016[filteredData6])['UV_Index'].max())



filteredData7 = uv2016.Date == "07/09/2016"

print ((uv2016[filteredData7])['UV_Index'].max())



filteredData8 = uv2016.Date == "08/09/2016"

print ((uv2016[filteredData8])['UV_Index'].max())



filteredData9 = uv2016.Date == "09/09/2016"

print ((uv2016[filteredData9])['UV_Index'].max())



filteredData10 = uv2016.Date == "10/09/2016"

print ((uv2016[filteredData10])['UV_Index'].max())



filteredData11 = uv2016.Date == "11/09/2016"

print ((uv2016[filteredData11])['UV_Index'].max())



filteredData12 = uv2016.Date == "12/09/2016"

print ((uv2016[filteredData12])['UV_Index'].max())



filteredData13 = uv2016.Date == "13/09/2016"

print ((uv2016[filteredData13])['UV_Index'].max())



filteredData14 = uv2016.Date == "14/09/2016"

print ((uv2016[filteredData14])['UV_Index'].max())



filteredData15 = uv2016.Date == "15/09/2016"

print ((uv2016[filteredData15])['UV_Index'].max())



filteredData16 = uv2016.Date == "16/09/2016"

print ((uv2016[filteredData16])['UV_Index'].max())



filteredData17 = uv2016.Date == "17/09/2016"

print ((uv2016[filteredData17])['UV_Index'].max())



filteredData18 = uv2016.Date == "18/09/2016"

print ((uv2016[filteredData18])['UV_Index'].max())



filteredData19 = uv2016.Date == "19/09/2016"

print ((uv2016[filteredData19])['UV_Index'].max())



filteredData20 = uv2016.Date == "20/09/2099"

print ((uv2016[filteredData20])['UV_Index'].max())



filteredData21 = uv2016.Date == "21/09/2016"

print ((uv2016[filteredData21])['UV_Index'].max())



filteredData22 = uv2016.Date == "22/09/2016"

print ((uv2016[filteredData22])['UV_Index'].max())



filteredData23 = uv2016.Date == "23/09/2016"

print ((uv2016[filteredData23])['UV_Index'].max())



filteredData24 = uv2016.Date == "24/09/2016"

print ((uv2016[filteredData24])['UV_Index'].max())



filteredData25 = uv2016.Date == "25/09/2016"

print ((uv2016[filteredData25])['UV_Index'].max())



filteredData26 = uv2016.Date == "26/09/2016"

print ((uv2016[filteredData26])['UV_Index'].max())



filteredData27 = uv2016.Date == "27/09/2016"

print ((uv2016[filteredData27])['UV_Index'].max())



filteredData28 = uv2016.Date == "28/09/2016"

print ((uv2016[filteredData28])['UV_Index'].max())



filteredData29 = uv2016.Date == "29/09/2016"

print ((uv2016[filteredData29])['UV_Index'].max())



filteredData30 = uv2016.Date == "30/09/2016"

print ((uv2016[filteredData30])['UV_Index'].max())



filteredData31 = uv2016.Date == "31/09/2016"

print ((uv2016[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2015 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2015.csv')



filteredData1 = uv2015.Date == "01/09/2015"

print((uv2015[filteredData1])['UV_Index'].max())



filteredData2 = uv2015.Date == "02/09/2015"

print ((uv2015[filteredData2])['UV_Index'].max())



filteredData3 = uv2015.Date == "03/09/2015"

print ((uv2015[filteredData3])['UV_Index'].max())



filteredData4 = uv2015.Date == "04/09/2015"

print ((uv2015[filteredData4])['UV_Index'].max())



filteredData5 = uv2015.Date == "05/09/2015"

print ((uv2015[filteredData5])['UV_Index'].max())



filteredData6 = uv2015.Date == "06/09/2015"

print ((uv2015[filteredData6])['UV_Index'].max())



filteredData7 = uv2015.Date == "07/09/2015"

print ((uv2015[filteredData7])['UV_Index'].max())



filteredData8 = uv2015.Date == "08/09/2015"

print ((uv2015[filteredData8])['UV_Index'].max())



filteredData9 = uv2015.Date == "09/09/2015"

print ((uv2015[filteredData9])['UV_Index'].max())



filteredData10 = uv2015.Date == "10/09/2015"

print ((uv2015[filteredData10])['UV_Index'].max())



filteredData11 = uv2015.Date == "11/09/2015"

print ((uv2015[filteredData11])['UV_Index'].max())



filteredData12 = uv2015.Date == "12/09/2015"

print ((uv2015[filteredData12])['UV_Index'].max())



filteredData13 = uv2015.Date == "13/09/2015"

print ((uv2015[filteredData13])['UV_Index'].max())



filteredData14 = uv2015.Date == "14/09/2015"

print ((uv2015[filteredData14])['UV_Index'].max())



filteredData15 = uv2015.Date == "15/09/2015"

print ((uv2015[filteredData15])['UV_Index'].max())



filteredData16 = uv2015.Date == "16/09/2015"

print ((uv2015[filteredData16])['UV_Index'].max())



filteredData17 = uv2015.Date == "17/09/2015"

print ((uv2015[filteredData17])['UV_Index'].max())



filteredData18 = uv2015.Date == "18/09/2015"

print ((uv2015[filteredData18])['UV_Index'].max())



filteredData19 = uv2015.Date == "19/09/2015"

print ((uv2015[filteredData19])['UV_Index'].max())



filteredData20 = uv2015.Date == "20/09/2099"

print ((uv2015[filteredData20])['UV_Index'].max())



filteredData21 = uv2015.Date == "21/09/2015"

print ((uv2015[filteredData21])['UV_Index'].max())



filteredData22 = uv2015.Date == "22/09/2015"

print ((uv2015[filteredData22])['UV_Index'].max())



filteredData23 = uv2015.Date == "23/09/2015"

print ((uv2015[filteredData23])['UV_Index'].max())



filteredData24 = uv2015.Date == "24/09/2015"

print ((uv2015[filteredData24])['UV_Index'].max())



filteredData25 = uv2015.Date == "25/09/2015"

print ((uv2015[filteredData25])['UV_Index'].max())



filteredData26 = uv2015.Date == "26/09/2015"

print ((uv2015[filteredData26])['UV_Index'].max())



filteredData27 = uv2015.Date == "27/09/2015"

print ((uv2015[filteredData27])['UV_Index'].max())



filteredData28 = uv2015.Date == "28/09/2015"

print ((uv2015[filteredData28])['UV_Index'].max())



filteredData29 = uv2015.Date == "29/09/2015"

print ((uv2015[filteredData29])['UV_Index'].max())



filteredData30 = uv2015.Date == "30/09/2015"

print ((uv2015[filteredData30])['UV_Index'].max())



filteredData31 = uv2015.Date == "31/09/2015"

print ((uv2015[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2014 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2014.csv')



filteredData1 = uv2014.Date == "01/09/2014"

print((uv2014[filteredData1])['UV_Index'].max())



filteredData2 = uv2014.Date == "02/09/2014"

print ((uv2014[filteredData2])['UV_Index'].max())



filteredData3 = uv2014.Date == "03/09/2014"

print ((uv2014[filteredData3])['UV_Index'].max())



filteredData4 = uv2014.Date == "04/09/2014"

print ((uv2014[filteredData4])['UV_Index'].max())



filteredData5 = uv2014.Date == "05/09/2014"

print ((uv2014[filteredData5])['UV_Index'].max())



filteredData6 = uv2014.Date == "06/09/2014"

print ((uv2014[filteredData6])['UV_Index'].max())



filteredData7 = uv2014.Date == "07/09/2014"

print ((uv2014[filteredData7])['UV_Index'].max())



filteredData8 = uv2014.Date == "08/09/2014"

print ((uv2014[filteredData8])['UV_Index'].max())



filteredData9 = uv2014.Date == "09/09/2014"

print ((uv2014[filteredData9])['UV_Index'].max())



filteredData10 = uv2014.Date == "10/09/2014"

print ((uv2014[filteredData10])['UV_Index'].max())



filteredData11 = uv2014.Date == "11/09/2014"

print ((uv2014[filteredData11])['UV_Index'].max())



filteredData12 = uv2014.Date == "12/09/2014"

print ((uv2014[filteredData12])['UV_Index'].max())



filteredData13 = uv2014.Date == "13/09/2014"

print ((uv2014[filteredData13])['UV_Index'].max())



filteredData14 = uv2014.Date == "14/09/2014"

print ((uv2014[filteredData14])['UV_Index'].max())



filteredData15 = uv2014.Date == "15/09/2014"

print ((uv2014[filteredData15])['UV_Index'].max())



filteredData16 = uv2014.Date == "16/09/2014"

print ((uv2014[filteredData16])['UV_Index'].max())



filteredData17 = uv2014.Date == "17/09/2014"

print ((uv2014[filteredData17])['UV_Index'].max())



filteredData18 = uv2014.Date == "18/09/2014"

print ((uv2014[filteredData18])['UV_Index'].max())



filteredData19 = uv2014.Date == "19/09/2014"

print ((uv2014[filteredData19])['UV_Index'].max())



filteredData20 = uv2014.Date == "20/09/2099"

print ((uv2014[filteredData20])['UV_Index'].max())



filteredData21 = uv2014.Date == "21/09/2014"

print ((uv2014[filteredData21])['UV_Index'].max())



filteredData22 = uv2014.Date == "22/09/2014"

print ((uv2014[filteredData22])['UV_Index'].max())



filteredData23 = uv2014.Date == "23/09/2014"

print ((uv2014[filteredData23])['UV_Index'].max())



filteredData24 = uv2014.Date == "24/09/2014"

print ((uv2014[filteredData24])['UV_Index'].max())



filteredData25 = uv2014.Date == "25/09/2014"

print ((uv2014[filteredData25])['UV_Index'].max())



filteredData26 = uv2014.Date == "26/09/2014"

print ((uv2014[filteredData26])['UV_Index'].max())



filteredData27 = uv2014.Date == "27/09/2014"

print ((uv2014[filteredData27])['UV_Index'].max())



filteredData28 = uv2014.Date == "28/09/2014"

print ((uv2014[filteredData28])['UV_Index'].max())



filteredData29 = uv2014.Date == "29/09/2014"

print ((uv2014[filteredData29])['UV_Index'].max())



filteredData30 = uv2014.Date == "30/09/2014"

print ((uv2014[filteredData30])['UV_Index'].max())



filteredData31 = uv2014.Date == "31/09/2014"

print ((uv2014[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2013 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2013.csv')



filteredData1 = uv2013.Date == "01/09/2013"

print((uv2013[filteredData1])['UV_Index'].max())



filteredData2 = uv2013.Date == "02/09/2013"

print ((uv2013[filteredData2])['UV_Index'].max())



filteredData3 = uv2013.Date == "03/09/2013"

print ((uv2013[filteredData3])['UV_Index'].max())



filteredData4 = uv2013.Date == "04/09/2013"

print ((uv2013[filteredData4])['UV_Index'].max())



filteredData5 = uv2013.Date == "05/09/2013"

print ((uv2013[filteredData5])['UV_Index'].max())



filteredData6 = uv2013.Date == "06/09/2013"

print ((uv2013[filteredData6])['UV_Index'].max())



filteredData7 = uv2013.Date == "07/09/2013"

print ((uv2013[filteredData7])['UV_Index'].max())



filteredData8 = uv2013.Date == "08/09/2013"

print ((uv2013[filteredData8])['UV_Index'].max())



filteredData9 = uv2013.Date == "09/09/2013"

print ((uv2013[filteredData9])['UV_Index'].max())



filteredData10 = uv2013.Date == "10/09/2013"

print ((uv2013[filteredData10])['UV_Index'].max())



filteredData11 = uv2013.Date == "11/09/2013"

print ((uv2013[filteredData11])['UV_Index'].max())



filteredData12 = uv2013.Date == "12/09/2013"

print ((uv2013[filteredData12])['UV_Index'].max())



filteredData13 = uv2013.Date == "13/09/2013"

print ((uv2013[filteredData13])['UV_Index'].max())



filteredData14 = uv2013.Date == "14/09/2013"

print ((uv2013[filteredData14])['UV_Index'].max())



filteredData15 = uv2013.Date == "15/09/2013"

print ((uv2013[filteredData15])['UV_Index'].max())



filteredData16 = uv2013.Date == "16/09/2013"

print ((uv2013[filteredData16])['UV_Index'].max())



filteredData17 = uv2013.Date == "17/09/2013"

print ((uv2013[filteredData17])['UV_Index'].max())



filteredData18 = uv2013.Date == "18/09/2013"

print ((uv2013[filteredData18])['UV_Index'].max())



filteredData19 = uv2013.Date == "19/09/2013"

print ((uv2013[filteredData19])['UV_Index'].max())



filteredData20 = uv2013.Date == "20/09/2099"

print ((uv2013[filteredData20])['UV_Index'].max())



filteredData21 = uv2013.Date == "21/09/2013"

print ((uv2013[filteredData21])['UV_Index'].max())



filteredData22 = uv2013.Date == "22/09/2013"

print ((uv2013[filteredData22])['UV_Index'].max())



filteredData23 = uv2013.Date == "23/09/2013"

print ((uv2013[filteredData23])['UV_Index'].max())



filteredData24 = uv2013.Date == "24/09/2013"

print ((uv2013[filteredData24])['UV_Index'].max())



filteredData25 = uv2013.Date == "25/09/2013"

print ((uv2013[filteredData25])['UV_Index'].max())



filteredData26 = uv2013.Date == "26/09/2013"

print ((uv2013[filteredData26])['UV_Index'].max())



filteredData27 = uv2013.Date == "27/09/2013"

print ((uv2013[filteredData27])['UV_Index'].max())



filteredData28 = uv2013.Date == "28/09/2013"

print ((uv2013[filteredData28])['UV_Index'].max())



filteredData29 = uv2013.Date == "29/09/2013"

print ((uv2013[filteredData29])['UV_Index'].max())



filteredData30 = uv2013.Date == "30/09/2013"

print ((uv2013[filteredData30])['UV_Index'].max())



filteredData31 = uv2013.Date == "31/09/2013"

print ((uv2013[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2012 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2012.csv')



filteredData1 = uv2012.Date == "01/09/2012"

print((uv2012[filteredData1])['UV_Index'].max())



filteredData2 = uv2012.Date == "02/09/2012"

print ((uv2012[filteredData2])['UV_Index'].max())



filteredData3 = uv2012.Date == "03/09/2012"

print ((uv2012[filteredData3])['UV_Index'].max())



filteredData4 = uv2012.Date == "04/09/2012"

print ((uv2012[filteredData4])['UV_Index'].max())



filteredData5 = uv2012.Date == "05/09/2012"

print ((uv2012[filteredData5])['UV_Index'].max())



filteredData6 = uv2012.Date == "06/09/2012"

print ((uv2012[filteredData6])['UV_Index'].max())



filteredData7 = uv2012.Date == "07/09/2012"

print ((uv2012[filteredData7])['UV_Index'].max())



filteredData8 = uv2012.Date == "08/09/2012"

print ((uv2012[filteredData8])['UV_Index'].max())



filteredData9 = uv2012.Date == "09/09/2012"

print ((uv2012[filteredData9])['UV_Index'].max())



filteredData10 = uv2012.Date == "10/09/2012"

print ((uv2012[filteredData10])['UV_Index'].max())



filteredData11 = uv2012.Date == "11/09/2012"

print ((uv2012[filteredData11])['UV_Index'].max())



filteredData12 = uv2012.Date == "12/09/2012"

print ((uv2012[filteredData12])['UV_Index'].max())



filteredData13 = uv2012.Date == "13/09/2012"

print ((uv2012[filteredData13])['UV_Index'].max())



filteredData14 = uv2012.Date == "14/09/2012"

print ((uv2012[filteredData14])['UV_Index'].max())



filteredData15 = uv2012.Date == "15/09/2012"

print ((uv2012[filteredData15])['UV_Index'].max())



filteredData16 = uv2012.Date == "16/09/2012"

print ((uv2012[filteredData16])['UV_Index'].max())



filteredData17 = uv2012.Date == "17/09/2012"

print ((uv2012[filteredData17])['UV_Index'].max())



filteredData18 = uv2012.Date == "18/09/2012"

print ((uv2012[filteredData18])['UV_Index'].max())



filteredData19 = uv2012.Date == "19/09/2012"

print ((uv2012[filteredData19])['UV_Index'].max())



filteredData20 = uv2012.Date == "20/09/2099"

print ((uv2012[filteredData20])['UV_Index'].max())



filteredData21 = uv2012.Date == "21/09/2012"

print ((uv2012[filteredData21])['UV_Index'].max())



filteredData22 = uv2012.Date == "22/09/2012"

print ((uv2012[filteredData22])['UV_Index'].max())



filteredData23 = uv2012.Date == "23/09/2012"

print ((uv2012[filteredData23])['UV_Index'].max())



filteredData24 = uv2012.Date == "24/09/2012"

print ((uv2012[filteredData24])['UV_Index'].max())



filteredData25 = uv2012.Date == "25/09/2012"

print ((uv2012[filteredData25])['UV_Index'].max())



filteredData26 = uv2012.Date == "26/09/2012"

print ((uv2012[filteredData26])['UV_Index'].max())



filteredData27 = uv2012.Date == "27/09/2012"

print ((uv2012[filteredData27])['UV_Index'].max())



filteredData28 = uv2012.Date == "28/09/2012"

print ((uv2012[filteredData28])['UV_Index'].max())



filteredData29 = uv2012.Date == "29/09/2012"

print ((uv2012[filteredData29])['UV_Index'].max())



filteredData30 = uv2012.Date == "30/09/2012"

print ((uv2012[filteredData30])['UV_Index'].max())



filteredData31 = uv2012.Date == "31/09/2012"

print ((uv2012[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2011 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2011.csv')



filteredData1 = uv2011.Date == "01/09/2011"

print((uv2011[filteredData1])['UV_Index'].max())



filteredData2 = uv2011.Date == "02/09/2011"

print ((uv2011[filteredData2])['UV_Index'].max())



filteredData3 = uv2011.Date == "03/09/2011"

print ((uv2011[filteredData3])['UV_Index'].max())



filteredData4 = uv2011.Date == "04/09/2011"

print ((uv2011[filteredData4])['UV_Index'].max())



filteredData5 = uv2011.Date == "05/09/2011"

print ((uv2011[filteredData5])['UV_Index'].max())



filteredData6 = uv2011.Date == "06/09/2011"

print ((uv2011[filteredData6])['UV_Index'].max())



filteredData7 = uv2011.Date == "07/09/2011"

print ((uv2011[filteredData7])['UV_Index'].max())



filteredData8 = uv2011.Date == "08/09/2011"

print ((uv2011[filteredData8])['UV_Index'].max())



filteredData9 = uv2011.Date == "09/09/2011"

print ((uv2011[filteredData9])['UV_Index'].max())



filteredData10 = uv2011.Date == "10/09/2011"

print ((uv2011[filteredData10])['UV_Index'].max())



filteredData11 = uv2011.Date == "11/09/2011"

print ((uv2011[filteredData11])['UV_Index'].max())



filteredData12 = uv2011.Date == "12/09/2011"

print ((uv2011[filteredData12])['UV_Index'].max())



filteredData13 = uv2011.Date == "13/09/2011"

print ((uv2011[filteredData13])['UV_Index'].max())



filteredData14 = uv2011.Date == "14/09/2011"

print ((uv2011[filteredData14])['UV_Index'].max())



filteredData15 = uv2011.Date == "15/09/2011"

print ((uv2011[filteredData15])['UV_Index'].max())



filteredData16 = uv2011.Date == "16/09/2011"

print ((uv2011[filteredData16])['UV_Index'].max())



filteredData17 = uv2011.Date == "17/09/2011"

print ((uv2011[filteredData17])['UV_Index'].max())



filteredData18 = uv2011.Date == "18/09/2011"

print ((uv2011[filteredData18])['UV_Index'].max())



filteredData19 = uv2011.Date == "19/09/2011"

print ((uv2011[filteredData19])['UV_Index'].max())



filteredData20 = uv2011.Date == "20/09/2099"

print ((uv2011[filteredData20])['UV_Index'].max())



filteredData21 = uv2011.Date == "21/09/2011"

print ((uv2011[filteredData21])['UV_Index'].max())



filteredData22 = uv2011.Date == "22/09/2011"

print ((uv2011[filteredData22])['UV_Index'].max())



filteredData23 = uv2011.Date == "23/09/2011"

print ((uv2011[filteredData23])['UV_Index'].max())



filteredData24 = uv2011.Date == "24/09/2011"

print ((uv2011[filteredData24])['UV_Index'].max())



filteredData25 = uv2011.Date == "25/09/2011"

print ((uv2011[filteredData25])['UV_Index'].max())



filteredData26 = uv2011.Date == "26/09/2011"

print ((uv2011[filteredData26])['UV_Index'].max())



filteredData27 = uv2011.Date == "27/09/2011"

print ((uv2011[filteredData27])['UV_Index'].max())



filteredData28 = uv2011.Date == "28/09/2011"

print ((uv2011[filteredData28])['UV_Index'].max())



filteredData29 = uv2011.Date == "29/09/2011"

print ((uv2011[filteredData29])['UV_Index'].max())



filteredData30 = uv2011.Date == "30/09/2011"

print ((uv2011[filteredData30])['UV_Index'].max())



filteredData31 = uv2011.Date == "31/09/2011"

print ((uv2011[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2010 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2010.csv')



filteredData1 = uv2010.Date == "01/09/2010"

print((uv2010[filteredData1])['UV_Index'].max())



filteredData2 = uv2010.Date == "02/09/2010"

print ((uv2010[filteredData2])['UV_Index'].max())



filteredData3 = uv2010.Date == "03/09/2010"

print ((uv2010[filteredData3])['UV_Index'].max())



filteredData4 = uv2010.Date == "04/09/2010"

print ((uv2010[filteredData4])['UV_Index'].max())



filteredData5 = uv2010.Date == "05/09/2010"

print ((uv2010[filteredData5])['UV_Index'].max())



filteredData6 = uv2010.Date == "06/09/2010"

print ((uv2010[filteredData6])['UV_Index'].max())



filteredData7 = uv2010.Date == "07/09/2010"

print ((uv2010[filteredData7])['UV_Index'].max())



filteredData8 = uv2010.Date == "08/09/2010"

print ((uv2010[filteredData8])['UV_Index'].max())



filteredData9 = uv2010.Date == "09/09/2010"

print ((uv2010[filteredData9])['UV_Index'].max())



filteredData10 = uv2010.Date == "10/09/2010"

print ((uv2010[filteredData10])['UV_Index'].max())



filteredData11 = uv2010.Date == "11/09/2010"

print ((uv2010[filteredData11])['UV_Index'].max())



filteredData12 = uv2010.Date == "12/09/2010"

print ((uv2010[filteredData12])['UV_Index'].max())



filteredData13 = uv2010.Date == "13/09/2010"

print ((uv2010[filteredData13])['UV_Index'].max())



filteredData14 = uv2010.Date == "14/09/2010"

print ((uv2010[filteredData14])['UV_Index'].max())



filteredData15 = uv2010.Date == "15/09/2010"

print ((uv2010[filteredData15])['UV_Index'].max())



filteredData16 = uv2010.Date == "16/09/2010"

print ((uv2010[filteredData16])['UV_Index'].max())



filteredData17 = uv2010.Date == "17/09/2010"

print ((uv2010[filteredData17])['UV_Index'].max())



filteredData18 = uv2010.Date == "18/09/2010"

print ((uv2010[filteredData18])['UV_Index'].max())



filteredData19 = uv2010.Date == "19/09/2010"

print ((uv2010[filteredData19])['UV_Index'].max())



filteredData20 = uv2010.Date == "20/09/2099"

print ((uv2010[filteredData20])['UV_Index'].max())



filteredData21 = uv2010.Date == "21/09/2010"

print ((uv2010[filteredData21])['UV_Index'].max())



filteredData22 = uv2010.Date == "22/09/2010"

print ((uv2010[filteredData22])['UV_Index'].max())



filteredData23 = uv2010.Date == "23/09/2010"

print ((uv2010[filteredData23])['UV_Index'].max())



filteredData24 = uv2010.Date == "24/09/2010"

print ((uv2010[filteredData24])['UV_Index'].max())



filteredData25 = uv2010.Date == "25/09/2010"

print ((uv2010[filteredData25])['UV_Index'].max())



filteredData26 = uv2010.Date == "26/09/2010"

print ((uv2010[filteredData26])['UV_Index'].max())



filteredData27 = uv2010.Date == "27/09/2010"

print ((uv2010[filteredData27])['UV_Index'].max())



filteredData28 = uv2010.Date == "28/09/2010"

print ((uv2010[filteredData28])['UV_Index'].max())



filteredData29 = uv2010.Date == "29/09/2010"

print ((uv2010[filteredData29])['UV_Index'].max())



filteredData30 = uv2010.Date == "30/09/2010"

print ((uv2010[filteredData30])['UV_Index'].max())



filteredData31 = uv2010.Date == "31/09/2010"

print ((uv2010[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2010 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2010.csv')



filteredData1 = uv2010.Date == "01/10/2010"

print((uv2010[filteredData1])['UV_Index'].max())



filteredData2 = uv2010.Date == "02/10/2010"

print ((uv2010[filteredData2])['UV_Index'].max())



filteredData3 = uv2010.Date == "03/10/2010"

print ((uv2010[filteredData3])['UV_Index'].max())



filteredData4 = uv2010.Date == "04/10/2010"

print ((uv2010[filteredData4])['UV_Index'].max())



filteredData5 = uv2010.Date == "05/10/2010"

print ((uv2010[filteredData5])['UV_Index'].max())



filteredData6 = uv2010.Date == "06/10/2010"

print ((uv2010[filteredData6])['UV_Index'].max())



filteredData7 = uv2010.Date == "07/10/2010"

print ((uv2010[filteredData7])['UV_Index'].max())



filteredData8 = uv2010.Date == "08/10/2010"

print ((uv2010[filteredData8])['UV_Index'].max())



filteredData9 = uv2010.Date == "09/10/2010"

print ((uv2010[filteredData9])['UV_Index'].max())



filteredData10 = uv2010.Date == "10/10/2010"

print ((uv2010[filteredData10])['UV_Index'].max())



filteredData11 = uv2010.Date == "11/10/2010"

print ((uv2010[filteredData11])['UV_Index'].max())



filteredData12 = uv2010.Date == "12/10/2010"

print ((uv2010[filteredData12])['UV_Index'].max())



filteredData13 = uv2010.Date == "13/10/2010"

print ((uv2010[filteredData13])['UV_Index'].max())



filteredData14 = uv2010.Date == "14/10/2010"

print ((uv2010[filteredData14])['UV_Index'].max())



filteredData15 = uv2010.Date == "15/10/2010"

print ((uv2010[filteredData15])['UV_Index'].max())



filteredData16 = uv2010.Date == "16/10/2010"

print ((uv2010[filteredData16])['UV_Index'].max())



filteredData17 = uv2010.Date == "17/10/2010"

print ((uv2010[filteredData17])['UV_Index'].max())



filteredData18 = uv2010.Date == "18/10/2010"

print ((uv2010[filteredData18])['UV_Index'].max())



filteredData19 = uv2010.Date == "19/10/2010"

print ((uv2010[filteredData19])['UV_Index'].max())



filteredData20 = uv2010.Date == "20/10/2109"

print ((uv2010[filteredData20])['UV_Index'].max())



filteredData21 = uv2010.Date == "21/10/2010"

print ((uv2010[filteredData21])['UV_Index'].max())



filteredData22 = uv2010.Date == "22/10/2010"

print ((uv2010[filteredData22])['UV_Index'].max())



filteredData23 = uv2010.Date == "23/10/2010"

print ((uv2010[filteredData23])['UV_Index'].max())



filteredData24 = uv2010.Date == "24/10/2010"

print ((uv2010[filteredData24])['UV_Index'].max())



filteredData25 = uv2010.Date == "25/10/2010"

print ((uv2010[filteredData25])['UV_Index'].max())



filteredData26 = uv2010.Date == "26/10/2010"

print ((uv2010[filteredData26])['UV_Index'].max())



filteredData27 = uv2010.Date == "27/10/2010"

print ((uv2010[filteredData27])['UV_Index'].max())



filteredData28 = uv2010.Date == "28/10/2010"

print ((uv2010[filteredData28])['UV_Index'].max())



filteredData29 = uv2010.Date == "29/10/2010"

print ((uv2010[filteredData29])['UV_Index'].max())



filteredData30 = uv2010.Date == "30/10/2010"

print ((uv2010[filteredData30])['UV_Index'].max())



filteredData31 = uv2010.Date == "31/10/2010"

print ((uv2010[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2011 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2011.csv')



filteredData1 = uv2011.Date == "01/10/2011"

print((uv2011[filteredData1])['UV_Index'].max())



filteredData2 = uv2011.Date == "02/10/2011"

print ((uv2011[filteredData2])['UV_Index'].max())



filteredData3 = uv2011.Date == "03/10/2011"

print ((uv2011[filteredData3])['UV_Index'].max())



filteredData4 = uv2011.Date == "04/10/2011"

print ((uv2011[filteredData4])['UV_Index'].max())



filteredData5 = uv2011.Date == "05/10/2011"

print ((uv2011[filteredData5])['UV_Index'].max())



filteredData6 = uv2011.Date == "06/10/2011"

print ((uv2011[filteredData6])['UV_Index'].max())



filteredData7 = uv2011.Date == "07/10/2011"

print ((uv2011[filteredData7])['UV_Index'].max())



filteredData8 = uv2011.Date == "08/10/2011"

print ((uv2011[filteredData8])['UV_Index'].max())



filteredData9 = uv2011.Date == "09/10/2011"

print ((uv2011[filteredData9])['UV_Index'].max())



filteredData10 = uv2011.Date == "10/10/2011"

print ((uv2011[filteredData10])['UV_Index'].max())



filteredData11 = uv2011.Date == "11/10/2011"

print ((uv2011[filteredData11])['UV_Index'].max())



filteredData12 = uv2011.Date == "12/10/2011"

print ((uv2011[filteredData12])['UV_Index'].max())



filteredData13 = uv2011.Date == "13/10/2011"

print ((uv2011[filteredData13])['UV_Index'].max())



filteredData14 = uv2011.Date == "14/10/2011"

print ((uv2011[filteredData14])['UV_Index'].max())



filteredData15 = uv2011.Date == "15/10/2011"

print ((uv2011[filteredData15])['UV_Index'].max())



filteredData16 = uv2011.Date == "16/10/2011"

print ((uv2011[filteredData16])['UV_Index'].max())



filteredData17 = uv2011.Date == "17/10/2011"

print ((uv2011[filteredData17])['UV_Index'].max())



filteredData18 = uv2011.Date == "18/10/2011"

print ((uv2011[filteredData18])['UV_Index'].max())



filteredData19 = uv2011.Date == "19/10/2011"

print ((uv2011[filteredData19])['UV_Index'].max())



filteredData20 = uv2011.Date == "20/10/2109"

print ((uv2011[filteredData20])['UV_Index'].max())



filteredData21 = uv2011.Date == "21/10/2011"

print ((uv2011[filteredData21])['UV_Index'].max())



filteredData22 = uv2011.Date == "22/10/2011"

print ((uv2011[filteredData22])['UV_Index'].max())



filteredData23 = uv2011.Date == "23/10/2011"

print ((uv2011[filteredData23])['UV_Index'].max())



filteredData24 = uv2011.Date == "24/10/2011"

print ((uv2011[filteredData24])['UV_Index'].max())



filteredData25 = uv2011.Date == "25/10/2011"

print ((uv2011[filteredData25])['UV_Index'].max())



filteredData26 = uv2011.Date == "26/10/2011"

print ((uv2011[filteredData26])['UV_Index'].max())



filteredData27 = uv2011.Date == "27/10/2011"

print ((uv2011[filteredData27])['UV_Index'].max())



filteredData28 = uv2011.Date == "28/10/2011"

print ((uv2011[filteredData28])['UV_Index'].max())



filteredData29 = uv2011.Date == "29/10/2011"

print ((uv2011[filteredData29])['UV_Index'].max())



filteredData30 = uv2011.Date == "30/10/2011"

print ((uv2011[filteredData30])['UV_Index'].max())



filteredData31 = uv2011.Date == "31/10/2011"

print ((uv2011[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2012 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2012.csv')



filteredData1 = uv2012.Date == "01/10/2012"

print((uv2012[filteredData1])['UV_Index'].max())



filteredData2 = uv2012.Date == "02/10/2012"

print ((uv2012[filteredData2])['UV_Index'].max())



filteredData3 = uv2012.Date == "03/10/2012"

print ((uv2012[filteredData3])['UV_Index'].max())



filteredData4 = uv2012.Date == "04/10/2012"

print ((uv2012[filteredData4])['UV_Index'].max())



filteredData5 = uv2012.Date == "05/10/2012"

print ((uv2012[filteredData5])['UV_Index'].max())



filteredData6 = uv2012.Date == "06/10/2012"

print ((uv2012[filteredData6])['UV_Index'].max())



filteredData7 = uv2012.Date == "07/10/2012"

print ((uv2012[filteredData7])['UV_Index'].max())



filteredData8 = uv2012.Date == "08/10/2012"

print ((uv2012[filteredData8])['UV_Index'].max())



filteredData9 = uv2012.Date == "09/10/2012"

print ((uv2012[filteredData9])['UV_Index'].max())



filteredData10 = uv2012.Date == "10/10/2012"

print ((uv2012[filteredData10])['UV_Index'].max())



filteredData11 = uv2012.Date == "11/10/2012"

print ((uv2012[filteredData11])['UV_Index'].max())



filteredData12 = uv2012.Date == "12/10/2012"

print ((uv2012[filteredData12])['UV_Index'].max())



filteredData13 = uv2012.Date == "13/10/2012"

print ((uv2012[filteredData13])['UV_Index'].max())



filteredData14 = uv2012.Date == "14/10/2012"

print ((uv2012[filteredData14])['UV_Index'].max())



filteredData15 = uv2012.Date == "15/10/2012"

print ((uv2012[filteredData15])['UV_Index'].max())



filteredData16 = uv2012.Date == "16/10/2012"

print ((uv2012[filteredData16])['UV_Index'].max())



filteredData17 = uv2012.Date == "17/10/2012"

print ((uv2012[filteredData17])['UV_Index'].max())



filteredData18 = uv2012.Date == "18/10/2012"

print ((uv2012[filteredData18])['UV_Index'].max())



filteredData19 = uv2012.Date == "19/10/2012"

print ((uv2012[filteredData19])['UV_Index'].max())



filteredData20 = uv2012.Date == "20/10/2109"

print ((uv2012[filteredData20])['UV_Index'].max())



filteredData21 = uv2012.Date == "21/10/2012"

print ((uv2012[filteredData21])['UV_Index'].max())



filteredData22 = uv2012.Date == "22/10/2012"

print ((uv2012[filteredData22])['UV_Index'].max())



filteredData23 = uv2012.Date == "23/10/2012"

print ((uv2012[filteredData23])['UV_Index'].max())



filteredData24 = uv2012.Date == "24/10/2012"

print ((uv2012[filteredData24])['UV_Index'].max())



filteredData25 = uv2012.Date == "25/10/2012"

print ((uv2012[filteredData25])['UV_Index'].max())



filteredData26 = uv2012.Date == "26/10/2012"

print ((uv2012[filteredData26])['UV_Index'].max())



filteredData27 = uv2012.Date == "27/10/2012"

print ((uv2012[filteredData27])['UV_Index'].max())



filteredData28 = uv2012.Date == "28/10/2012"

print ((uv2012[filteredData28])['UV_Index'].max())



filteredData29 = uv2012.Date == "29/10/2012"

print ((uv2012[filteredData29])['UV_Index'].max())



filteredData30 = uv2012.Date == "30/10/2012"

print ((uv2012[filteredData30])['UV_Index'].max())



filteredData31 = uv2012.Date == "31/10/2012"

print ((uv2012[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2013 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2013.csv')



filteredData1 = uv2013.Date == "01/10/2013"

print((uv2013[filteredData1])['UV_Index'].max())



filteredData2 = uv2013.Date == "02/10/2013"

print ((uv2013[filteredData2])['UV_Index'].max())



filteredData3 = uv2013.Date == "03/10/2013"

print ((uv2013[filteredData3])['UV_Index'].max())



filteredData4 = uv2013.Date == "04/10/2013"

print ((uv2013[filteredData4])['UV_Index'].max())



filteredData5 = uv2013.Date == "05/10/2013"

print ((uv2013[filteredData5])['UV_Index'].max())



filteredData6 = uv2013.Date == "06/10/2013"

print ((uv2013[filteredData6])['UV_Index'].max())



filteredData7 = uv2013.Date == "07/10/2013"

print ((uv2013[filteredData7])['UV_Index'].max())



filteredData8 = uv2013.Date == "08/10/2013"

print ((uv2013[filteredData8])['UV_Index'].max())



filteredData9 = uv2013.Date == "09/10/2013"

print ((uv2013[filteredData9])['UV_Index'].max())



filteredData10 = uv2013.Date == "10/10/2013"

print ((uv2013[filteredData10])['UV_Index'].max())



filteredData11 = uv2013.Date == "11/10/2013"

print ((uv2013[filteredData11])['UV_Index'].max())



filteredData12 = uv2013.Date == "12/10/2013"

print ((uv2013[filteredData12])['UV_Index'].max())



filteredData13 = uv2013.Date == "13/10/2013"

print ((uv2013[filteredData13])['UV_Index'].max())



filteredData14 = uv2013.Date == "14/10/2013"

print ((uv2013[filteredData14])['UV_Index'].max())



filteredData15 = uv2013.Date == "15/10/2013"

print ((uv2013[filteredData15])['UV_Index'].max())



filteredData16 = uv2013.Date == "16/10/2013"

print ((uv2013[filteredData16])['UV_Index'].max())



filteredData17 = uv2013.Date == "17/10/2013"

print ((uv2013[filteredData17])['UV_Index'].max())



filteredData18 = uv2013.Date == "18/10/2013"

print ((uv2013[filteredData18])['UV_Index'].max())



filteredData19 = uv2013.Date == "19/10/2013"

print ((uv2013[filteredData19])['UV_Index'].max())



filteredData20 = uv2013.Date == "20/10/2109"

print ((uv2013[filteredData20])['UV_Index'].max())



filteredData21 = uv2013.Date == "21/10/2013"

print ((uv2013[filteredData21])['UV_Index'].max())



filteredData22 = uv2013.Date == "22/10/2013"

print ((uv2013[filteredData22])['UV_Index'].max())



filteredData23 = uv2013.Date == "23/10/2013"

print ((uv2013[filteredData23])['UV_Index'].max())



filteredData24 = uv2013.Date == "24/10/2013"

print ((uv2013[filteredData24])['UV_Index'].max())



filteredData25 = uv2013.Date == "25/10/2013"

print ((uv2013[filteredData25])['UV_Index'].max())



filteredData26 = uv2013.Date == "26/10/2013"

print ((uv2013[filteredData26])['UV_Index'].max())



filteredData27 = uv2013.Date == "27/10/2013"

print ((uv2013[filteredData27])['UV_Index'].max())



filteredData28 = uv2013.Date == "28/10/2013"

print ((uv2013[filteredData28])['UV_Index'].max())



filteredData29 = uv2013.Date == "29/10/2013"

print ((uv2013[filteredData29])['UV_Index'].max())



filteredData30 = uv2013.Date == "30/10/2013"

print ((uv2013[filteredData30])['UV_Index'].max())



filteredData31 = uv2013.Date == "31/10/2013"

print ((uv2013[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2014 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2014.csv')



filteredData1 = uv2014.Date == "01/10/2014"

print((uv2014[filteredData1])['UV_Index'].max())



filteredData2 = uv2014.Date == "02/10/2014"

print ((uv2014[filteredData2])['UV_Index'].max())



filteredData3 = uv2014.Date == "03/10/2014"

print ((uv2014[filteredData3])['UV_Index'].max())



filteredData4 = uv2014.Date == "04/10/2014"

print ((uv2014[filteredData4])['UV_Index'].max())



filteredData5 = uv2014.Date == "05/10/2014"

print ((uv2014[filteredData5])['UV_Index'].max())



filteredData6 = uv2014.Date == "06/10/2014"

print ((uv2014[filteredData6])['UV_Index'].max())



filteredData7 = uv2014.Date == "07/10/2014"

print ((uv2014[filteredData7])['UV_Index'].max())



filteredData8 = uv2014.Date == "08/10/2014"

print ((uv2014[filteredData8])['UV_Index'].max())



filteredData9 = uv2014.Date == "09/10/2014"

print ((uv2014[filteredData9])['UV_Index'].max())



filteredData10 = uv2014.Date == "10/10/2014"

print ((uv2014[filteredData10])['UV_Index'].max())



filteredData11 = uv2014.Date == "11/10/2014"

print ((uv2014[filteredData11])['UV_Index'].max())



filteredData12 = uv2014.Date == "12/10/2014"

print ((uv2014[filteredData12])['UV_Index'].max())



filteredData13 = uv2014.Date == "13/10/2014"

print ((uv2014[filteredData13])['UV_Index'].max())



filteredData14 = uv2014.Date == "14/10/2014"

print ((uv2014[filteredData14])['UV_Index'].max())



filteredData15 = uv2014.Date == "15/10/2014"

print ((uv2014[filteredData15])['UV_Index'].max())



filteredData16 = uv2014.Date == "16/10/2014"

print ((uv2014[filteredData16])['UV_Index'].max())



filteredData17 = uv2014.Date == "17/10/2014"

print ((uv2014[filteredData17])['UV_Index'].max())



filteredData18 = uv2014.Date == "18/10/2014"

print ((uv2014[filteredData18])['UV_Index'].max())



filteredData19 = uv2014.Date == "19/10/2014"

print ((uv2014[filteredData19])['UV_Index'].max())



filteredData20 = uv2014.Date == "20/10/2109"

print ((uv2014[filteredData20])['UV_Index'].max())



filteredData21 = uv2014.Date == "21/10/2014"

print ((uv2014[filteredData21])['UV_Index'].max())



filteredData22 = uv2014.Date == "22/10/2014"

print ((uv2014[filteredData22])['UV_Index'].max())



filteredData23 = uv2014.Date == "23/10/2014"

print ((uv2014[filteredData23])['UV_Index'].max())



filteredData24 = uv2014.Date == "24/10/2014"

print ((uv2014[filteredData24])['UV_Index'].max())



filteredData25 = uv2014.Date == "25/10/2014"

print ((uv2014[filteredData25])['UV_Index'].max())



filteredData26 = uv2014.Date == "26/10/2014"

print ((uv2014[filteredData26])['UV_Index'].max())



filteredData27 = uv2014.Date == "27/10/2014"

print ((uv2014[filteredData27])['UV_Index'].max())



filteredData28 = uv2014.Date == "28/10/2014"

print ((uv2014[filteredData28])['UV_Index'].max())



filteredData29 = uv2014.Date == "29/10/2014"

print ((uv2014[filteredData29])['UV_Index'].max())



filteredData30 = uv2014.Date == "30/10/2014"

print ((uv2014[filteredData30])['UV_Index'].max())



filteredData31 = uv2014.Date == "31/10/2014"

print ((uv2014[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2015 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2015.csv')



filteredData1 = uv2015.Date == "01/10/2015"

print((uv2015[filteredData1])['UV_Index'].max())



filteredData2 = uv2015.Date == "02/10/2015"

print ((uv2015[filteredData2])['UV_Index'].max())



filteredData3 = uv2015.Date == "03/10/2015"

print ((uv2015[filteredData3])['UV_Index'].max())



filteredData4 = uv2015.Date == "04/10/2015"

print ((uv2015[filteredData4])['UV_Index'].max())



filteredData5 = uv2015.Date == "05/10/2015"

print ((uv2015[filteredData5])['UV_Index'].max())



filteredData6 = uv2015.Date == "06/10/2015"

print ((uv2015[filteredData6])['UV_Index'].max())



filteredData7 = uv2015.Date == "07/10/2015"

print ((uv2015[filteredData7])['UV_Index'].max())



filteredData8 = uv2015.Date == "08/10/2015"

print ((uv2015[filteredData8])['UV_Index'].max())



filteredData9 = uv2015.Date == "09/10/2015"

print ((uv2015[filteredData9])['UV_Index'].max())



filteredData10 = uv2015.Date == "10/10/2015"

print ((uv2015[filteredData10])['UV_Index'].max())



filteredData11 = uv2015.Date == "11/10/2015"

print ((uv2015[filteredData11])['UV_Index'].max())



filteredData12 = uv2015.Date == "12/10/2015"

print ((uv2015[filteredData12])['UV_Index'].max())



filteredData13 = uv2015.Date == "13/10/2015"

print ((uv2015[filteredData13])['UV_Index'].max())



filteredData14 = uv2015.Date == "14/10/2015"

print ((uv2015[filteredData14])['UV_Index'].max())



filteredData15 = uv2015.Date == "15/10/2015"

print ((uv2015[filteredData15])['UV_Index'].max())



filteredData16 = uv2015.Date == "16/10/2015"

print ((uv2015[filteredData16])['UV_Index'].max())



filteredData17 = uv2015.Date == "17/10/2015"

print ((uv2015[filteredData17])['UV_Index'].max())



filteredData18 = uv2015.Date == "18/10/2015"

print ((uv2015[filteredData18])['UV_Index'].max())



filteredData19 = uv2015.Date == "19/10/2015"

print ((uv2015[filteredData19])['UV_Index'].max())



filteredData20 = uv2015.Date == "20/10/2109"

print ((uv2015[filteredData20])['UV_Index'].max())



filteredData21 = uv2015.Date == "21/10/2015"

print ((uv2015[filteredData21])['UV_Index'].max())



filteredData22 = uv2015.Date == "22/10/2015"

print ((uv2015[filteredData22])['UV_Index'].max())



filteredData23 = uv2015.Date == "23/10/2015"

print ((uv2015[filteredData23])['UV_Index'].max())



filteredData24 = uv2015.Date == "24/10/2015"

print ((uv2015[filteredData24])['UV_Index'].max())



filteredData25 = uv2015.Date == "25/10/2015"

print ((uv2015[filteredData25])['UV_Index'].max())



filteredData26 = uv2015.Date == "26/10/2015"

print ((uv2015[filteredData26])['UV_Index'].max())



filteredData27 = uv2015.Date == "27/10/2015"

print ((uv2015[filteredData27])['UV_Index'].max())



filteredData28 = uv2015.Date == "28/10/2015"

print ((uv2015[filteredData28])['UV_Index'].max())



filteredData29 = uv2015.Date == "29/10/2015"

print ((uv2015[filteredData29])['UV_Index'].max())



filteredData30 = uv2015.Date == "30/10/2015"

print ((uv2015[filteredData30])['UV_Index'].max())



filteredData31 = uv2015.Date == "31/10/2015"

print ((uv2015[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2016 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2016.csv')



filteredData1 = uv2016.Date == "01/10/2016"

print((uv2016[filteredData1])['UV_Index'].max())



filteredData2 = uv2016.Date == "02/10/2016"

print ((uv2016[filteredData2])['UV_Index'].max())



filteredData3 = uv2016.Date == "03/10/2016"

print ((uv2016[filteredData3])['UV_Index'].max())



filteredData4 = uv2016.Date == "04/10/2016"

print ((uv2016[filteredData4])['UV_Index'].max())



filteredData5 = uv2016.Date == "05/10/2016"

print ((uv2016[filteredData5])['UV_Index'].max())



filteredData6 = uv2016.Date == "06/10/2016"

print ((uv2016[filteredData6])['UV_Index'].max())



filteredData7 = uv2016.Date == "07/10/2016"

print ((uv2016[filteredData7])['UV_Index'].max())



filteredData8 = uv2016.Date == "08/10/2016"

print ((uv2016[filteredData8])['UV_Index'].max())



filteredData9 = uv2016.Date == "09/10/2016"

print ((uv2016[filteredData9])['UV_Index'].max())



filteredData10 = uv2016.Date == "10/10/2016"

print ((uv2016[filteredData10])['UV_Index'].max())



filteredData11 = uv2016.Date == "11/10/2016"

print ((uv2016[filteredData11])['UV_Index'].max())



filteredData12 = uv2016.Date == "12/10/2016"

print ((uv2016[filteredData12])['UV_Index'].max())



filteredData13 = uv2016.Date == "13/10/2016"

print ((uv2016[filteredData13])['UV_Index'].max())



filteredData14 = uv2016.Date == "14/10/2016"

print ((uv2016[filteredData14])['UV_Index'].max())



filteredData15 = uv2016.Date == "15/10/2016"

print ((uv2016[filteredData15])['UV_Index'].max())



filteredData16 = uv2016.Date == "16/10/2016"

print ((uv2016[filteredData16])['UV_Index'].max())



filteredData17 = uv2016.Date == "17/10/2016"

print ((uv2016[filteredData17])['UV_Index'].max())



filteredData18 = uv2016.Date == "18/10/2016"

print ((uv2016[filteredData18])['UV_Index'].max())



filteredData19 = uv2016.Date == "19/10/2016"

print ((uv2016[filteredData19])['UV_Index'].max())



filteredData20 = uv2016.Date == "20/10/2109"

print ((uv2016[filteredData20])['UV_Index'].max())



filteredData21 = uv2016.Date == "21/10/2016"

print ((uv2016[filteredData21])['UV_Index'].max())



filteredData22 = uv2016.Date == "22/10/2016"

print ((uv2016[filteredData22])['UV_Index'].max())



filteredData23 = uv2016.Date == "23/10/2016"

print ((uv2016[filteredData23])['UV_Index'].max())



filteredData24 = uv2016.Date == "24/10/2016"

print ((uv2016[filteredData24])['UV_Index'].max())



filteredData25 = uv2016.Date == "25/10/2016"

print ((uv2016[filteredData25])['UV_Index'].max())



filteredData26 = uv2016.Date == "26/10/2016"

print ((uv2016[filteredData26])['UV_Index'].max())



filteredData27 = uv2016.Date == "27/10/2016"

print ((uv2016[filteredData27])['UV_Index'].max())



filteredData28 = uv2016.Date == "28/10/2016"

print ((uv2016[filteredData28])['UV_Index'].max())



filteredData29 = uv2016.Date == "29/10/2016"

print ((uv2016[filteredData29])['UV_Index'].max())



filteredData30 = uv2016.Date == "30/10/2016"

print ((uv2016[filteredData30])['UV_Index'].max())



filteredData31 = uv2016.Date == "31/10/2016"

print ((uv2016[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2017 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2017.csv')



filteredData1 = uv2017.Date == "01/10/2017"

print((uv2017[filteredData1])['UV_Index'].max())



filteredData2 = uv2017.Date == "02/10/2017"

print ((uv2017[filteredData2])['UV_Index'].max())



filteredData3 = uv2017.Date == "03/10/2017"

print ((uv2017[filteredData3])['UV_Index'].max())



filteredData4 = uv2017.Date == "04/10/2017"

print ((uv2017[filteredData4])['UV_Index'].max())



filteredData5 = uv2017.Date == "05/10/2017"

print ((uv2017[filteredData5])['UV_Index'].max())



filteredData6 = uv2017.Date == "06/10/2017"

print ((uv2017[filteredData6])['UV_Index'].max())



filteredData7 = uv2017.Date == "07/10/2017"

print ((uv2017[filteredData7])['UV_Index'].max())



filteredData8 = uv2017.Date == "08/10/2017"

print ((uv2017[filteredData8])['UV_Index'].max())



filteredData9 = uv2017.Date == "09/10/2017"

print ((uv2017[filteredData9])['UV_Index'].max())



filteredData10 = uv2017.Date == "10/10/2017"

print ((uv2017[filteredData10])['UV_Index'].max())



filteredData11 = uv2017.Date == "11/10/2017"

print ((uv2017[filteredData11])['UV_Index'].max())



filteredData12 = uv2017.Date == "12/10/2017"

print ((uv2017[filteredData12])['UV_Index'].max())



filteredData13 = uv2017.Date == "13/10/2017"

print ((uv2017[filteredData13])['UV_Index'].max())



filteredData14 = uv2017.Date == "14/10/2017"

print ((uv2017[filteredData14])['UV_Index'].max())



filteredData15 = uv2017.Date == "15/10/2017"

print ((uv2017[filteredData15])['UV_Index'].max())



filteredData16 = uv2017.Date == "16/10/2017"

print ((uv2017[filteredData16])['UV_Index'].max())



filteredData17 = uv2017.Date == "17/10/2017"

print ((uv2017[filteredData17])['UV_Index'].max())



filteredData18 = uv2017.Date == "18/10/2017"

print ((uv2017[filteredData18])['UV_Index'].max())



filteredData19 = uv2017.Date == "19/10/2017"

print ((uv2017[filteredData19])['UV_Index'].max())



filteredData20 = uv2017.Date == "20/10/2109"

print ((uv2017[filteredData20])['UV_Index'].max())



filteredData21 = uv2017.Date == "21/10/2017"

print ((uv2017[filteredData21])['UV_Index'].max())



filteredData22 = uv2017.Date == "22/10/2017"

print ((uv2017[filteredData22])['UV_Index'].max())



filteredData23 = uv2017.Date == "23/10/2017"

print ((uv2017[filteredData23])['UV_Index'].max())



filteredData24 = uv2017.Date == "24/10/2017"

print ((uv2017[filteredData24])['UV_Index'].max())



filteredData25 = uv2017.Date == "25/10/2017"

print ((uv2017[filteredData25])['UV_Index'].max())



filteredData26 = uv2017.Date == "26/10/2017"

print ((uv2017[filteredData26])['UV_Index'].max())



filteredData27 = uv2017.Date == "27/10/2017"

print ((uv2017[filteredData27])['UV_Index'].max())



filteredData28 = uv2017.Date == "28/10/2017"

print ((uv2017[filteredData28])['UV_Index'].max())



filteredData29 = uv2017.Date == "29/10/2017"

print ((uv2017[filteredData29])['UV_Index'].max())



filteredData30 = uv2017.Date == "30/10/2017"

print ((uv2017[filteredData30])['UV_Index'].max())



filteredData31 = uv2017.Date == "31/10/2017"

print ((uv2017[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2018 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2018.csv')





filteredData1 = uv2018.Date == "01/10/2018"

print((uv2018[filteredData1])['UV_Index'].max())



filteredData2 = uv2018.Date == "02/10/2018"

print ((uv2018[filteredData2])['UV_Index'].max())



filteredData3 = uv2018.Date == "03/10/2018"

print ((uv2018[filteredData3])['UV_Index'].max())



filteredData4 = uv2018.Date == "04/10/2018"

print ((uv2018[filteredData4])['UV_Index'].max())



filteredData5 = uv2018.Date == "05/10/2018"

print ((uv2018[filteredData5])['UV_Index'].max())



filteredData6 = uv2018.Date == "06/10/2018"

print ((uv2018[filteredData6])['UV_Index'].max())



filteredData7 = uv2018.Date == "07/10/2018"

print ((uv2018[filteredData7])['UV_Index'].max())



filteredData8 = uv2018.Date == "08/10/2018"

print ((uv2018[filteredData8])['UV_Index'].max())



filteredData9 = uv2018.Date == "09/10/2018"

print ((uv2018[filteredData9])['UV_Index'].max())



filteredData10 = uv2018.Date == "10/10/2018"

print ((uv2018[filteredData10])['UV_Index'].max())



filteredData11 = uv2018.Date == "11/10/2018"

print ((uv2018[filteredData11])['UV_Index'].max())



filteredData12 = uv2018.Date == "12/10/2018"

print ((uv2018[filteredData12])['UV_Index'].max())



filteredData13 = uv2018.Date == "13/10/2018"

print ((uv2018[filteredData13])['UV_Index'].max())



filteredData14 = uv2018.Date == "14/10/2018"

print ((uv2018[filteredData14])['UV_Index'].max())



filteredData15 = uv2018.Date == "15/10/2018"

print ((uv2018[filteredData15])['UV_Index'].max())



filteredData16 = uv2018.Date == "16/10/2018"

print ((uv2018[filteredData16])['UV_Index'].max())



filteredData17 = uv2018.Date == "17/10/2018"

print ((uv2018[filteredData17])['UV_Index'].max())



filteredData18 = uv2018.Date == "18/10/2018"

print ((uv2018[filteredData18])['UV_Index'].max())



filteredData19 = uv2018.Date == "19/10/2018"

print ((uv2018[filteredData19])['UV_Index'].max())



filteredData20 = uv2018.Date == "20/10/2109"

print ((uv2018[filteredData20])['UV_Index'].max())



filteredData21 = uv2018.Date == "21/10/2018"

print ((uv2018[filteredData21])['UV_Index'].max())



filteredData22 = uv2018.Date == "22/10/2018"

print ((uv2018[filteredData22])['UV_Index'].max())



filteredData23 = uv2018.Date == "23/10/2018"

print ((uv2018[filteredData23])['UV_Index'].max())



filteredData24 = uv2018.Date == "24/10/2018"

print ((uv2018[filteredData24])['UV_Index'].max())



filteredData25 = uv2018.Date == "25/10/2018"

print ((uv2018[filteredData25])['UV_Index'].max())



filteredData26 = uv2018.Date == "26/10/2018"

print ((uv2018[filteredData26])['UV_Index'].max())



filteredData27 = uv2018.Date == "27/10/2018"

print ((uv2018[filteredData27])['UV_Index'].max())



filteredData28 = uv2018.Date == "28/10/2018"

print ((uv2018[filteredData28])['UV_Index'].max())



filteredData29 = uv2018.Date == "29/10/2018"

print ((uv2018[filteredData29])['UV_Index'].max())



filteredData30 = uv2018.Date == "30/10/2018"

print ((uv2018[filteredData30])['UV_Index'].max())



filteredData31 = uv2018.Date == "31/10/2018"

print ((uv2018[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2019 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2019.csv')



filteredData1 = uv2019.Date == "01/10/2019"

print((uv2019[filteredData1])['UV_Index'].max())



filteredData2 = uv2019.Date == "02/10/2019"

print ((uv2019[filteredData2])['UV_Index'].max())



filteredData3 = uv2019.Date == "03/10/2019"

print ((uv2019[filteredData3])['UV_Index'].max())



filteredData4 = uv2019.Date == "04/10/2019"

print ((uv2019[filteredData4])['UV_Index'].max())



filteredData5 = uv2019.Date == "05/10/2019"

print ((uv2019[filteredData5])['UV_Index'].max())



filteredData6 = uv2019.Date == "06/10/2019"

print ((uv2019[filteredData6])['UV_Index'].max())



filteredData7 = uv2019.Date == "07/10/2019"

print ((uv2019[filteredData7])['UV_Index'].max())



filteredData8 = uv2019.Date == "08/10/2019"

print ((uv2019[filteredData8])['UV_Index'].max())



filteredData9 = uv2019.Date == "09/10/2019"

print ((uv2019[filteredData9])['UV_Index'].max())



filteredData10 = uv2019.Date == "10/10/2019"

print ((uv2019[filteredData10])['UV_Index'].max())



filteredData11 = uv2019.Date == "11/10/2019"

print ((uv2019[filteredData11])['UV_Index'].max())



filteredData12 = uv2019.Date == "12/10/2019"

print ((uv2019[filteredData12])['UV_Index'].max())



filteredData13 = uv2019.Date == "13/10/2019"

print ((uv2019[filteredData13])['UV_Index'].max())



filteredData14 = uv2019.Date == "14/10/2019"

print ((uv2019[filteredData14])['UV_Index'].max())



filteredData15 = uv2019.Date == "15/10/2019"

print ((uv2019[filteredData15])['UV_Index'].max())



filteredData16 = uv2019.Date == "16/10/2019"

print ((uv2019[filteredData16])['UV_Index'].max())



filteredData17 = uv2019.Date == "17/10/2019"

print ((uv2019[filteredData17])['UV_Index'].max())



filteredData18 = uv2019.Date == "18/10/2019"

print ((uv2019[filteredData18])['UV_Index'].max())



filteredData19 = uv2019.Date == "19/10/2019"

print ((uv2019[filteredData19])['UV_Index'].max())



filteredData20 = uv2019.Date == "20/10/2109"

print ((uv2019[filteredData20])['UV_Index'].max())



filteredData21 = uv2019.Date == "21/10/2019"

print ((uv2019[filteredData21])['UV_Index'].max())



filteredData22 = uv2019.Date == "22/10/2019"

print ((uv2019[filteredData22])['UV_Index'].max())



filteredData23 = uv2019.Date == "23/10/2019"

print ((uv2019[filteredData23])['UV_Index'].max())



filteredData24 = uv2019.Date == "24/10/2019"

print ((uv2019[filteredData24])['UV_Index'].max())



filteredData25 = uv2019.Date == "25/10/2019"

print ((uv2019[filteredData25])['UV_Index'].max())



filteredData26 = uv2019.Date == "26/10/2019"

print ((uv2019[filteredData26])['UV_Index'].max())



filteredData27 = uv2019.Date == "27/10/2019"

print ((uv2019[filteredData27])['UV_Index'].max())



filteredData28 = uv2019.Date == "28/10/2019"

print ((uv2019[filteredData28])['UV_Index'].max())



filteredData29 = uv2019.Date == "29/10/2019"

print ((uv2019[filteredData29])['UV_Index'].max())



filteredData30 = uv2019.Date == "30/10/2019"

print ((uv2019[filteredData30])['UV_Index'].max())



filteredData31 = uv2019.Date == "31/10/2019"

print ((uv2019[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2019 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2019.csv')



filteredData1 = uv2019.Date == "01/11/2019"

print((uv2019[filteredData1])['UV_Index'].max())



filteredData2 = uv2019.Date == "02/11/2019"

print ((uv2019[filteredData2])['UV_Index'].max())



filteredData3 = uv2019.Date == "03/11/2019"

print ((uv2019[filteredData3])['UV_Index'].max())



filteredData4 = uv2019.Date == "04/11/2019"

print ((uv2019[filteredData4])['UV_Index'].max())



filteredData5 = uv2019.Date == "05/11/2019"

print ((uv2019[filteredData5])['UV_Index'].max())



filteredData6 = uv2019.Date == "06/11/2019"

print ((uv2019[filteredData6])['UV_Index'].max())



filteredData7 = uv2019.Date == "07/11/2019"

print ((uv2019[filteredData7])['UV_Index'].max())



filteredData8 = uv2019.Date == "08/11/2019"

print ((uv2019[filteredData8])['UV_Index'].max())



filteredData9 = uv2019.Date == "09/11/2019"

print ((uv2019[filteredData9])['UV_Index'].max())



filteredData10 = uv2019.Date == "10/11/2019"

print ((uv2019[filteredData10])['UV_Index'].max())



filteredData11 = uv2019.Date == "11/11/2019"

print ((uv2019[filteredData11])['UV_Index'].max())



filteredData12 = uv2019.Date == "12/11/2019"

print ((uv2019[filteredData12])['UV_Index'].max())



filteredData13 = uv2019.Date == "13/11/2019"

print ((uv2019[filteredData13])['UV_Index'].max())



filteredData14 = uv2019.Date == "14/11/2019"

print ((uv2019[filteredData14])['UV_Index'].max())



filteredData15 = uv2019.Date == "15/11/2019"

print ((uv2019[filteredData15])['UV_Index'].max())



filteredData16 = uv2019.Date == "16/11/2019"

print ((uv2019[filteredData16])['UV_Index'].max())



filteredData17 = uv2019.Date == "17/11/2019"

print ((uv2019[filteredData17])['UV_Index'].max())



filteredData18 = uv2019.Date == "18/11/2019"

print ((uv2019[filteredData18])['UV_Index'].max())



filteredData19 = uv2019.Date == "19/11/2019"

print ((uv2019[filteredData19])['UV_Index'].max())



filteredData20 = uv2019.Date == "20/11/2119"

print ((uv2019[filteredData20])['UV_Index'].max())



filteredData21 = uv2019.Date == "21/11/2019"

print ((uv2019[filteredData21])['UV_Index'].max())



filteredData22 = uv2019.Date == "22/11/2019"

print ((uv2019[filteredData22])['UV_Index'].max())



filteredData23 = uv2019.Date == "23/11/2019"

print ((uv2019[filteredData23])['UV_Index'].max())



filteredData24 = uv2019.Date == "24/11/2019"

print ((uv2019[filteredData24])['UV_Index'].max())



filteredData25 = uv2019.Date == "25/11/2019"

print ((uv2019[filteredData25])['UV_Index'].max())



filteredData26 = uv2019.Date == "26/11/2019"

print ((uv2019[filteredData26])['UV_Index'].max())



filteredData27 = uv2019.Date == "27/11/2019"

print ((uv2019[filteredData27])['UV_Index'].max())



filteredData28 = uv2019.Date == "28/11/2019"

print ((uv2019[filteredData28])['UV_Index'].max())



filteredData29 = uv2019.Date == "29/11/2019"

print ((uv2019[filteredData29])['UV_Index'].max())



filteredData30 = uv2019.Date == "30/11/2019"

print ((uv2019[filteredData30])['UV_Index'].max())



filteredData31 = uv2019.Date == "31/11/2019"

print ((uv2019[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2018 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2018.csv')



filteredData1 = uv2018.Date == "01/11/2018"

print((uv2018[filteredData1])['UV_Index'].max())



filteredData2 = uv2018.Date == "02/11/2018"

print ((uv2018[filteredData2])['UV_Index'].max())



filteredData3 = uv2018.Date == "03/11/2018"

print ((uv2018[filteredData3])['UV_Index'].max())



filteredData4 = uv2018.Date == "04/11/2018"

print ((uv2018[filteredData4])['UV_Index'].max())



filteredData5 = uv2018.Date == "05/11/2018"

print ((uv2018[filteredData5])['UV_Index'].max())



filteredData6 = uv2018.Date == "06/11/2018"

print ((uv2018[filteredData6])['UV_Index'].max())



filteredData7 = uv2018.Date == "07/11/2018"

print ((uv2018[filteredData7])['UV_Index'].max())



filteredData8 = uv2018.Date == "08/11/2018"

print ((uv2018[filteredData8])['UV_Index'].max())



filteredData9 = uv2018.Date == "09/11/2018"

print ((uv2018[filteredData9])['UV_Index'].max())



filteredData10 = uv2018.Date == "10/11/2018"

print ((uv2018[filteredData10])['UV_Index'].max())



filteredData11 = uv2018.Date == "11/11/2018"

print ((uv2018[filteredData11])['UV_Index'].max())



filteredData12 = uv2018.Date == "12/11/2018"

print ((uv2018[filteredData12])['UV_Index'].max())



filteredData13 = uv2018.Date == "13/11/2018"

print ((uv2018[filteredData13])['UV_Index'].max())



filteredData14 = uv2018.Date == "14/11/2018"

print ((uv2018[filteredData14])['UV_Index'].max())



filteredData15 = uv2018.Date == "15/11/2018"

print ((uv2018[filteredData15])['UV_Index'].max())



filteredData16 = uv2018.Date == "16/11/2018"

print ((uv2018[filteredData16])['UV_Index'].max())



filteredData17 = uv2018.Date == "17/11/2018"

print ((uv2018[filteredData17])['UV_Index'].max())



filteredData18 = uv2018.Date == "18/11/2018"

print ((uv2018[filteredData18])['UV_Index'].max())



filteredData19 = uv2018.Date == "19/11/2018"

print ((uv2018[filteredData19])['UV_Index'].max())



filteredData20 = uv2018.Date == "20/11/2119"

print ((uv2018[filteredData20])['UV_Index'].max())



filteredData21 = uv2018.Date == "21/11/2018"

print ((uv2018[filteredData21])['UV_Index'].max())



filteredData22 = uv2018.Date == "22/11/2018"

print ((uv2018[filteredData22])['UV_Index'].max())



filteredData23 = uv2018.Date == "23/11/2018"

print ((uv2018[filteredData23])['UV_Index'].max())



filteredData24 = uv2018.Date == "24/11/2018"

print ((uv2018[filteredData24])['UV_Index'].max())



filteredData25 = uv2018.Date == "25/11/2018"

print ((uv2018[filteredData25])['UV_Index'].max())



filteredData26 = uv2018.Date == "26/11/2018"

print ((uv2018[filteredData26])['UV_Index'].max())



filteredData27 = uv2018.Date == "27/11/2018"

print ((uv2018[filteredData27])['UV_Index'].max())



filteredData28 = uv2018.Date == "28/11/2018"

print ((uv2018[filteredData28])['UV_Index'].max())



filteredData29 = uv2018.Date == "29/11/2018"

print ((uv2018[filteredData29])['UV_Index'].max())



filteredData30 = uv2018.Date == "30/11/2018"

print ((uv2018[filteredData30])['UV_Index'].max())



filteredData31 = uv2018.Date == "31/11/2018"

print ((uv2018[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2017 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2017.csv')



filteredData1 = uv2017.Date == "01/11/2017"

print((uv2017[filteredData1])['UV_Index'].max())



filteredData2 = uv2017.Date == "02/11/2017"

print ((uv2017[filteredData2])['UV_Index'].max())



filteredData3 = uv2017.Date == "03/11/2017"

print ((uv2017[filteredData3])['UV_Index'].max())



filteredData4 = uv2017.Date == "04/11/2017"

print ((uv2017[filteredData4])['UV_Index'].max())



filteredData5 = uv2017.Date == "05/11/2017"

print ((uv2017[filteredData5])['UV_Index'].max())



filteredData6 = uv2017.Date == "06/11/2017"

print ((uv2017[filteredData6])['UV_Index'].max())



filteredData7 = uv2017.Date == "07/11/2017"

print ((uv2017[filteredData7])['UV_Index'].max())



filteredData8 = uv2017.Date == "08/11/2017"

print ((uv2017[filteredData8])['UV_Index'].max())



filteredData9 = uv2017.Date == "09/11/2017"

print ((uv2017[filteredData9])['UV_Index'].max())



filteredData10 = uv2017.Date == "10/11/2017"

print ((uv2017[filteredData10])['UV_Index'].max())



filteredData11 = uv2017.Date == "11/11/2017"

print ((uv2017[filteredData11])['UV_Index'].max())



filteredData12 = uv2017.Date == "12/11/2017"

print ((uv2017[filteredData12])['UV_Index'].max())



filteredData13 = uv2017.Date == "13/11/2017"

print ((uv2017[filteredData13])['UV_Index'].max())



filteredData14 = uv2017.Date == "14/11/2017"

print ((uv2017[filteredData14])['UV_Index'].max())



filteredData15 = uv2017.Date == "15/11/2017"

print ((uv2017[filteredData15])['UV_Index'].max())



filteredData16 = uv2017.Date == "16/11/2017"

print ((uv2017[filteredData16])['UV_Index'].max())



filteredData17 = uv2017.Date == "17/11/2017"

print ((uv2017[filteredData17])['UV_Index'].max())



filteredData18 = uv2017.Date == "18/11/2017"

print ((uv2017[filteredData18])['UV_Index'].max())



filteredData19 = uv2017.Date == "19/11/2017"

print ((uv2017[filteredData19])['UV_Index'].max())



filteredData20 = uv2017.Date == "20/11/2119"

print ((uv2017[filteredData20])['UV_Index'].max())



filteredData21 = uv2017.Date == "21/11/2017"

print ((uv2017[filteredData21])['UV_Index'].max())



filteredData22 = uv2017.Date == "22/11/2017"

print ((uv2017[filteredData22])['UV_Index'].max())



filteredData23 = uv2017.Date == "23/11/2017"

print ((uv2017[filteredData23])['UV_Index'].max())



filteredData24 = uv2017.Date == "24/11/2017"

print ((uv2017[filteredData24])['UV_Index'].max())



filteredData25 = uv2017.Date == "25/11/2017"

print ((uv2017[filteredData25])['UV_Index'].max())



filteredData26 = uv2017.Date == "26/11/2017"

print ((uv2017[filteredData26])['UV_Index'].max())



filteredData27 = uv2017.Date == "27/11/2017"

print ((uv2017[filteredData27])['UV_Index'].max())



filteredData28 = uv2017.Date == "28/11/2017"

print ((uv2017[filteredData28])['UV_Index'].max())



filteredData29 = uv2017.Date == "29/11/2017"

print ((uv2017[filteredData29])['UV_Index'].max())



filteredData30 = uv2017.Date == "30/11/2017"

print ((uv2017[filteredData30])['UV_Index'].max())



filteredData31 = uv2017.Date == "31/11/2017"

print ((uv2017[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2016 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2016.csv')



filteredData1 = uv2016.Date == "01/11/2016"

print((uv2016[filteredData1])['UV_Index'].max())



filteredData2 = uv2016.Date == "02/11/2016"

print ((uv2016[filteredData2])['UV_Index'].max())



filteredData3 = uv2016.Date == "03/11/2016"

print ((uv2016[filteredData3])['UV_Index'].max())



filteredData4 = uv2016.Date == "04/11/2016"

print ((uv2016[filteredData4])['UV_Index'].max())



filteredData5 = uv2016.Date == "05/11/2016"

print ((uv2016[filteredData5])['UV_Index'].max())



filteredData6 = uv2016.Date == "06/11/2016"

print ((uv2016[filteredData6])['UV_Index'].max())



filteredData7 = uv2016.Date == "07/11/2016"

print ((uv2016[filteredData7])['UV_Index'].max())



filteredData8 = uv2016.Date == "08/11/2016"

print ((uv2016[filteredData8])['UV_Index'].max())



filteredData9 = uv2016.Date == "09/11/2016"

print ((uv2016[filteredData9])['UV_Index'].max())



filteredData10 = uv2016.Date == "10/11/2016"

print ((uv2016[filteredData10])['UV_Index'].max())



filteredData11 = uv2016.Date == "11/11/2016"

print ((uv2016[filteredData11])['UV_Index'].max())



filteredData12 = uv2016.Date == "12/11/2016"

print ((uv2016[filteredData12])['UV_Index'].max())



filteredData13 = uv2016.Date == "13/11/2016"

print ((uv2016[filteredData13])['UV_Index'].max())



filteredData14 = uv2016.Date == "14/11/2016"

print ((uv2016[filteredData14])['UV_Index'].max())



filteredData15 = uv2016.Date == "15/11/2016"

print ((uv2016[filteredData15])['UV_Index'].max())



filteredData16 = uv2016.Date == "16/11/2016"

print ((uv2016[filteredData16])['UV_Index'].max())



filteredData17 = uv2016.Date == "17/11/2016"

print ((uv2016[filteredData17])['UV_Index'].max())



filteredData18 = uv2016.Date == "18/11/2016"

print ((uv2016[filteredData18])['UV_Index'].max())



filteredData19 = uv2016.Date == "19/11/2016"

print ((uv2016[filteredData19])['UV_Index'].max())



filteredData20 = uv2016.Date == "20/11/2119"

print ((uv2016[filteredData20])['UV_Index'].max())



filteredData21 = uv2016.Date == "21/11/2016"

print ((uv2016[filteredData21])['UV_Index'].max())



filteredData22 = uv2016.Date == "22/11/2016"

print ((uv2016[filteredData22])['UV_Index'].max())



filteredData23 = uv2016.Date == "23/11/2016"

print ((uv2016[filteredData23])['UV_Index'].max())



filteredData24 = uv2016.Date == "24/11/2016"

print ((uv2016[filteredData24])['UV_Index'].max())



filteredData25 = uv2016.Date == "25/11/2016"

print ((uv2016[filteredData25])['UV_Index'].max())



filteredData26 = uv2016.Date == "26/11/2016"

print ((uv2016[filteredData26])['UV_Index'].max())



filteredData27 = uv2016.Date == "27/11/2016"

print ((uv2016[filteredData27])['UV_Index'].max())



filteredData28 = uv2016.Date == "28/11/2016"

print ((uv2016[filteredData28])['UV_Index'].max())



filteredData29 = uv2016.Date == "29/11/2016"

print ((uv2016[filteredData29])['UV_Index'].max())



filteredData30 = uv2016.Date == "30/11/2016"

print ((uv2016[filteredData30])['UV_Index'].max())



filteredData31 = uv2016.Date == "31/11/2016"

print ((uv2016[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2015 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2015.csv')



filteredData1 = uv2015.Date == "01/11/2015"

print((uv2015[filteredData1])['UV_Index'].max())



filteredData2 = uv2015.Date == "02/11/2015"

print ((uv2015[filteredData2])['UV_Index'].max())



filteredData3 = uv2015.Date == "03/11/2015"

print ((uv2015[filteredData3])['UV_Index'].max())



filteredData4 = uv2015.Date == "04/11/2015"

print ((uv2015[filteredData4])['UV_Index'].max())



filteredData5 = uv2015.Date == "05/11/2015"

print ((uv2015[filteredData5])['UV_Index'].max())



filteredData6 = uv2015.Date == "06/11/2015"

print ((uv2015[filteredData6])['UV_Index'].max())



filteredData7 = uv2015.Date == "07/11/2015"

print ((uv2015[filteredData7])['UV_Index'].max())



filteredData8 = uv2015.Date == "08/11/2015"

print ((uv2015[filteredData8])['UV_Index'].max())



filteredData9 = uv2015.Date == "09/11/2015"

print ((uv2015[filteredData9])['UV_Index'].max())



filteredData10 = uv2015.Date == "10/11/2015"

print ((uv2015[filteredData10])['UV_Index'].max())



filteredData11 = uv2015.Date == "11/11/2015"

print ((uv2015[filteredData11])['UV_Index'].max())



filteredData12 = uv2015.Date == "12/11/2015"

print ((uv2015[filteredData12])['UV_Index'].max())



filteredData13 = uv2015.Date == "13/11/2015"

print ((uv2015[filteredData13])['UV_Index'].max())



filteredData14 = uv2015.Date == "14/11/2015"

print ((uv2015[filteredData14])['UV_Index'].max())



filteredData15 = uv2015.Date == "15/11/2015"

print ((uv2015[filteredData15])['UV_Index'].max())



filteredData16 = uv2015.Date == "16/11/2015"

print ((uv2015[filteredData16])['UV_Index'].max())



filteredData17 = uv2015.Date == "17/11/2015"

print ((uv2015[filteredData17])['UV_Index'].max())



filteredData18 = uv2015.Date == "18/11/2015"

print ((uv2015[filteredData18])['UV_Index'].max())



filteredData19 = uv2015.Date == "19/11/2015"

print ((uv2015[filteredData19])['UV_Index'].max())



filteredData20 = uv2015.Date == "20/11/2119"

print ((uv2015[filteredData20])['UV_Index'].max())



filteredData21 = uv2015.Date == "21/11/2015"

print ((uv2015[filteredData21])['UV_Index'].max())



filteredData22 = uv2015.Date == "22/11/2015"

print ((uv2015[filteredData22])['UV_Index'].max())



filteredData23 = uv2015.Date == "23/11/2015"

print ((uv2015[filteredData23])['UV_Index'].max())



filteredData24 = uv2015.Date == "24/11/2015"

print ((uv2015[filteredData24])['UV_Index'].max())



filteredData25 = uv2015.Date == "25/11/2015"

print ((uv2015[filteredData25])['UV_Index'].max())



filteredData26 = uv2015.Date == "26/11/2015"

print ((uv2015[filteredData26])['UV_Index'].max())



filteredData27 = uv2015.Date == "27/11/2015"

print ((uv2015[filteredData27])['UV_Index'].max())



filteredData28 = uv2015.Date == "28/11/2015"

print ((uv2015[filteredData28])['UV_Index'].max())



filteredData29 = uv2015.Date == "29/11/2015"

print ((uv2015[filteredData29])['UV_Index'].max())



filteredData30 = uv2015.Date == "30/11/2015"

print ((uv2015[filteredData30])['UV_Index'].max())



filteredData31 = uv2015.Date == "31/11/2015"

print ((uv2015[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2014 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2014.csv')



filteredData1 = uv2014.Date == "01/11/2014"

print((uv2014[filteredData1])['UV_Index'].max())



filteredData2 = uv2014.Date == "02/11/2014"

print ((uv2014[filteredData2])['UV_Index'].max())



filteredData3 = uv2014.Date == "03/11/2014"

print ((uv2014[filteredData3])['UV_Index'].max())



filteredData4 = uv2014.Date == "04/11/2014"

print ((uv2014[filteredData4])['UV_Index'].max())



filteredData5 = uv2014.Date == "05/11/2014"

print ((uv2014[filteredData5])['UV_Index'].max())



filteredData6 = uv2014.Date == "06/11/2014"

print ((uv2014[filteredData6])['UV_Index'].max())



filteredData7 = uv2014.Date == "07/11/2014"

print ((uv2014[filteredData7])['UV_Index'].max())



filteredData8 = uv2014.Date == "08/11/2014"

print ((uv2014[filteredData8])['UV_Index'].max())



filteredData9 = uv2014.Date == "09/11/2014"

print ((uv2014[filteredData9])['UV_Index'].max())



filteredData10 = uv2014.Date == "10/11/2014"

print ((uv2014[filteredData10])['UV_Index'].max())



filteredData11 = uv2014.Date == "11/11/2014"

print ((uv2014[filteredData11])['UV_Index'].max())



filteredData12 = uv2014.Date == "12/11/2014"

print ((uv2014[filteredData12])['UV_Index'].max())



filteredData13 = uv2014.Date == "13/11/2014"

print ((uv2014[filteredData13])['UV_Index'].max())



filteredData14 = uv2014.Date == "14/11/2014"

print ((uv2014[filteredData14])['UV_Index'].max())



filteredData15 = uv2014.Date == "15/11/2014"

print ((uv2014[filteredData15])['UV_Index'].max())



filteredData16 = uv2014.Date == "16/11/2014"

print ((uv2014[filteredData16])['UV_Index'].max())



filteredData17 = uv2014.Date == "17/11/2014"

print ((uv2014[filteredData17])['UV_Index'].max())



filteredData18 = uv2014.Date == "18/11/2014"

print ((uv2014[filteredData18])['UV_Index'].max())



filteredData19 = uv2014.Date == "19/11/2014"

print ((uv2014[filteredData19])['UV_Index'].max())



filteredData20 = uv2014.Date == "20/11/2119"

print ((uv2014[filteredData20])['UV_Index'].max())



filteredData21 = uv2014.Date == "21/11/2014"

print ((uv2014[filteredData21])['UV_Index'].max())



filteredData22 = uv2014.Date == "22/11/2014"

print ((uv2014[filteredData22])['UV_Index'].max())



filteredData23 = uv2014.Date == "23/11/2014"

print ((uv2014[filteredData23])['UV_Index'].max())



filteredData24 = uv2014.Date == "24/11/2014"

print ((uv2014[filteredData24])['UV_Index'].max())



filteredData25 = uv2014.Date == "25/11/2014"

print ((uv2014[filteredData25])['UV_Index'].max())



filteredData26 = uv2014.Date == "26/11/2014"

print ((uv2014[filteredData26])['UV_Index'].max())



filteredData27 = uv2014.Date == "27/11/2014"

print ((uv2014[filteredData27])['UV_Index'].max())



filteredData28 = uv2014.Date == "28/11/2014"

print ((uv2014[filteredData28])['UV_Index'].max())



filteredData29 = uv2014.Date == "29/11/2014"

print ((uv2014[filteredData29])['UV_Index'].max())



filteredData30 = uv2014.Date == "30/11/2014"

print ((uv2014[filteredData30])['UV_Index'].max())



filteredData31 = uv2014.Date == "31/11/2014"

print ((uv2014[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2013 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2013.csv')



filteredData1 = uv2013.Date == "01/11/2013"

print((uv2013[filteredData1])['UV_Index'].max())



filteredData2 = uv2013.Date == "02/11/2013"

print ((uv2013[filteredData2])['UV_Index'].max())



filteredData3 = uv2013.Date == "03/11/2013"

print ((uv2013[filteredData3])['UV_Index'].max())



filteredData4 = uv2013.Date == "04/11/2013"

print ((uv2013[filteredData4])['UV_Index'].max())



filteredData5 = uv2013.Date == "05/11/2013"

print ((uv2013[filteredData5])['UV_Index'].max())



filteredData6 = uv2013.Date == "06/11/2013"

print ((uv2013[filteredData6])['UV_Index'].max())



filteredData7 = uv2013.Date == "07/11/2013"

print ((uv2013[filteredData7])['UV_Index'].max())



filteredData8 = uv2013.Date == "08/11/2013"

print ((uv2013[filteredData8])['UV_Index'].max())



filteredData9 = uv2013.Date == "09/11/2013"

print ((uv2013[filteredData9])['UV_Index'].max())



filteredData10 = uv2013.Date == "10/11/2013"

print ((uv2013[filteredData10])['UV_Index'].max())



filteredData11 = uv2013.Date == "11/11/2013"

print ((uv2013[filteredData11])['UV_Index'].max())



filteredData12 = uv2013.Date == "12/11/2013"

print ((uv2013[filteredData12])['UV_Index'].max())



filteredData13 = uv2013.Date == "13/11/2013"

print ((uv2013[filteredData13])['UV_Index'].max())



filteredData14 = uv2013.Date == "14/11/2013"

print ((uv2013[filteredData14])['UV_Index'].max())



filteredData15 = uv2013.Date == "15/11/2013"

print ((uv2013[filteredData15])['UV_Index'].max())



filteredData16 = uv2013.Date == "16/11/2013"

print ((uv2013[filteredData16])['UV_Index'].max())



filteredData17 = uv2013.Date == "17/11/2013"

print ((uv2013[filteredData17])['UV_Index'].max())



filteredData18 = uv2013.Date == "18/11/2013"

print ((uv2013[filteredData18])['UV_Index'].max())



filteredData19 = uv2013.Date == "19/11/2013"

print ((uv2013[filteredData19])['UV_Index'].max())



filteredData20 = uv2013.Date == "20/11/2119"

print ((uv2013[filteredData20])['UV_Index'].max())



filteredData21 = uv2013.Date == "21/11/2013"

print ((uv2013[filteredData21])['UV_Index'].max())



filteredData22 = uv2013.Date == "22/11/2013"

print ((uv2013[filteredData22])['UV_Index'].max())



filteredData23 = uv2013.Date == "23/11/2013"

print ((uv2013[filteredData23])['UV_Index'].max())



filteredData24 = uv2013.Date == "24/11/2013"

print ((uv2013[filteredData24])['UV_Index'].max())



filteredData25 = uv2013.Date == "25/11/2013"

print ((uv2013[filteredData25])['UV_Index'].max())



filteredData26 = uv2013.Date == "26/11/2013"

print ((uv2013[filteredData26])['UV_Index'].max())



filteredData27 = uv2013.Date == "27/11/2013"

print ((uv2013[filteredData27])['UV_Index'].max())



filteredData28 = uv2013.Date == "28/11/2013"

print ((uv2013[filteredData28])['UV_Index'].max())



filteredData29 = uv2013.Date == "29/11/2013"

print ((uv2013[filteredData29])['UV_Index'].max())



filteredData30 = uv2013.Date == "30/11/2013"

print ((uv2013[filteredData30])['UV_Index'].max())



filteredData31 = uv2013.Date == "31/11/2013"

print ((uv2013[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2012 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2012.csv')



filteredData1 = uv2012.Date == "01/11/2012"

print((uv2012[filteredData1])['UV_Index'].max())



filteredData2 = uv2012.Date == "02/11/2012"

print ((uv2012[filteredData2])['UV_Index'].max())



filteredData3 = uv2012.Date == "03/11/2012"

print ((uv2012[filteredData3])['UV_Index'].max())



filteredData4 = uv2012.Date == "04/11/2012"

print ((uv2012[filteredData4])['UV_Index'].max())



filteredData5 = uv2012.Date == "05/11/2012"

print ((uv2012[filteredData5])['UV_Index'].max())



filteredData6 = uv2012.Date == "06/11/2012"

print ((uv2012[filteredData6])['UV_Index'].max())



filteredData7 = uv2012.Date == "07/11/2012"

print ((uv2012[filteredData7])['UV_Index'].max())



filteredData8 = uv2012.Date == "08/11/2012"

print ((uv2012[filteredData8])['UV_Index'].max())



filteredData9 = uv2012.Date == "09/11/2012"

print ((uv2012[filteredData9])['UV_Index'].max())



filteredData10 = uv2012.Date == "10/11/2012"

print ((uv2012[filteredData10])['UV_Index'].max())



filteredData11 = uv2012.Date == "11/11/2012"

print ((uv2012[filteredData11])['UV_Index'].max())



filteredData12 = uv2012.Date == "12/11/2012"

print ((uv2012[filteredData12])['UV_Index'].max())



filteredData13 = uv2012.Date == "13/11/2012"

print ((uv2012[filteredData13])['UV_Index'].max())



filteredData14 = uv2012.Date == "14/11/2012"

print ((uv2012[filteredData14])['UV_Index'].max())



filteredData15 = uv2012.Date == "15/11/2012"

print ((uv2012[filteredData15])['UV_Index'].max())



filteredData16 = uv2012.Date == "16/11/2012"

print ((uv2012[filteredData16])['UV_Index'].max())



filteredData17 = uv2012.Date == "17/11/2012"

print ((uv2012[filteredData17])['UV_Index'].max())



filteredData18 = uv2012.Date == "18/11/2012"

print ((uv2012[filteredData18])['UV_Index'].max())



filteredData19 = uv2012.Date == "19/11/2012"

print ((uv2012[filteredData19])['UV_Index'].max())



filteredData20 = uv2012.Date == "20/11/2119"

print ((uv2012[filteredData20])['UV_Index'].max())



filteredData21 = uv2012.Date == "21/11/2012"

print ((uv2012[filteredData21])['UV_Index'].max())



filteredData22 = uv2012.Date == "22/11/2012"

print ((uv2012[filteredData22])['UV_Index'].max())



filteredData23 = uv2012.Date == "23/11/2012"

print ((uv2012[filteredData23])['UV_Index'].max())



filteredData24 = uv2012.Date == "24/11/2012"

print ((uv2012[filteredData24])['UV_Index'].max())



filteredData25 = uv2012.Date == "25/11/2012"

print ((uv2012[filteredData25])['UV_Index'].max())



filteredData26 = uv2012.Date == "26/11/2012"

print ((uv2012[filteredData26])['UV_Index'].max())



filteredData27 = uv2012.Date == "27/11/2012"

print ((uv2012[filteredData27])['UV_Index'].max())



filteredData28 = uv2012.Date == "28/11/2012"

print ((uv2012[filteredData28])['UV_Index'].max())



filteredData29 = uv2012.Date == "29/11/2012"

print ((uv2012[filteredData29])['UV_Index'].max())



filteredData30 = uv2012.Date == "30/11/2012"

print ((uv2012[filteredData30])['UV_Index'].max())



filteredData31 = uv2012.Date == "31/11/2012"

print ((uv2012[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2011 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2011.csv')



filteredData1 = uv2011.Date == "01/11/2011"

print((uv2011[filteredData1])['UV_Index'].max())



filteredData2 = uv2011.Date == "02/11/2011"

print ((uv2011[filteredData2])['UV_Index'].max())



filteredData3 = uv2011.Date == "03/11/2011"

print ((uv2011[filteredData3])['UV_Index'].max())



filteredData4 = uv2011.Date == "04/11/2011"

print ((uv2011[filteredData4])['UV_Index'].max())



filteredData5 = uv2011.Date == "05/11/2011"

print ((uv2011[filteredData5])['UV_Index'].max())



filteredData6 = uv2011.Date == "06/11/2011"

print ((uv2011[filteredData6])['UV_Index'].max())



filteredData7 = uv2011.Date == "07/11/2011"

print ((uv2011[filteredData7])['UV_Index'].max())



filteredData8 = uv2011.Date == "08/11/2011"

print ((uv2011[filteredData8])['UV_Index'].max())



filteredData9 = uv2011.Date == "09/11/2011"

print ((uv2011[filteredData9])['UV_Index'].max())



filteredData10 = uv2011.Date == "10/11/2011"

print ((uv2011[filteredData10])['UV_Index'].max())



filteredData11 = uv2011.Date == "11/11/2011"

print ((uv2011[filteredData11])['UV_Index'].max())



filteredData12 = uv2011.Date == "12/11/2011"

print ((uv2011[filteredData12])['UV_Index'].max())



filteredData13 = uv2011.Date == "13/11/2011"

print ((uv2011[filteredData13])['UV_Index'].max())



filteredData14 = uv2011.Date == "14/11/2011"

print ((uv2011[filteredData14])['UV_Index'].max())



filteredData15 = uv2011.Date == "15/11/2011"

print ((uv2011[filteredData15])['UV_Index'].max())



filteredData16 = uv2011.Date == "16/11/2011"

print ((uv2011[filteredData16])['UV_Index'].max())



filteredData17 = uv2011.Date == "17/11/2011"

print ((uv2011[filteredData17])['UV_Index'].max())



filteredData18 = uv2011.Date == "18/11/2011"

print ((uv2011[filteredData18])['UV_Index'].max())



filteredData19 = uv2011.Date == "19/11/2011"

print ((uv2011[filteredData19])['UV_Index'].max())



filteredData20 = uv2011.Date == "20/11/2119"

print ((uv2011[filteredData20])['UV_Index'].max())



filteredData21 = uv2011.Date == "21/11/2011"

print ((uv2011[filteredData21])['UV_Index'].max())



filteredData22 = uv2011.Date == "22/11/2011"

print ((uv2011[filteredData22])['UV_Index'].max())



filteredData23 = uv2011.Date == "23/11/2011"

print ((uv2011[filteredData23])['UV_Index'].max())



filteredData24 = uv2011.Date == "24/11/2011"

print ((uv2011[filteredData24])['UV_Index'].max())



filteredData25 = uv2011.Date == "25/11/2011"

print ((uv2011[filteredData25])['UV_Index'].max())



filteredData26 = uv2011.Date == "26/11/2011"

print ((uv2011[filteredData26])['UV_Index'].max())



filteredData27 = uv2011.Date == "27/11/2011"

print ((uv2011[filteredData27])['UV_Index'].max())



filteredData28 = uv2011.Date == "28/11/2011"

print ((uv2011[filteredData28])['UV_Index'].max())



filteredData29 = uv2011.Date == "29/11/2011"

print ((uv2011[filteredData29])['UV_Index'].max())



filteredData30 = uv2011.Date == "30/11/2011"

print ((uv2011[filteredData30])['UV_Index'].max())



filteredData31 = uv2011.Date == "31/11/2011"

print ((uv2011[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2010 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2010.csv')



filteredData1 = uv2010.Date == "01/11/2010"

print((uv2010[filteredData1])['UV_Index'].max())



filteredData2 = uv2010.Date == "02/11/2010"

print ((uv2010[filteredData2])['UV_Index'].max())



filteredData3 = uv2010.Date == "03/11/2010"

print ((uv2010[filteredData3])['UV_Index'].max())



filteredData4 = uv2010.Date == "04/11/2010"

print ((uv2010[filteredData4])['UV_Index'].max())



filteredData5 = uv2010.Date == "05/11/2010"

print ((uv2010[filteredData5])['UV_Index'].max())



filteredData6 = uv2010.Date == "06/11/2010"

print ((uv2010[filteredData6])['UV_Index'].max())



filteredData7 = uv2010.Date == "07/11/2010"

print ((uv2010[filteredData7])['UV_Index'].max())



filteredData8 = uv2010.Date == "08/11/2010"

print ((uv2010[filteredData8])['UV_Index'].max())



filteredData9 = uv2010.Date == "09/11/2010"

print ((uv2010[filteredData9])['UV_Index'].max())



filteredData10 = uv2010.Date == "10/11/2010"

print ((uv2010[filteredData10])['UV_Index'].max())



filteredData11 = uv2010.Date == "11/11/2010"

print ((uv2010[filteredData11])['UV_Index'].max())



filteredData12 = uv2010.Date == "12/11/2010"

print ((uv2010[filteredData12])['UV_Index'].max())



filteredData13 = uv2010.Date == "13/11/2010"

print ((uv2010[filteredData13])['UV_Index'].max())



filteredData14 = uv2010.Date == "14/11/2010"

print ((uv2010[filteredData14])['UV_Index'].max())



filteredData15 = uv2010.Date == "15/11/2010"

print ((uv2010[filteredData15])['UV_Index'].max())



filteredData16 = uv2010.Date == "16/11/2010"

print ((uv2010[filteredData16])['UV_Index'].max())



filteredData17 = uv2010.Date == "17/11/2010"

print ((uv2010[filteredData17])['UV_Index'].max())



filteredData18 = uv2010.Date == "18/11/2010"

print ((uv2010[filteredData18])['UV_Index'].max())



filteredData19 = uv2010.Date == "19/11/2010"

print ((uv2010[filteredData19])['UV_Index'].max())



filteredData20 = uv2010.Date == "20/11/2119"

print ((uv2010[filteredData20])['UV_Index'].max())



filteredData21 = uv2010.Date == "21/11/2010"

print ((uv2010[filteredData21])['UV_Index'].max())



filteredData22 = uv2010.Date == "22/11/2010"

print ((uv2010[filteredData22])['UV_Index'].max())



filteredData23 = uv2010.Date == "23/11/2010"

print ((uv2010[filteredData23])['UV_Index'].max())



filteredData24 = uv2010.Date == "24/11/2010"

print ((uv2010[filteredData24])['UV_Index'].max())



filteredData25 = uv2010.Date == "25/11/2010"

print ((uv2010[filteredData25])['UV_Index'].max())



filteredData26 = uv2010.Date == "26/11/2010"

print ((uv2010[filteredData26])['UV_Index'].max())



filteredData27 = uv2010.Date == "27/11/2010"

print ((uv2010[filteredData27])['UV_Index'].max())



filteredData28 = uv2010.Date == "28/11/2010"

print ((uv2010[filteredData28])['UV_Index'].max())



filteredData29 = uv2010.Date == "29/11/2010"

print ((uv2010[filteredData29])['UV_Index'].max())



filteredData30 = uv2010.Date == "30/11/2010"

print ((uv2010[filteredData30])['UV_Index'].max())



filteredData31 = uv2010.Date == "31/11/2010"

print ((uv2010[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2010 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2010.csv')



filteredData1 = uv2010.Date == "01/12/2010"

print((uv2010[filteredData1])['UV_Index'].max())



filteredData2 = uv2010.Date == "02/12/2010"

print ((uv2010[filteredData2])['UV_Index'].max())



filteredData3 = uv2010.Date == "03/12/2010"

print ((uv2010[filteredData3])['UV_Index'].max())



filteredData4 = uv2010.Date == "04/12/2010"

print ((uv2010[filteredData4])['UV_Index'].max())



filteredData5 = uv2010.Date == "05/12/2010"

print ((uv2010[filteredData5])['UV_Index'].max())



filteredData6 = uv2010.Date == "06/12/2010"

print ((uv2010[filteredData6])['UV_Index'].max())



filteredData7 = uv2010.Date == "07/12/2010"

print ((uv2010[filteredData7])['UV_Index'].max())



filteredData8 = uv2010.Date == "08/12/2010"

print ((uv2010[filteredData8])['UV_Index'].max())



filteredData9 = uv2010.Date == "09/12/2010"

print ((uv2010[filteredData9])['UV_Index'].max())



filteredData10 = uv2010.Date == "10/12/2010"

print ((uv2010[filteredData10])['UV_Index'].max())



filteredData11 = uv2010.Date == "11/12/2010"

print ((uv2010[filteredData11])['UV_Index'].max())



filteredData12 = uv2010.Date == "12/12/2010"

print ((uv2010[filteredData12])['UV_Index'].max())



filteredData13 = uv2010.Date == "13/12/2010"

print ((uv2010[filteredData13])['UV_Index'].max())



filteredData14 = uv2010.Date == "14/12/2010"

print ((uv2010[filteredData14])['UV_Index'].max())



filteredData15 = uv2010.Date == "15/12/2010"

print ((uv2010[filteredData15])['UV_Index'].max())



filteredData16 = uv2010.Date == "16/12/2010"

print ((uv2010[filteredData16])['UV_Index'].max())



filteredData17 = uv2010.Date == "17/12/2010"

print ((uv2010[filteredData17])['UV_Index'].max())



filteredData18 = uv2010.Date == "18/12/2010"

print ((uv2010[filteredData18])['UV_Index'].max())



filteredData19 = uv2010.Date == "19/12/2010"

print ((uv2010[filteredData19])['UV_Index'].max())



filteredData20 = uv2010.Date == "20/12/2129"

print ((uv2010[filteredData20])['UV_Index'].max())



filteredData21 = uv2010.Date == "21/12/2010"

print ((uv2010[filteredData21])['UV_Index'].max())



filteredData22 = uv2010.Date == "22/12/2010"

print ((uv2010[filteredData22])['UV_Index'].max())



filteredData23 = uv2010.Date == "23/12/2010"

print ((uv2010[filteredData23])['UV_Index'].max())



filteredData24 = uv2010.Date == "24/12/2010"

print ((uv2010[filteredData24])['UV_Index'].max())



filteredData25 = uv2010.Date == "25/12/2010"

print ((uv2010[filteredData25])['UV_Index'].max())



filteredData26 = uv2010.Date == "26/12/2010"

print ((uv2010[filteredData26])['UV_Index'].max())



filteredData27 = uv2010.Date == "27/12/2010"

print ((uv2010[filteredData27])['UV_Index'].max())



filteredData28 = uv2010.Date == "28/12/2010"

print ((uv2010[filteredData28])['UV_Index'].max())



filteredData29 = uv2010.Date == "29/12/2010"

print ((uv2010[filteredData29])['UV_Index'].max())



filteredData30 = uv2010.Date == "30/12/2010"

print ((uv2010[filteredData30])['UV_Index'].max())



filteredData31 = uv2010.Date == "31/12/2010"

print ((uv2010[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2011 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2011.csv')



filteredData1 = uv2011.Date == "01/12/2011"

print((uv2011[filteredData1])['UV_Index'].max())



filteredData2 = uv2011.Date == "02/12/2011"

print ((uv2011[filteredData2])['UV_Index'].max())



filteredData3 = uv2011.Date == "03/12/2011"

print ((uv2011[filteredData3])['UV_Index'].max())



filteredData4 = uv2011.Date == "04/12/2011"

print ((uv2011[filteredData4])['UV_Index'].max())



filteredData5 = uv2011.Date == "05/12/2011"

print ((uv2011[filteredData5])['UV_Index'].max())



filteredData6 = uv2011.Date == "06/12/2011"

print ((uv2011[filteredData6])['UV_Index'].max())



filteredData7 = uv2011.Date == "07/12/2011"

print ((uv2011[filteredData7])['UV_Index'].max())



filteredData8 = uv2011.Date == "08/12/2011"

print ((uv2011[filteredData8])['UV_Index'].max())



filteredData9 = uv2011.Date == "09/12/2011"

print ((uv2011[filteredData9])['UV_Index'].max())



filteredData10 = uv2011.Date == "10/12/2011"

print ((uv2011[filteredData10])['UV_Index'].max())



filteredData11 = uv2011.Date == "11/12/2011"

print ((uv2011[filteredData11])['UV_Index'].max())



filteredData12 = uv2011.Date == "12/12/2011"

print ((uv2011[filteredData12])['UV_Index'].max())



filteredData13 = uv2011.Date == "13/12/2011"

print ((uv2011[filteredData13])['UV_Index'].max())



filteredData14 = uv2011.Date == "14/12/2011"

print ((uv2011[filteredData14])['UV_Index'].max())



filteredData15 = uv2011.Date == "15/12/2011"

print ((uv2011[filteredData15])['UV_Index'].max())



filteredData16 = uv2011.Date == "16/12/2011"

print ((uv2011[filteredData16])['UV_Index'].max())



filteredData17 = uv2011.Date == "17/12/2011"

print ((uv2011[filteredData17])['UV_Index'].max())



filteredData18 = uv2011.Date == "18/12/2011"

print ((uv2011[filteredData18])['UV_Index'].max())



filteredData19 = uv2011.Date == "19/12/2011"

print ((uv2011[filteredData19])['UV_Index'].max())



filteredData20 = uv2011.Date == "20/12/2129"

print ((uv2011[filteredData20])['UV_Index'].max())



filteredData21 = uv2011.Date == "21/12/2011"

print ((uv2011[filteredData21])['UV_Index'].max())



filteredData22 = uv2011.Date == "22/12/2011"

print ((uv2011[filteredData22])['UV_Index'].max())



filteredData23 = uv2011.Date == "23/12/2011"

print ((uv2011[filteredData23])['UV_Index'].max())



filteredData24 = uv2011.Date == "24/12/2011"

print ((uv2011[filteredData24])['UV_Index'].max())



filteredData25 = uv2011.Date == "25/12/2011"

print ((uv2011[filteredData25])['UV_Index'].max())



filteredData26 = uv2011.Date == "26/12/2011"

print ((uv2011[filteredData26])['UV_Index'].max())



filteredData27 = uv2011.Date == "27/12/2011"

print ((uv2011[filteredData27])['UV_Index'].max())



filteredData28 = uv2011.Date == "28/12/2011"

print ((uv2011[filteredData28])['UV_Index'].max())



filteredData29 = uv2011.Date == "29/12/2011"

print ((uv2011[filteredData29])['UV_Index'].max())



filteredData30 = uv2011.Date == "30/12/2011"

print ((uv2011[filteredData30])['UV_Index'].max())



filteredData31 = uv2011.Date == "31/12/2011"

print ((uv2011[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2012 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2012.csv')



filteredData1 = uv2012.Date == "01/12/2012"

print((uv2012[filteredData1])['UV_Index'].max())



filteredData2 = uv2012.Date == "02/12/2012"

print ((uv2012[filteredData2])['UV_Index'].max())



filteredData3 = uv2012.Date == "03/12/2012"

print ((uv2012[filteredData3])['UV_Index'].max())



filteredData4 = uv2012.Date == "04/12/2012"

print ((uv2012[filteredData4])['UV_Index'].max())



filteredData5 = uv2012.Date == "05/12/2012"

print ((uv2012[filteredData5])['UV_Index'].max())



filteredData6 = uv2012.Date == "06/12/2012"

print ((uv2012[filteredData6])['UV_Index'].max())



filteredData7 = uv2012.Date == "07/12/2012"

print ((uv2012[filteredData7])['UV_Index'].max())



filteredData8 = uv2012.Date == "08/12/2012"

print ((uv2012[filteredData8])['UV_Index'].max())



filteredData9 = uv2012.Date == "09/12/2012"

print ((uv2012[filteredData9])['UV_Index'].max())



filteredData10 = uv2012.Date == "10/12/2012"

print ((uv2012[filteredData10])['UV_Index'].max())



filteredData11 = uv2012.Date == "11/12/2012"

print ((uv2012[filteredData11])['UV_Index'].max())



filteredData12 = uv2012.Date == "12/12/2012"

print ((uv2012[filteredData12])['UV_Index'].max())



filteredData13 = uv2012.Date == "13/12/2012"

print ((uv2012[filteredData13])['UV_Index'].max())



filteredData14 = uv2012.Date == "14/12/2012"

print ((uv2012[filteredData14])['UV_Index'].max())



filteredData15 = uv2012.Date == "15/12/2012"

print ((uv2012[filteredData15])['UV_Index'].max())



filteredData16 = uv2012.Date == "16/12/2012"

print ((uv2012[filteredData16])['UV_Index'].max())



filteredData17 = uv2012.Date == "17/12/2012"

print ((uv2012[filteredData17])['UV_Index'].max())



filteredData18 = uv2012.Date == "18/12/2012"

print ((uv2012[filteredData18])['UV_Index'].max())



filteredData19 = uv2012.Date == "19/12/2012"

print ((uv2012[filteredData19])['UV_Index'].max())



filteredData20 = uv2012.Date == "20/12/2129"

print ((uv2012[filteredData20])['UV_Index'].max())



filteredData21 = uv2012.Date == "21/12/2012"

print ((uv2012[filteredData21])['UV_Index'].max())



filteredData22 = uv2012.Date == "22/12/2012"

print ((uv2012[filteredData22])['UV_Index'].max())



filteredData23 = uv2012.Date == "23/12/2012"

print ((uv2012[filteredData23])['UV_Index'].max())



filteredData24 = uv2012.Date == "24/12/2012"

print ((uv2012[filteredData24])['UV_Index'].max())



filteredData25 = uv2012.Date == "25/12/2012"

print ((uv2012[filteredData25])['UV_Index'].max())



filteredData26 = uv2012.Date == "26/12/2012"

print ((uv2012[filteredData26])['UV_Index'].max())



filteredData27 = uv2012.Date == "27/12/2012"

print ((uv2012[filteredData27])['UV_Index'].max())



filteredData28 = uv2012.Date == "28/12/2012"

print ((uv2012[filteredData28])['UV_Index'].max())



filteredData29 = uv2012.Date == "29/12/2012"

print ((uv2012[filteredData29])['UV_Index'].max())



filteredData30 = uv2012.Date == "30/12/2012"

print ((uv2012[filteredData30])['UV_Index'].max())



filteredData31 = uv2012.Date == "31/12/2012"

print ((uv2012[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2013 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2013.csv')



filteredData1 = uv2013.Date == "01/12/2013"

print((uv2013[filteredData1])['UV_Index'].max())



filteredData2 = uv2013.Date == "02/12/2013"

print ((uv2013[filteredData2])['UV_Index'].max())



filteredData3 = uv2013.Date == "03/12/2013"

print ((uv2013[filteredData3])['UV_Index'].max())



filteredData4 = uv2013.Date == "04/12/2013"

print ((uv2013[filteredData4])['UV_Index'].max())



filteredData5 = uv2013.Date == "05/12/2013"

print ((uv2013[filteredData5])['UV_Index'].max())



filteredData6 = uv2013.Date == "06/12/2013"

print ((uv2013[filteredData6])['UV_Index'].max())



filteredData7 = uv2013.Date == "07/12/2013"

print ((uv2013[filteredData7])['UV_Index'].max())



filteredData8 = uv2013.Date == "08/12/2013"

print ((uv2013[filteredData8])['UV_Index'].max())



filteredData9 = uv2013.Date == "09/12/2013"

print ((uv2013[filteredData9])['UV_Index'].max())



filteredData10 = uv2013.Date == "10/12/2013"

print ((uv2013[filteredData10])['UV_Index'].max())



filteredData11 = uv2013.Date == "11/12/2013"

print ((uv2013[filteredData11])['UV_Index'].max())



filteredData12 = uv2013.Date == "12/12/2013"

print ((uv2013[filteredData12])['UV_Index'].max())



filteredData13 = uv2013.Date == "13/12/2013"

print ((uv2013[filteredData13])['UV_Index'].max())



filteredData14 = uv2013.Date == "14/12/2013"

print ((uv2013[filteredData14])['UV_Index'].max())



filteredData15 = uv2013.Date == "15/12/2013"

print ((uv2013[filteredData15])['UV_Index'].max())



filteredData16 = uv2013.Date == "16/12/2013"

print ((uv2013[filteredData16])['UV_Index'].max())



filteredData17 = uv2013.Date == "17/12/2013"

print ((uv2013[filteredData17])['UV_Index'].max())



filteredData18 = uv2013.Date == "18/12/2013"

print ((uv2013[filteredData18])['UV_Index'].max())



filteredData19 = uv2013.Date == "19/12/2013"

print ((uv2013[filteredData19])['UV_Index'].max())



filteredData20 = uv2013.Date == "20/12/2129"

print ((uv2013[filteredData20])['UV_Index'].max())



filteredData21 = uv2013.Date == "21/12/2013"

print ((uv2013[filteredData21])['UV_Index'].max())



filteredData22 = uv2013.Date == "22/12/2013"

print ((uv2013[filteredData22])['UV_Index'].max())



filteredData23 = uv2013.Date == "23/12/2013"

print ((uv2013[filteredData23])['UV_Index'].max())



filteredData24 = uv2013.Date == "24/12/2013"

print ((uv2013[filteredData24])['UV_Index'].max())



filteredData25 = uv2013.Date == "25/12/2013"

print ((uv2013[filteredData25])['UV_Index'].max())



filteredData26 = uv2013.Date == "26/12/2013"

print ((uv2013[filteredData26])['UV_Index'].max())



filteredData27 = uv2013.Date == "27/12/2013"

print ((uv2013[filteredData27])['UV_Index'].max())



filteredData28 = uv2013.Date == "28/12/2013"

print ((uv2013[filteredData28])['UV_Index'].max())



filteredData29 = uv2013.Date == "29/12/2013"

print ((uv2013[filteredData29])['UV_Index'].max())



filteredData30 = uv2013.Date == "30/12/2013"

print ((uv2013[filteredData30])['UV_Index'].max())



filteredData31 = uv2013.Date == "31/12/2013"

print ((uv2013[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2014 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2014.csv')



filteredData1 = uv2014.Date == "01/12/2014"

print((uv2014[filteredData1])['UV_Index'].max())



filteredData2 = uv2014.Date == "02/12/2014"

print ((uv2014[filteredData2])['UV_Index'].max())



filteredData3 = uv2014.Date == "03/12/2014"

print ((uv2014[filteredData3])['UV_Index'].max())



filteredData4 = uv2014.Date == "04/12/2014"

print ((uv2014[filteredData4])['UV_Index'].max())



filteredData5 = uv2014.Date == "05/12/2014"

print ((uv2014[filteredData5])['UV_Index'].max())



filteredData6 = uv2014.Date == "06/12/2014"

print ((uv2014[filteredData6])['UV_Index'].max())



filteredData7 = uv2014.Date == "07/12/2014"

print ((uv2014[filteredData7])['UV_Index'].max())



filteredData8 = uv2014.Date == "08/12/2014"

print ((uv2014[filteredData8])['UV_Index'].max())



filteredData9 = uv2014.Date == "09/12/2014"

print ((uv2014[filteredData9])['UV_Index'].max())



filteredData10 = uv2014.Date == "10/12/2014"

print ((uv2014[filteredData10])['UV_Index'].max())



filteredData11 = uv2014.Date == "11/12/2014"

print ((uv2014[filteredData11])['UV_Index'].max())



filteredData12 = uv2014.Date == "12/12/2014"

print ((uv2014[filteredData12])['UV_Index'].max())



filteredData13 = uv2014.Date == "13/12/2014"

print ((uv2014[filteredData13])['UV_Index'].max())



filteredData14 = uv2014.Date == "14/12/2014"

print ((uv2014[filteredData14])['UV_Index'].max())



filteredData15 = uv2014.Date == "15/12/2014"

print ((uv2014[filteredData15])['UV_Index'].max())



filteredData16 = uv2014.Date == "16/12/2014"

print ((uv2014[filteredData16])['UV_Index'].max())



filteredData17 = uv2014.Date == "17/12/2014"

print ((uv2014[filteredData17])['UV_Index'].max())



filteredData18 = uv2014.Date == "18/12/2014"

print ((uv2014[filteredData18])['UV_Index'].max())



filteredData19 = uv2014.Date == "19/12/2014"

print ((uv2014[filteredData19])['UV_Index'].max())



filteredData20 = uv2014.Date == "20/12/2129"

print ((uv2014[filteredData20])['UV_Index'].max())



filteredData21 = uv2014.Date == "21/12/2014"

print ((uv2014[filteredData21])['UV_Index'].max())



filteredData22 = uv2014.Date == "22/12/2014"

print ((uv2014[filteredData22])['UV_Index'].max())



filteredData23 = uv2014.Date == "23/12/2014"

print ((uv2014[filteredData23])['UV_Index'].max())



filteredData24 = uv2014.Date == "24/12/2014"

print ((uv2014[filteredData24])['UV_Index'].max())



filteredData25 = uv2014.Date == "25/12/2014"

print ((uv2014[filteredData25])['UV_Index'].max())



filteredData26 = uv2014.Date == "26/12/2014"

print ((uv2014[filteredData26])['UV_Index'].max())



filteredData27 = uv2014.Date == "27/12/2014"

print ((uv2014[filteredData27])['UV_Index'].max())



filteredData28 = uv2014.Date == "28/12/2014"

print ((uv2014[filteredData28])['UV_Index'].max())



filteredData29 = uv2014.Date == "29/12/2014"

print ((uv2014[filteredData29])['UV_Index'].max())



filteredData30 = uv2014.Date == "30/12/2014"

print ((uv2014[filteredData30])['UV_Index'].max())



filteredData31 = uv2014.Date == "31/12/2014"

print ((uv2014[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2015 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2015.csv')



filteredData1 = uv2015.Date == "01/12/2015"

print((uv2015[filteredData1])['UV_Index'].max())



filteredData2 = uv2015.Date == "02/12/2015"

print ((uv2015[filteredData2])['UV_Index'].max())



filteredData3 = uv2015.Date == "03/12/2015"

print ((uv2015[filteredData3])['UV_Index'].max())



filteredData4 = uv2015.Date == "04/12/2015"

print ((uv2015[filteredData4])['UV_Index'].max())



filteredData5 = uv2015.Date == "05/12/2015"

print ((uv2015[filteredData5])['UV_Index'].max())



filteredData6 = uv2015.Date == "06/12/2015"

print ((uv2015[filteredData6])['UV_Index'].max())



filteredData7 = uv2015.Date == "07/12/2015"

print ((uv2015[filteredData7])['UV_Index'].max())



filteredData8 = uv2015.Date == "08/12/2015"

print ((uv2015[filteredData8])['UV_Index'].max())



filteredData9 = uv2015.Date == "09/12/2015"

print ((uv2015[filteredData9])['UV_Index'].max())



filteredData10 = uv2015.Date == "10/12/2015"

print ((uv2015[filteredData10])['UV_Index'].max())



filteredData11 = uv2015.Date == "11/12/2015"

print ((uv2015[filteredData11])['UV_Index'].max())



filteredData12 = uv2015.Date == "12/12/2015"

print ((uv2015[filteredData12])['UV_Index'].max())



filteredData13 = uv2015.Date == "13/12/2015"

print ((uv2015[filteredData13])['UV_Index'].max())



filteredData14 = uv2015.Date == "14/12/2015"

print ((uv2015[filteredData14])['UV_Index'].max())



filteredData15 = uv2015.Date == "15/12/2015"

print ((uv2015[filteredData15])['UV_Index'].max())



filteredData16 = uv2015.Date == "16/12/2015"

print ((uv2015[filteredData16])['UV_Index'].max())



filteredData17 = uv2015.Date == "17/12/2015"

print ((uv2015[filteredData17])['UV_Index'].max())



filteredData18 = uv2015.Date == "18/12/2015"

print ((uv2015[filteredData18])['UV_Index'].max())



filteredData19 = uv2015.Date == "19/12/2015"

print ((uv2015[filteredData19])['UV_Index'].max())



filteredData20 = uv2015.Date == "20/12/2129"

print ((uv2015[filteredData20])['UV_Index'].max())



filteredData21 = uv2015.Date == "21/12/2015"

print ((uv2015[filteredData21])['UV_Index'].max())



filteredData22 = uv2015.Date == "22/12/2015"

print ((uv2015[filteredData22])['UV_Index'].max())



filteredData23 = uv2015.Date == "23/12/2015"

print ((uv2015[filteredData23])['UV_Index'].max())



filteredData24 = uv2015.Date == "24/12/2015"

print ((uv2015[filteredData24])['UV_Index'].max())



filteredData25 = uv2015.Date == "25/12/2015"

print ((uv2015[filteredData25])['UV_Index'].max())



filteredData26 = uv2015.Date == "26/12/2015"

print ((uv2015[filteredData26])['UV_Index'].max())



filteredData27 = uv2015.Date == "27/12/2015"

print ((uv2015[filteredData27])['UV_Index'].max())



filteredData28 = uv2015.Date == "28/12/2015"

print ((uv2015[filteredData28])['UV_Index'].max())



filteredData29 = uv2015.Date == "29/12/2015"

print ((uv2015[filteredData29])['UV_Index'].max())



filteredData30 = uv2015.Date == "30/12/2015"

print ((uv2015[filteredData30])['UV_Index'].max())



filteredData31 = uv2015.Date == "31/12/2015"

print ((uv2015[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2016 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2016.csv')



filteredData1 = uv2016.Date == "01/12/2016"

print((uv2016[filteredData1])['UV_Index'].max())



filteredData2 = uv2016.Date == "02/12/2016"

print ((uv2016[filteredData2])['UV_Index'].max())



filteredData3 = uv2016.Date == "03/12/2016"

print ((uv2016[filteredData3])['UV_Index'].max())



filteredData4 = uv2016.Date == "04/12/2016"

print ((uv2016[filteredData4])['UV_Index'].max())



filteredData5 = uv2016.Date == "05/12/2016"

print ((uv2016[filteredData5])['UV_Index'].max())



filteredData6 = uv2016.Date == "06/12/2016"

print ((uv2016[filteredData6])['UV_Index'].max())



filteredData7 = uv2016.Date == "07/12/2016"

print ((uv2016[filteredData7])['UV_Index'].max())



filteredData8 = uv2016.Date == "08/12/2016"

print ((uv2016[filteredData8])['UV_Index'].max())



filteredData9 = uv2016.Date == "09/12/2016"

print ((uv2016[filteredData9])['UV_Index'].max())



filteredData10 = uv2016.Date == "10/12/2016"

print ((uv2016[filteredData10])['UV_Index'].max())



filteredData11 = uv2016.Date == "11/12/2016"

print ((uv2016[filteredData11])['UV_Index'].max())



filteredData12 = uv2016.Date == "12/12/2016"

print ((uv2016[filteredData12])['UV_Index'].max())



filteredData13 = uv2016.Date == "13/12/2016"

print ((uv2016[filteredData13])['UV_Index'].max())



filteredData14 = uv2016.Date == "14/12/2016"

print ((uv2016[filteredData14])['UV_Index'].max())



filteredData15 = uv2016.Date == "15/12/2016"

print ((uv2016[filteredData15])['UV_Index'].max())



filteredData16 = uv2016.Date == "16/12/2016"

print ((uv2016[filteredData16])['UV_Index'].max())



filteredData17 = uv2016.Date == "17/12/2016"

print ((uv2016[filteredData17])['UV_Index'].max())



filteredData18 = uv2016.Date == "18/12/2016"

print ((uv2016[filteredData18])['UV_Index'].max())



filteredData19 = uv2016.Date == "19/12/2016"

print ((uv2016[filteredData19])['UV_Index'].max())



filteredData20 = uv2016.Date == "20/12/2129"

print ((uv2016[filteredData20])['UV_Index'].max())



filteredData21 = uv2016.Date == "21/12/2016"

print ((uv2016[filteredData21])['UV_Index'].max())



filteredData22 = uv2016.Date == "22/12/2016"

print ((uv2016[filteredData22])['UV_Index'].max())



filteredData23 = uv2016.Date == "23/12/2016"

print ((uv2016[filteredData23])['UV_Index'].max())



filteredData24 = uv2016.Date == "24/12/2016"

print ((uv2016[filteredData24])['UV_Index'].max())



filteredData25 = uv2016.Date == "25/12/2016"

print ((uv2016[filteredData25])['UV_Index'].max())



filteredData26 = uv2016.Date == "26/12/2016"

print ((uv2016[filteredData26])['UV_Index'].max())



filteredData27 = uv2016.Date == "27/12/2016"

print ((uv2016[filteredData27])['UV_Index'].max())



filteredData28 = uv2016.Date == "28/12/2016"

print ((uv2016[filteredData28])['UV_Index'].max())



filteredData29 = uv2016.Date == "29/12/2016"

print ((uv2016[filteredData29])['UV_Index'].max())



filteredData30 = uv2016.Date == "30/12/2016"

print ((uv2016[filteredData30])['UV_Index'].max())



filteredData31 = uv2016.Date == "31/12/2016"

print ((uv2016[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2017 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2017.csv')



filteredData1 = uv2017.Date == "01/12/2017"

print((uv2017[filteredData1])['UV_Index'].max())



filteredData2 = uv2017.Date == "02/12/2017"

print ((uv2017[filteredData2])['UV_Index'].max())



filteredData3 = uv2017.Date == "03/12/2017"

print ((uv2017[filteredData3])['UV_Index'].max())



filteredData4 = uv2017.Date == "04/12/2017"

print ((uv2017[filteredData4])['UV_Index'].max())



filteredData5 = uv2017.Date == "05/12/2017"

print ((uv2017[filteredData5])['UV_Index'].max())



filteredData6 = uv2017.Date == "06/12/2017"

print ((uv2017[filteredData6])['UV_Index'].max())



filteredData7 = uv2017.Date == "07/12/2017"

print ((uv2017[filteredData7])['UV_Index'].max())



filteredData8 = uv2017.Date == "08/12/2017"

print ((uv2017[filteredData8])['UV_Index'].max())



filteredData9 = uv2017.Date == "09/12/2017"

print ((uv2017[filteredData9])['UV_Index'].max())



filteredData10 = uv2017.Date == "10/12/2017"

print ((uv2017[filteredData10])['UV_Index'].max())



filteredData11 = uv2017.Date == "11/12/2017"

print ((uv2017[filteredData11])['UV_Index'].max())



filteredData12 = uv2017.Date == "12/12/2017"

print ((uv2017[filteredData12])['UV_Index'].max())



filteredData13 = uv2017.Date == "13/12/2017"

print ((uv2017[filteredData13])['UV_Index'].max())



filteredData14 = uv2017.Date == "14/12/2017"

print ((uv2017[filteredData14])['UV_Index'].max())



filteredData15 = uv2017.Date == "15/12/2017"

print ((uv2017[filteredData15])['UV_Index'].max())



filteredData16 = uv2017.Date == "16/12/2017"

print ((uv2017[filteredData16])['UV_Index'].max())



filteredData17 = uv2017.Date == "17/12/2017"

print ((uv2017[filteredData17])['UV_Index'].max())



filteredData18 = uv2017.Date == "18/12/2017"

print ((uv2017[filteredData18])['UV_Index'].max())



filteredData19 = uv2017.Date == "19/12/2017"

print ((uv2017[filteredData19])['UV_Index'].max())



filteredData20 = uv2017.Date == "20/12/2129"

print ((uv2017[filteredData20])['UV_Index'].max())



filteredData21 = uv2017.Date == "21/12/2017"

print ((uv2017[filteredData21])['UV_Index'].max())



filteredData22 = uv2017.Date == "22/12/2017"

print ((uv2017[filteredData22])['UV_Index'].max())



filteredData23 = uv2017.Date == "23/12/2017"

print ((uv2017[filteredData23])['UV_Index'].max())



filteredData24 = uv2017.Date == "24/12/2017"

print ((uv2017[filteredData24])['UV_Index'].max())



filteredData25 = uv2017.Date == "25/12/2017"

print ((uv2017[filteredData25])['UV_Index'].max())



filteredData26 = uv2017.Date == "26/12/2017"

print ((uv2017[filteredData26])['UV_Index'].max())



filteredData27 = uv2017.Date == "27/12/2017"

print ((uv2017[filteredData27])['UV_Index'].max())



filteredData28 = uv2017.Date == "28/12/2017"

print ((uv2017[filteredData28])['UV_Index'].max())



filteredData29 = uv2017.Date == "29/12/2017"

print ((uv2017[filteredData29])['UV_Index'].max())



filteredData30 = uv2017.Date == "30/12/2017"

print ((uv2017[filteredData30])['UV_Index'].max())



filteredData31 = uv2017.Date == "31/12/2017"

print ((uv2017[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2018 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2018.csv')



filteredData1 = uv2018.Date == "01/12/2018"

print((uv2018[filteredData1])['UV_Index'].max())



filteredData2 = uv2018.Date == "02/12/2018"

print ((uv2018[filteredData2])['UV_Index'].max())



filteredData3 = uv2018.Date == "03/12/2018"

print ((uv2018[filteredData3])['UV_Index'].max())



filteredData4 = uv2018.Date == "04/12/2018"

print ((uv2018[filteredData4])['UV_Index'].max())



filteredData5 = uv2018.Date == "05/12/2018"

print ((uv2018[filteredData5])['UV_Index'].max())



filteredData6 = uv2018.Date == "06/12/2018"

print ((uv2018[filteredData6])['UV_Index'].max())



filteredData7 = uv2018.Date == "07/12/2018"

print ((uv2018[filteredData7])['UV_Index'].max())



filteredData8 = uv2018.Date == "08/12/2018"

print ((uv2018[filteredData8])['UV_Index'].max())



filteredData9 = uv2018.Date == "09/12/2018"

print ((uv2018[filteredData9])['UV_Index'].max())



filteredData10 = uv2018.Date == "10/12/2018"

print ((uv2018[filteredData10])['UV_Index'].max())



filteredData11 = uv2018.Date == "11/12/2018"

print ((uv2018[filteredData11])['UV_Index'].max())



filteredData12 = uv2018.Date == "12/12/2018"

print ((uv2018[filteredData12])['UV_Index'].max())



filteredData13 = uv2018.Date == "13/12/2018"

print ((uv2018[filteredData13])['UV_Index'].max())



filteredData14 = uv2018.Date == "14/12/2018"

print ((uv2018[filteredData14])['UV_Index'].max())



filteredData15 = uv2018.Date == "15/12/2018"

print ((uv2018[filteredData15])['UV_Index'].max())



filteredData16 = uv2018.Date == "16/12/2018"

print ((uv2018[filteredData16])['UV_Index'].max())



filteredData17 = uv2018.Date == "17/12/2018"

print ((uv2018[filteredData17])['UV_Index'].max())



filteredData18 = uv2018.Date == "18/12/2018"

print ((uv2018[filteredData18])['UV_Index'].max())



filteredData19 = uv2018.Date == "19/12/2018"

print ((uv2018[filteredData19])['UV_Index'].max())



filteredData20 = uv2018.Date == "20/12/2129"

print ((uv2018[filteredData20])['UV_Index'].max())



filteredData21 = uv2018.Date == "21/12/2018"

print ((uv2018[filteredData21])['UV_Index'].max())



filteredData22 = uv2018.Date == "22/12/2018"

print ((uv2018[filteredData22])['UV_Index'].max())



filteredData23 = uv2018.Date == "23/12/2018"

print ((uv2018[filteredData23])['UV_Index'].max())



filteredData24 = uv2018.Date == "24/12/2018"

print ((uv2018[filteredData24])['UV_Index'].max())



filteredData25 = uv2018.Date == "25/12/2018"

print ((uv2018[filteredData25])['UV_Index'].max())



filteredData26 = uv2018.Date == "26/12/2018"

print ((uv2018[filteredData26])['UV_Index'].max())



filteredData27 = uv2018.Date == "27/12/2018"

print ((uv2018[filteredData27])['UV_Index'].max())



filteredData28 = uv2018.Date == "28/12/2018"

print ((uv2018[filteredData28])['UV_Index'].max())



filteredData29 = uv2018.Date == "29/12/2018"

print ((uv2018[filteredData29])['UV_Index'].max())



filteredData30 = uv2018.Date == "30/12/2018"

print ((uv2018[filteredData30])['UV_Index'].max())



filteredData31 = uv2018.Date == "31/12/2018"

print ((uv2018[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2019 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2019.csv')



filteredData1 = uv2019.Date == "01/12/2019"

print((uv2019[filteredData1])['UV_Index'].max())



filteredData2 = uv2019.Date == "02/12/2019"

print ((uv2019[filteredData2])['UV_Index'].max())



filteredData3 = uv2019.Date == "03/12/2019"

print ((uv2019[filteredData3])['UV_Index'].max())



filteredData4 = uv2019.Date == "04/12/2019"

print ((uv2019[filteredData4])['UV_Index'].max())



filteredData5 = uv2019.Date == "05/12/2019"

print ((uv2019[filteredData5])['UV_Index'].max())



filteredData6 = uv2019.Date == "06/12/2019"

print ((uv2019[filteredData6])['UV_Index'].max())



filteredData7 = uv2019.Date == "07/12/2019"

print ((uv2019[filteredData7])['UV_Index'].max())



filteredData8 = uv2019.Date == "08/12/2019"

print ((uv2019[filteredData8])['UV_Index'].max())



filteredData9 = uv2019.Date == "09/12/2019"

print ((uv2019[filteredData9])['UV_Index'].max())



filteredData10 = uv2019.Date == "10/12/2019"

print ((uv2019[filteredData10])['UV_Index'].max())



filteredData11 = uv2019.Date == "11/12/2019"

print ((uv2019[filteredData11])['UV_Index'].max())



filteredData12 = uv2019.Date == "12/12/2019"

print ((uv2019[filteredData12])['UV_Index'].max())



filteredData13 = uv2019.Date == "13/12/2019"

print ((uv2019[filteredData13])['UV_Index'].max())



filteredData14 = uv2019.Date == "14/12/2019"

print ((uv2019[filteredData14])['UV_Index'].max())



filteredData15 = uv2019.Date == "15/12/2019"

print ((uv2019[filteredData15])['UV_Index'].max())



filteredData16 = uv2019.Date == "16/12/2019"

print ((uv2019[filteredData16])['UV_Index'].max())



filteredData17 = uv2019.Date == "17/12/2019"

print ((uv2019[filteredData17])['UV_Index'].max())



filteredData18 = uv2019.Date == "18/12/2019"

print ((uv2019[filteredData18])['UV_Index'].max())



filteredData19 = uv2019.Date == "19/12/2019"

print ((uv2019[filteredData19])['UV_Index'].max())



filteredData20 = uv2019.Date == "20/12/2129"

print ((uv2019[filteredData20])['UV_Index'].max())



filteredData21 = uv2019.Date == "21/12/2019"

print ((uv2019[filteredData21])['UV_Index'].max())



filteredData22 = uv2019.Date == "22/12/2019"

print ((uv2019[filteredData22])['UV_Index'].max())



filteredData23 = uv2019.Date == "23/12/2019"

print ((uv2019[filteredData23])['UV_Index'].max())



filteredData24 = uv2019.Date == "24/12/2019"

print ((uv2019[filteredData24])['UV_Index'].max())



filteredData25 = uv2019.Date == "25/12/2019"

print ((uv2019[filteredData25])['UV_Index'].max())



filteredData26 = uv2019.Date == "26/12/2019"

print ((uv2019[filteredData26])['UV_Index'].max())



filteredData27 = uv2019.Date == "27/12/2019"

print ((uv2019[filteredData27])['UV_Index'].max())



filteredData28 = uv2019.Date == "28/12/2019"

print ((uv2019[filteredData28])['UV_Index'].max())



filteredData29 = uv2019.Date == "29/12/2019"

print ((uv2019[filteredData29])['UV_Index'].max())



filteredData30 = uv2019.Date == "30/12/2019"

print ((uv2019[filteredData30])['UV_Index'].max())



filteredData31 = uv2019.Date == "31/12/2019"

print ((uv2019[filteredData31])['UV_Index'].max())

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2010 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2010.csv')



filteredData29 = ((uv2010.Date >= "01/02/2010") &  (uv2010.Date <= "29/02/2010"))

# print (filteredData29)

print ((uv2010[filteredData29])['UV_Index'].max())
import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure



uv2016 = pd.read_csv('../input/tsv-uv-index/uv-townsville-2016.csv')



# filteredDataJan = uv2016[(uv2016["Date"] >= "01/01/2016") & (uv2016["Date"] <= "31/01/2016")]

# print (filteredDataJan)

# print((uv2016[filteredData1])['UV_Index'].max())



for i in range ("01/01/2016","31/01/2016"):

    ['UV_Index'].max()



#     filteredDataJan = uv2016.Date == "27/01/2017"

#     print ((uv2017[filteredDataJan])['UV_Index'].max())



# Date_and_Time = uv2010[(uv2010["Date"]=="01/01/2010") & (uv2008["Time"]>='06:00') & (uv2008["Time"]<='18:00')]

# print (Date_and_Time)



# filteredDataJan = uv2016[(uv2016["Date"] <= '31/01/2016')]

# (uv2016[filteredDataJan])['UV_Index'].max()

# (uv2016[filteredDataJan])['UV_Index'].value_counts().sort_index().plot.bar()

# print (filteredDataJan)
# Average daily maximum UV levels by month



import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure



data = {'Month':['Jan', 'Feb','Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], 

        'UV_Index':[14.0, 14.0, 12.3, 9.7, 7.3, 6.2, 6.6, 8.4, 10.4, 11.7, 12.9, 13.4]}



# new_ds = pd.DataFrame.from_dict(data)

new_ds = pd.DataFrame(data)

print(new_ds)



# new_ds['Percentage Value'] = ((new_ds['count'] / new_ds['count'].sum()) * 100).round(2)

# print(new_ds)



new_ds.plot(kind='bar',ylim =(0,15), x='Month',y='UV_Index', color='brown')

plt.title("Average daily maximum UV levels by month", y=1.05);

plt.ylabel("UV Index", labelpad=14)