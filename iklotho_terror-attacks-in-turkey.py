# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import shapefile



import matplotlib.pyplot as plt

%matplotlib inline

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/gtd/globalterrorismdb_0617dist.csv',

                  encoding='ISO-8859-1',

                  usecols=[0, 1, 2, 3, 8, 11, 13, 14,26,27, 35,58, 84, 100, 103])
#renaming selected columns

data = data.rename(columns={'eventid':'id', 'iyear':'year', 'imonth':'month', 'iday':'day',

             'country_txt':'country', 'provstate':'state', 'targtype1_txt':'target',

             'weapsubtype1_txt':'weapon','gname':'name', 'nkillter':'fatalities', 'nwoundte':'injuries'})
data.head()
data['fatalities'] = data['fatalities'].fillna(0).astype(int)

data['injuries'] = data['injuries'].fillna(0).astype(int)

terror_turkey = data[(data.country == 'Turkey')]
terror_turkey.head()
terror_turkey['day'][terror_turkey.day == 0] = 1

terror_turkey['month'][terror_turkey.month == 0] = 1

terror_turkey['date'] = pd.to_datetime(terror_turkey[['day', 'month', 'year']])
terror_turkey.head()
terror_turkey = terror_turkey.sort_values(['fatalities','injuries'],ascending=False) #sorting by values
terror_turkey_na = terror_turkey.dropna()
counted = terror_turkey_na.groupby(terror_turkey_na.state).size()

sumofcounted = counted.sort_values(ascending=False).sum()

terror_rate = counted.sort_values(ascending=False) / sumofcounted

print (terror_rate.head()) # probability of a province getting attacked by terrorist

print ("most safest provine: " + terror_rate.reset_index().sort_values(by=0,ascending=True).iloc[0]['state'])
terror_turkey.groupby('name').size().sort_values(ascending=False)[:10]
grouped_year = terror_turkey.groupby('year').size()

grouped_year.plot()
def xrange(x):



    return iter(range(x))


def readLoc(fname):

    f = open(fname, 'r')

    lines = f.readlines()

    f.close()

    x = list()

    y = list()

    for line in lines:

        comp = line.strip('\n').split(',')

        x.append(float(comp[1]))

        y.append(float(comp[0]))

    return x, y



## load shapefile

sf = shapefile.Reader('../input/turkeyshp/TUR_adm1.shp')

years = []

plt.figure(figsize=(18,6))

for shape in sf.shapeRecords():

    

    # end index of each components of map

    l = shape.shape.parts

    

    len_l = len(l)  # how many parts of countries i.e. land and islands

    x = [i[0] for i in shape.shape.points[:]] # list of latitude

    y = [i[1] for i in shape.shape.points[:]] # list of longitude

    l.append(len(x)) # ensure the closure of the last component

    for k in xrange(len_l):

        # draw each component of map.

        # l[k] to l[k + 1] is the range of points that make this component

        plt.plot(x[l[k]:l[k + 1]],y[l[k]:l[k + 1]], 'k-')



county_xs, county_ys = readLoc('../input/turkeyshp/user_density.txt')

plt.plot(county_xs,county_ys,'ro')

# display

plt.show()