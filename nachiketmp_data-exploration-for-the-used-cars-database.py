# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



import matplotlib.pyplot as plt

% matplotlib inline

import datetime

import calendar

import seaborn as sns
#Setting the datetime format while importing data

fmt = '%Y-%m-%d %H:%M:%S'

dateparse = lambda dates: pd.datetime.strptime(dates, fmt)

rawautodf = pd.read_csv('../input/autos.csv', sep=',',encoding='Latin1',parse_dates = ['dateCrawled','dateCreated','lastSeen'], date_parser = dateparse)
rawautodf.head()
print ("The original dtype of dateCrawled is " + str(rawautodf.dateCrawled.dtype)) # checked datatype

#Let's keep it
autodf = rawautodf.drop('name',axis = 1)

print( "Size of the dataset - " + str(len(autodf)))
print(autodf.groupby('seller').size())

#We can live without the 3 gewerblich. Let's analyze the data with only the private sellers

autodf = autodf[autodf['seller'] != 'gewerblich']

autodf = autodf.drop('seller',axis = 1)

print( "\nSize of the dataset - " + str(len(autodf)))
print(autodf.groupby('offerType').size())

#Same here. Let's drop the 12

autodf = autodf[autodf['offerType'] != 'Gesuch']

autodf = autodf.drop('offerType',axis = 1)

print( "\nSize of the dataset - " + str(len(autodf)))
#plt.plot()

#autodf.price.plot(kind='hist',bins=100)

#plt.show()

sns.boxplot(autodf.price)
# are there any cars with 0 value?

print ("How low priced priced cars are there?  " + str(len(autodf[autodf.price < 100 ])))

print ("\nHow many highly price-listed cars are there?  " + str(len(autodf[autodf.price > 150000 ])))

#print autodf[(autodf.price < 0) | (autodf.price > 150000)]

print ("\nWell, that's ridiculous. Let's get rid of them.")

autodf = autodf[autodf.price > 100]

autodf = autodf[autodf.price < 50000]

print( "\nSize of the dataset - " + str(len(autodf)))
#plt.plot()

#autodf.price.plot(kind='hist',bins=100)

#plt.legend()

#plt.show()

sns.boxplot(autodf.price)
sns.distplot(autodf.price)
autodf = autodf.drop('abtest',1)
autodf.info()
p = sns.factorplot('vehicleType',data=autodf,kind='count')

p.set_xticklabels(rotation=30) #letitbe
print( "\nSize of the dataset - " + str(len(autodf)))

# I only want to consier for 1980 to 2017

autodf = autodf[(autodf.yearOfRegistration >= 1990) & (autodf.yearOfRegistration < 2017)]

print( "\nSize of the dataset - " + str(len(autodf)))
p = sns.factorplot('yearOfRegistration',data=autodf,kind='count')

p.set_xticklabels(rotation=90)
autodf.gearbox = autodf.gearbox.astype('category')

plt.figure(1)

plt.plot()

autodf.gearbox.astype('category').value_counts().plot(kind='bar')

plt.show()



sns.factorplot('gearbox',data=autodf,kind='count',hue='yearOfRegistration')
group = autodf.groupby('yearOfRegistration')

temp_df = group.gearbox.value_counts()

#temp_df.head(1)

plt.plot()

temp_df.plot(kind='bar')

plt.show()
print ((str((float(len(autodf.powerPS[autodf.powerPS > 300])) / len(autodf.powerPS)) * 100) + " %"))

print (len(autodf.powerPS[autodf.powerPS > 300]))

#print len(autodf.powerPS)

print ("Number of zeros = " + str(len(autodf.powerPS[autodf.powerPS == 0])))



print( "\nSize of the dataset - " + str(len(autodf)))

autodf = autodf[(autodf.powerPS <= 300) & (autodf.powerPS > 50)]

print( "\nSize of the dataset - " + str(len(autodf)))
#autodf.powerPS.hist(bins=100)

sns.distplot(autodf.powerPS)
sns.boxplot(autodf.powerPS)
autodf.model = autodf.model.astype('category')
sns.factorplot('model',data=autodf,kind='count',size=10)

#toomuch data. Let's keep it anyway. We can try to implement machine learning algorithms (I'm a noob)
print (autodf.kilometer.count())

print (max(autodf.kilometer))

print (min(autodf.kilometer))

sns.boxplot(autodf.kilometer)
autodf['monthOfRegistration'] = autodf.monthOfRegistration.astype('category')
sns.factorplot('monthOfRegistration',data=autodf,kind='count')
#sns.boxplot(autodf.monthOfRegistration)
sns.factorplot('fuelType',data=autodf,kind='count')
p = sns.factorplot('brand',kind='count',data=autodf,size=10)

p.set_xticklabels(rotation=90)
x = pd.DataFrame(autodf.notRepairedDamage.value_counts())

print ("Percentage of cars not repaired = " + str(round( x.ix['ja'] * 100 /x.ix['nein'],2)) + " %")
sns.factorplot('notRepairedDamage',data=autodf,kind='count')
#
#autodf.postalCode.value_counts().plot(kind='bar)
#Letitbe
#
#