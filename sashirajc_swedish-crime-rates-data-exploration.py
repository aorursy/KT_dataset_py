# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/reported.csv')
data.shape
plt.plot(data['Year'],data['crimes.total'])

plt.xlabel('Year')

plt.ylabel('Number of crimes')

plt.show()



plt.plot(data['Year'],data['population'])

plt.xlabel('Year')

plt.ylabel('Population')

plt.show()
plt.plot(data['Year'],data['rape'])

plt.plot(data['Year'],data['sexual.offenses'])

plt.plot(data['Year'],data['crimes.total'])

plt.xlabel('Year')

plt.ylabel('Number of crimes')

plt.show()



plt.plot(data['Year'],data['rape'])

plt.plot(data['Year'],data['sexual.offenses'])

plt.xlabel('Year')

plt.ylabel('Number of crimes')

plt.show()
plt.plot(data['Year'],data['robbery'])

plt.plot(data['Year'],data['burglary'])

plt.plot(data['Year'],data['vehicle.theft'])

plt.plot(data['Year'],data['house.theft'])

plt.plot(data['Year'],data['shop.theft'])

plt.plot(data['Year'],data['out.of.vehicle.theft'])

plt.xlabel('Year')

plt.ylabel('Number of thefts/robberies')

plt.legend()

plt.show()



plt.plot(data['Year'],data['robbery'])

plt.plot(data['Year'],data['burglary'])

plt.plot(data['Year'],data['vehicle.theft'])

plt.plot(data['Year'],data['house.theft'])

plt.plot(data['Year'],data['shop.theft'])

plt.plot(data['Year'],data['out.of.vehicle.theft'])

plt.plot(data['Year'],data['crimes.total'])

plt.xlabel('Year')

plt.ylabel('Number of thefts/robberies')

plt.legend()

plt.show()
plt.plot(data['Year'],data['narcotics'])

plt.plot(data['Year'],data['drunk.driving'])

plt.xlabel('Year')

plt.ylabel('Number of crimes')

plt.legend()

plt.show()



plt.plot(data['Year'],data['narcotics'])

plt.plot(data['Year'],data['drunk.driving'])

plt.plot(data['Year'],data['crimes.total'])

plt.xlabel('Year')

plt.ylabel('Number of crimes')

plt.legend()

plt.show()
data['Sexual Crimes'] = data['rape'] + data['sexual.offenses']

data['Theft'] = data['robbery'] + data['burglary'] + data['vehicle.theft'] + data['house.theft'] + data['shop.theft'] + data['out.of.vehicle.theft']

data['Personal Crimes'] = data['crimes.person'] + data['assault'] + data['murder']

data['Penal Code'] = data['crimes.penal.code'] + data['other.penal.crimes']

data['Substance Abuse'] = data['drunk.driving'] + data['narcotics']
plt.plot(data['Year'],data['Sexual Crimes'])

plt.plot(data['Year'],data['Theft'])

plt.plot(data['Year'],data['Personal Crimes'])

plt.plot(data['Year'],data['Penal Code'])

plt.plot(data['Year'],data['Substance Abuse'])

plt.plot(data['Year'],data['fraud'])

plt.plot(data['Year'],data['criminal.damage'])

plt.xlabel('Year')

plt.ylabel('Number of Crimes')

plt.legend()

plt.show()