# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input/flavors_of_cacao.csv"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

df = pd.read_csv("../input/flavors_of_cacao.csv")
divider = 70

rowcnt = round((df.shape[0] * divider) / 100)
cities = []

def getCity(c):

  if(c in cities):

    return cities.index(c)

  else:

    return -1

  

def addCity(c):

  if not (c in cities):

    cities.append(c)



companies = []

def getCompany(c):

  if(c in companies):

    return companies.index(c)

  else:

    return -1

  

def addCompany(c):

  if not (c in companies):

    companies.append(c)



origins = []

def getOrigin(o):

  if(o in origins):

    return origins.index(o)

  else:

    return -1

  

def addOrigin(o):

  if not (o in origins):

    origins.append(o)
X = []

Y = []



newDf = df[:rowcnt]



for index, row in newDf.iterrows():

  percentStr = row['Cocoa\nPercent']

  rating = round(row['Rating'])

  location = row['Company\nLocation']

  company = row[0]

  origin = row[1]



  addCity(location)

  addCompany(company)

  addOrigin(origin)

  

  percent = float(percentStr.replace("%",""));

  X.append([percent, getCity(location), getCompany(company), getOrigin(origin)])

  Y.append(rating)
from sklearn import svm

clf = svm.SVC()

clf.fit(X, Y)
successCount  = 0;

failCount = 0;

testDF = df[rowcnt:]

for index, row in testDF.iterrows():

  percentStr = row['Cocoa\nPercent']

  rating = round(row['Rating'])

  location = row['Company\nLocation']

  company = row[0]

  origin = row[1]



  percent = float(percentStr.replace("%",""));

  

  pre = clf.predict([[percent, getCity(location), getCompany(company), getOrigin(origin)]])[0]

  

  if(pre==rating):

    successCount = successCount + 1

  else:

    failCount = failCount + 1

  

suPercent = (successCount / (successCount + failCount))*100



print(suPercent)