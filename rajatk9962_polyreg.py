!cd /kaggle/input/covid19-global-forecasting-week-3/
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import csv

filename = "train.csv"

  

# initializing the titles and rows list 

fields = [] 

rows = [] 

  

# reading csv file 

with open('../input/covid19-global-forecasting-week-3/train.csv', 'r') as csvfile: 

    # creating a csv reader object 

    csvreader = csv.reader(csvfile) 

      

    # extracting field names through first row 

    fields = next(csvreader) 

  

    # extracting each data row one by one 

    for row in csvreader: 

        rows.append(row) 

  

    # get total number of rows 

    print("Total no. of rows: %d"%(csvreader.line_num)) 

countries=[]

datecol=[]

dates=[]

res=[]

for row in rows:

    row[2]=row[1]+' '+row[2]

    del row[1]

    #del row[1]

    del row[4]

    countries.append(row[1])

    dates.append(row[2])   

[res.append(x) for x in countries if x not in res] 

countries=res

res=[]

[res.append(x) for x in dates if x not in res] 

dates=res

with open('/kaggle/working/cases_split.csv', 'w', newline='') as csvfile: 

    csvwriter = csv.writer(csvfile)

    countries[:0] = ['Date']

    csvwriter.writerow(countries)

    for date in dates:

        date=[date]

        datecol.append(date)



    for date in dates :

        for row in rows:

        

            temprow=[]

            if row[2]==date:

                for datecolrow in datecol:

                    if datecolrow[0]==date:

                        if type(datecolrow)==str:

                            temprow=[datecolrow]

                        else:

                            temprow=datecolrow

                        temprow.append(row[3]) 



                        datecolrow=temprow  

                    

    csvwriter.writerows(datecol)
!pwd
!ls

import csv

filename = "../input/covid19-global-forecasting-week-3/train.csv"

  

# initializing the titles and rows list 

fields = [] 

rows = [] 

  

# reading csv file 

with open(filename, 'r') as csvfile: 

    # creating a csv reader object 

    csvreader = csv.reader(csvfile) 

      

    # extracting field names through first row 

    fields = next(csvreader) 

  

    # extracting each data row one by one 

    for row in csvreader: 

        rows.append(row) 

  

    # get total number of rows 

    print("Total no. of rows: %d"%(csvreader.line_num)) 

countries=[]

datecol=[]

dates=[]

res=[]

for row in rows:

    row[2]=row[1]+' '+row[2]

    del row[1]

    #del row[1]

    del row[3]

    countries.append(row[1])

    dates.append(row[2])   

[res.append(x) for x in countries if x not in res] 

countries=res

res=[]

[res.append(x) for x in dates if x not in res] 

dates=res

with open('/kaggle/working/deaths_split.csv', 'w', newline='') as csvfile: 

    csvwriter = csv.writer(csvfile)

    countries[:0] = ['Date']

    csvwriter.writerow(countries)

    for date in dates:

        date=[date]

        datecol.append(date)



    for date in dates :

        for row in rows:

        

            temprow=[]

            if row[2]==date:

                for datecolrow in datecol:

                    if datecolrow[0]==date:

                        if type(datecolrow)==str:

                            temprow=[datecolrow]

                        else:

                            temprow=datecolrow

                        temprow.append(row[3]) 



                        datecolrow=temprow  

                    

    csvwriter.writerows(datecol)
# Polynomial Regression

%matplotlib inline

# Importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import csv

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression

# Importing the dataset

dataset = pd.read_csv('/kaggle/working/cases_split.csv')

#X = dataset.iloc[:, 0:1].values

X=[]

result=[]



def listtostr(s):  

    res = float(str(s)[1:-1])

    return (res)



for i in range(1,len(dataset.columns)):

    X=[]

    y=[]

    y = dataset.iloc[:, i].values

    for n in range(1,len(y)+1):

        X.append([n])

    #print(y)

    #print(X)

    #X = X[40:]

    #y = y[40:]

    # Fitting Polynomial Regression to the dataset

    

    poly_reg = PolynomialFeatures(degree = 4)

    X_poly = poly_reg.fit_transform(X)

    poly_reg.fit(X_poly, y)

    lin_reg_2 = LinearRegression()

    lin_reg_2.fit(X_poly, y)

    

    

    



    # Visualising the Polynomial Regression results

#     plt.scatter(X, y, color = 'red')

#     predline=[]

#     for n in range(1,108):

#         predline.append([n])

#     #predline = predline[40:]    

#     plt.plot(predline, lin_reg_2.predict(poly_reg.fit_transform(predline)), color = 'blue')

    

#     plt.title(dataset.columns[i])

#     plt.xlabel('Days')

#     plt.ylabel('Cases')

#     plt.show()

    for n in range(65,108):

        result.append(listtostr(lin_reg_2.predict(poly_reg.fit_transform([[n]]))))

    

ids=[]        

for i in range(1,len(result)+1):

    ids.append(i)

    

df = pd.DataFrame()

df["ForecastId"] = pd.Series(ids)

df["ConfirmedCases"] = pd.Series(result)

# Polynomial Regression

%matplotlib inline

# Importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import csv

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression

# Importing the dataset

dataset = pd.read_csv('/kaggle/working/deaths_split.csv')

#X = dataset.iloc[:, 0:1].values

X=[]

result=[]



def listtostr(s):  

    res = float(str(s)[1:-1])

    return (res)



for i in range(1,len(dataset.columns)):

    X=[]

    y=[]

    y = dataset.iloc[:, i].values

    for n in range(1,len(y)+1):

        X.append([n])

    #print(y)

    #print(X)

    #X = X[40:]

    #y = y[40:]

    # Fitting Polynomial Regression to the dataset

    

    poly_reg = PolynomialFeatures(degree = 4)

    X_poly = poly_reg.fit_transform(X)

    poly_reg.fit(X_poly, y)

    lin_reg_2 = LinearRegression()

    lin_reg_2.fit(X_poly, y)

    

    

    



    # Visualising the Polynomial Regression results

#     plt.scatter(X, y, color = 'red')

#     predline=[]

#     for n in range(1,108):

#         predline.append([n])

#     #predline = predline[40:]    

#     plt.plot(predline, lin_reg_2.predict(poly_reg.fit_transform(predline)), color = 'blue')

    

#     plt.title(dataset.columns[i])

#     plt.xlabel('Days')

#     plt.ylabel('Cases')

#     plt.show()

    for n in range(65,108):

        result.append(listtostr(lin_reg_2.predict(poly_reg.fit_transform([[n]]))))

        

df["Fatalities"] = pd.Series(result)

df.to_csv("submission.csv", index=False)


