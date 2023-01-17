# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import glob

import pandas

import pandasql



def find_file():

    return glob.glob("../input/**/*.xlsx", recursive=True)[0]



def run_sql(query):

    return pandasql.sqldf(query, globals())



Apartment = pandas.read_excel(find_file(), sheet_name="Apartment")

print(Apartment)



Apartment = run_sql("""

    select *

    from Apartment

    where ApartmentAmenities='Yes'

""")



print(Apartment)





Apartment = pandas.read_excel(find_file(), sheet_name="Apartment")

print(Apartment)



Broker = pandas.read_excel(find_file(), sheet_name="Broker")

print(Broker)



Apartment = run_sql("""

    select Apartment.ApartmentName,Apartment.ApartmentType, Apartment.BrokerID

    from Apartment

    INNER JOIN Broker ON Apartment.BrokerID=Broker.BrokerID

""")



print(Apartment)



Apartment = pandas.read_excel(find_file(), sheet_name="Apartment")

print(Apartment)



AmenitiesID = pandas.read_excel(find_file(), sheet_name="AmenitiesID")

print(AmenitiesID)



Broker = pandas.read_excel(find_file(), sheet_name="Broker")

print(Broker)



Location = pandas.read_excel(find_file(), sheet_name="Location")

print(Location)



Price = pandas.read_excel(find_file(), sheet_name="Price")

print(Price)



run_sql("""

    select *

    from Apartment,AmenitiesID,Broker,Location,Price

""")



print


