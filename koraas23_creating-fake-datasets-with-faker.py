# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
!pip install Faker 

from faker import Faker
# Create fake data using Faker

fake_data = Faker()
# Let's create a list of 200 fake names first

def create_names(n):

    name = []

    for _ in range(0, n):

        name.append(fake_data.name())

    return name
name = create_names(200)
# Check the first 10 results

name[:10]
# Create a list of countries

def create_country(n):

    nation = []

    for _ in range(0, n):

        nation.append(fake_data.country())

    return nation
country = create_country(200)
country[:10]
# Create a list of age from 21 to 98 (integers)

age = np.random.randint(21, 99, size = 200, dtype = 'int')
# Create a list fo fake dates

def create_date(n):

    member_since = []

    for _ in range(0, n):

        member_since.append(fake_data.date_this_century())

    return member_since
date = create_date(200)
# Create a list of fake jobs

def create_job(n):

    job = []

    for _ in range(0, n):

        job.append(fake_data.job())

    return job
occupation = create_job(200)
# Create a list of fake credit card types

def create_credit(n):

    card = []

    for _ in range(0, n):

        card.append(fake_data.credit_card_provider())

    return card

card = create_credit(200)
# Create a fake location in US

def create_location(n):

    location = []

    for _ in range(0, n):

        location.append(fake_data.local_latlng(country_code = "US"))

    return location
location = create_location(200)
# Create a list of values 0 and 1, where 1 = married

married = np.random.randint(0,2, size = 200)
# Create a list of randomly picked races from the list of 4 races

def create_race(n):

    race = []

    for _ in range(0, n):

        race.append(fake_data.random_element(elements = ("White", "Hispanic", "Black", "Asian")))

    return race
races = create_race(200)
# Now create 3 list of integers just to have some numerical data

salary = np.random.randint(10000, 120000, size = 200, dtype = 'int')

savings = np.random.randint(0, 50000, size = 200, dtype = 'int')

rent = np.random.randint(500, 3001, size = 200, dtype = 'int')
# Now, connect all columns into one data set

data = pd.DataFrame(list(zip(name, country, age, date, occupation, card, location, married,

                            races, salary, savings, rent)),

                   columns = ['Name', 'Place_of_Birth', 'Age', 'Member_Since', 'Job',

                              'Card', 'Location', 'Married', 'Race', 'Salary', 'Savings',

                              'Rent'])
# Let's see how the data looks like

data.head()
data.info()
# Perfect, let's create another set based on time

datelist = pd.date_range('2010-01-01', periods = 200)

sales = np.random.randint(100, 1100, size = 200, dtype = 'int')

profit = np.random.uniform(10000, 1000000, size = 200)



data2 = pd.DataFrame(list(zip(datelist, sales, profit)),

                   columns = ['Date', 'Sales', 'Profit'])
data2.head()