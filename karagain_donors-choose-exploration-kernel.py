%matplotlib inline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))

# Univariate Table Exploration
def introexplore(csv):
    print(csv.info())
    
    _, length = csv.shape
    print('\n', 'Dataset Dimensions')
    print(_, length)

    for i in range(length):
        print('\n', csv.columns[i])
        print(csv.iloc[:, i].value_counts().head())
donors = pd.read_csv('../input/Donors.csv')
donors_c = donors.copy() # cleaned dataframe
introexplore(donors)
# Check whether the zipcode and city difference are correct.
donors.loc[donors['Donor Zip'] == 606 , 'Donor City'].value_counts()
donors['Donor Zip'].value_counts().head()
donors[donors['Donor Zip'] == 606 ]['Donor State'].value_counts()
donors[donors['Donor Zip'] == '606' ]['Donor State'].value_counts().head()
donors_c['Donor Zip'] = donors_c['Donor Zip'].astype(str)
donors_c[donors_c['Donor Zip'] == 606 ]['Donor State'].value_counts()
donors_c[donors_c['Donor Zip'] == '606' ]['Donor State'].value_counts().head()
# find a positive 606 value in original dataset
donors[donors['Donor Zip'] == 606 ].head()
# Pull up the same value in the "cleaned" dataset
donors_c.iloc[2097169, :]

donors_c.iloc[2097169, 4]
try:
    donors_c['Donor Zip'] = donors_c['Donor Zip'].astype(int).astype(str)
except:
    print("ValueError: invalid literal for int() with base 10: 'nan'")
donors_c = donors.copy()
donors_c.loc[donors_c['Donor Zip'].notnull(), 'Donor Zip'].value_counts().tail(20)
donors_c.loc[donors_c['Donor Zip']== 'n19', :]
del donors_c
prop = donors['Donor Is Teacher'].value_counts(normalize=True)
print(prop)
prop.plot(kind='bar')
plt.title('Is the Donor a Teacher?')
plt.show()
donors[donors['Donor Zip'].isnull()].count()
prop = donors.loc[donors['Donor Zip'].isnull(), 'Donor Is Teacher'].value_counts(normalize=True)
prop
prop.plot(kind='bar')
plt.title('Anonymous location, Is the Donor a Teacher?')
plt.show()
donors.loc[donors['Donor Zip'].isnull(), 'Donor State'].value_counts(normalize=True).head()
donors['Donor City'].value_counts().head(20).plot(kind='barh')
plt.title('Top 20 Cities with most Donors')
plt.xlabel('Number of Donors')
plt.gca().invert_yaxis()
donors['Donor State'].value_counts().head(20).plot(kind='barh')
plt.title('Top 20 States with most Donors')
plt.xlabel('Number of Donors')
plt.gca().invert_yaxis()
%reset
%matplotlib inline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))

# Univariate Table Exploration
def introexplore(csv):
    print(csv.info())
    
    _, length = csv.shape
    print('\n', 'Dataset Dimensions')
    print(_, length)

    for i in range(length):
        print('\n', csv.columns[i])
        print(csv.iloc[:, i].value_counts().head())
donations = pd.read_csv('../input/Donations.csv')
introexplore(donations)
fdonor = donations.groupby('Donor ID')['Donation Amount'].agg(['count', 'sum']).sort_values('count', ascending=False).head(10)
fdonor
fdonor.plot(kind='bar')
plt.ylim((0,100000))
donations.groupby('Donor ID')['Project ID'].value_counts().head(10)
donations['Donation Amount'].value_counts().head(10)
default = donations['Donation Amount'].value_counts().head(10)
default.plot(kind='bar')
plt.title('Donation Amount')
plt.ylabel('frequency')
plt.xlabel('Dollars Donated')
def percentdonationoptional(num):
    print('\n', f'{num} dollars')
    print(donations.loc[donations['Donation Amount'] == num, 'Donation Included Optional Donation'].value_counts())
    print('percentage')
    print(donations.loc[donations['Donation Amount'] == num, 'Donation Included Optional Donation'].value_counts(normalize=True))
print(donations['Donation Included Optional Donation'].value_counts())
print('percentage')
print(donations['Donation Included Optional Donation'].value_counts(normalize=True))
for i in default.index:
    percentdonationoptional(i)
donations["Donation Received Date"] = donations["Donation Received Date"].astype("datetime64[ns]")
# donations['Donation Received Date'].value_counts().plot(kind='bar')
# The code above will kill the computer since there are too many bars to create. Friendly Warning
donations['Donation Received Date'].dt.month.value_counts().sort_index().plot(kind='bar')
plt.title('Donations Received by Month')
plt.xlabel('Month')
plt.ylabel('Counts')
%reset
%matplotlib inline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))

# Univariate Table Exploration
def introexplore(csv):
    print(csv.info())
    
    _, length = csv.shape
    print('\n', 'Dataset Dimensions')
    print(_, length)

    for i in range(length):
        print('\n', csv.columns[i])
        print(csv.iloc[:, i].value_counts().head())
schools = pd.read_csv('../input/Schools.csv')
introexplore(schools)
# look at interesting columns
intcolumns = [2, 3, 4, 6, 7, 8]
def plotgrids(df, columns):
    
    fig = plt.figure(figsize=(15, 15))
    head = 10
    colnum = 3
    
    for i in range(len(columns)):
        
        increment = 0
        if len(columns)%colnum != 0:
            increment = 1
            
        rownum = len(columns)//colnum + increment
        graphnum = int(f'{rownum}{colnum}{i+1}')
        plt.subplot(graphnum)
        df.iloc[:, columns[i]].value_counts(normalize=True).head(head).plot(kind='bar')
        plt.title(df.iloc[:, columns[i]].name)
plotgrids(schools, intcolumns)
%matplotlib inline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))

# Univariate Table Exploration
def introexplore(csv):
    print(csv.info())
    
    _, length = csv.shape
    print('\n', 'Dataset Dimensions')
    print(_, length)

    for i in range(length):
        print('\n', csv.columns[i])
        print(csv.iloc[:, i].value_counts().head())
resources = pd.read_csv('../input/Resources.csv')
introexplore(resources)
%matplotlib inline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))

# Univariate Table Exploration
def introexplore(csv):
    print(csv.info())
    
    _, length = csv.shape
    print('\n', 'Dataset Dimensions')
    print(_, length)

    for i in range(length):
        print('\n', csv.columns[i])
        print(csv.iloc[:, i].value_counts().head())
teachers = pd.read_csv('../input/Teachers.csv')
introexplore(teachers)
%matplotlib inline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))

# Univariate Table Exploration
def introexplore(csv):
    print(csv.info())
    
    _, length = csv.shape
    print('\n', 'Dataset Dimensions')
    print(_, length)

    for i in range(length):
        print('\n', csv.columns[i])
        print(csv.iloc[:, i].value_counts().head())
projects = pd.read_csv('../input/Projects.csv', nrows=500)
introexplore(projects)





