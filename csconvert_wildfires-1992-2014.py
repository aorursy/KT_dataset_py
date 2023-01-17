# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sqlite3 as sql #To interface with SQL



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
conn = sql.connect(r'../input/FPA_FOD_20170508.sqlite')  #SQL connection to the sqlite file

c = conn.cursor()  #Cursor pointing to my connection object



container = c.execute('SELECT * FROM Fires;')  #Data pull from sqlite into python

WildFireData = container.fetchall()  



columns = c.execute('PRAGMA table_info(Fires);')  #Data pull to get the columns. My initial data pull doesn't grab the column lables.

WildFireColumns = columns.fetchall()



FireColumns = list(zip(*WildFireColumns))  #Strip the columns from all of the other information I don't need

print(FireColumns[1])  #Display my columns for reference
WildFires = pd.DataFrame(WildFireData)  #Turn data pull into a DaraFrame



WildFires.columns = FireColumns[1]  #Rename my columns to the column labels I pulled



Discovery_Date_Column = pd.Series(WildFires.DISCOVERY_DATE)  #The date columns were in Julian Days. 

                                                             #I need to pull them and rename them to change them and update my DataFrame

Cont_Date_Column = pd.Series(WildFires.CONT_DATE)

Discovery_Date_Column.rename("DISCOVERY_DATE")

Cont_Date_Column.rename("CONT_DATE")
def convertJD(JD):      #This function I converted from one I found on Wikipedia. It converts the Julian days to a calendar format that me and Tableau are more

    B = 274277          #familar with.

    C = (-38)

    j = 1401

    y = 4716

    m = 2

    n = 12

    r = 4

    p = 1461

    v = 3

    u = 5

    s = 153

    w = 2

    

    f = JD + j + (((4 * JD + B) // 146097) * 3) // 4 + C

    e = r * f + v

    g = (e % p) // r

    h = u * g + w

    D = (h % s) // u + 1

    M = ((h // s + m) % n) + 1

    Y = (e // p) - y + (n + m - M) // n

    return (str(int(D)) + '/' + str(int(M)) + '/' + str(int(Y)))



Discovery_Date_Column = pd.DataFrame(WildFires.DISCOVERY_DATE.apply(convertJD))

Cont_Date_Null = WildFires.CONT_DATE.notnull().values

Cont_Date_Column = WildFires.CONT_DATE[Cont_Date_Null].apply(convertJD)

#Cont_Date_Column = WildFires.DISCOVERY_DATE[WildFires.CONT_DATE.isnull()].values

#Cont_Date_Column = pd.DataFrame(WildFires.CONT_DATE.fillna(0).apply(convertJD))
WildFires.update(Discovery_Date_Column)  #These statements update my date column with the converted dates, and write the dataframe to a .csv file.

WildFires.update(Cont_Date_Column)

WildFires.to_csv('WildFires.csv',index=False)