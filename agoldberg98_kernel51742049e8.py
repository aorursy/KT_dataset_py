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



CastMember = pandas.read_excel(find_file(), sheet_name="CastMember")

print(CastMember)



CastMember = run_sql("""

    select *

    from CastMember

    where CastMemberPreviousMarriages='yes'

""")



print(CastMember)





CastMember = pandas.read_excel(find_file(), sheet_name="CastMember")

print(CastMember)



Country = pandas.read_excel(find_file(), sheet_name="Country")

print(Country)



CastMember = run_sql("""

    select CastMember.CastMember,CastMember.CastMemberAge,CastMember.CountryID

    from CastMember

    INNER JOIN Country ON CastMember.CountryID=Country.CountryID

""")



print(CastMember)



CastMember = pandas.read_excel(find_file(), sheet_name="CastMember")

print(CastMember)



Country = pandas.read_excel(find_file(), sheet_name="Country")

print(Country)



Relationship = pandas.read_excel(find_file(), sheet_name="Relationship")

print(Relationship)



Industry = pandas.read_excel(find_file(), sheet_name="Industry")

print(Industry)



CommunicationStyle = pandas.read_excel(find_file(), sheet_name="CommunicationStyle")

print(CommunicationStyle)



run_sql("""

    select *

    from CastMember,Country,Relationship,Industry,CommunicationStyle

""")



print
