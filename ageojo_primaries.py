# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

#Ignore the seaborn warnings.
import warnings
warnings.filterwarnings("ignore");

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats

#Import data and see what states it has values for so far.
df = pd.read_csv('../input/primary_results.csv')
df.state.unique()
df.head()
import sqlalchemy
from sqlalchemy import select

# establishconnection w. sqlite db
db_name = "database.sqlite"
# actually first param of .read_sql only accepts queries or table namess

engine = create_engine('sqlite:///' + db_name)
session = sessionmaker()
session.configure(bind=engine) #in gfm had,  MegaData(bind=engine)
#Base.metadata.create_all(engine)  #for when setting up db?
metadata = MetaData(bind=engine)
tables = ["primary_results", "county_facts", "county_facts_dictionary"]

dbs = ["primary_results", "county_facts", "county_facts_dictionary"]


primary_results = pd.read_sql_table(table_name=tables[0], con=engine,\
                              columns=["state", "state_abbreviation", "county", "fips", "party", "candidate", "votes", "fraction_votes"]) 



import os
#path = "/Users/Amelie/Dropbox/0_Data_Science_Projects/Primary_2016_Kaggle"
#os.chdir(path)
#os.getcwd()  #can type pwd for same result but unicode

import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")
import string
import datetime
from datetime import datetime

import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import scipy
import scipy.stats

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

import matplotlib as plt
import seaborn as sns
sns.set(style="white", color_codes=True) #python 3.0, color_codes=True) 

import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy import MetaData, Column, Table, Integer, String, Text 
from sqlalchemy import ForeignKey, func, DateTime, select

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, backref, mapper, aliased
from sqlalchemy.sql import exists
#from sqlalchemy import and_  ### use and_() not python 'and'
#from sqlalchemy import or_  ### use or_() not pythong 'or'
