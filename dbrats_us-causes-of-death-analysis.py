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




# Any results you write to the current directory are saved as output.
#Read in the death records to pandas df
deaths = pd.read_csv('../input/DeathRecords.csv')
codes = pd.read_csv('../input/Icd10Code.csv')
manners = pd.read_csv('../input/MannerOfDeath.csv')
icd10 = pd.read_csv('../input/Icd10Code.csv')
age = pd.read_csv('../input/AgeType.csv')
race = pd.read_csv('../input/Race.csv')
loc= pd.read_csv('../input/PlaceOfDeathAndDecedentsStatus.csv')

df_lst = [deaths,codes,manners,icd10,age,race]
for elem in df_lst:
    print(elem.columns)
import sqlite3 as sqlite
import pandas as pd


tablesToIgnore = ["sqlite_sequence"]

outputFilename = None

def Print(msg):
    
    if (outputFilename != None):
        outputFile = open(outputFilename,'a')
        #print( >> outputFile, msg)
        outputFile.close()
    else:
        print(msg)
        


def Describe(dbFile):
    connection = sqlite.connect(dbFile)
    cursor = connection.cursor()
    
    Print("TableName\tColumns\tRows\tCells")

    totalTables = 0
    totalColumns = 0
    totalRows = 0
    totalCells = 0
    
    # Get List of Tables:      
    tableListQuery = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY Name"
    cursor.execute(tableListQuery)
    tables = map(lambda t: t[0], cursor.fetchall())
    
    for table in tables:
    
        if (table in tablesToIgnore):
            continue            
            
        columnsQuery = "PRAGMA table_info(%s)" % table
        cursor.execute(columnsQuery)
        numberOfColumns = len(cursor.fetchall())
        
        rowsQuery = "SELECT Count() FROM %s" % table
        cursor.execute(rowsQuery)
        numberOfRows = cursor.fetchone()[0]
        
        numberOfCells = numberOfColumns*numberOfRows
        
        Print("%s\t%d\t%d\t%d" % (table, numberOfColumns, numberOfRows, numberOfCells))
        
        totalTables += 1
        totalColumns += numberOfColumns
        totalRows += numberOfRows
        totalCells += numberOfCells

    Print( "" )
    Print( "Number of Tables:\t%d" % totalTables )
    Print( "Total Number of Columns:\t%d" % totalColumns )
    Print( "Total Number of Rows:\t%d" % totalRows )
    Print( "Total Number of Cells:\t%d" % totalCells )
        
    cursor.close()
    connection.close()   


db = sqlite.connect('database.sqlite')
cursor = db.cursor()

tableListQuery = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY Name"
cursor.execute(tableListQuery)
tables = map(lambda t: t[0], cursor.fetchall())

Describe('database.sqlite')
codes.head()
deaths.head(1)
import seaborn as sns


sns.distplot(deaths['Age'])
deaths['Age'].describe()

unknown_age = deaths[deaths['Age'] > 200]
non_usresidents = deaths[deaths['ResidentStatus'] == 4]
usresidents = deaths[deaths['ResidentStatus'] != 4]
sns.distplot(usresidents['Age'])
sns.distplot(non_usresidents['Age'])
deaths['Icd10Code']
icd10.columns= ['Icd10Code', 'dx']
df = pd.merge(deaths, icd10, how='left', on='Icd10Code')

counts = df[['Icd10Code', 'Id']].groupby(['Icd10Code'], as_index=False).count()
rare =counts[counts['Id'] < 100]
rare.columns = ['Icd10Code','count']
counts =counts[counts['Id'] > 100]
counts.columns = ['Icd10Code','count']
counts.head()
dx = df[['Icd10Code','Age']].groupby(['Icd10Code'], as_index=False).mean()
dx = pd.merge(dx,counts, how='inner', on='Icd10Code')
dx = pd.merge(dx,icd10, how='inner', on='Icd10Code')
dx.head()
most_common_causes = dx.sort_values(by=['count'],ascending=False).head(100)

from ggplot import *

most_common_causes['senior_illness'] = most_common_causes['Age'] >= 65
most_common_causes['senior_illness'] = most_common_causes['senior_illness'].astype(int)
most_common_causes

ggplot(aes(x='count',  fill='senior_illness'), data=most_common_causes) + \
    geom_density(alpha=0.6)
sns.distplot(most_common_causes['Age'])
geom_text(position = 'identity', stat = 'identity')

ggplot(most_common_causes, aes(x='Age', y='count', label='dx')) +\
    geom_text(hjust=0.15, vjust=0.1) +\
    geom_point()
young = most_common_causes[most_common_causes['Age'] < 60]
geom_text(position = 'identity', stat = 'identity')

ggplot(young, aes(x='Age', y='count', label='dx')) +\
    geom_text(hjust=0.15, vjust=0.1) +\
    geom_point()
rare = pd.merge(rare, icd10, how='left', on='Icd10Code')
uncommon_samp = rare[['count','dx']].sample(10)
uncommon_samp[['count','dx']]
sns.barplot(x="count", y="dx", data=uncommon_samp);
ms = df[df['Icd10Code']=='G35']
ms=ms[['Age','Sex','NumberOfRecordAxisConditions','CurrentDataYear','AgeRecode12']]

ggplot(aes(x='Age', fill='Sex'), data=ms) + \
    geom_density(alpha=0.6)

ggplot(aes(x='Age',  fill='Sex'), data=ms) + \
    geom_histogram(alpha=0.6)
sns.jointplot(x='Age', y='NumberOfRecordAxisConditions', data=ms, kind='reg')
