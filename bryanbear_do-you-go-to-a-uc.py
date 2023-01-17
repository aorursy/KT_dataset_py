%matplotlib inline
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

conn = sqlite3.connect('../input/database.sqlite')
c = conn.cursor()
df = pd.read_sql_query("""
SELECT INSTNM College 
FROM Scorecard 
WHERE Year=2000 
AND INSTNM like 'University of California%' 
AND PREDDEG = 'Predominantly bachelor''s-degree granting'""", conn)
for i in range(2010,2014):
    column = pd.read_sql_query("""
    SELECT GRAD_DEBT_MDN FROM Scorecard 
    WHERE Year="""+str(i)+""" 
    AND INSTNM like 'University of California%' 
    AND PREDDEG = 'Predominantly bachelor''s-degree granting'""", conn)
    df[str(i)]=column
conn.close()


df.plot(kind='bar')
leg = plt.legend( loc = 'lower right')
ax = plt.subplot()
ax.set_ylabel('Graduation Debt')
ax.set_xlabel('School')
ax.set_xticklabels(['UCB', 'UCD', 'UCI', 'UCLA', 'UCR', 'UCSD', 'UCSF', 'UCSB', 'UCSC'])
plt.show()
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

conn = sqlite3.connect('../input/database.sqlite')
c = conn.cursor()
df = pd.read_sql_query("""
SELECT INSTNM College 
FROM Scorecard 
WHERE Year=2000 
AND INSTNM like 'University of California%' 
AND PREDDEG = 'Predominantly bachelor''s-degree granting'""", conn)
for i in range(2010,2014):
    column = pd.read_sql_query("""
    SELECT TUITIONFEE_IN FROM Scorecard 
    WHERE Year="""+str(i)+""" 
    AND INSTNM like 'University of California%' 
    AND PREDDEG = 'Predominantly bachelor''s-degree granting'""", conn)
    df[str(i)]=column
conn.close()


df.plot(kind='bar')
leg = plt.legend( loc = 'lower right')
ax = plt.subplot()
ax.set_ylabel('Tuition Fee')
ax.set_xlabel('School')
ax.set_xticklabels(['UCB', 'UCD', 'UCI', 'UCLA', 'UCR', 'UCSD', 'UCSF', 'UCSB', 'UCSC'])
plt.show()
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

conn = sqlite3.connect('../input/database.sqlite')
c = conn.cursor()
df = pd.read_sql_query("""SELECT INSTNM College, 
md_earn_wne_p10
FROM Scorecard 
WHERE INSTNM like 'University of California%' 
AND Year=2011 
AND PREDDEG = 'Predominantly bachelor''s-degree granting'""", conn)
conn.close()

df.plot(kind='bar')
leg = plt.legend( loc = 'lower right')
ax = plt.subplot()
ax.set_ylabel('Income 10 Years after Graduation')
ax.set_xlabel('School')
ax.set_xticklabels(['UCB', 'UCD', 'UCI', 'UCLA', 'UCR', 'UCSD', 'UCSF', 'UCSB', 'UCSC'])
plt.show()