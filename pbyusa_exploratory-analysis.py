# The data comes as the raw data files, a transformed CSV file, and a SQLite database

import pandas as pd
import sqlite3

# You can read in the SQLite datbase like this
import sqlite3
con = sqlite3.connect('../input/database.sqlite')
sample = pd.read_sql_query("""
SELECT INSTNM,
       COSTT4_A AverageCostOfAttendance,
       Year
FROM Scorecard
WHERE INSTNM='Duke University'""", con)
print(sample)

# You can read a CSV file like this
scorecard = pd.read_csv("../input/Scorecard.csv")
print(scorecard)

# It's yours to take from here!

repayment_rate_sql= '''SELECT
                           INSTNM InstitutionName,
                           COSTT4_A AverageCostofAttendance,
                           GRAD_DEBT_MDN_SUPP,
                           GRAD_DEBT_MDN10YR_SUPP, 
                           NOPELL_RPY_3YR_RT_SUPP,
                           FEMALE_RPY_3YR_RT_SUPP,
                           MALE_RPY_3YR_RT_SUPP,            
                           RPY_3YR_RT_SUPP 
                           
                        FROM Scorecard'''

#In this section, we will look schools with top 10 repayment rates 3 years after leaving school.
#read repayment rates dataset with variables established above.
rpyrates = pd.read_sql(repayment_rate_sql ,con)
#convert object data types to float
rpyrates = rpyrates.convert_objects(convert_numeric = True)
#rpyrates.dtypes
#drop missing values
rpyrates = rpyrates.dropna()
top_10_rpyrates = rpyrates.sort_values(['RPY_3YR_RT_SUPP'],ascending=[False])
print (top_10_rpyrates[:10])
#bottom repayment rates
print (top_10_rpyrates[-10:])

