import sqlite3
import pandas as pd
conn = sqlite3.connect('../input/FPA_FOD_20170508.sqlite')
df = pd.read_sql(
                       """
                       
                        SELECT *
                        from fires 
                        
                       """, con=conn)
pd.read_sql("""

SELECT *
FROM fires
LIMIT 100

""",con = conn)
pd.read_sql("""

SELECT *
FROM fires
WHERE STATE = 'CA'

""",con = conn)
pd.read_sql("""

SELECT SOURCE_REPORTING_UNIT_NAME,count(*) as [count]
FROM fires
GROUP BY SOURCE_REPORTING_UNIT_NAME

""",con = conn)
