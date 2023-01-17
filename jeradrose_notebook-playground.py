import sqlite3

con = sqlite3.connect('../input/database.sqlite')
cursor = con.cursor()

for row in cursor.execute('SELECT Year, COUNT(*) FROM NationalNames GROUP BY Year ORDER BY Year'):
    print(str(row[0])+','+str(row[1]))
