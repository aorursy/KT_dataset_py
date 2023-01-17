import sqlite3

 

from sqlite3 import Error

 

def sql_connection():

 

    try:

 

        con = sqlite3.connect('mydatabase.db')

 

        return con

 

    except Error:

 

        print(Error)

 

def sql_table(con):

 

    cursorObj = con.cursor()

 

    cursorObj.execute("CREATE TABLE trial(Student text,CGPA integer,Branch text,Jobs_Offered integer,Package_in_lakhs integer,Recruiter text)")

 

    con.commit()

 

con = sql_connection()

 

sql_table(con)
con = sqlite3.connect('mydatabase.db')

cursorObj = con.cursor()

cursorObj.execute("INSERT INTO trial VALUES('Ashish',8.2,'CE',2,8,'IOCL')")

cursorObj.execute("INSERT INTO trial VALUES('Nilesh',7.8,'PE',1,10,'ONGC')")

cursorObj.execute("INSERT INTO trial VALUES('Ronit',8.4,'PE',1,8.5,'HPCL')")

cursorObj.execute("INSERT INTO trial VALUES('Kapil',7.2,'CE',2,7,'BPCL')")

cursorObj.execute("INSERT INTO trial VALUES('Vishal',8,'PE',1,9.5,'ONGC')")

cursorObj.execute("INSERT INTO trial VALUES('Aman',9,'CE',2,9,'IOCL')")

cursorObj.execute("INSERT INTO trial VALUES('Sourabh',8.6,'CE',1,7.2,'ONGC')")

cursorObj.execute("INSERT INTO trial VALUES('Rajveer',7.9,'PE',2,10,'HPCL')")

cursorObj.execute("INSERT INTO trial VALUES('Shivam',8.8,'CE',1,11,'IOCL')")

cursorObj.execute("INSERT INTO trial VALUES('Abhimanyu',8,'PE',1,9,'ONGC')")

cursorObj.execute('SELECT * FROM trial')

rows = cursorObj.fetchall()

for row in rows:

        print(row)
# Both branch included with cutoff CGPA 8 and no student with 2 job offers.

cursorObj.execute('SELECT Student FROM trial WHERE CGPA>=8 AND Jobs_Offered<2')

rows = cursorObj.fetchall()

for row in rows:

        print(row)
# Only for PE branch with cutoff CGPA 8 and no student with 2 job offers.

cursorObj.execute('SELECT Student FROM trial WHERE CGPA>=8 AND Jobs_Offered<2 AND Branch="PE"')

rows = cursorObj.fetchall()

for row in rows:

        print(row)
# Similarly for CE branch with cutoff CGPA 8 and no student with 2 job offers.

cursorObj.execute('SELECT Student FROM trial WHERE CGPA>=8 AND Jobs_Offered<2 AND Branch="CE"')

rows = cursorObj.fetchall()

for row in rows:

        print(row)
# We will check if there is any student with 0 jobs offered.

cursorObj.execute('SELECT COUNT(*) FROM trial WHERE Jobs_Offered=0')

rows = cursorObj.fetchall()

for row in rows:

        print(row)
cursorObj.execute('SELECT MAX(Package_in_lakhs),MIN(Package_in_lakhs),AVG(Package_in_lakhs) FROM trial')

rows = cursorObj.fetchall()

for row in rows:

        print(row)
cursorObj.execute('SELECT COUNT(*),Recruiter FROM trial GROUP BY Recruiter ORDER BY Recruiter desc')

rows = cursorObj.fetchall()

for row in rows:

        print(row)