import numpy as np 

import pandas as pd 

import sqlite3

import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

def sql_connection(databaseName):

    try:

        connection=sqlite3.connect(databaseName)

        print('Connection Successfull!!!')

    except:

        print(Error)

    finally:

        #connection.close()

        return connection

        

connection=sql_connection('mydatabase.db')

cursor=connection.cursor()
cursor.executescript("""

CREATE TABLE employee( 

firstname text, 

lastname text, 

age integer

);

CREATE TABLE Book( 

title text, 

author text 

);

INSERT INTO 

Book(title, author) 

VALUES ( 

'Dan Clarke''s GFG Detective Agency', 

'Sean Simpsons' 

);

INSERT INTO 

Employee(firstname,lastname,age)

VALUES(

'John',

'Doe',

27

);

""")
connection.commit()
sql_select_query='SELECT * FROM Employee'

cursor.execute(sql_select_query)
result=cursor.fetchall()

print(result)