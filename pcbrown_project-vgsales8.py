import numpy as np # linear algebra



import os

import time





def debugInfoBasic():

    #list CSVs in our input folder

    print(os.listdir("../input"))

    print(os.listdir("../working/"))



    

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



class csvLoader:

    my_csv = None



    def __init__(self, csv_file):

        

        #check csv exists, then load

        if(os.path.isfile(csv_file)):

            print("file found... loading")

            #try load into pandas DataFrame

            try:

                self.my_csv = pd.read_csv(csv_file)

            except:

                print("csv failed to load! exiting...")

                exit()

                

            print("loading complete")

        else:

            print("csv load failed... exiting")

            exit()



            

    def getDataFrame(self):

        #if(self.my_csv != None):

        #    return self.my_csv

        #else:

        #    print("Cant return dataframe: null")

        #    return False

        return self.my_csv

    

    def getRow(self,row):

        row = None

        try:

            row = self.my_csv.iloc[row,:]  #row and all columns

        except:

            return False

        return row

    

    def dataTest(self):

        # look at a few rows of the nfl_data file. I can see a handful of missing data already!

        #print(self.my_csv.sample(5))

        # get the number of missing data points per column

        missing_values_count = self.my_csv.isnull().sum()

        # look at the # of missing points in the first ten columns

        missing_values_count[0:10]

        for i in range(1,10):

            print("Missing values" + str(i) + ":" + str(missing_values_count[i]))



    #https://stackoverflow.com/questions/13413590/how-to-drop-rows-of-pandas-dataframe-whose-value-in-a-certain-column-is-nan        

    def filterEssentialColumnsNonNull(self,nonNullColumnList):

        

        for columnName in nonNullColumnList:

            #take only rows where specified column is not NA

            self.my_csv = self.my_csv[self.my_csv[columnName].notna()]



    def filterAllLowercase(self):

        #self.my_csv.applymap(lambda s:s.lower() if type(s) == str else s)

        #self.my_csv.concat([df[col].astype(str).str.lower() for col in df.columns], axis=1)

        self.my_csv.astype(str).apply(lambda x: x.str.lower())

                                      









import sqlite3

from sqlite3 import Error



class sqlLiteHandler:

    conn = None



    

    def __init__(self, db_file):

        #self.conn = create_connection(db_file) #was acting up for some reason?

        try:

            self.conn = sqlite3.connect(db_file, timeout=10)

        except Error as e:

            print(e)

            

    def test_connection(self):

        if(self.conn == None):

            return false

        else:

            return true



    def create_connection(db_file):

        conn = None

        try:

            conn = sqlite3.connect(db_file)

            return conn

        except Error as e:

            print(e)

        return conn

    



    def sqlite_exec(self, sql_cmd):

        try:

            cur = self.conn.cursor()

            if(cur.execute(sql_cmd)):

                return True

            else:

                return False

        except Error as e:

            print(e)

            return False

        return False



    def get_table_info(self, table):

        cur = self.conn.cursor()

        #prints each column and type

        cur.execute('PRAGMA TABLE_INFO(' + table + ')')

        rows = cur.fetchall()

        for row in rows:

            print(row)



    def select_all_table(self, table):

        cur = self.conn.cursor()

        cur.execute("SELECT * FROM " + table)

        rows = cur.fetchall()

        for row in rows:

            print(row)

            

    def select_table_row(self, table, row):

        cur = self.conn.cursor()

        sql_query = "SELECT * FROM " + str(table) + " WHERE rank = " + str(row) + ";"

        cur.execute(sql_query)

        rows = cur.fetchall()

        for row in rows:

            print(row)



    def pandasDF2sqlite(self, table,pandas_dataframe):

        cur = self.conn.cursor()

        #cur.execute('''DROP TABLE IF EXISTS vgsales''')

        pandas_dataframe.to_sql('vgsales', self.conn, if_exists='replace', index=False) # - writes the pd.df to SQLIte DB

        sql_query = "select * from " + str(table)

        pd.read_sql(sql_query, self.conn)

        self.conn.commit()

        #self.conn.close()

        

    def sqlite2pandasDF(self,table_name):

        sql_query = "SELECT * FROM " + table_name

        df = pd.read_sql_query(sql_query, self.conn)

        return df



    def addEntry(self, sql_insert_row, entry):



        cur = self.conn.cursor()

        if(cur.execute(sql_insert_row, entry)):

            return True

        else:

            return False

        return False







def dataLoadingPhase():

    csvFile = "../input/videogamesales/vgsales.csv"

    nonNullList = [ "Name", "Platform", "Year" ] # list of columns that cannot be null, else discard



    print("Initializing csv: " + csvFile)

    csv0 = csvLoader(csvFile)

    print("filtering out bad data...")

    csv0.filterEssentialColumnsNonNull(nonNullList)

    print("setting all to lowercase...")

    csv0.filterAllLowercase()

    #csv0.dataTest()





    #VARS

    database = r"pythonsqlite1.db"

    sql_vgsales_tablename = "vgsales"

    sql_create_vgsales_table = """CREATE TABLE IF NOT EXISTS vgsales (

                                rank integer PRIMARY KEY,

                                name text NOT NULL,

                                platform text NOT NULL,

                                year integer NOT NULL,

                                genre text NOT NULL,

                                publisher text,

                                na_sales int NOT NULL,

                                eu_sales int NOT NULL,

                                jp_sales int NOT NULL,

                                other_sales int NOT NULL,

                                global_sales int NOT NULL

                            );"""

    sql_insert_vgsales_row = """INSERT INTO vgsales(Rank,Name,Platform,Year,Genre,Publisher,NA_Sales,EU_Sales,JP_Sales,Other_Sales,Global_Sales)

                             VALUES(?,?,?,?,?,?,?,?,?,?,?)"""



    print("Initializing sqlLite3 handler...")

    sqlite0 = sqlLiteHandler(database)

    print("Creating table " + sql_vgsales_tablename)

    sqlite0.sqlite_exec(sql_create_vgsales_table)

    print("Printing: TableColumns")

    sqlite0.get_table_info(sql_vgsales_tablename)



    print("Manual data add test...")

    sql_entry01 = (1,'Wii Sports','Wii',2006,'Sports','Nintendo',41.49,29.02,3.77,8.46,82.74)

    sqlite0.addEntry(sql_insert_vgsales_row, sql_entry01)



    print("\nPushing all pandas dataframe data to SQLite3")  

    sqlite0.pandasDF2sqlite(sql_vgsales_tablename, csv0.getDataFrame())





    #sqlite0.select_all_table(sql_vgsales_tablename) # select every single line

    sqlite0.select_table_row(sql_vgsales_tablename,1) # select and print 1 row







def dataVisualizationPhase():

    # We'll also import seaborn, a Python graphing library

    import warnings # current version of seaborn generates a bunch of warnings that we'll ignore

    warnings.filterwarnings("ignore")

    import seaborn as sns

    import matplotlib.pyplot as plt

    sns.set(style="white", color_codes=True)



    database = r"pythonsqlite1.db"

    print("loading table " + "\"vgsales\"" + " from " + database)

    sqlite = sqlLiteHandler(database)

    df = sqlite.sqlite2pandasDF("vgsales")

    

    print("initializing plots...")

    myPlot = df.plot(kind="scatter", x="Genre", y="Global_Sales", figsize=(16,4))

    myPlot.tick_params(axis='x', rotation=45) #rotate xaxis lables 45degrees



    myPlot0 = df.plot(kind="scatter", x="Platform", y="Global_Sales", figsize=(16,4))

    myPlot0.tick_params(axis='x', rotation=90) #rotate xaxis lables 90degrees



def Main():

    #debugInfoBasic()

    dataLoadingPhase()

    dataVisualizationPhase()

    

Main()