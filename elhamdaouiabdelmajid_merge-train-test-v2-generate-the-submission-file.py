import numpy as np
import pandas as pd
import os
import gc
import sqlite3
import json
from pandas.io.json import json_normalize

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# # Number of rows, train
# n = 0
# with open("../input/train_v2.csv", "r") as f:
#     for l in f:
#         if n==0:
#             print(l)
#             break
#         n += 1
#         if not(n % 100000):
#             print(n," :: date:",l.strip().split(",")[2])
# print("TRAIN CONTAINS %d-1 rows"%n)
N_ROWS = 10
# 'customDimensions','hits' may are multi dict into a list (w'll stock them as STR)
JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']

def json_normalize_data(df):
    for column in JSON_COLUMNS: 
        column_as_df = json_normalize(df[column]) 
        column_as_df.columns = [f"{column}_{subcolumn}".replace('.','_') for subcolumn in column_as_df.columns] 
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    return df

train = pd.read_csv("../input/train_v2.csv",  nrows=N_ROWS, 
                    converters={column: json.loads for column in JSON_COLUMNS}, 
                    dtype={"fullVisitorId": "str"})
test = pd.read_csv("../input/test_v2.csv", nrows=N_ROWS, 
                    converters={column: json.loads for column in JSON_COLUMNS}, 
                    dtype={'fullVisitorId': 'str'})

train = json_normalize_data(train)
test = json_normalize_data(test)

train.shape,test.shape


columns_types = {'date': 'TEXT', }
for c in set(train.columns) - set(['date']):
    columns_types[c] = 'TEXT' # YOU CAN CHANGE THE TYPES 
cols = [c for c in train.columns]
#cols
del train
del test
gc.collect()
del gc.garbage[:]
# -*- coding:utf8 -*-
import sqlite3

columns_types.update(ID='INTEGER', ISTRAIN='INTEGER')

class DataBase:
    """"""
    
    fileCsvTrain = "../input/train_v2.csv"
    fileCsvTest = "../input/test_v2.csv"
    fileTrainDb = "data.sq3"
    columns = ['ID',] + cols + ['ISTRAIN',]
    
    #=======================================================#
    def __init__(self, data="train"):
        """init"""
        self.dataset=data
        self.bdd_file=DataBase.fileTrainDb
        self.table = "train_test_data"
        self.nb_rows = 0
        #create file
#         with open(self.bdd_file, 'w') as f:
#             pass
        self.connexion = sqlite3.connect(self.bdd_file)
        self.cursor = self.connexion.cursor()
        
    #=======================================================#   
    def create_table(self):
        """"""
        req = "CREATE TABLE IF NOT EXISTS "+self.table+\
                            " ("+ ','.join(['%s %s'%(c,columns_types[c])\
                                            for c in DataBase.columns])+")"
        #print(req)
        self.cursor.execute(req)
        self.connexion.commit()
        
    #=======================================================#
    def file_csv_to_SQLite(self):
        """"""
        nb_rows = 0
        #train_v2
        print('train v2 ...')
        for dt in pd.read_csv(DataBase.fileCsvTrain,chunksize=5*10**4, iterator=True,
                               converters={column: json.loads for column in JSON_COLUMNS}, 
                    dtype={"fullVisitorId": "str"}):
            data = json_normalize_data(dt.copy())
            print("chunk, ",data.shape)
            for index, row in data.iterrows():
                nb_rows += 1
                row_values = [nb_rows] + [row[c] for c in DataBase.columns[1:-1]] + [1]
                self.cursor.execute("INSERT INTO " + self.table +\
                                    "(%s) VALUES(%s)"%(
                        ','.join([str(c) for c in DataBase.columns]),
                    ','.join(['?' for c in DataBase.columns]),
                ), tuple([str(v) for v in row_values]))
            del data
            gc.collect()
            del gc.garbage[:]
                
                
                
        #test_v2
        print("test v2 ...")
        for dt in pd.read_csv(DataBase.fileCsvTest,chunksize=5*10**4, iterator=True,
                               converters={column: json.loads for column in JSON_COLUMNS}, 
                    dtype={"fullVisitorId": "str"}):
            data = json_normalize_data(dt.copy())
            print("chunk, ",data.shape)
            for index, row in data.iterrows():
                nb_rows += 1
                row_values = [nb_rows] + [row[c] for c in DataBase.columns[1:-1]] + [0]
                self.cursor.execute("INSERT INTO " + self.table +\
                                    "(%s) VALUES(%s)"%(
                        ','.join([str(c) for c in DataBase.columns]),
                    ','.join(['?' for c in DataBase.columns]),
                ), tuple([str(v) for v in row_values]))
            del data
            gc.collect()
            del gc.garbage[:]
            
        print("Done storing data with %d rows" % nb_rows)
        self.nb_rows = nb_rows
        self.connexion.commit()
        
        
        
    #=======================================================#    
    def close_connexion_cursor(self):
        """"""
        self.cursor.close()
        self.connexion.close()
        print("Connexion and Cursor closed")
        
    #=======================================================#
    def select_all_rows(self):
        """"""
        self.cursor.execute("SELECT * FROM " + self.table + "")
        for l in self.cursor:
            yield l

    #=======================================================#
    def get_line_by_index(self,index):
        """"""
        self.cursor.execute("SELECT * FROM " + self.table + " where id=?", (str(index),))
        for line in self.cursor:
            return line
        return None
    
    #=======================================================#
    def get_lines_between(self, start_id, end_id):
        """"""
        self.cursor.execute("SELECT * FROM " + self.table + " where id>=? and id<=?",(str(start_id),str(end_id)))
        return self.cursor.fetchall()

    #=======================================================#
    def create_index_column(self, column='id'):
        """C"""
        self.cursor.execute("CREATE INDEX id_index on " + self.table + " (%s)"%column)
        self.connexion.commit()
        print("% indexed" + str(column))
        
    #=======================================================#
    def clear_table(self):
        """Vider une table"""
        self.cursor.execute("DELETE FROM " + self.table + "")
        self.connexion.commit()

    #=======================================================#   
    def get_features_names(self):
        """"""
        return DataBase.columns[1:-1]
    
    def generate_init_sub(self):
        self.cursor.execute("SELECT DISTINCT fullVisitorId FROM " + self.table)
        sub = pd.DataFrame.from_dict({'fullVisitorId':[l[0] for l in self.cursor.fetchall()]})
        sub['PredictedLogRevenue'] = 0
        return sub
db = DataBase()
db.create_table()
db.file_csv_to_SQLite()
db.create_index_column()
submission = db.generate_init_sub()
db.close_connexion_cursor()

del db
gc.collect()
submission.to_csv("to_submit.csv", index=False)