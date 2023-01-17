import sqlite3
import json
from datetime import datetime

timeframe = '2018-5-18'
sql_transaction = []

connection = sqlite3.connect('{}.db'.format(timeframe))
c = connection.cursor()

def create_table():
    c.execute("""CREATE TABLE IF NOT EXISTS parent_reply(parent_id TEXT PRIMARY KEY, 
    comment_id TEXT UNIQUE, parent TEXT,comment TEXT, subreddit TEXT, unix INT, score INT)""")
    
def format_data(data):
    data = data.replace("\n"," newlinechar ").replace("\r"," newlinechar ").replace('"',"'")
    return data

def find_existing_score(pid):
    try:
        sql = "SELECT score FROM parent_reply WHERE parent_id = '{}' LIMIT 1".format(pid)
        c.execute(sql)
        result = c.fetchone()
        if result != None:
            return result[0]
        else: 
            return False
    except Exception as e:
        #print("find_parent",e)
        return False

def acceptable(data):
    if len(data.split(' ')) > 50 or len(data) < 1:
        return False
    elif len(data) > 1000:
        return False
    elif data == '[deleted]' or data == '[removed]':
        return False
    else:
        return True
    
def find_parent(pid):
    try:
        sql = "SELECT comment FROM parent_reply WHERE comment_id = '{}' LIMIT 1".format(pid)
        c.execute(sql)
        result = c.fetchone()
        if result != None:
            return result[0]
        else: 
            return False
    except Exception as e:
        #print("find_parent",e)
        return False
#3    
def transaction_bldr(sql):
    global sql_transaction
    sql_transaction.append(sql)
    if len(sql_transaction) > 1000: 
        c.execute('BEGIN TRANSACTION') 
        for s in sql_transaction: 
            try: 
                c.execute(s) 
            except: 
                pass 
        connection.commit() 
        sql_transaction = []
#2    
def sql_insert_replace_comment(commentid,parentid,parent,comment,subreddit,time,score):
    try:
        sql = """UPDATE parent_reply SET parent_id = ?, comment_id = ?, parent = ?, comment = ?, subreddit = ?,
        unix = ?, score = ? WHERE parent_id =?;""".format(parentid, commentid, parent, comment, 
                                                          subreddit, int(time), score, parentid)
        transaction_bldr(sql)
    except Exception as e:
        print('s-UPDATA insertion',str(e))

def sql_insert_has_parent(commentid,parentid,parent,comment,subreddit,time,score):
    try:
        sql = """INSERT INTO parent_reply (parent_id, comment_id, parent, comment, subreddit, 
        unix, score) VALUES ("{}","{}","{}","{}","{}",{},{});""".format(parentid, commentid, parent, comment, 
                                                                        subreddit, int(time), score)
        transaction_bldr(sql)
    except Exception as e:
        print('s-PARENT insertion',str(e))
        
def sql_insert_no_parent(commentid,parentid,comment,subreddit,time,score):
    try:
        sql = """INSERT INTO parent_reply (parent_id, comment_id, comment, subreddit,
        unix, score) VALUES ("{}","{}","{}","{}",{},{});""".format(parentid, commentid, comment, 
                                                                   subreddit, int(time), score) 
        transaction_bldr(sql)
    except Exception as e:
        print('s-NO_PARENT insertion',str(e))
    
if __name__ == "__main__":
    create_table()
    row_counter = 0
    paired_rows = 0
        
    with open("C:/Users/user/Desktop/chatbot/RC_2015-01".format(timeframe.split('-')[0],timeframe),buffering=1000) as f:
        for row in f:
            #print(row)
            row_counter += 1
            row = json.loads(row)
            parent_id = row['parent_id']
            body = format_data(row['body'])
            created_utc = row['created_utc']
            score = row['score']
            subreddit = row['subreddit']
            comment_id = row['name']
            parent_data = find_parent(parent_id)
            
            if score >= 2:
                #1
                if acceptable(body):
                    existing_comment_score = find_existing_score(parent_id)
                if  existing_comment_score:
                    if score > existing_comment_score:
                        sql_insert_replace_comment(comment_id,parent_id,parent_data,body,subreddit,created_utc,score)
                else:
                    if parent_data:
                        sql_insert_has_parent(comment_id,parent_id,parent_data,body,subreddit,created_utc,score)
                        #5
                        paired_rows += 1
                    else:
                        sql_insert_no_parent(comment_id,parent_id,body,subreddit,created_utc,score)
            #4
            if row_counter % 100000 == 0:
                print('Total Rows Read: {}, Paired Rows: {}, Time:{}'.format(row_counter, paired_rows, str(datetime.now())))