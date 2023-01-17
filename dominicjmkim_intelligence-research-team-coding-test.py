import pandas as pd

import sqlite3



conn = sqlite3.connect('example.db')



calls_json = '{"call_id":{"0":100,"1":101,"2":102},"created_at":{"0":20200511,"1":20200610,"2":20200511},"customer_id":{"0":1,"1":1,"2":4},"driver_id":{"0":2,"1":11,"2":3},"fare":{"0":4500,"1":10000,"2":6700}}'

calls = pd.read_json(calls_json)

calls.to_sql("calls", conn, if_exists='replace', index=False)



drivers_json = '{"driver_id":{"0":1,"1":2,"2":3,"3":4},"joined_at":{"0":20200501,"1":20190701,"2":20180702,"3":20200711},"kind":{"0":"\\uc77c\\ubc18","1":"\\ubaa8\\ubc94","2":"\\uc77c\\ubc18","3":"\\uc77c\\ubc18"},"name":{"0":"\\uae40\\uc544\\ubb34\\uac1c","1":"\\uc774\\uc544\\ubb34\\uac1c","2":"\\ud64d\\uc544\\ubb34\\uac1c","3":"\\ubc15\\uc544\\ubb34\\uac1c"}}'

drivers = pd.read_json(drivers_json)

drivers.to_sql("drivers", conn, if_exists='replace', index=False)



users_json = '{"customer_id":{"0":1,"1":2,"2":3,"3":4},"joined_at":{"0":20200211,"1":20190501,"2":20120502,"3":20200611},"name":{"0":"\\uae40\\uae38\\ub3d9","1":"\\uc774\\uae38\\ub3d9","2":"\\ud64d\\uae38\\ub3d9","3":"\\ubc15\\uae38\\ub3d9"}}'

users = pd.read_json(users_json)

users.to_sql("users", conn, if_exists='replace', index=False)



def run_query(query, conn):

    try:

        display(pd.read_sql_query(query, conn))

    except:

        pass
# Table: calls

"""

call_id: 콜의 고유 id (int)

created_at: 콜 생성 시간, yyyymmdd (string)

customer_id: 고객 고유 id (int)

driver_id: 기사 고유 id (int)

fare: 콜의 요금 (int)

"""



print("Table: calls")

display(calls)



# Table: drivers

"""

driver_id: 기사 고유 id (int)

joined_at: 기사의 서비스 가입 일자, yyyymmdd (string)

kind: 기사가 운행하는 택시의 종류 (string)

name: 기사 이름 (string)

"""



print("Table: drivers")

display(drivers)



# Table: users

"""

customer_id: 고객 고유 id (int)

joined_at: 승객의 서비스 가입 일자, yyyymmdd (string)

name: 승객 이름 (string)

"""



print("Table: users")

display(users)
#Q1-1-1:월별 콜 발생 건수

answer = """



"""



run_query(answer, conn)
#Q1-1-2: 월별 운행기사 수

answer = """



"""



run_query(answer, conn)
#Q1-1-3: 월별 기사당 운임

answer = """



"""



run_query(answer, conn)
#Q1-2. 7월 운행한 모범 기사 중 가장 많이 운행한 기사?

answer = """



"""



run_query(answer, conn)
#Q1-3. 고객 가입월에 따른 월별 재이용률 (코호트)

answer = """



"""



run_query(answer, conn)
#Q-Add-1-1:월별 콜 발생 건수

#Q-Add-1-2: 월별 운행기사 수

#Q-Add-1-3: 월별 기사당 운임

#Q-Add-2. 7월 운행한 모범 기사 중 가장 많이 운행한 기사?

#Q-Add-3. 고객 가입월에 따른 월별 재이용률 (코호트)

# Q2: Answer
