import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import sqlite3

cnx = sqlite3.connect('/kaggle/input/8anu-climbing-logbook/database.sqlite')
query = """

SELECT F510.user_id, 

F511.date_511a - F510.date_510a as months_510_511,

F512.date_512a - F511.date_511a as months_511_512,

F513.date_513a - F512.date_512a as months_512_513

FROM 

	(SELECT user_id, min(date) / (3600*24*30) as date_510a 

    -- to convert to months we calculate --> date / (3600*24*30)

	from ascent

	where date != 0 and grade_id == 36 --5.10a

	GROUP BY user_id, grade_id) as F510

LEFT OUTER JOIN 

		(SELECT user_id, min(date) / (3600*24*30) as date_511a

		from ascent

		where date != 0 and grade_id == 44 --5.11a

		GROUP BY user_id, grade_id) as F511,

		(SELECT user_id, min(date) / (3600*24*30) as date_512a

		from ascent

		where date != 0 and grade_id == 51 --5.12a

		GROUP BY user_id, grade_id) as F512,

		(SELECT user_id, min(date) / (3600*24*30) as date_513a

		from ascent

		where date != 0 and grade_id == 59 --5.13a

		GROUP BY user_id, grade_id) as F513

ON (F510.user_id == F511.user_id and

F510.user_id == F512.user_id and

F510.user_id == F513.user_id)

where (months_510_511 >= 0 and months_511_512 >= 0 and months_512_513 >=0)

"""
df = pd.read_sql_query(query, cnx)
df.describe()