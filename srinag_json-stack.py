# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
!pip install moz_sql_parser

import ast

from moz_sql_parser import parse

import json
sql = """ SELECT * FROM user join department on (user.id = department.id) WHERE id = (select id from candidate where id = 123 or id = 456) """

sql
dump = json.dumps(parse(sql))



jsn = ast.literal_eval(dump)



jsn
s=[]

f = 0

i = 0



def rec(d,s,f):

    

    if "select" in d.keys():

        f=f+1

    

    for i in d.keys():

        if type(d[i]).__name__ == "dict":

            tmp_list = []

            for j in d[i].keys():

                tmp_list.append(j)

            s.append((i,tmp_list[0],f)) 

            rec(d[i],s,f) 

        elif type(d[i]).__name__ == "list":

            if(type(d[i][1]).__name__ == "dict" and type(d[i][0]).__name__ != "dict"):

                s.append((i,d[i][0],f))

                rec(d[i][1],s,f)

            elif type(d[i][0]).__name__ == "dict" and type(d[i][1]).__name__ != "dict":

                rec(d[i][0],s,f)

            elif type(d[i][0]).__name__ == "dict" and type(d[i][1]).__name__ == "dict":

                rec(d[i][1],s,f)

                rec(d[i][0],s,f)

            else:

                s.append((i,d[i],f))

        else:

            s.append((i,d[i],f))



rec(jsn,s,f)

    

s

    

 