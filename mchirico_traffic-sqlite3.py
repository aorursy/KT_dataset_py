# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np

import seaborn

import pandas as pd

import matplotlib

import sqlite3



f={}

f[0] = pd.read_csv("../input/accident.csv")

f[1] = pd.read_csv("../input/person.csv")

c = sqlite3.connect(":memory:")

f[0].to_sql("accident",c)

f[1].to_sql("person",c)



df = pd.read_sql("select * from accident a, person p where a.ST_CASE=p.ST_CASE limit 3;",c)





#df= pd.read_sql("SELECT title, COUNT(*) as Incidents FROM nine11 GROUP BY title ORDER BY Incidents DESC LIMIT 20;", c)

#df.plot(kind='bar', x='title', y='Incidents')
df.head()