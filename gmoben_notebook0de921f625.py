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
with open("../input/hashes.txt") as f:

    print(f.readlines())
import sqlite3



conn = sqlite3.connect('../input/database.sqlite')



c = conn.cursor()

c.execute("SELECT name FROM sqlite_master WHERE type='table';")



c.fetchall()
c.execute('PRAGMA table_info([Emails])')

c.fetchall()
def fetchall(c, statement):

    c.execute(statement)

    return c.fetchall()
fetchall(c, 'Select * from Emails limit 1')
import pandas as pd



emails = pd.read_csv('../input/Emails.csv')



emails
from sklearn.feature_extraction.text import CountVectorizer



#CountVectorizer(emails.MetadataTo)



print(emails.MetadataTo)