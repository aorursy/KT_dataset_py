# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


eemail= pd.read_csv("../input/enron-email-dataset/emails.csv")
eemail.head()
eemail.describe()
import email
#msg = eemail['message'][517398]
#msg
type(eemail["message"])
print(eemail["message"][517398])
eemail["message"][39]
eemail["file"][49]

msg1 = eemail['message'][0]
msg1
e = email.message_from_string(msg1)

e["subject"]
e["X-From"]
e["X-Folder"]
e["Date"]
e["X-To"]
e["To"]
e["From"]

headers = ["subject","X-From","X-To","X-Folder","From","To","Msg-Content"]
for h in headers:
    arr = []
    arr.append(e[h])
    print(arr)

headers = ["subject","X-From","X-Folder","Msg-Content"]
e.get_payload()


df3 = pd.DataFrame([[e["subject"], e["X-From"], e["X-Folder"],e.get_payload()]],
                   columns= headers)
df3

e["X-To"][:e["X-To"].find('<')]
msgs = eemail['message']
headers = ["subject","Date","X-From","X-To","X-Folder","From","To","Msg-Content"]

def getheaders(msgs):
    table = []
    for msg in msgs:
        e = email.message_from_string(msg)
        arr = []
        for h in headers:
            if h == "Msg-Content":
                e.get_payload()
                arr.append(e.get_payload())
            elif (h == "X-From" or h == "X-To") and e[h]:
                arr.append(e[h][:e[h].find('<')])
            else:
                arr.append(e[h])
        table.append(arr)
    return table
            
res = getheaders(msgs)
res
     
df_new = pd.DataFrame(res,
                   columns= headers)
df_new
df_new.to_csv('df_new.csv',index=False)
df_new[100010:100025]
df_new.loc[df_new['subject']=='CONFIDENTIAL']
df_new.loc[df_new['subject']=='RE: CONFIDENTIAL']
print(eemail["message"][90109])
df_new["X-From"].value_counts()
email_subset = df_new.iloc[497401:, :]
email_subset.info()
email_subset.isnull().sum()
df_new.loc[df_new['subject']=='RE: CONFIDENTIAL'].count()
df_new.loc[df_new['subject']=='CONFIDENTIAL'].count()
print(eemail['message'][213366])
