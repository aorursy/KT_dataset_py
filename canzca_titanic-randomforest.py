# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
targetdata=pd.read_csv("/kaggle/input/titanic/gender_submission.csv")

targetdata.head()
df=pd.read_csv("/kaggle/input/titanic/train.csv")

df.head()
inputs=df.drop(["PassengerId","Survived","Name","SibSp","Parch","Ticket","Cabin","Embarked"],axis=1)

target=df["Survived"]
tst=pd.read_csv("/kaggle/input/titanic/test.csv")

tstinputs=tst.drop(["PassengerId","Name","SibSp","Parch","Ticket","Cabin","Embarked"],axis=1)

tstinputs.head()
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import MinMaxScaler 

le_sex=LabelEncoder()

tstinputs["sex_ln"]=le_sex.fit_transform(tstinputs["Sex"])

tstinputs.head()
tstinputs=tstinputs.drop(["Sex"],axis=1)

tstinputs.head()
scaller=MinMaxScaler((0,1))

rescalex=scaller.fit_transform(tstinputs)

tstinputs=pd.DataFrame(rescalex)

tstinputs.head()
inputs=inputs.fillna(method="backfill")

inputs.info()
le_sex=LabelEncoder()

inputs["sex_ln"]=le_sex.fit_transform(inputs["Sex"])

inputs.head()
newinputs=inputs.drop(["Sex"],axis=1)

newinputs.head()
from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler(feature_range=(0,1))

recsaler=scaler.fit_transform(newinputs)

newinputs=pd.DataFrame(recsaler)

newinputs.head()
from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier()

model.fit(newinputs,target)
tstinputs=tstinputs.fillna(method="backfill")

tstinputs=tstinputs.fillna(method="ffill")

tstinputs.head()
pred=model.predict(tstinputs)

pred
targetdata["PassengerId"]
d = {'PassengerId':targetdata["PassengerId"] , 'Survived': pred}

mysubmission=pd.DataFrame(d)

mysubmission.info()

# import the modules we'll need

from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv()

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



# create a random sample dataframe

df = pd.DataFrame(d)



# create a link to download the dataframe

create_download_link(df)
