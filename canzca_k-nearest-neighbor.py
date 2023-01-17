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
df=pd.read_csv("/kaggle/input/titanic/train.csv")

df.isnull().sum()
df.info()
df=df.fillna(method="bfill")
df.isnull().sum()
df.info()
dftrain=df.drop(["PassengerId","Name","Ticket","Cabin","Embarked","Survived","Sex"],axis=1)
dftrain.info()
from sklearn.preprocessing import MinMaxScaler 

scaller=MinMaxScaler((0,1))

rescalex=scaller.fit_transform(dftrain)

traininputs=pd.DataFrame(rescalex)

traininputs.head()
dftarget=df["Survived"]
from sklearn.neighbors import KNeighborsClassifier

model=KNeighborsClassifier(n_neighbors=7)

model.fit(traininputs,dftarget)
dftest=pd.read_csv("/kaggle/input/titanic/test.csv")

dftest.info()
dftest=dftest.fillna(method="ffill")
dftest=dftest.drop(["PassengerId","Name","Ticket","Cabin","Embarked","Sex"],axis=1)

dftest.info()
predict=model.predict(dftest)

predict



sub=pd.read_csv("/kaggle/input/titanic/gender_submission.csv")

sub.head()
d = {'PassengerId':sub["PassengerId"] , 'Survived': predict}

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

df = pd.DataFrame(mysubmission)



# create a link to download the dataframe

create_download_link(df)