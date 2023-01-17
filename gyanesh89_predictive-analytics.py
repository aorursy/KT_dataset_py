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
d=pd.read_csv("/kaggle/input/absenteeismatwork/Absenteeism_at_work.csv")
d.head()
d.columns
d.isna().sum()
np.array([d["Absenteeism time in hours"]==0]).sum()
x=d.groupby("ID")["Absenteeism time in hours"].sum()
(x==0).sum()
d["ID"].nunique()/len(d)
x.plot(kind="bar")
d["Absenteeism time in hours"].plot(kind="box")
d["Absenteeism time in hours"]=d["Absenteeism time in hours"].apply(lambda x: x/10 if x>=20 else x)
d["Absenteeism time in hours"].plot(kind="kde")
x=d.groupby("ID")["Absenteeism time in hours"].sum()
x.plot(kind="bar")
d.columns
d[["Reason for absence","Absenteeism time in hours"]].plot(kind="bar")
d.describe()

d.Seasons=d.Seasons.astype(str)
d.dtypes
x=d.groupby(["Seasons","Body mass index"])["Absenteeism time in hours"].sum()
x[x["Seasons"]=="3"].plot()
d["Body mass index"].describe()
d["BMI"]=pd.cut(d["Body mass index"],[19,25,38],labels=("Fit","OvW"))
d[["BMI","Body mass index"]]