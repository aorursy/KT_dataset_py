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
df=pd.read_csv("/kaggle/input/consumer/Edureka_Consumer_Complaints_train.csv")
df.head(2)
df.shape
print("The total count of missing values is",df.isna().sum().sum())
missing_val_col=pd.DataFrame(df.isna().sum())
missing_val_col.rename(columns={0:"missing_val_count"},inplace=True)
print("Out of",df.shape[1],"columns, below are the columns having missing values")
missing_val_col[~(missing_val_col["missing_val_count"]==0)]
nan_rows_df=pd.DataFrame(df.apply(lambda x: sum(x.isna()),axis=1))
nan_rows_count=nan_rows_df[~(nan_rows_df[0]==0)].count().tolist()[0]
print("Out of",df.shape[0],"rows,",nan_rows_count,"have missing values")
df.info()
column=df.columns.tolist()
##Dropping Complaint ID as it's of type int
column.remove("Complaint ID")

for i in column:
    print("# ",i)
    print("\t a. No of unique value is :",df[i].nunique() )
    print("\t b. Most frequently occuring",i,"is :",df[i].value_counts().reset_index()["index"].tolist()[0])
print("Below are the top 10 issues raised by consumers :")
df["Issue"].value_counts().reset_index().head(10)["index"]
print("Below are the top 7 products which receive a higher number of complaints :")
df["Product"].value_counts().reset_index().head(7)["index"]
df["Company"].value_counts()
print("Below are the medium for complaint registration :")
df["Submitted via"].unique().tolist()
print("The complaints came from",df["State"].nunique(),"states and below are the list of reigons.")
df["State"].unique()
df["month"]=df["Date received"].apply(lambda x: str(x).split("-")[1])
df.groupby("month")["month"].count().sort_values(ascending=False)
import datetime
import calendar
df["weekday"]=df["Date received"].apply(lambda date:calendar.day_name[datetime.datetime.strptime(date,"%Y-%m-%d").weekday()])
df.groupby("weekday")["weekday"].count().sort_values(ascending=False)
df["Company public response"].value_counts()
d1=df[df["Timely response?"]=="Yes"]
len(d1[d1["Consumer disputed?"]=="Yes"])
len(d1[d1["Consumer disputed?"]=="No"])
