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
import pandas as pd

data=pd.read_csv("../input/input-data/input.csv")
data.head()
data.head()
data2=pd.read_excel("../input/final-data/DATA1.xlsx")
data2.head()
data3=pd.read_csv("../input/depramnetal-data/Department_Information.txt",sep="|")
data3.head()
data4=pd.read_sas("../input/sas-dataset/employee_information.sas7bdat")
data4
data3.to_csv("mydata.csv")
my_data=data2.to_excel("excel_data.xlsx")
data3.to_csv("text_data.txt")
data3.columns
def fun1(x):

    for i in x:

        data3["new_id"]=int(i[6:9])

fun1(data3.Department_ID)        
data3.head()
data3.dtypes
data3["new_id_of_char"]=data3.new_id.astype("str")
data3.head()
data3.dtypes
#convert into standar time date format
import datetime as datetime
data3["DOE"]=pd.to_datetime(data3.DOE)
data3.dtypes
def fun2(x):

    for i in x:

        data3["DOE_Date"]=i.date()

        data3["DOE_year"]=i.year

        data3["DOE_month"]=i.month

        data3["DOE_day"]=i.weekday_name

fun2(data3.DOE)        

data3
from datetime import datetime

def fun3(x):

    for i in x:

        z=datetime.strftime(i,"%m/%d/%Y")

        print(z)

fun3(data3.DOE)        

        
from datetime import datetime

def fun4(x):

    for i in x:

        z=datetime.strftime(i,"%d/%m/%Y")

        print(z)

fun4(data3.DOE) 
from datetime import datetime

def fun5(x):

    for i in x:

        z=datetime.strftime(i,"%b %d ,%Y")

        print(z)

fun5(data3.DOE) 
from datetime import datetime

def fun4(x):

    for i in x:

        z=datetime.strftime(i,"%d-%m-%Y")

        print(z)

fun4(data3.DOE) 