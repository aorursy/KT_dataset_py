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
import numpy as np

import pandas as pd 

import os

print(os.listdir("../input"))
#Setting input directory

os.chdir("../input")



#Importing from *.txt

Depart_Info = pd.read_csv("Department_Information.txt" , sep =  "|")

Depart_Info.head()

Employee_Info= pd.read_csv("Employee_Information.txt" , sep =  "|")

Employee_Info.head()

Employee_Info.info()

Student_cons_info= pd.read_csv("Student_Counceling_Information.txt" , sep =  "|")



#reteriving observations of top five rows

Student_cons_info.head()
#Importing data from *.csv

Depart_inf = pd.read_csv("Department_Information.csv")

Depart_inf.head()
#Importing data *.xlsx

Employ_Info = pd.read_excel("Employee_Information.xlsx")

Employ_Info.head()
#Importing from *.sas7bdat

department_info= pd.read_sas('department_information.sas7bdat')

department_info.head()
#Export to  *.txt

Student_cons_info.to_csv("../Student_Counceling_Information.txt")

#Export to *.csv 

Depart_inf.to_csv("../Department_Information.csv")
#Export to *.xlsx

Employ_Info.to_excel("../Employee_Information.xlsx")
#Export to *.sas7bdat 

department_info.to_sas("../department_information.sas7bdat")


