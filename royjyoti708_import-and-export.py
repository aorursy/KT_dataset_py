#Importing Dataset

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
