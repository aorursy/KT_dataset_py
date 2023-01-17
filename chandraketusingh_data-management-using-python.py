import numpy as np

import pandas as pd 

import os

print(os.listdir("../input"))

os.chdir("../input")
department_information= pd.read_sas('department_information.sas7bdat')

employee_information= pd.read_sas('employee_information.sas7bdat')
Department_Information = pd.read_csv("Department_Information.csv")

Employee_Information = pd.read_csv("Employee_Information.csv")

Student_Counceling_Information = pd.read_csv("Student_Counceling_Information.csv")

Department_Information = pd.read_excel("Department_Information.xlsx")

Employee_Information = pd.read_excel("Employee_Information.xlsx")

Student_Counceling_Information = pd.read_excel("Student_Counceling_Information.xlsx")
""" To read text file the same function will be used """

""" Because CSV is a secial case of txt file where file is delimited by default system seprator """

""" Default system seprator changes from one country to another """



Department_Information = pd.read_csv("Department_Information.txt" , sep =  "|")

Employee_Information = pd.read_csv("Employee_Information.txt" , sep =  "|")

Student_Counceling_Information = pd.read_csv("Student_Counceling_Information.txt" , sep =  "|")
Department_Information.head()

Employee_Information.head()
Department_Information.shape

Department_Information.shape

Employee_Information.shape

Student_Counceling_Information.shape
Department_Information.to_csv("../Department_Information.csv")

Employee_Information.to_csv("../Employee_Information.csv")

Student_Counceling_Information.to_csv("../Student_Counceling_Information.csv")
Department_Information.to_excel("../Department_Information.xlsx")

Employee_Information.to_excel("../Employee_Information.xlsx")

Student_Counceling_Information.to_excel("../Student_Counceling_Information.xlsx")