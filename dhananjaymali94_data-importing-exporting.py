import pandas as pd

import numpy as np

import os
print(os.listdir("../input"))
# Importing csv data .

Department_Information = pd.read_csv("../input/student-performance-dataset/Department_Information.csv")

Department_Information
Department_Information.shape
Employee_Information = pd.read_csv("../input/student-performance-dataset/Employee_Information.csv")

Employee_Information
Employee_Information.shape
Student_Counceling_Information = pd.read_csv('../input/student-performance-dataset/Student_Counceling_Information.csv')

Student_Counceling_Information
Student_Counceling_Information.shape
Student_Performance_Data = pd.read_csv('../input/student-performance-dataset/Student_Performance_Data.csv')

Student_Performance_Data
Student_Performance_Data.shape
# Importing xlsx data.

Department_Information1 = pd.read_excel("../input/student-performance-dataset/Department_Information.xlsx")

Department_Information1
Student_Counceling_Information1 = pd.read_excel('../input/student-performance-dataset/Student_Counceling_Information.xlsx')

Student_Counceling_Information1
Student_Performance_Data1 = pd.read_excel('../input/student-performance-dataset/Student_Performance_Data.xlsx')

Student_Performance_Data1
# Importing txt file with sep = '|'

Department_Information_2=open('../input/student-performance-dataset/Department_Information.txt',"r",)

for line in Department_Information_2:

    field=line.split("|")

    field1=field[0]

    field2=field[1]

    field3=field[2]

    print(field1+" "+field2+" "+field3)
Employee_Information_2=open('../input/student-performance-dataset/Employee_Information.txt',"r",)

for line in Employee_Information_2:

    field=line.split("|")

    field1=field[0]

    field2=field[1]

    field3=field[2]

    field4=field[3]

    print(field1+" "+field2+" "+field3+" "+field4)
Student_Counceling_Information_2=open('../input/student-performance-dataset/Student_Counceling_Information.txt',"r",)

for line in Student_Counceling_Information_2:

    field=line.split("|")

    field1=field[0]

    field2=field[1]

    field3=field[2]

    field4=field[3]

    field5=field[4]

    print(field1+" "+field2+" "+field3+" "+field4+" "+field5)
Student_Performance_Data_2=open('../input/student-performance-dataset/Student_Performance_Data.txt',"r",)

for line in Student_Performance_Data_2:

    field=line.split("|")

    field1=field[0]

    field2=field[1]

    field3=field[2]

    field4=field[3]

    field5=field[4]

    print(field1+" "+field2+" "+field3+" "+field4+" "+field5)
# Importing txt file without sep=""

file=open('../input/Frogfood.txt',"r",)

print(file.read())