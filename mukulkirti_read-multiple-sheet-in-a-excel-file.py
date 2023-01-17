import pandas as pd
#load xlsx files in sheet variable

sheet=pd.ExcelFile(r'/kaggle/input/ExcelTestData1.xlsx')
#fetch sheet name in each xlsx fle

sheetName=sheet.sheet_names

sheetName
#parse the exls data and load int variable

   

file1_data=sheet.parse(sheetName[0])

file2_data=sheet.parse(sheetName[0])

    

print(file1_data)    

print(file2_data)
