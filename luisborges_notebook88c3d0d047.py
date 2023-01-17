import xlrd



dataset = ("../input/covid19-10092020/COVID-19_10-09-2020.xlsx")



workbook = xlrd.open_workbook(dataset)

sheet = workbook.sheet_by_index(0)



sheet.cell_value(0, 0)



for i in range(sheet.ncols):

    print(sheet.cell_value(0, i))
import pandas as pd



dataset = pd.read_excel("../input/covid19-10092020/COVID-19_10-09-2020.xlsx")



for col_name in dataset.columns: 

    print(col_name)
