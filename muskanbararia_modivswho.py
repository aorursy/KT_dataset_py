import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import re
#getting all the sheet_names
df_read = pd.read_excel('../input/convertcsv.xlsx', sheet_name='Sheet 1')
print("Column headings:")
sheet_names=[]
for index, row in df_read.iterrows():
    sheet_names.append(row[0])
print(len(sheet_names))
#Formatting the sheet names
sheet=[]
for each in sheet_names:
    word=  re.findall(r"[a-zA-Z]+", each)
    stn=''
    for each2 in word:
        stn+=each2+" "
    sheet.append(stn[:-1])
print(sheet)
#getting the data array
temp=[]

for row in df_read.iterrows():
    index, data = row
    temp.append(data.tolist())
for i in range(0,len(temp)):
    temp[i]=temp[i][2:92]

#checking if all data is populated
print(len(temp)==len(sheet))
#Populating sheets 
path = "write.xlsx"
writer = pd.ExcelWriter(path, engine='openpyxl')
for j in range(0,len(sheet)):
    df=pd.DataFrame(columns=["Month-Year","Index"])
    year=2011
    month=3
    i=0
    while year!=2019:
        month+=1
        if month==13:
            month=1
            year+=1
        i+=1
        df = df.append({'Month-Year': str(month)+"-"+str(year),'Index':temp[j][i]}, ignore_index=True)
        if month==8 and year==2018:
            break
    df.to_excel(writer, sheet[j], index=False)
writer.save()
excel_file = 'line.xlsx'
writer = pd.ExcelWriter(excel_file, engine='xlsxwriter')
for each in sheet:
    df = pd.read_excel('write.xlsx', sheet_name=each)
    sheet_name=each
    if len(each)>30:
        sheet_name=each[:30]
    
    df.to_excel(writer, sheet_name=sheet_name, index=False)
    workbook = writer.book
    worksheet = writer.sheets[sheet_name]

    # Create a chart object.
    chart = workbook.add_chart({'type': 'line'})
    
    # Configure the series of the chart from the dataframe data.

    chart.add_series({
        'categories': [sheet_name, 1, 0, 90, 0],
        'values':     [sheet_name, 1, 1, 90, 1],
    })

    # Configure the chart axes.
    chart.set_x_axis({'name': 'Index', 'position_axis': 'on_tick'})
    chart.set_y_axis({'name': 'Value', 'major_gridlines': {'visible': False},'min':100})

    # Turn off chart legend. It is on by default in Excel.
    chart.set_legend({'position': 'none'})

    # Insert the chart into the worksheet.
    worksheet.insert_chart('D2', chart)
writer.save()
