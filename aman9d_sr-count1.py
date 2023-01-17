import pandas as pd

Total_data = pd.read_csv("../input/export_123.csv", index_col=False)

data_pendingCus = Total_data[Total_data['Status']=="Pending Customer"]

b = data_pendingCus['Assignment Group'].value_counts().index

final = pd.ExcelWriter("final.xlsx")

data_pendingCus.to_excel(final,index=False,sheet_name= "Sheet1" )

Incident_with_Expected_Finish_Date = data_pendingCus[data_pendingCus["Expected Finish Date"].isnull() == False]

Incident_with_Expected_Finish_Date.to_excel(final,index=False,sheet_name= "Incident_with_EFD" )

Incident_without_Expected_Finish_Date = data_pendingCus[data_pendingCus["Expected Finish Date"].isnull() == True]

Incident_without_Expected_Finish_Date.to_excel(final,index=False,sheet_name= "Incident_without_EFD" )

for i in range(1,len(b)):

    #print(i)

    if len(b[i])>30:

        a = b[i]

        a = a.split(" SUPPORT ")

        a = a[1]

    else:

        a = b[i]

    df = data_pendingCus[data_pendingCus["Assignment Group"]==b[i]]

    df.to_excel(final,index=False, sheet_name=a)