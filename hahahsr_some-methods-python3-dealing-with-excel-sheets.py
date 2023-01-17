import os
import xlrd
import pandas as pd
path='F:\\kaggle\\kernels\\Excel aggregating'
excels=os.listdir(path) #see how many files in the folder
a = list(excels)
for i in range(len(excels)):
    a[i] = path + excels[i] #add each Excel Sheet in the list 'a'

newfloder='F:\\Python\\' # new folder to save data
sheet_names=list(excels) # new list
for i in range(len(excels)):
    sheet_names[i]=xlrd.open_workbook(excels[i]).sheet_names()
    for j in range(len(sheet_names[i])):
        if 'calve' in sheet_names[i][j]:
            df=pd.read_excel(excels[i],sheet_name=sheet_names[i][j],encoding='utf8')
            df.to_csv(newfile+excels[i]+sheet_names[i][j]+'.csv',encoding="utf_8_sig",index=False)
files=os.listdir(newfolder) #the csv data we transformed
SaveFile_Name='agg.csv'
SaveFile_path="F:\\Python\\agg"
df1 = pd.read_csv(newfolder+ files[0])   
#read the first file and save it in new path
df1['name']=calvefiles[0] # define a column to indicate which file the data from
df2=df1.loc[:,["ID","parity","date","afc","name"]] #the header you want
df2.to_csv(SaveFile_path+'\\'+SaveFile_Name,encoding="utf_8_sig",index=False)

#iterations for files in the folder
try:
    for i in range(1,len(files)):
        df1 = pd.read_csv(newfolder+ files[i])
        df1['name']=calvefiles[1]
        df2=df1.loc[:,["ID","parity","date","afc","name"]]
        df2.to_csv(SaveFile_path+'\\'+SaveFile_Name,encoding="utf_8_sig",index=False, header=False, mode='a+')
except Exception as err:
    print(err)
    print(files[i])     #print the filename that has problem
df1 = pd.read_excel(newfolder+ files[0],sheet_name="calve") #you can even choose the sheet  
#read the first file and save it in new path
df1['name']=calvefiles[0] # define a column to indicate which file the data from
df2=df1.loc[:,["ID","parity","date","afc","name"]] #the header you want
df2.to_csv(SaveFile_path+'\\'+SaveFile_Name,encoding="utf_8_sig",index=False)

#iterations for files in the folder
try:
    for i in range(1,len(files)):
        df1 = pd.read_excel(newfolder+ files[i],sheet_name="calve")
        df1['name']=calvefiles[1]
        df2=df1.loc[:,["ID","parity","date","afc","name"]]
        df2.to_csv(SaveFile_path+'\\'+SaveFile_Name,encoding="utf_8_sig",index=False, header=False, mode='a+')
except Exception as err:
    print(err)
    print(files[i])     #print the filename that has problem