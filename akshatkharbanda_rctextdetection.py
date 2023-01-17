# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
from numpy import nan as Nan
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # plotting

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/rcdata/RC'):
    i=0
    #for filename in filenames:
        #print(str(i),os.path.join(dirname, filename))
        #i=i+1
        
# Text detection in images required packages 
import cv2 
import pytesseract 
from PIL import Image
from zipfile import ZipFile
import boto3

#Data extraction from text
import re


# Any results you write to the current directory are saved as output.
fdf={"IMG_NAME":[]}
for img in filenames:
    fdf['IMG_NAME'].append(img)
fdf= pd.DataFrame.from_dict(fdf, orient='columns', dtype=None, columns=None)   
fdf
#Analyzing noise pattern on a random image
img = cv2.imread('/kaggle/input/rcdata/RC/txt_mudit_b11_1361.jpg',0)
height = np.size(img, 0)
width = np.size(img, 1)
print(height,width)
row, col = img.shape
gauss = np.random.normal(10,10,(row,col))
noisy = img + gauss
smooth_part = noisy[:30, :30]

plt.subplot(221),plt.imshow(noisy,cmap = 'gray')
plt.title('Noisy Image'), plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(smooth_part,cmap = 'gray')
plt.title('Smooth Part'), plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.hist(noisy.ravel(),256,[0,256])#; plt.show()
plt.title('Noisy Image Histogram'), plt.xticks([]), plt.yticks([])
plt.subplot(224),plt.hist(smooth_part.ravel(),256,[0,256])#; plt.show()
plt.title('Estimated Noise Distribution'), plt.xticks([]), plt.yticks([])
plt.show()
i=0

for img in filenames:
 # A text file is created and flushed 
 i= i+1
 file = open("r%d.txt" %i, "w+") 
 file.write("") 
 file.close() 
 
 img1 = img
 img = cv2.imread('/kaggle/input/rcdata/RC/%s' % img)

 pil_img = Image.fromarray(img)
 pil_img.save('/kaggle/working/%s' % img1 ,dpi=(300,300))

# Document
 documentName = '/kaggle/working/%s' % img1 

# Read document content
 with open(documentName, 'rb') as document:
    imageBytes = (document.read())

# Amazon Textract client
 textract = boto3.client('textract', region_name='us-west-2',  aws_access_key_id='AKIAIFS32F3H4AFXVQYA', aws_secret_access_key='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')
 

# Call Amazon Textract
 response = textract.detect_document_text(Document={'Bytes': imageBytes})

#print(response)

# Print detected text
 for item in response["Blocks"]:
    if item["BlockType"] == "LINE":
        print ('\033[94m' +  item["Text"] + '\033[0m')
         # Open the file in append mode 
        file = open("r%d.txt" %i, "a")
        # Appending the text into file 
        file.write(item["Text"])
        file.write("\n")
        # Close the file 
        file.close
df0={"REG_DATE":[] ,"REG_UPTO":[] }
for i in range(len(filenames)):
 i=i+1
 k=0
# Open the file that you want to search 
 f = open("/kaggle/working/r%d.txt" %i, "r")

# Will contain the entire content of the file as a string
 content = f.read()

# The regex pattern that I created
 pattern = "\d{1,2}[/-]\w{2,}[/-]\d{4}" 

# Will return all the strings that are matched
 dates = re.findall(pattern, content)
 if len(dates)==0:
    df0["REG_DATE"].append(Nan)
    df0["REG_UPTO"].append(Nan)
 elif len(dates)==1:
   for date in dates:
     if "-" in date:
        day, month, year = date.split("-")
     else:
        day, month, year = date.split("/")
     if int(year)>2020:
         df0["REG_DATE"].append(Nan)
         df0["REG_UPTO"].append(date)
     else:
        df0["REG_DATE"].append(date)
        df0["REG_UPTO"].append(Nan)
 else:
    df0["REG_DATE"].append(dates[0])
    df0["REG_UPTO"].append(dates[1])
 f.close()
df0= pd.DataFrame.from_dict(df0, orient='columns', dtype=None, columns=None) 
df0["REG_DATE"] = df0["REG_DATE"].astype('datetime64')
df0["REG_UPTO"] = df0["REG_UPTO"].astype('datetime64')
for i in range(len(df0.REG_DATE)):
    try:
     if int(df0.REG_DATE[i].strftime("%Y"))>int(df0.REG_UPTO[i].strftime("%Y")):
        m=df0.REG_DATE[i]
        df0.REG_DATE[i]=df0.REG_UPTO[i]
        df0.REG_UPTO[i]=m
        df0["REG_DATE"]= df0["REG_DATE"].dt.strftime('%d/%m/%Y')
        df0["REG_UPTO"]= df0["REG_UPTO"].dt.strftime('%d/%m/%Y')
    except:
        continue
df0
df1={"MFG_DATE":[]}
for i in range(len(filenames)):
 i=i+1
# Open the file that you want to search 
 f = open("/kaggle/working/r%d.txt" %i, "r")

# Will contain the entire content of the file as a string
 content = f.read()

# The regex pattern that I created
 pattern = "[^\/]\d{1,2}[/]\d{4}" 

# Will return all the strings that are matched
 dates = re.findall(pattern, content)
 if len(dates)==0:
    df1["MFG_DATE"].append(Nan)
 else:
    k=0
    for i in range(len(dates)):
      month,year= dates[i].split("/")
      if int(year)<2020 and int(month)<13 and k==0:
          df1["MFG_DATE"].append(dates[i])
          k=1
 f.close()
import datetime
df1= pd.DataFrame.from_dict(df1, orient='columns', dtype=None, columns=None)
try:
 df1["MFG_DATE"] = df1["MFG_DATE"].astype('datetime64[ns]')
 df1["MFG_DATE"]= df1["MFG_DATE"].dt.strftime('%m/%Y')
except: 
 pass
df1
df2={"CHASSIS_NUM":[]}
for i in range(len(filenames)):
 i=i+1
# Open the file that you want to search 
 f = open("/kaggle/working/r%d.txt" %i, "r")

# Will contain the entire content of the file as a string
 content = f.read()

# The regex pattern that I created
 pattern = "\w{9,11}\s?\d{5,6}" 

# Will return all the strings that are matched
 chas = re.findall(pattern, content)
 if len(chas)==0:
    df2["CHASSIS_NUM"].append(Nan)
 else:
    if len(chas[0])<17 and chas[0][0].isdigit()==False and chas[0][0]!="M":
        df2["CHASSIS_NUM"].append("M"+chas[0])
    elif len(chas[0])>16 and chas[0][0].isdigit()==False and chas[0][0]!="M":
        df2["CHASSIS_NUM"].append("M"+chas[0][1:])
    else:
        df2["CHASSIS_NUM"].append(chas[0])
 f.close()
df2= pd.DataFrame.from_dict(df2, orient='columns', dtype=None, columns=None)
df2
df3={"REGN_NUM":[]}
for i in range(len(filenames)):
 i=i+1
# Open the file that you want to search 
 f = open("/kaggle/working/r%d.txt" %i, "r")

# Will contain the entire content of the file as a string
 content = f.read()

# The regex pattern that I created
 pattern = "[A-Z]{2,3}\d\w[-]?[A-Z][-]?\D?\d{4}" 

# Will return all the strings that are matched
 regnum = re.findall(pattern, content)
 if len(regnum)==0:
    df3["REGN_NUM"].append(Nan)
 else:
    df3["REGN_NUM"].append(regnum[0])
 f.close()
df3= pd.DataFrame.from_dict(df3, orient='columns', dtype=None, columns=None)
df3
df4={"ENG_NUM":[]}
for i in range(len(filenames)):
 i=i+1
# Open the file that you want to search 
 f = open("/kaggle/working/r%d.txt" %i, "r")

# Will contain the entire content of the file as a string
 content = f.read()

# The regex pattern that I created
 pattern = "[A-Z][0-9]\w{3,4}\s?\d{5,6}"

# Will return all the strings that are matched
 engnum = re.findall(pattern, content)
 if len(engnum)==0:
    df4["ENG_NUM"].append(Nan)
 else:
    df4["ENG_NUM"].append(engnum[-1])
 f.close()
df4= pd.DataFrame.from_dict(df4, orient='columns', dtype=None, columns=None)
df4
df5={"NAME":[]}
for i in range(len(filenames)):
 c=0
 i=i+1
# Open the file that you want to search 
 f = open("/kaggle/working/r%d.txt" %i, "r")

# Will contain the entire content of the file as a string
 content = f.readlines()

# The regex pattern that we created
 pattern = "[A-Z]{4,13}\s?[A-Z]{1,13}\s?[A-Z]{1,13}?$" 

# Will return all the strings that are matched but don't belong to list of words in matches
 for m in range(len(content)):
  matches=['UPTO', 'MOTOR','STATION','ROAD','COLOR','SILV','GRE','BLACK','WHEEL','VIHAR','STREET','BAGH','APTS','HOUSE',
          'ENCLAVE','NORTH','EAST','SOUTH','WEST','PARK','RESIDENT','GARDEN','WHITE','BLU','RED','MARG','LICENCE','HOSPITAL','AUTO','SILKY',
          'SLS','SCHOOL','GOVERNMENT','REGISTRATION','AUTHORITY','CHOCO','NOTCH','ADDRESS','VEHICLE','OF','INDIA','LTD','SALOON','SEATING',
          'COLOUR','PETROL','DIESEL','MUL','TATAEL','VILLAGE','INDIGO','STANDING','SEAT','MAGMA','APPT','PEARL','PVT','CERTIFICATE','FORM',
          'VEHICLE','MFG','NAME']
  nam = re.findall(pattern, content[m])
  if len(nam)!=0 and (any(x in str(nam[0]) for x in matches)==False):
   df5["NAME"].append(nam[-1])
   c=1
   break
 if c==0:
    df5["NAME"].append(Nan)
 f.close()
df5= pd.DataFrame.from_dict(df5, orient='columns', dtype=None, columns=None)
df5
df=pd.concat([fdf,df3,df2,df5,df4,df0.REG_DATE,df1], axis=1)
df
df.to_csv('train_output.csv', index=False) 