'''Write a Script(R/Python) which fetch the detail from the CSV and create an email template.'''





import pandas as pd

import numpy as np



import pandas as pd

df = pd.read_csv("../input/sample_email.csv",encoding='windows-1254')
import csv

with open("../input/sample_email.csv",encoding='windows-1254', newline='') as f:

    reader = csv.reader(f)

    for row in reader:

        print(row)

import csv



inputfile = csv.reader(open("../input/sample_email.csv",'r',encoding='windows-1254'))



for row in inputfile:

    print(row[2])
import email

    
import csv



f = open("../input/sample_email.csv",encoding='windows-1254')

csv_f = csv.reader(f)



for row in csv_f:

    print(row[12])
import csv



f = open("../input/sample_email.csv",encoding='windows-1254')

csv_f = csv.reader(f)



for columns in csv_f:

    print(columns[12])

    

import re

 

fileToRead = 'readText.txt'

fileToWrite = 'emailExtracted.txt'

 

 

delimiterInFile = [',', ';']

 

def validateEmail(strEmail):

    # .* Zero or more characters of any type. 

    if re.match("(.*)@(.*).(.*)", strEmail):

        return True

    return False

 

def writeFile(listData):

    file = open(fileToWrite, 'w+')

    strData = ""

    for item in listData:

        strData = strData+item+'\n'

    file.write(strData)

 

listEmail = []

file = open("../input/sample_email.csv", 'r',encoding='windows-1254') 

listLine = file.readlines()

for itemLine in listLine:

    item =str(itemLine)

    for delimeter in delimiterInFile:

        item = item.replace(str(delimeter),' ')

     

    wordList = item.split()

    for word in wordList:

        if(validateEmail(word)):

            listEmail.append(word)

 

if listEmail:

    uniqEmail = set(listEmail)

    print(len(uniqEmail),"emails collected!")

    writeFile(uniqEmail)

else:

    print("No email found.")
listEmail
len(listEmail)
keywods_search = ['.com']

email_list = []

for i in range(len(df)):

    if ".com" in df["Email Boby"].iloc[i]:

        email_list.append(df["Email Boby"].iloc[i])



final_email_list = []

for i in range(len(email_list)):

    ".com" in email_list[i].split()

    final_email_list.append([j for j in email_list[i].split() if keywods_search[0] in j])

final_email_list
number_list = []

for i in range(len(df)):

    if "-" in df["Email Boby"].iloc[i]:

        number_list.append(df["Email Boby"].iloc[i])

       



#final_email_list = []

#for i in range(len(email_list)):

#    keywods_search[0] in email_list[i].split()

#    final_email_list.append([j for j in email_list[i].split() if keywods_search[0] in j])
number_list
bad_chars = ['-', '()', '!', "*"] 

  

# initializing test string  

test_string = "I have a nearly unreadable font in my Warehouse Software under Export-File Options for 2 days.Please do contact me at 504-621-8927"

  

# printing original string  

print ("Original String : " + test_string) 

  

# using join() + generator to  

# remove bad_chars  

test_string = ''.join(i for i in test_string if not i in bad_chars) 

  

# printing resultant string  

print ("Resultant list is : " + str(test_string)) 
len(number_list)
import re

def transform_record(record):

  new_record = re.search(r"([0-9]{3}-)([0-9]{3}-[0-9]{4}|[0-9]{7})",record)

  if new_record!=None:

     return new_record[1]+new_record[2]

x=['I have a nearly unreadable font in my Warehouse Software under Export-File Options for 2 days.Please do contact me at 504-621-8927',

 'on my detailsheet http://www.rhein-main-versand.de since yesterday evening some Calibri texts are replaced by blue speech bubbles (?!), i.a. also the imprint',

 'I would like to use the studio version, but I can not exchange the preview images, even though they were deleted in the online organizer - i. E. I have changed the order of the products or deleted products so that the preview always shows the first product, but nothing happens.Reach out me at (541) 754-3010',

 'If I then want to pull down the new process (example: Gelsenkirchen-Projekt2.dat) in overview mode with the mouse, to attach it to the first process, I only get the line 1 (data), which is line 2 (order numbers) disappeared.',

 'Unfortunately I can not import CASH-builds into the DAT-procedure without any problem. The order-number line remains empty (ZD recordings work). Reach out me at calbares@gmail.com',

 'There is no order number in SFC.Call me at 123-456-7890',

 'have since yesterday Sales Firstclass 19 Special Edition installed-registered and then I have imported the sales files from the memory card and when viewing the process, simply no order number comes.',

 '- When burning a product preview with products in the IT & OFFICE catalog 12 & 13 The error message appears at step 7 of 9 - Disc Copy is generated. Call me at +1 (123) 456-7890',

 '!!! MBK exception in CopyWorkspace: 13 - process failed !!!']

y=[]

for i in x:

    t=transform_record(i)

    if t!=None:

        y.append(t)

print(y)

    

    
import re

def transform_record(record):

  new_record = re.search(r"([0-9]{3}-|[(][0-9]{3}[)] )([0-9]{3}-[0-9]{4})",record)

  if new_record!=None:

     return new_record[1]+new_record[2]

x=['I have a nearly unreadable font in my Warehouse Software under Export-File Options for 2 days.Please do contact me at 504-621-8927',

 'on my detailsheet http://www.rhein-main-versand.de since yesterday evening some Calibri texts are replaced by blue speech bubbles (?!), i.a. also the imprint',

 'I would like to use the studio version, but I can not exchange the preview images, even though they were deleted in the online organizer - i. E. I have changed the order of the products or deleted products so that the preview always shows the first product, but nothing happens.Reach out me at (541) 754-3010',

 'If I then want to pull down the new process (example: Gelsenkirchen-Projekt2.dat) in overview mode with the mouse, to attach it to the first process, I only get the line 1 (data), which is line 2 (order numbers) disappeared.',

 'Unfortunately I can not import CASH-builds into the DAT-procedure without any problem. The order-number line remains empty (ZD recordings work). Reach out me at calbares@gmail.com',

 'There is no order number in SFC.Call me at 123-456-7890',

 'have since yesterday Sales Firstclass 19 Special Edition installed-registered and then I have imported the sales files from the memory card and when viewing the process, simply no order number comes.',

 '- When burning a product preview with products in the IT & OFFICE catalog 12 & 13 The error message appears at step 7 of 9 - Disc Copy is generated. Call me at +1 (123) 456-7890',

 '!!! MBK exception in CopyWorkspace: 13 - process failed !!!']

y=[]

for i in x:

    t=transform_record(i)

    if t!=None:

        y.append(t)

print(y)





import csv

with open('number.csv', 'w', newline='') as file:

    writer = csv.writer(file)

    writer.writerow(['504-621-8927', '(541) 754-3010', '123-456-7890', '(123) 456-7890'])

    
import csv

with open('email.csv', 'w', newline='') as file:

    writer = csv.writer(file)

    writer.writerow([['asergi[at]apple.com'],

 ['asergi@yahoo.com'],

 ['calbares@gmail.com'],

 ['asergi@gmail.com']])

    
import csv

with open('number_2.csv', 'w', newline='') as file:

    writer = csv.writer(file)

    writer.writerow(['504-621-8927'])

    
df2 = pd.read_csv("email.csv")
df2.head()
df3 = pd.read_csv("number.csv")
df3.head()
df4 = pd.read_csv("number_2.csv")
df4
import smtplib

def gmailsend(email_id, subject, name,name_2,email_body,email_sender,password_sender):

    server = 'smtp.gmail.com'

    port = 587

    sender = email_sender

    password = password_sender

    receivers = email_id

    fromPerson = name_2

    subject = subject

    message = "From:"+fromPerson+"\nTo:"+receivers+"\nMIME-Version: 1.0\nContent-Type: text/html\nSubject: "+subject+"\n\n\n\nHi "+name+"<br><br>"+email_body +"<br><br>"+"Thanks,"+"<br><br>"+name_2

    smtpObj = smtplib.SMTP(server,int(port))

    smtpObj.starttls()

    smtpObj.login(sender,password) 

    smtpObj.sendmail(sender,receivers,message.encode("utf8"))

    print("Mail Sent Successfully")



 
name_2 = input('Enter the name of sender')

email_sender = input('Enter your mail id:')

password_sender = input('Enter your password:')

for i in range(len(df)):



    email_id = df["email"].iloc[i]

    name = df["first_name"].iloc[i]

    subject = df["subject"].iloc[i]

    email_body = df["Email Boby"].iloc[i]

    gmailsend(email_id, subject, name,name_2,email_body,email_sender,password_sender) 

        