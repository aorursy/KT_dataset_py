# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
with open('../input/fraudulent-email-corpus/fradulent_emails.txt','r',encoding='utf-8',errors='ignore') as file:
        email_text = file.read()
print (email_text)
#Extract the from address
import re

for line in re.findall('From:.*', email_text):
    print(line)
#Isolate name
match = re.findall("From:.*", email_text)
#print (match)
for line in match:
     print (line)
     name = re.findall('\s(.*?)<', line) #The question mark makes the preceding token in the regular expression optional. colou?r matches both colour and color. The question mark is called a quantifier.
     print (name)
     name_final =  [i.replace('"', '') for i in name]
     print (name_final)
#Get email-id
for line in match:
     print(re.findall("\w\S*@.*\w", line))
#Isolate the username
for line in match:
     print(re.findall("(\w\S*)@", line))
#Isolate the domain name
for line in match:
     print(re.findall("@(.*\w)", line))