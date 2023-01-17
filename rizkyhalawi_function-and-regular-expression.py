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
#import library yang dibutuhkan

import re

#function email_check

def email_check(input):

    match = re.search('(?=^((?!-).)*$)(?=[^0-9])((?=^((?!\.\d).)*$)|(?=.*_))',input)

    if match:

        print('Pass')

    else:

        print('Not Pass')

#Masukkan daftar email ke dalam list

emails = ['my-name@someemail.com', 'myname@someemail.com','my.name@someemail.com',

'my.name2019@someemail.com', 'my.name.2019@someemail.com',

'somename.201903@someemail.com','my_name.201903@someemail.com',

'201903myname@someemail.com', '201903.myname@someemail.com']

#Looping untuk pengecekan Pass atau Not Pass

for email in emails :

    email_check(email)