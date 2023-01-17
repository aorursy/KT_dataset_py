# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
print('hello')
import os

import pandas as pd

from zipfile import ZipFile



#Unzipping the file

# zip_file_directory = input('Enter the path to the zip file')

# zip_file_name = zip_file_directory[zip_file_directory.find()]

zip_file_name = input("Enter the name of the file  ")



#taking only the filename from filename.zip

rootdir = zip_file_name[:zip_file_name.find(".")]



with ZipFile(zip_file_name, 'r') as zipObj:

    zipObj.extractall(rootdir)

print("successfully extracted")



#Searching through each folder and files inside the directory

dir_list = []

for subdir, dirs, files in os.walk(rootdir):

    for file in files:

        new_file = os.path.join(subdir,file)

        dir_list.append(new_file)

print(dir_list)



#Creating the csv foldername to store the csv file



path = rootdir+"_csv"

mode = 0o666

os.mkdir(path, mode) 



#converting to csv with the save file name as the .sas7bdat file name



for ele in dir_list:

    df1 = pd.read_sas(ele)

    df1.to_csv(path+"\\"+ele[ele.find("\\")+1:ele.find(".")]+".csv",index = False) 

    print(ele)

    print(df1.head(5))