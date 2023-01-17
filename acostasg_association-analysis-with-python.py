# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

import csv

import re

from datetime import datetime



#Normalize shopping data to input csv

#Association Analysis with Python

#http://aimotion.blogspot.com.es/2013/01/machine-learning-and-data-mining.html

#https://github.com/acostasg/normalize_shopping_data



data_set = []

count_shopping = 0



def get_date_from_row():

    global data_raw, match

    data_raw = row.pop(0)

    match = re.search(r'\d{1,2}/\d{1}/\d{4}', data_raw)





def get_first_value():

    global date

    date = datetime.strptime(match.group(), '%d/%m/%Y').date()

    fist_value = data_raw.replace(match.group(), '')

    data_set.append([date, count_shopping, fist_value])





def add_value_to_array():

    if value.strip():

        data_set.append([date, count_shopping, value.strip()])





with open('../input/dataset.csv', newline='') as f:

    reader = csv.reader(f)

    for row in reader:

        get_date_from_row()



        if match:

            count_shopping = count_shopping + 1

            get_first_value()



            for value in row:

                add_value_to_array()



# Any results you write to the current directory are saved as output.

with open("dataset_group.csv", "w+") as f:

    writer = csv.writer(f)

    writer.writerows(data_set)

    

#TODO to analisys




