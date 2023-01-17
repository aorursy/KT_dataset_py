# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



data = []



import csv

import json

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        file_name = os.path.join(dirname, filename)

        print(file_name)

        print('Done!')



# Any results you write to the current directory are saved as output.

def load_tokens(tokens_file):

    try:

        with open(tokens_file) as csvDataFile:

            csvReader = csv.reader(csvDataFile)

            

            for i,row in enumerate(csvReader):

                

                data.insert(i,json.loads(json.dumps({'country':row[0].strip(), 'population':row[2].strip()})))

                

    except IOError:

        print("Error: Tokens file [{}] is not found.".format(tokens_file))

        exit()







load_tokens(file_name)



print(data)
