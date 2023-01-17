# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



def clean_date(date):

    if len(date.split("."))==3:

        return date

    else:

        return 0

    

    

    





GSAF = "../input/attacks.csv"



Data = pd.read_csv(GSAF, encoding = 'ISO-8859-1')



Data['Date'] = Data['Date'].astype(str)



Data['CleanDate'] = Data['Date'].apply(clean_date)



#Data = Data[Data.CleanDate != 0]



print("Date",Data.Date.value_counts())

print("CleanDate",Data.CleanDate.value_counts())



# Any results you write to the current directory are saved as output.