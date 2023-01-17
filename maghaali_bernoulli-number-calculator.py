# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import timeit                  #for calculate time of running program

import pandas as pd            #for create xlsx file and write on that

from sympy import bernoulli    #for calculate bernoulli



def bernoulli_numbers(n):

    start = timeit.default_timer()   #start time

    

    ber_arra = []  #create array for write bernoulli numbers on that

    # with this for loop we write bernoulli numbers on ber_arra[] 

    for i in range(n):

        ber_arra.append(bernoulli(i))

    # now, write ber_arra[] on panda dataframes

    df = pd.DataFrame({'Bernoulli':ber_arra})



    # Create a Pandas Excel writer using XlsxWriter as the engine.

    writer = pd.ExcelWriter('bernoulli.xlsx', engine='xlsxwriter')



    # Convert the dataframe to an XlsxWriter Excel object.

    df.to_excel(writer, sheet_name='bernoulli')



    # Close the Pandas Excel writer and output the Excel file.

    writer.save()

    stop = timeit.default_timer()

    print('Time: ', stop - start) 

#calculate 20 bernoulli number and write thos on excel file

bernoulli_numbers(20)
# now read our file and write on consule

bernoulli_file_location = 'bernoulli.xlsx'

reader = pd.read_excel(bernoulli_file_location)

print(reader)