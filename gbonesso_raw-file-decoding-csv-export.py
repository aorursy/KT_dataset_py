# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import time

from datetime import datetime



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

print(check_output(["ls", "../input/COTAHIST_A2017"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Here we'll specify the columns width

widths = [2, 8, 2, 12, 3, 12, 

          10, 3, 4, 13, 13, 13, 13, 13, 13, 13, 5, 18, 

          18, 13, 1, 8, 7, 13, 12, 3]



# and the columns names, based in the original specification

column_names = ['TIPREG', 'DATPRE', 'CODBDI', 'CODNEG', 'TPMERC', 'NOMRES', 'ESPECI', 'PRAZOT', 

                'MODREF', 'PREABE', 'PREMAX', 'PREMIN', 'PREMED', 'PREULT', 'PREOFC', 'PREOFV', 

                'TOTNEG', 'QUATOT', 'VOLTOT', 'PREEXE', 'INDOPC', 'DATVEN', 'FATCOT', 'PTOEXE', 

                'CODISI', 'DISMES']
# Most of the prices are defined with two decimals. 

# This function is used to adjust this while loading...

def convert_price(s):

    return (float(s) / 100.0)



# The date fields are in the format YYYYMMDD

def convert_date(d):

    struct = time.strptime(d, '%Y%m%d')

    dt = datetime.fromtimestamp(time.mktime(struct))

    return(dt)
# Specify dtype while loading

dtype_dict = {

    'TOTNEG':np.int32

}



# Use the functions defined above to convert data while loading

convert_dict = {

    'DATPRE':convert_date, 

    'PREABE':convert_price, 'PREMAX':convert_price, 

    'PREMIN':convert_price, 

    'PREMED':convert_price, 'PREULT':convert_price, 'PREOFC':convert_price, 

    'PREOFV':convert_price,

    'DATVEN':convert_date, 

}
# Load the raw file

def load_and_preprocess(file_path):

    df = pd.read_fwf(

        file_path, 

        widths=widths, 

        names=column_names, 

        dtype=dtype_dict, 

        converters=convert_dict,

        skiprows=1,              # Skip the header row

        skipfooter=1             # Skip the footer row

    )

    return(df)
# Read all files and concatenate in one Dataframe

df1 = load_and_preprocess("../input/COTAHIST_A2017/COTAHIST_A2017.TXT")

df2 = df1.append(load_and_preprocess("../input/COTAHIST_A2016/COTAHIST_A2016.TXT"), ignore_index=True)

df3 = df2.append(load_and_preprocess("../input/COTAHIST_A2015/COTAHIST_A2015.TXT"), ignore_index=True)

df4 = df3.append(load_and_preprocess("../input/COTAHIST_A2014/COTAHIST_A2014.TXT"), ignore_index=True)

df5 = df4.append(load_and_preprocess("../input/COTAHIST_A2013/COTAHIST_A2013.TXT"), ignore_index=True)

df6 = df5.append(load_and_preprocess("../input/COTAHIST_A2012/COTAHIST_A2012.TXT"), ignore_index=True)

df7 = df6.append(load_and_preprocess("../input/COTAHIST_A2011/COTAHIST_A2011.TXT"), ignore_index=True)

df8 = df7.append(load_and_preprocess("../input/COTAHIST_A2010/COTAHIST_A2010.TXT"), ignore_index=True)

df9 = df8.append(load_and_preprocess("../input/COTAHIST_A2009/COTAHIST_A2009.TXT"), ignore_index=True)

df  = df9.append(load_and_preprocess("../input/COTAHIST_A2018/COTAHIST_A2018.TXT"), ignore_index=True)

pd.set_option('display.max_columns', 26)

df.head()
df.to_csv('COTAHIST_A2009_to_A2018P.csv')