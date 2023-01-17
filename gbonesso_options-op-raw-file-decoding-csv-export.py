# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import time
from datetime import datetime

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# and the columns names, based in the original specification
column_names = [
    'REC_TYP', #Record Type
    'UND_ASS', #Underlying Asset
    'ASS_SPE', #Underlying Asset Specification
    'EXP_DAT', #Expiration Date
    'SER_NUM', #Serie Number
    'OPT_TYP', #Option Type (070 = Call; 080=Put)
    'OPT_TIC', #Ticker
    'QUO_FAC', #Quotation factor
    'STK_PRI', #Strike Price
    'COV_POS', #Covered Position
    'LOC_POS', #Locked Position
    'UNC_POS', #Uncovered Position
    'TOT_POS', #Total Position
    'HOL_QTY', #Holder Quantity
    'SEL_QTY', #Seller Quantity
    'DIST',    #Distribution???
    'OPT_STY'  #Option Style'
]
# Header columns names, based in the original specification
header_column_names = [
    'REC_TYP', #Record Type
    'POS_DAT', #Position Date
    'PRO_DAT', #Process Date
    'PRO_TIM'  #Process Time
]
# Most of the prices are defined with two decimals. 
# This function is used to adjust this while loading...
def convert_price(s):
    return (float(s) / 100.0)

# The date fields are in the format YYYYMMDD
def convert_date(d):
    struct = time.strptime(d, '%Y%m%d')
    dt = datetime.fromtimestamp(time.mktime(struct))
    return(dt)
# Specify header dtype while loading
header_dtype_dict = {
    'REC_TYP':np.int32
}

# Use the functions defined above to convert data while loading
header_convert_dict = {
    'POS_DAT':convert_date, 
    'PRO_DAT':convert_date
}
# Specify dtype while loading
dtype_dict = {
    'REC_TYP':np.int32
}

# Use the functions defined above to convert data while loading
convert_dict = {
    'EXP_DAT':convert_date#, 
    #'STK_PRI':convert_price
}
# Fuction to get the position date from header
def get_position_date(file_path):
    df = pd.read_csv(
        file_path, 
        names=header_column_names, 
        dtype=header_dtype_dict, 
        converters=header_convert_dict,
        nrows=1, # Read just one line, the header...
        delimiter ='|'
    )
    #print(df.head())
    return(df['POS_DAT'].values[0])
# Load the raw file
def load_and_preprocess(file_path):
    position_date = get_position_date(file_path) # Get position date from header
    df = pd.read_csv(
        file_path, 
        names=column_names, 
        dtype=dtype_dict, 
        converters=convert_dict,
        skiprows=1,               # Skip the header row
        #skipfooter=1             # Skip the footer row
        delimiter ='|'
    )
    df.insert(0, 'POS_DAT', position_date) # insert a new column with position date
    return(df)
# Read all files and concatenate in one Dataframe
df = pd.DataFrame() # Creates an empty Dataframe to append all the raw files 

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        df_temp = load_and_preprocess(os.path.join(dirname, filename))
        print(os.path.join(dirname, filename), df_temp.shape[0]) # shape[0] = Number of rows...
        df  = df.append(df_temp, ignore_index=True)
        
print('Total df rows: ', df.shape[0])
df.head()
df.groupby(['POS_DAT']).count()
df.to_csv(
    'CONCATENATED.csv',
    index = False # Export without creating an index column
)