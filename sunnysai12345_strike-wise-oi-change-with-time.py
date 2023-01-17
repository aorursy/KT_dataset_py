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
%matplotlib inline

import pandas as pd

df = pd.read_csv("/kaggle/input/nse-future-and-options-dataset-3m/3mfanddo.csv")
def transformEntry(source):  

    months = {

        'Jan': '01',

        'Feb': '02',

        'Mar': '03',

        'Apr': '04',

        'May': '05',

        'Jun': '06',

        'Jul': '07',

        'Aug': '08',

        'Sep': '09',

        'Oct': '10',

        'Nov': '11',

        'Dec': '12',

        'JAN': '01',

        'FEB': '02',

        'MAR': '03',

        'APR': '04',

        'MAY': '05',

        'JUN': '06',

        'JUL': '07',

        'AUG': '08',

        'SEP': '09',

        'OCT': '10',

        'NOV': '11',

        'DEC': '12'

    };

    

    date = source;

    date = date.split('-');

    #return date[2]+ "-" + nmonths[date[1]] + "-" +date[1];

    if (len(date) == 3):

        y = date[2];

        if (int(date[2]) < 100):

            y = '20' + date[2];



        m = months[date[1]];

        d = date[0];

        cdate = y + "-" + m + "-" + d;

        return cdate; 

    

    

#transformEntry(""11-Nov-17")

df['TIMESTAMP'] = df['TIMESTAMP'].map(transformEntry)

#print(df.columns)

#print(df.EXPIRY_DT.unique())

strikes = [30000,30500,31000,315000,32000]

tdf = df[(df.SYMBOL  == "BANKNIFTY") & (df.OPTION_TYP == "CE") & (df.EXPIRY_DT == "28-Nov-2019") & (df['STRIKE_PR'].isin(strikes))]

#df.reset_index(inplace=True)

pivot_df = tdf.pivot(index='TIMESTAMP', columns='STRIKE_PR', values='OPEN_INT')

pivot_df.plot.bar(stacked=True, figsize=(10,7))