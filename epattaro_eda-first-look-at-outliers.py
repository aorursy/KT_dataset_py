# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/DIRTY_DEPUTIES.csv')
df.columns
## TOP 10 SPENDERS (VALUE) ##



df.groupby('deputy_name')['refund_value'].sum().sort_values(ascending=False)[:10]
## TOP 10 SPENDERS (FREQUENCY) ##



df.groupby('deputy_name')['refund_value'].count().sort_values(ascending=False)[:10]
## TOP 10 HIGHEST TOTAL VALUE COMPANIES ##



df.groupby('company_name')['refund_value'].sum().sort_values(ascending=False)[:10]
## TOP 10 MOST USED COMPANIES ##



df.groupby('company_name')['refund_value'].count().sort_values(ascending=False)[:10]
## TOP 10 SPENDERS BY AREA ##



for area in sorted (set (df['refund_description'])):

    

    print ('='*10, area, '='*10)

    print (df.loc[df['refund_description']==area].groupby('deputy_name')['refund_value'].sum().sort_values(ascending=False)[:10])

    print ('\n')
## TOP 10 FREQUENCY BY AREA ##



for area in sorted (set (df['refund_description'])):

    

    print ('='*10, area, '='*10)

    print (df.loc[df['refund_description']==area].groupby('deputy_name')['refund_value'].count().sort_values(ascending=False)[:10])

    print ('\n')
## TOP SPENDING POLITICAL PARTY ##

# note: deputy_state and political_party columns are switched

df.groupby('deputy_state')['refund_value'].sum().sort_values(ascending=False)[:10]
## TOP SPENDING POLITICAL PARTY PER DEPUTY ##



temp = pd.DataFrame(df.groupby('deputy_state')['refund_value'].sum().sort_values(ascending=False))



for index in temp.index:

    

    temp.loc[index, 'refund_value'] = temp.loc[index, 'refund_value'] / float(len(set(df.loc[df['deputy_state']==index,'deputy_name'])))



print (temp.sort_values('refund_value', ascending=False)[:10])