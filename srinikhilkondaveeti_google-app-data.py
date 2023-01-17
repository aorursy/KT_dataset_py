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
appdt=pd.read_csv('/kaggle/input/google-play-store-apps/googleplaystore.csv')
for i in range(len(appdt['Reviews'])):

    if 'm' in appdt['Reviews'][i].lower():

        appdt['Reviews'][i]=float(appdt['Reviews'][i][:-1])*1e6

    else:

        appdt['Reviews'][i]=float(appdt['Reviews'][i])

        

        
appdt.sort_values(by='Reviews',ascending=False,inplace=True)
answer=appdt['Reviews'].max()
answer
app_answer=appdt[appdt['Reviews']==appdt['Reviews'].max()]
app_answer
appdt.head(10)