# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



df = pd.read_csv('../input/vgsales.csv')



genrelist = ['Shooter', 'Platform', 'Racing', 'Sports', 'Puzzle', 'Misc', 'Role-Playing', 'Action', 'Simulation']



for genre in genrelist:

    tempNA = df.loc[df['Genre'] == genre, 'NA_Sales'].sum()

    tempEU = df.loc[df['Genre'] == genre, 'EU_Sales'].sum()

    tempJP = df.loc[df['Genre'] == genre, 'JP_Sales'].sum()

    

    print('NA sales for '+genre+ '='+str(tempNA))

    print('EU sales for '+genre+ '='+str(tempEU))

    print('JP sales for '+genre+ '='+str(tempJP))



# Any results you write to the current directory are saved as output.