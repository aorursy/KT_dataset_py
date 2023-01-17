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
data = pd.read_csv("../input/Speed Dating Data.csv", encoding="ISO-8859-1")

data.head(10)
print(data['pid'])
mod_data = data.groupby('iid').sum()

print (mod_data.reset_index())

#print(mod_data['match'])

#plt.figure()

#plt.scatter(x=mod_data.reset_index()['iid'], y=mod_data['match'])

plt.figure()

plt.scatter(mod_data['attr'], mod_data['match'])

attr_data = mod_data.groupby('attr').sum()

plt.figure()

plt.scatter(x=attr_data.reset_index()['attr'], y=attr_data['match'])
