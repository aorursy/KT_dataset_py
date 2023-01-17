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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

#df = pd.DataFrame({'a':[check_output]})

#df['a'].values.tolist()

#names = ['Bob','Jessica','Mary','John','Mel']

#births = [968, 155, 77, 578, 973]

#print (births);

#BabyDataSet = list(zip(names,births))

#BabyDataSet

#df = pd.DataFrame(data = BabyDataSet, columns=['Names', 'Births'])

location = r'../input/creditcard.csv'

my_data = pd.read_csv(location)

my_data = pd.read_csv

print (my_data)

my_data