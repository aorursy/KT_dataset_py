# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as pyplot



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



df = pd.read_csv('../input/en.openfoodfacts.org.products.tsv',sep='\t',low_memory=False)
data = np.transpose(df[['fat_100g','energy_100g']].dropna().as_matrix())
filter = np.logical_and(data[0]>20 , data[1]<4000)

data = data[:,filter]
training_data_x,training_data_y = data[0][900:1000],data[1][900:1000]

test_data_x,test_data_y = data[0][1000:1100],data[1][1000:1100]
pyplot.scatter(test_data_x,test_data_y)

pyplot.scatter(training_data_x,training_data_y)