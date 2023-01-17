# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



#for charts

import seaborn as sns



# Any results you write to the current directory are saved as output.



#import the dataset

data = pd.read_csv('../input/cereal.csv')
gfx = sns.countplot(data['mfr']).set_title('Cereals by Manufacturer')
import matplotlib.pyplot as plt



pc_data = data['mfr'].value_counts()

pc_labels = data['mfr'].value_counts().index.tolist()



explode = (0.4, 0, 0.2, 0,0,0,0)



plt.pie(pc_data,labels = pc_labels,explode=explode, autopct='%.0f%%', shadow=True)