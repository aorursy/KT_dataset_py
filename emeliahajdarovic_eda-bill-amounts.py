# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("../input/default-of-credit-card-clients-dataset/UCI_Credit_Card.csv")

df

df.columns
from mpl_toolkits.mplot3d import axes3d

import matplotlib.pyplot as plt

import numpy as np

from matplotlib import style

from matplotlib import pyplot as plt



bills= ['BILL_AMT1', 'BILL_AMT2','BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']

i=0



for col in bills:

    plt.figure()

    colors=['#DCE22A','#4214FB','#D11ED4','#EEA1D8','#15BF1A','#027074']

    m=df[col].hist(color=colors[i],bins=10)

    plt.title(col)

    plt.xlabel("Bill Amount")

    plt.ylabel("Frequency")

    i+=1
bills= ['BILL_AMT1', 'BILL_AMT2','BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']

i=0



mean_list=[]

std_list=[]

max_list=[]

min_list=[]



    

for col in bills:

    print(col,":")

    m=df[col].mean()

    print("mean: ",m)

    s=df[col].std()

    print("std: ",s)

    mi=df[col].min()

    print("min: ",mi)

    ma=df[col].max()

    print("max: ",ma)

    print("count of non n/a: ", df[col].count(),"\n")#check for 30000

    

    #append

    mean_list.append(m)

    std_list.append(s)

    min_list.append(mi)

    max_list.append(ma)

    

#plot the stats

import numpy as np

from matplotlib.pylab import plt #load plot library

# indicate the output of plotting function is printed to the notebook

%matplotlib inline 

    

x = ['APR','MAY','JUN','JUL','AUG','SEP']

y_1 = mean_list

y_2 = std_list

y_3 = max_list

y_4 = min_list

plt.figure(figsize=(8, 8))

plt.plot(x, y_1,color="#FBF527",label='mean')#yellow

plt.plot(x, y_2,color="#EEA1D8",label='std')#pink

plt.plot(x, y_3,color="#15BF1A",label='max')#green

plt.plot(x, y_4,color="#027074",label='min')#blue

plt.legend()

plt.show()



from matplotlib import pyplot as plt

#list for monthes to display

bills= ['BILL_AMT1', 'BILL_AMT2','BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']

mean_list=[]

j=0#counter



import os

os.system("")



for col in bills:

    #get stats

    d=df[col].mean()

    mean_list.append(d)

        

        

plt.figure()

plt.plot(['APR','MAY','JUN','JUL','AUG','SEP'], mean_list) 

plt.title("BILL AMT avgs")

plt.xlabel("Month")

plt.ylabel("Avg")

plt.show()