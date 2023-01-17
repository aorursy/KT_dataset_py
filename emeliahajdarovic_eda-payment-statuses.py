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
import h2o

h2o.init()

data=h2o.import_file("../input/default-of-credit-card-clients-dataset/UCI_Credit_Card.csv")

data.head()
rows=data["ID"].dim[0]

rows
pay_columns = ['PAY_0','PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

pay_columns
from matplotlib import pyplot as plt

#list for monthes to display

mon=['April','May', 'June', 'July', 'August', 'September']

stat=[-2,-1,0,1,2,3]

j=0#counter



import os

os.system("")



for s in stat:

    j=0#counter

    CVIOLET = '\33[35m'

    print (CVIOLET+"RATES FOR STATUS",s,":\n")

    default_percent_list=[]

    

    for col in pay_columns:

        #config group

        c=df[df[col]==s]

        

        #get stats

        i=len(c["ID"])

        p=(i/rows)*100

        #default rate within people with status "s"

        d=(c["default.payment.next.month"].sum()/i)*100

        default_percent_list.append(d)



        #print

        print (mon[j],":",i ,"people")

        print ("frequency rate",":",p, "%")

        print ("default rate",":",d, "%\n")

        j+=1

        

    print("LINE GRAPH FOR STATUS",s,":\n")

    plt.figure()

    plt.plot([1,2,3,4,5,6], default_percent_list) 

    plt.title("default rate for status above")

    plt.xlabel("Month")

    plt.ylabel("Default Percentage")

    plt.show()

        

        

        
from matplotlib import pyplot as plt

from matplotlib.pyplot import plot

#list for monthes to display

mon=['April','May', 'June', 'July', 'August', 'September']

stat=[-2,-1,0,2,3]

#skip 1 because too little amount of frequencies 

j=0#counter



for s in stat:

    k=0;#counter for monthes

    for col in pay_columns:

        #config group

        c=df[df[col]==s]

        plt.figure()

        colors=['#4214FB','#D11ED4']

        c["default.payment.next.month"].value_counts().plot(kind='bar', color=colors)

        plot

        t=mon[k],s

        plt.title(t)

        plt.xlabel("Default Status")

        plt.ylabel("Frequency")

        k+=1

    

    