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
from matplotlib import pyplot as plt



status= ['PAY_0','PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']



for col in status:

    plt.figure()

    colors=['#FB3C14','#4214FB','#D11ED4','#67F1BB','#DCE22A']

    p=df[col].value_counts().plot(kind='bar', color=colors)

    plt.title(col)

    plt.xlabel("Payment Status")

    plt.ylabel("Frequency")

    print(p)

    

    

    
df["AGE"].value_counts()
plt.figure(figsize=(20,10))

colors=['#FB3C14','#4214FB','#D11ED4','#67F1BB','#DCE22A','#4EED4C','#F99F16']

p=df["AGE"].value_counts().sort_index(ascending=True).plot(kind='bar', color=colors)

plt.title('Age Distribution')

plt.xlabel("Age")

plt.ylabel("Frequency")

print(p)
#get all people with defaults

default=df[df['default.payment.next.month']==1]

default



plt.figure(figsize=(20,10))

colors=['#FB3C14','#4214FB','#D11ED4','#67F1BB','#DCE22A','#4EED4C','#F99F16']

p=default["AGE"].value_counts().sort_index(ascending=True).plot(kind='bar', color=colors)

plt.title('Default vs. Age Distribution')

plt.xlabel("Age")

plt.ylabel("Defaults")

print(p)
students = df[df["EDUCATION"]==2]

students
plt.figure()

colors=['#FB3C14','#4214FB','#D11ED4','#67F1BB','#DCE22A','#4EED4C','#F99F16']

p=df["EDUCATION"].value_counts().sort_index(ascending=True).plot(kind='bar', color=colors)

plt.title('Education Distribution')

plt.xlabel("Education Status")

plt.ylabel("Frequency")

print(p)
#get all people with defaults

default=df[df['default.payment.next.month']==1]

default



plt.figure()

colors=['#FB3C14','#4214FB','#D11ED4','#67F1BB','#DCE22A','#4EED4C','#F99F16']

p=default["EDUCATION"].value_counts().sort_index(ascending=True).plot(kind='bar', color=colors)

plt.title('Default vs. Education Distribution')

plt.xlabel("Education Status")

plt.ylabel("Defaults")

print(p)
alt_students=students[students["PAY_6"].isin([0,-2])]

alt_students
s=alt_students["default.payment.next.month"].sum()

percentage=(s/df["default.payment.next.month"].sum())*100

percentage