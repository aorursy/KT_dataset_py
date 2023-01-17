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
import h2o

h2o.init()

from matplotlib import pyplot as plt
data=h2o.import_file("../input/default-of-credit-card-clients-dataset/UCI_Credit_Card.csv")

data.head()
data.describe()
data.columns
data["default.payment.next.month"].table()
data["PAY_0"].table()
df=pd.read_csv("../input/default-of-credit-card-clients-dataset/UCI_Credit_Card.csv")

df
students=df[df["EDUCATION"]==2]
df = df.rename(columns={'PAY_0': 'PAY_1'})

df
students = df[df["EDUCATION"]==2]

students
for i in range(1,7):

    plt.figure()

    colors=['#FB3C14','#4214FB','#D11ED4','#67F1BB','#DCE22A']

    p=students["PAY_"+ str(i)].value_counts().plot(kind='bar', color=colors)

    plt.title("PAY_"+ str(i))

    plt.xlabel("Payment Status")

    plt.ylabel("Frequency")

    print(p)

    
plt.figure()

colors=['#67F1BB','#DCE22A']

p=students["default.payment.next.month"].value_counts().plot(kind='bar', color=colors)

plt.title("College Educated People Defaults")

plt.xlabel("Default Status")

plt.ylabel("Frequency")

print(p)

s=students["default.payment.next.month"].sum()

percentage=(s/df["default.payment.next.month"].sum())*100

percentage