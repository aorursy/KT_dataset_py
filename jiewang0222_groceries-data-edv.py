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
grocery=pd.read_csv('/kaggle/input/groceries-dataset/Groceries_dataset.csv')
grocery.info()

grocery['Date']=pd.to_datetime(grocery['Date'])
date_diff=grocery['Date'].max()-grocery['Date'].min()

date_diff
print(grocery.head(20))
gr=grocery.groupby('Member_number').count().sort_values(['Date','itemDescription'],ascending=False)



print(gr.head(5))

# create a list of our conditions

conditions = [

    (gr['Date']  > 0) & (gr['Date']  <=10),

    (gr['Date'] > 10) & (gr['Date'] <= 20),

    (gr['Date'] > 20)

    ]



# create a list of the values we want to assign for each condition

values = ['low','middle','high']



# create a new column and use np.select to assign values to it using our lists as arguments

gr['frequency']=np.select(conditions, values)

gr.head(5)
import matplotlib.pyplot as plt

gr1=gr.groupby('frequency').count()

gr1.sort_values('Date',ascending=False).plot.bar()

plt.title('counting for frequency')

plt.show()
gr1.head(5)
gr1['fraction']=gr1['Date']/gr1['Date'].sum()

# Creating plot 

fig = plt.figure(figsize =(10, 7)) 

plt.pie(gr1['fraction'], labels = gr1.index,autopct='%1.2f') 

plt.legend(title='Frequency') 

# show plot 

plt.show() 