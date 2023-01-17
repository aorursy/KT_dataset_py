
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


data=pd.read_csv("/kaggle/input/fifa19/data.csv")
data.shape
data.columns
pd.set_option("max_columns",100)
data.tail()
data=data.drop("Unnamed: 0",axis=1)
pd.set_option("max_rows",100)
data.isnull().sum()

x = data.pivot_table(index = 'Nationality',
                     values = 'Age',
                     aggfunc = 'mean').plot(kind='line')
import numpy as np
def convert_fun(val):
    x = val.replace('€', '')
    if 'M' in x:
        x = float(x.replace('M', ''))*1000000
    elif 'K' in val:
        x = float(x.replace('K', ''))*1000
    return float(x)

data['Value']=data['Value'].apply(convert_fun)
data['Wage']=data['Wage'].apply(convert_fun)
#data['Release Clause']=data['Release Clause'].apply(convert_fun)
pd.set_option("max_rows",1000)
x=data[['Name','Nationality','Wage']]
x=x.groupby(['Nationality']).agg('max').sort_values(by='Wage',ascending=False).head(20)
x.style.background_gradient(cmap='Blues')
data['Age'].min()


data[data['Age']==16][['Name','Nationality','Age','Wage']].sort_values(by='Wage', ascending=False).head(5)


y=data[['Preferred Foot','Skill Moves']]
y=y.groupby(['Preferred Foot']).agg('sum').sort_values(by='Skill Moves',ascending=False).head(20)
y.style.background_gradient(cmap='Blues')

x=data[data['Contract Valid Until'] =='2018']
x=x[['ID','Nationality']]
y=x.groupby('Nationality').agg('count').sort_values(by='ID', ascending=False).head(10)
y
x=data[data['Nationality'] =='India']
x['Name']
x=data[data['ID'] ==158023]
y=data[data['ID']==20801]
x=x[['Name','International Reputation','Wage']]
y=y[['Name','International Reputation','Wage']]
z=x.append(y)
z.style.background_gradient(cmap='Blues')
x=data[['Club','Wage','International Reputation']]
y=x.groupby('Club').agg('sum').sort_values(by='Wage', ascending=False).head(10)
y.style.background_gradient(cmap='Wistia')
x=data[['Position','Wage']]
y=x.groupby('Position').agg('sum').sort_values(by='Wage', ascending=False).head(30)
y.style.background_gradient(cmap='Wistia')
data.iloc[data.groupby(data['Position'])['Overall']
          .idxmax()][['Position', 'Name', 'Age', 'Club', 'Nationality','Overall']]
