import pandas as pd

from math import pi

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

df = pd.read_csv('../input/Pokemon/Pokemon.csv')
df.columns = df.columns.str.upper().str.replace('_', '')

df.head()

df = df.set_index('NAME')
df.index = df.index.str.replace(".*(?=Mega)", "")

df=df.drop(['#'],axis=1)

df['TYPE 2'].fillna(df['TYPE 1'], inplace=True)
data=df

data.drop(["TYPE 1", "TYPE 2","TOTAL","GENERATION","LEGENDARY"], axis = 1, inplace = True)

Attributes =list(data)

AttNo = len(Attributes)

name = input("Enter name of first pokemon : ") 

print(name)

values = data.loc[name].tolist()

values += values [:1]



angles = [n / float(AttNo) * 2 * pi for n in range(AttNo)]

angles += angles [:1]

name2 = input("Enter name of second pokemon: ") 

print(name2)

values2 = data.loc[name2].tolist()

values2 += values2 [:1]



angles2 = [n / float(AttNo) * 2 * pi for n in range(AttNo)]

angles2 += angles2 [:1]



ax = plt.subplot(111, polar=True)



plt.xticks(angles[:-1],Attributes)



ax.plot(angles,values)

ax.fill(angles, values, 'blue', alpha=0.1)



ax.plot(angles2,values2)

ax.fill(angles2, values2, 'red', alpha=0.1)



#Rather than use a title, individual text points are added

plt.figtext(0.2,0.9,name,color="red")

plt.figtext(0.2,0.85,"vs")

plt.figtext(0.2,0.8,name2,color="blue")

plt.show()