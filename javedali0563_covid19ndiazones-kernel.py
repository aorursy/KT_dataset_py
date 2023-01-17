
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
%matplotlib inline
plt.style.use('fivethirtyeight')


# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

df
color = ["#008000","#FFA500","#FF0000"]
df['Zone'].value_counts(normalize=True).plot(kind='bar',color=color)
state_grp = df.groupby(['State'])
states = []
for state in state_grp:
    states.append(state[0])
sta = []
for state in states:
    add=dict(state_grp.get_group(state)['Zone'].value_counts())
    sorted_dict = dict(sorted(add.items()))
    sta.append(sorted_dict)
    country = dict(zip(states,sta))
orange = []
for state in country:
    
    if 'Orange Zone' in country[state]:
        orange.append(country[state]['Orange Zone'])
    else:
        orange.append(0)
green = []
for state in country:
    
    if 'Green Zone' in country[state]:
        green.append(country[state]['Green Zone'])
    else:
        green.append(0)
red = []
for state in country:
    
    if 'Red Zone' in country[state]:
        red.append(country[state]['Red Zone'])
    else:
        red.append(0)
plt.barh(states,green,color="#008000",label="Green Zone")
plt.legend()
plt.tight_layout()
plt.ylabel("States")
plt.rcParams['figure.figsize'] = [50/2.54, 40/2.54]
plt.xlabel("Number of Zones")
plt.show()
plt.barh(states,orange,color="#FFA500",label="Orange Zone")
plt.legend()
plt.tight_layout()
plt.ylabel("States")
plt.rcParams['figure.figsize'] = [50/2.54, 40/2.54]
plt.xlabel("Number of Zones")
plt.show()
plt.barh(states,red,color="#FF0000",label="Red Zone")
plt.legend()
plt.tight_layout()
plt.ylabel("States")
plt.rcParams['figure.figsize'] = [50/2.54, 40/2.54]
plt.xlabel("Number of Zones")
plt.show()
