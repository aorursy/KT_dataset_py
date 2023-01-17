#Bar graph for average Happiness Score in Western Europe
import pandas as pd

import matplotlib.pyplot as plt 

%matplotlib inline

import matplotlib.mlab as mlab

import numpy as np



data=pd.read_csv('../input/2016.csv')

data.columns.values

w_eu= data[data['Region']=='Western Europe']

w_eu_avg_economy= w_eu['Happiness Score'].mean()

group=w_eu.groupby(by='Country', as_index=False)['Happiness Score'].mean()





fig=plt.figure()



labels = group['Country']

pos= np.arange(len(labels))+1

avg_score = group['Happiness Score']

colors=('mediumslateblue','mediumslateblue','mediumslateblue','greenyellow','mediumslateblue','mediumslateblue','mediumslateblue','mediumslateblue','mediumslateblue','mediumslateblue','mediumslateblue','mediumslateblue','mediumslateblue','mediumslateblue','mediumslateblue','mediumslateblue','mediumslateblue','mediumslateblue','mediumslateblue','mediumslateblue')

width=1/1.8,

plt.bar([p + width for p in pos],  avg_score, width=width,color=colors)

fig.set_facecolor('#E6D7D7')







plt.xticks(pos + width+0.2, group['Country'], rotation=90)

plt.yticks(range(0, 9))

plt.ylabel('Score')

plt.title('Average Happiness in Western Europe')

plt.axhline(y=6.68, xmin=0, xmax=22, hold=None, color='red')

plt.annotate('Mean = 6.686', xy=(2,1), xytext=(22, 6.2),color='red', size=6.5)

plt.grid(b=True, which='major', color='grey', linestyle='--', alpha=0.35)

plt.tight_layout()





plt.show()

plt.savefig('erstebar.jpg')
