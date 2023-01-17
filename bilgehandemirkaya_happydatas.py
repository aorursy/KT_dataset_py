import os
import pandas as pd

data_2015 = pd.read_csv('../input/world-happiness/2015.csv')

data_2016 = pd.read_csv('../input/world-happiness/2016.csv')

data_2017 = pd.read_csv('../input/world-happiness/2017.csv')

data_2018 = pd.read_csv('../input/world-happiness/2018.csv')

data_2019 = pd.read_csv('../input/world-happiness/2019.csv')
data_2018.shape
data_2019.head()
data_2018.tail()
import seaborn as sns

import matplotlib.pyplot as plt



data_2017[['Country', 'Generosity']].sort_values(by = 'Generosity',

                                                ascending = False).head(10)

data_2015[['Country', 'Generosity']].sort_values(by = 'Generosity',

                                                ascending = False).head(10)
data_2016[['Country', 'Generosity']].sort_values(by = 'Generosity',

                                                ascending = False).head(10)
data_2015[['Country', 'Generosity']].sort_values(by = 'Generosity',

                                                ascending = False).head(10)
data_2018[['Country or region', 'Generosity']].sort_values(by = 'Generosity',

                                                ascending = False).head(10)
data_2019[['Country or region', 'Generosity']].sort_values(by = 'Generosity',

                                                ascending = False).head(10)
df_row = pd.concat([data_2019, data_2018], axis=1)
df_row
frames = [data_2018.head(),data_2019.head()]

df_keys = pd.concat(frames, keys=['2018', '2019'])



df_keys
data = {"Country or region 2019":["Finland","South Sudan"],"Score2019":["7.769","2,853"],"GDP Per Capita19":["1.340","0.306"]}

df1 = pd.DataFrame(data, columns=["Country or region 2019","Score2019","GDP Per Capita19"])

df1
data2 = {"Country or region 2018":["Finland","South Sudan"],"Score2018":["7.632","3,254"],"GDP Per Capita18":["1.305","0.024"]}

df2 = pd.DataFrame(data2, columns=["Country or region 2018","Score2018","GDP Per Capita18"])

df2
df_concat = pd.concat([df1,df2], axis=1)

df_concat
import matplotlib.pyplot as plt

import numpy as np
labels = ['2018 Score','2019 Score','2018 GDP','2019 GDP']



Finland_means = [7.632, 7.769, 1.305, 1.340]

SSudan_means = [3.254, 2.853, 0.024, 0.306]



x = np.arange(len(labels)) 

width = 0.35  



fig, ax = plt.subplots()

rects1 = ax.bar(x - width/2, Finland_means, width, label='Finland')

rects2 = ax.bar(x + width/2, SSudan_means, width, label='SSudan')



ax.set_ylabel('amount')

ax.set_title('2018-2019 Finland vs South Sudan')

ax.set_xticks(x)

ax.set_xticklabels(labels)

ax.legend()





def autolabel(rects):

    for rect in rects:

        height = rect.get_height()

        ax.annotate('{}'.format(height),

                    xy=(rect.get_x() + rect.get_width() / 2, height),

                    xytext=(0, 3),  # 3 points vertical offset

                    textcoords="offset points",

                    ha='center', va='bottom')





autolabel(rects1)

autolabel(rects2)



fig.tight_layout()



plt.show()

import pandas as pd

from pandas import DataFrame

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure as fig



Data = {'Country': ['Finland','Norway','Denmark','Iceland','Netherlands'],

        'GDP_Per_Capita': [7.769,7.600,7.554,7.494,7.488]

       }

  

df = DataFrame(Data,columns=['Country','GDP_Per_Capita'])

df.plot(x ='Country', y='GDP_Per_Capita',width=0.5 , kind = 'bar', figsize=(8,9))

plt.rcParams["figure.figsize"] = (15,6)

plt.rcParams["figure.dpi"] = 75.

plt.ylim(7,8)

plt.show()

data_2019.groupby(by ='Score').max()
import matplotlib.pyplot as plt



data_2016.head()
import matplotlib.pyplot as plt



plt.figure(figsize=(50,76))

sns.countplot(x="Region", data = data_2016)    

sns.set(font_scale=2.8) 

plt.savefig('saving-a-seaborn-plot-as-pdf-file.pdf', dpi=300, figsize=(12,15))
sns.distplot(data_2019.Generosity)
sns.distplot(data_2019.Score)