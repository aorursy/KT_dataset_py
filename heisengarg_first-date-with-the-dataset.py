import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from IPython.display import Image

from IPython.core.display import HTML 

Image(url= "http://images.memes.com/meme/592575")
df=pd.read_csv('../input/creditcard.csv')
df.shape
df.head()
df.describe()
plt.figure(figsize=(35,20))

plt.plot(df.drop(['Time','Amount','Class'],axis=1))

plt.legend(df.drop(['Time','Amount','Class',],axis=1).columns)

plt.show()
Image(url= "http://s2.quickmeme.com/img/db/dbc97d3b537a3b38f323b2cd9e97228de9342018e72bb18e3b36ec235a8783f5.jpg")
f, (ax1, ax2) = plt.subplots(1, 2)

ax1.set_title('Feature V15')

#ax1.set_yaxis('Value')

ax1.plot(df['V15'])



ax2.set_title('Amount')

#ax2.yaxis('Value')

ax2.plot(df['Amount'])



f.set_figheight(5)

f.set_figwidth(15)

Image(url='https://i.imgflip.com/1htaug.jpg')
plt.figure(figsize=(20,8))

plt.ylim(0,4000)

plt.plot(df['Amount'])

plt.plot(df[df['Class']==1]['Amount'],'ro')

plt.show()