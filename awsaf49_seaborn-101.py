# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

print('Setup complete')



# Any results you write to the current directory are saved as output.
tips_data = sns.load_dataset('tips')

tips_data.head()
print(tips_data.corr())

print(sns.get_dataset_names())
plt.figure(figsize=(14,6))

def sinplot(flip = 1):

   x = np.linspace(0, 14, 100)

   for i in range(1, 5): 

      plt.plot(x, np.sin(x + i * .5) * (7 - i) * flip)

sinplot()

plt.figure(figsize=(14,6))

plt.show()
import numpy as np

from matplotlib import pyplot as plt

def sinplot(flip = 1):

   x = np.linspace(0, 14, 100)

   for i in range(1, 5):

      plt.plot(x, np.sin(x + i * .5) * (7 - i) * flip)

import seaborn as sb

sb.set()

sinplot()

plt.show()
import numpy as np

from matplotlib import pyplot as plt

def sinplot(flip=1):

   x = np.linspace(0, 14, 100)

   for i in range(1, 5):

      plt.plot(x, np.sin(x + i * .5) * (7 - i) * flip)

import seaborn as sns

sns.set_style("ticks")

sns.set_context('talk')#it sets the marker size and the figure size

sinplot()

plt.show()
print(sns.axes_style)
sns.color_palette(palette = None, n_colors = None, desat = None)
from matplotlib import pyplot as plt

import seaborn as sb

current_palette = sb.color_palette(palette='Purples')

sb.palplot(current_palette)

plt.show()
from matplotlib import pyplot as plt

import seaborn as sb

current_palette = sb.color_palette()

sb.palplot(sb.color_palette("Greens"))

plt.show()

sb.palplot(sb.color_palette('Purples'))
from matplotlib import pyplot as plt

import seaborn as sb

current_palette = sb.color_palette()

sb.palplot(sb.color_palette("BrBG", 7))

plt.show()
sns.palplot(sns.color_palette("hls", 8))

sns.palplot(sns.color_palette("husl", 8))

sns.palplot(sns.color_palette())
import pandas as pd

import seaborn as sb

from matplotlib import pyplot as plt

plt.figure(figsize=(6,4))

sb.set_palette("muted")

sb.set_style("darkgrid")

df = sb.load_dataset('iris')

sb.distplot(df['petal_length'],kde = False)#kde = True dile density dekhabe

plt.show()
plt.figure(figsize=(6,4))

titanic_data = sb.load_dataset('titanic')

sns.set_palette('deep')

sns.distplot(a=titanic_data['survived'],kde =False)#hue and palette nai

print(titanic_data['survived'].value_counts())
plt.figure(figsize=(6,4))

titanic=sns.load_dataset('titanic')    

age1=titanic['age'].dropna()

print(age1.head())

sns.distplot(a = age1,kde=False,color = 'blue')#kde = true dile frequencey er jaigai density ashbe

plt.show()

#here titanic['sex'] will give error as it requires some numerical values(data) as it calculates distribution. Male and Female isn't

#numerical
import pandas as pd

import seaborn as sb

from matplotlib import pyplot as plt



df = sb.load_dataset('iris')

plt.figure(figsize=(3,2))

sns.set_context('talk')

sb.jointplot(x = 'petal_length',y = 'petal_width',data = df,kind='reg',dropna= True)#kind : { “scatter” | “reg” | “resid” | “kde” | “hex” },

plt.show()



sb.jointplot(x='age',y='survived',data=titanic,kind='reg',dropna=True,color='purple')

#hue and palette nai
import pandas as pd

import seaborn as sb

from matplotlib import pyplot as plt

#plt.figure(figsize=(12,8))

df = sb.load_dataset('iris')

sb.set_style("ticks")

sb.set_context("talk")

sb.pairplot(data=df,hue = 'species',diag_kind = "kde",palette = "husl",height=2,aspect=1.5)#kind : {‘scatter’, ‘reg’} diag_kind : {‘auto’, ‘hist’, ‘kde’},

#dropna : boolean

plt.show()
import pandas as pd

import seaborn as sb

from matplotlib import pyplot as plt

df = sb.load_dataset('iris')

sb.set_palette('Set1')

plt.figure(figsize=(12,8))

sb.set_style('ticks')



sb.stripplot(x = "species", y = "petal_length", data = df)

plt.show()

plt.figure(figsize=(16,8))

sb.stripplot(x = "species", y = "petal_length", data = df,jitter=False)#jitter adds some noise in the data
import pandas as pd

import seaborn as sb

from matplotlib import pyplot as plt

df = sb.load_dataset('iris')

plt.figure(figsize=(12,8))

sb.swarmplot(x = "species", y = "petal_length", data = df,palette='Set1')

plt.show()
import pandas as pd

import seaborn as sb

from matplotlib import pyplot as plt

df = sb.load_dataset('iris')

sb.set_palette('husl')

sb.boxplot(x = "species", y = "petal_length", data = df)

plt.show()
import pandas as pd

import seaborn as sb

from matplotlib import pyplot as plt

df = sb.load_dataset('tips')

sb.set_palette('husl')

sb.violinplot(x = "day", y = "total_bill", data=df)

plt.show()



sb.violinplot(x='survived',y='age',hue='sex',data=titanic,palette='husl')
import pandas as pd

import seaborn as sb

from matplotlib import pyplot as plt

df = sb.load_dataset('tips')

sb.set_context('talk')

plt.figure(figsize=(12,8))

sb.violinplot(x = "day", y = "total_bill",hue = 'sex', data = df,palette='husl')

plt.show()
import pandas as pd

import seaborn as sb

from matplotlib import pyplot as plt

df = sb.load_dataset('titanic')

sb.set_context('talk')

plt.figure(figsize=(6,4))

sb.barplot(x = "sex", y = "survived", hue = "class", data = df,palette='husl')#categories are in X and Hue and Y axis contains . Survived is a categorial but it has a numerical value

#the continuous variable

plt.show()

sb.barplot(x = "sex", y = "age", data = df,palette='husl')

plt.show()

#Y axis a string thakle: unsupported operand type(s) for /: 'str' and 'int'
import pandas as pd

import seaborn as sb

from matplotlib import pyplot as plt

df = sb.load_dataset('titanic')

sb.set_context('talk')

plt.figure(figsize=(12,8))

sb.countplot(x = "class", y=None,hue='sex', data = df, palette = "husl");

plt.show()

sb.countplot(x = "survived", y=None,hue='sex', data = df, palette = "hls");

plt.show()
import pandas as pd

import seaborn as sb

from matplotlib import pyplot as plt

df = sb.load_dataset('titanic')

sb.set_context('talk')

plt.figure(figsize=(12,8))

sb.pointplot(x = "sex", y = "survived", hue = "class", data = df)

plt.show()







tips = sb.load_dataset("tips")

sb.pointplot(x = "day", y = "total_bill",hue = 'sex', data = tips,palette='husl')

plt.show()



#this plot is showing us the trend of getting tips in everyday
import pandas as pd

import seaborn as sb

from matplotlib import pyplot as plt

sb.reset_defaults()

df = sb.load_dataset('exercise')

sb.catplot(x = "time", y = "pulse", hue = "kind", kind = 'point', col = "diet", data = df,palette='husl');#The kind of plot to draw (corresponds to the name of a categorical plotting function. 

#Options are: “point”, “bar”, “strip”, “swarm”, “box”, “violin”, or “boxen”,"count".

# Variables= x,y,col,row,hue

plt.show()



sns.catplot(x="alive", col="class", col_wrap=3,data=titanic[titanic['class'].notnull()],kind="count", height=3, aspect=0.8)

plt.show()   



sns.catplot(x="sex", col="class",hue='survived', col_wrap=3,data=titanic[titanic['class'].notnull()],kind="count", height=3, aspect=0.8)

plt.show()    
sns.catplot(x="class", hue="who", col="alive",data=titanic, kind="count",height=4, aspect=.7,palette='Set1');
sns.catplot(x="deck", kind="count", palette="ch:.25", data=titanic);
sb.scatterplot(x="total_bill", y="tip",hue='sex', data=tips)
import pandas as pd

import seaborn as sb

from matplotlib import pyplot as plt

df = sb.load_dataset('tips')

sb.regplot(x = "total_bill", y = "tip", data = df)

sb.lmplot(x = "total_bill", y = "tip",order=4, data = df)

plt.show()
sns.regplot(x="size", y="total_bill", data=tips, x_jitter=.1)

sns.regplot(x="size", y="total_bill", data=tips,x_estimator=np.mean)
ans = sns.load_dataset("anscombe")

ax = sns.regplot(x="x", y="y", data=ans.loc[ans.dataset == "III"],robust=True, ci=None)
tips["big_tip"] = (tips.tip / tips.total_bill) > .175

ax = sns.regplot(x="total_bill", y="big_tip", data=tips,logistic=True, n_boot=500, y_jitter=.03)

plt.show()

sns.regplot(x='age',y='survived',data=titanic_data,logistic=True,y_jitter=0.03)

plt.show()
df.head()

sns.catplot(x="day", y="total_bill", hue="smoker",col="time", aspect=.6, kind="swarm", data=df);
tips = sns.load_dataset('tips')

sns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips);

sns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips, markers=["o", "x"], palette="Set1");
sns.lmplot(x="total_bill", y="tip", hue="smoker", col="time", data=tips);
sns.lmplot(x="total_bill", y="tip", hue="smoker",col="time", row="sex", data=tips);#I can input 5 variables

sns.lmplot(x="size", y="tip", data=tips, x_jitter=.05);#one vairable is descrete

#x_jitter will add some noise thtat will help us to regression
anscombe = sns.load_dataset("anscombe")

sns.lmplot(x="x", y="y", data=anscombe.query("dataset == 'II'"),order=2, ci=None, scatter_kws={"s": 80});
a = sns.FacetGrid( titanic_data, hue = 'survived', aspect=4 ,palette='Set1')

a.map(sns.kdeplot, 'age', shade= True )

a.set(xlim=(0 , titanic_data['age'].max()))

a.add_legend()
h = sns.FacetGrid(titanic_data, row = 'sex', col = 'class', hue = 'survived',palette='Set1')

h.map(plt.hist, 'age', alpha = .75)

h.add_legend()



















