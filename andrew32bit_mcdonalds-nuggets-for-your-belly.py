%matplotlib inline

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import warnings

warnings.filterwarnings('ignore')

from IPython.display import clear_output, Image, display

md = pd.read_csv('../input/menu.csv')

md.head(5)





# head of our fatty menu
md.describe()

#look at the mean of callories!368 is a lot
x = max(md['Cholesterol'])

md[(md.Cholesterol ==x)]

#the most dangerous for your heart breakfasts.It can be your last ...Cholesterol is a cause of heart attacks
s=max(md['Sugars'])

md[(md.Sugars ==s)]

#hmm,look at this cause of high sugar in the blood!McFlurry!
#oops,we forgot about checking the null values.Anyway we dont have them.lucky

print(md.isnull().any())
md.groupby('Category')['Item'].count().plot(kind='bar')

#the most common used items(Coffee and tea)
sns.boxplot(data= md, x = 'Category',y = 'Dietary Fiber')

plt.tight_layout

plt.show()
Max_Cal = md.groupby('Category').max().sort_values('Calories',ascending=False)

sns.swarmplot(data =Max_Cal, x= Max_Cal.index,y = 'Calories',hue ='Item',size =10 )

plt.tight_layout()
measures = ['Calories', 'Total Fat', 'Cholesterol','Sodium', 'Sugars', 'Carbohydrates']



for m in measures:   

    plot = sns.violinplot(x="Category", y=m, data=md)

    plt.setp(plot.get_xticklabels(), size=7)

    plt.title(m)

    plt.show()
for m in measures:

    g = sns.factorplot(x="Category", y=m,data=md, kind="swarm",size=5, aspect=2.5);
carb = max(md['Carbohydrates'])

display(carb)#i want to try display function lol
md[(md.Carbohydrates ==carb)]

#looks like i will order large chocolate shake after my gym lol=))
#Conlusion.If u dont want to die from heart attack-dont order 'Chicken McNuggets (40 piece)+southern Style crispy chicken+McFlurry with M&M’s Candies (Medium)'.If u want to be fat -u can just order Chocolate Shake (Large) twice or more in a day  lol .

#im sorry for luck of graphics,i will add some soon.