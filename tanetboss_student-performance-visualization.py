import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # this is used for the plot the graph 
import seaborn as sns # used for plot interactive graph.

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
from pylab import rcParams
# figure size in inches
rcParams['figure.figsize'] = 11.7,8.27

import ptitprince as pt

%matplotlib inline
## Read file

data = pd.read_csv("../input/StudentsPerformance.csv")
data.head()

sns.set(style="white",font_scale=1.5)
dx="parental level of education"; dy="math score"; ort="v"; pal = "Set2"; sigma = .2
ax=pt.RainCloud(x = dx, y = dy, 
                data = data, 
                palette = 'Set2', 
                width_viol = .4,
                width_box = .2,
                figsize = (9,6), orient = 'h',
               move = .0)
plt.title('Parent\'s Education VS Child\'s Math Score',size = 20)

sns.set(style="white",font_scale=1.5)
dx="parental level of education"; dy="reading score"; ort="v"; pal = "Set2"; sigma = .2
ax=pt.RainCloud(x = dx, y = dy, 
                data = data, 
                palette = 'Set2', 
                width_viol = .4,
                width_box = .2,
                figsize = (9,6), orient = 'h',
               move = .0)

plt.title('Parent\'s Education VS Child\'s Reading Score',size = 20)

sns.set(style="white",font_scale=1.5)
dx="parental level of education"; dy="writing score"; ort="v"; pal = "Set2"; sigma = .2
ax=pt.RainCloud(x = dx, y = dy, 
                data = data, 
                palette = 'Set2', 
                width_viol = .4,
                width_box = .2,
                figsize = (9,6), orient = 'h',
               move = .0)
plt.title('Parent\'s Education VS Child\'s Writing Score',size = 20)

data.gender.value_counts()
rcParams['figure.figsize'] = 9.7,6.27

ax = sns.countplot(x="gender", data= data, palette = "Pastel1",linewidth=2, edgecolor=sns.color_palette("dark"))

ax = sns.countplot(x="gender",hue = 'race/ethnicity', data=data,
                   palette = "Pastel2",
                   linewidth=3,
                  edgecolor=sns.color_palette("dark"))

rcParams['figure.figsize'] = 11.7,6.27

fig, axs = plt.subplots(ncols=3)
fig = plt.subplots_adjust( wspace=0.7)
sns.boxplot(x="gender",y="math score",data=data ,palette = "Set1", ax=axs[0])
sns.boxplot(x="gender",y="reading score",data=data ,palette = "Set1", ax=axs[1])
sns.boxplot(x="gender",y="writing score",data=data ,palette = "Set1", ax=axs[2])

%matplotlib inline
sns.set_style("white")
sns.pairplot(data.iloc[:], hue='test preparation course', palette='Set3');
%matplotlib inline
sns.set_style("white")
sns.pairplot(data.iloc[:], hue='lunch', palette='Pastel1');

