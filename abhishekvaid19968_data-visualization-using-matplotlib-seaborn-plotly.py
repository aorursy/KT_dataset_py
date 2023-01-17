import pandas as pd
# using titanic dataset
train = pd.read_csv('../input/titanic/train.csv')
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import plotly.express as px


def bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True, figsize=(10,5))
bar_chart('Survived')
def Horizontal_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='barh',stacked=True, figsize=(10,5))
Horizontal_chart('Survived')
def bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True, figsize=(10,5))
bar_chart('Sex')
def bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=False, figsize=(10,5))
bar_chart('Sex')
import seaborn as sns
sns.set(style="darkgrid")
ax = sns.countplot(x="Sex", data=train)
train["Age"].isnull()
train['Age']=train['Age'].fillna(train['Age'].mean())
train["Age"].min()
train["Age"].max()
fig = plt.figure(figsize = (10,10))
data = train["Age"]
plt.hist(data, range=(0.42,80), bins=10, color='r', edgecolor='black')
plt.show()
fig = plt.figure(figsize = (15,15))
ax = fig.gca()
train.hist(ax=ax, color='r', edgecolor='black')
plt.show()
import matplotlib.pyplot as plt
import numpy as np
 
# create data
x = train['Fare']
y = data
 
# Big bins
plt.hist2d(x, y, bins=(1,80), cmap=plt.cm.jet)
#plt.show()
 
plt.show()


plt.hist2d(x, y, bins=(1, 80), cmap=plt.cm.Reds)
plt.show()
fig = plt.figure(figsize=(16, 10), dpi= 80)
grid = plt.GridSpec(4, 4, hspace=0.5, wspace=0.2)


ax_main = fig.add_subplot(grid[:-1, :-1])
ax_right = fig.add_subplot(grid[:-1, -1], xticklabels=[], yticklabels=[])
ax_bottom = fig.add_subplot(grid[-1, 0:-1], xticklabels=[], yticklabels=[])

ax_main.scatter('Age', 'Fare', alpha=.9, data=train, cmap="tab10", edgecolors='gray', linewidths=.5)


ax_bottom.hist(train.Age, 40, histtype='stepfilled', orientation='vertical', color='deeppink')
ax_bottom.invert_yaxis()


ax_right.hist(train.Fare, 40, histtype='stepfilled', orientation='horizontal', color='deeppink')


ax_main.set(title='Scatterplot with Histograms Age to Fare visualization', xlabel='Age', ylabel='Fare')
ax_main.title.set_fontsize(20)
for item in ([ax_main.xaxis.label, ax_main.yaxis.label] + ax_main.get_xticklabels() + ax_main.get_yticklabels()):
    item.set_fontsize(14)

xlabels = ax_main.get_xticks().tolist()
ax_main.set_xticklabels(xlabels)
plt.show()
 
data.head
fig = plt.figure(figsize=(16, 10), dpi= 80)
grid = plt.GridSpec(4, 4, hspace=0.5, wspace=0.2)

# Define the axes
ax_main = fig.add_subplot(grid[:-1, :-1])
ax_right = fig.add_subplot(grid[:-1, -1], xticklabels=[], yticklabels=[])
ax_bottom = fig.add_subplot(grid[-1, 0:-1], xticklabels=[], yticklabels=[])

# Scatterplot on main ax
ax_main.scatter('Age', 'Fare', alpha=.9, data=train, cmap="Set1", edgecolors='purple', linewidths=.5)

# Add a graph in each part
sns.boxplot(train.Fare, ax=ax_right, orient="v",color='deeppink' )
sns.boxplot(train.Age, ax=ax_bottom, orient="h", color='deeppink')

# Decorations ------------------
# Remove x axis name for the boxplot
ax_bottom.set(xlabel='')
ax_right.set(ylabel='')

# Main Title, Xlabel and YLabel
ax_main.set(title='Scatterplot with Boxplot Age to Fare visualization', xlabel='Age', ylabel='Fare')

# Set font size of different components
ax_main.title.set_fontsize(20)
for item in ([ax_main.xaxis.label, ax_main.yaxis.label] + ax_main.get_xticklabels() + ax_main.get_yticklabels()):
    item.set_fontsize(14)

plt.show()
plt.figure(figsize=(20,10), dpi= 216)
sns.heatmap(train.corr(), xticklabels=train.corr().columns, yticklabels=train.corr().columns, cmap='RdYlGn', center=0, annot=True)

# Decorations
plt.title('Correlogram of Titanic Dataset', fontsize=22)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
sns.set(style="darkgrid")
g = sns.FacetGrid(train, row="Sex", col="Survived", margin_titles=True, size=10)
bins = np.linspace(0, 60, 13)
g.map(plt.hist, "Fare", color="blueviolet", bins=bins)
train
df = pd.read_csv("../input/who-data-base-for-public-use/2020-Full database.csv")['Journal']
from wordcloud import WordCloud, ImageColorGenerator

word = ','.join(map(str, df))
from wordcloud import WordCloud , STOPWORDS
stopwords = set(STOPWORDS)
wordcloud = WordCloud(background_color = "Black", max_words = 200, stopwords = stopwords).generate(word)
plt.figure(1,figsize=(20, 20))
plt.axis('off')
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.show()
sns.set(style="whitegrid")
ax = sns.violinplot(x=train["Age"])
fig = plt.figure(figsize=(16, 10), dpi= 80)
grid = plt.GridSpec(4, 4, hspace=0.5, wspace=0.2)

# Define the axes
ax_main = fig.add_subplot(grid[:-1, :-1])
ax_right = fig.add_subplot(grid[:-1, -1], xticklabels=[], yticklabels=[])
ax_bottom = fig.add_subplot(grid[-1, 0:-1], xticklabels=[], yticklabels=[])

# Scatterplot on main ax
ax_main.scatter('Age', 'Fare', alpha=.9, data=train, cmap="Set1", edgecolors='purple', linewidths=.5)

# Add a graph in each part

sns.violinplot(train.Fare,ax=ax_right, orient="v",color='deeppink')
sns.violinplot(train.Age,ax=ax_bottom, orient="h",color='deeppink')
# Decorations ------------------
# Remove x axis name for the boxplot
ax_bottom.set(xlabel='')
ax_right.set(ylabel='')

# Main Title, Xlabel and YLabel
ax_main.set(title='Scatterplot with Boxplot Age to Fare visualization', xlabel='Age', ylabel='Fare')

# Set font size of different components
ax_main.title.set_fontsize(20)
for item in ([ax_main.xaxis.label, ax_main.yaxis.label] + ax_main.get_xticklabels() + ax_main.get_yticklabels()):
    item.set_fontsize(14)

plt.show()

sns.set(style="whitegrid")
ax = sns.violinplot(x=train.Pclass, y=train.Sex, data=train, palette='Paired')

sns.set(style="whitegrid")
ax = sns.violinplot(x=train.Pclass, y=train.Age, data=train, palette="bright")

plt.figure(figsize=(15,10))
ax = sns.violinplot(x=train.Pclass, y=train.Age, hue=train.Sex,split = True,
                    data=train, palette="Set1")
plt.figure(figsize=(15,10))
ax = sns.swarmplot(x="Sex", y="Age", data=train, palette="bright")
ax = sns.violinplot(x="Sex", y="Age", data=train, palette="Set1")
sns.set(style="ticks")
tips = sns.load_dataset("tips")
g = sns.relplot(x="total_bill", y="tip", hue="day", data=tips)
fig = plt.figure(figsize=(10, 7), dpi= 80)
sns.distplot(train['Age'], bins=10, kde=True, color = 'red')
df = train['Pclass'].value_counts()
 
# make the plot
df.plot(kind='pie', subplots=True, figsize=(8, 8))

df = train['Pclass'].value_counts()
df
names='1', '2', '3',
size=[216,184,491]
 
# create a figure and set different background
fig = plt.figure(figsize=(16, 10), dpi= 80)
fig.patch.set_facecolor('black')
 
# Change color of text
plt.rcParams['text.color'] = 'white'
 
# Create a circle for the center of the plot
my_circle=plt.Circle( (0,0), 0.7, color='black')
 
# Pieplot + circle on it
plt.pie(size, labels=df)
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.show()



import numpy as np
import matplotlib.pyplot as plt

def sunburst(nodes, total=np.pi * 2, offset=0, level=0, ax=None):
    fig = plt.figure(figsize=(16, 10), dpi= 80)
    ax = ax or plt.subplot(111, projection='polar')

    if level == 0 and len(nodes) == 1:
        label, value, subnodes = nodes[0]
        ax.bar([0], [0.5], [np.pi * 2])
        ax.text(0, 0, label, ha='center', va='center')
        sunburst(subnodes, total=value, level=level + 1, ax=ax)
    elif nodes:
        d = np.pi * 2 / total
        labels = []
        widths = []
        local_offset = offset
        for label, value, subnodes in nodes:
            labels.append(label)
            widths.append(value * d)
            sunburst(subnodes, total=total, offset=local_offset,
                     level=level + 1, ax=ax)
            local_offset += value
        values = np.cumsum([offset * d] + widths[:-1])
        heights = [1] * len(nodes)
        bottoms = np.zeros(len(nodes)) + level - 0.5
        rects = ax.bar(values, heights, widths, bottoms, linewidth=1,
                       edgecolor='white', align='edge')
        for rect, label in zip(rects, labels):
            x = rect.get_x() + rect.get_width() / 2
            y = rect.get_y() + rect.get_height() / 2
            rotation = (90 + (360 - np.degrees(x) % 180)) % 360
            ax.text(x, y, label, rotation=rotation, ha='center', va='center') 

    if level == 0:
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location('N')
        ax.set_axis_off()

data = [
    ('/', 100, [
        ('Survived', 68, [
            ('Female', 46, []),
            ('Male', 22, []),
            
        ]),
        ('Dead', 32, [
                ('Female',15 , []),
                ('male', 17, []),
            
                

        ]),
          
        ]),
]

sunburst(data)
df = train['Sex'].value_counts()
df
survived = train[train['Survived']==0]['Sex'].value_counts()
survived



import plotly.express as px
fig = px.sunburst(train, path=["Pclass",'Survived'],
                  color='Pclass', hover_data=['Survived'],
                  color_continuous_scale='rainbow')
fig.show()
fig = plt.figure(figsize=(10, 7), dpi= 80)
plt.scatter(train.Age,train.Fare,color='red',edgecolors='purple')

fig = plt.figure(figsize=(10, 7), dpi= 80)
x = train.Age
y = train.Fare
m, b = np.polyfit(x, y, 1)
plt.scatter(train.Age,train.Fare,color='white',edgecolors='orange')
plt.plot(x, m*x + b,color='red' )
fig = plt.figure(figsize=(11, 8), dpi= 80)
!pip install python-ternary
import ternary

fig, tax = ternary.figure(scale=200)
fig.set_size_inches(10, 7)

tax.scatter(train[['Fare', 'Age', 'Pclass']].values,color='deeppink',edgecolors='purple')
tax.gridlines(multiple=20)
tax.get_axes().axis('off')

fig = plt.figure(figsize=(11, 8), dpi= 80)
import ternary

fig, tax = ternary.figure(scale=200)
fig.set_size_inches(10, 7)

tax.scatter(train[['Fare', 'PassengerId', 'Pclass']].values,color='green',edgecolors='green')
tax.gridlines(multiple=20)
tax.get_axes().axis('off')

grd = train.groupby(["Survived"])[["Pclass","Fare","Age"]].mean().reset_index()
f, ax = plt.subplots(figsize=(100, 30))

plt.plot(grd.Survived,grd.Pclass,color="blue")
plt.plot(grd.Survived,grd.Fare,color="black")
plt.plot(grd.Survived,grd.Age,color="red")
plt.xticks(fontsize=50)
plt.yticks(fontsize=50)
train
plt.figure(figsize=(14,8))
sns.set_style("darkgrid")
sns.kdeplot(train['Age'],label="Age" ,shade=True, color='gold')
plt.figure(figsize=(23,12), dpi= 216)
x = train.Age[:]
y = x+train.Fare[:]
z = x+train.Survived[:]
z=z*z
 
# Change color with c and alpha. I map the color to the X axis value.
plt.scatter(x, y, z, c=x, cmap="plasma", alpha=0.4, edgecolors="grey", linewidth=2)
 
# Add titles (main and on axis)
plt.xlabel("Age")
plt.ylabel("Fare")
plt.title("A colored bubble plot")
 
plt.show()

plt.figure(figsize=(23,12))
x = train.Age[:100]
y = x+train.Fare[:100]
z = x+train.Pclass[:100]
z=z*z
 
# Change color with c and alpha. I map the color to the X axis value.
plt.scatter(x, y, z, c=x, cmap="plasma", alpha=0.4, edgecolors="grey", linewidth=2)
 
# Add titles (main and on axis)
plt.xlabel("the X axis")
plt.ylabel("the Y axis")
plt.title("A colored bubble plot")
 
plt.show()




plt.figure(figsize=(20,10), dpi= 216)
sns.boxplot(x= train["Age"] , y= train["Sex"], palette="Set1")
plt.figure(figsize=(23,12), dpi= 216)
sns.boxplot(x= train["Age"] , y= train["Sex"], hue=train["Pclass"], data=train, palette="Set1")
plt.figure(figsize=(20,10), dpi= 216)
sns.boxplot(x= train["Age"] , y= train["Sex"], palette="Set1", notch=True)
plt.figure(figsize=(20,10), dpi= 216)
ax = sns.boxplot(x= train["Age"] , y= train["Sex"], data=train)
ax = sns.swarmplot(x= train["Age"] , y= train["Sex"], data=train, color="yellow")
plt.figure(figsize=(20,10), dpi= 216)
ax = sns.swarmplot(x= train["Sex"] , y= train["Age"], data=train, color="red")
plt.figure(figsize=(20,10), dpi= 216)
ax = sns.boxenplot(x= train["Sex"] , y= train["Age"], hue=train["Pclass"],
                 data=train, palette="plasma")
plt.figure(figsize=(15,11))
sns.set_style("darkgrid")
sns.jointplot(x= train["Fare"] , y= train["Age"], kind='scatter', color="red", size=10)

plt.figure(figsize=(15,11))
sns.set_style("darkgrid")
sns.jointplot(x= train["Fare"] , y= train["Age"], kind='resid', color="blue",size=10 )

plt.figure(figsize=(15,11))
sns.set_style("darkgrid")
sns.jointplot(x= train["Fare"] , y= train["Age"], kind='hex', color="green", size=10)

plt.figure(figsize=(15,11))
sns.set_style("darkgrid")
sns.jointplot(x= train["Fare"] , y= train["Age"], kind='kde', color="deeppink", size=10)
train
import plotly.express as px


funnel = train.sort_values(by=['Fare'],ascending=False)

fig = px.funnel(funnel, x='Fare', y='Embarked')

fig.show()
sns.clustermap(train.corr(), center=0, cmap="plasma",
               linewidths=.75, figsize=(18, 14))
sns.pairplot(train)
plt.figure(figsize=(15,11))
Strip = sns.stripplot(x="Sex", y="Age", data=train)
Cat = sns.catplot(x="Sex", y="Age", data=train, size=10)
plt.figure(figsize=(15,11))
sns.residplot(train.Fare, train.Age, lowess=True, color="red")

sns.lmplot(x = 'Fare' ,y= 'Age', data=train, size=10)
# using new placement data set to represent an AREA PLOT, Also going to use the Titanic Data Set Blow it
Place = pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
Place.plot.area(y=['ssc_p','hsc_p','degree_p','mba_p'],alpha=0.4,figsize=(18, 12), cmap = 'plasma')
train.plot.area(y=['Fare','Age','Pclass','PassengerId'],alpha=0.4,figsize=(18, 12), cmap = 'plasma');
train
from yellowbrick.features import ParallelCoordinates
from yellowbrick.datasets import load_occupancy

# Load the classification data set
X, y = load_occupancy()

# Specify the features of interest and the classes of the target
features = [
    "temperature", "relative humidity", "light", "CO2", "humidity"
]
classes = ["unoccupied", "occupied"]

# Instantiate the visualizer
visualizer = ParallelCoordinates(
    classes=classes, features=features, sample=0.05, shuffle=True
)

# Fit and transform the data to the visualizer
visualizer.fit_transform(X, y)

# Finalize the title and axes then display the visualization
visualizer.show()
Place
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
 
# First way to call the 2 group Venn diagram:
plt.figure(figsize=(25,21))
venn2(Place['ssc_p'], set_labels = ('Group A', 'Group B'))
plt.show()

import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3, venn3_circles

plt.figure(figsize=(25,21))
venn3(Place['ssc_p'])
plt.show()
# Import the library
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
 
# Custom text labels: change the label of group A
plt.figure(figsize=(25,21))
v=venn3(subsets = (10, 8, 22, 6,9,4,2), set_labels = ('Group A', 'Group B', 'Group C'))
v.get_label_by_id('A').set_text('My Favourite group!')
plt.show()
 
# Line style: can be 'dashed' or 'dotted' for example
plt.figure(figsize=(25,21))
v=venn3(subsets = (10, 8, 22, 6,9,4,2), set_labels = ('Group A', 'Group B', 'Group C'))
c=venn3_circles(subsets = (10, 8, 22, 6,9,4,2), linestyle='dashed', linewidth=1, color="grey")
plt.show()
 
# Change one group only
plt.figure(figsize=(25,21))
v=venn3(subsets = (10, 8, 22, 6,9,4,2), set_labels = ('Group A', 'Group B', 'Group C'))
c=venn3_circles(subsets = (10, 8, 22, 6,9,4,2), linestyle='dashed', linewidth=1, color="grey")
c[0].set_lw(8.0)
c[0].set_ls('dotted')
c[0].set_color('skyblue')
plt.show()
 
# Color
v.get_patch_by_id('100').set_alpha(1.0)
v.get_patch_by_id('100').set_color('white')
plt.show()

Place
Place = pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
Place.degree_t.value_counts()
Place.degree_t.value_counts()
import warnings
warnings.filterwarnings("ignore")
sns.FacetGrid(Place, hue="hsc_p", size=10) \
   .map(sns.kdeplot, "salary") \
   .add_legend()
plt.ioff() 
Place
k = Place.drop(["ssc_b","gender",'hsc_b', 'hsc_s', 'workex', 'specialisation', 'status','degree_t'], axis=1)
k['salary']=k['salary'].fillna(k['salary'].mean())
k
from pandas.plotting import andrews_curves
andrews_curves(k,"salary")
sns.factorplot('Pclass','Age',data=train,size=10 )
plt.ioff()
plt.show()
!pip install windrose
import warnings
warnings.filterwarnings("ignore")
from windrose import WindroseAxes
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np

# Create wind speed and direction variables

ws = np.random.random(500) * 6
wd = np.random.random(500) * 360
ax = WindroseAxes.from_ax()
ax.bar(wd, ws, normed=True, opening=0.8, edgecolor='white')
ax.set_legend()
windgraph = px.data.wind()
fig = px.bar_polar(windgraph, r="frequency", theta="direction",
                   color="strength")
fig.show()
plt.figure(figsize=(14,12))
from math import pi
df = pd.DataFrame({
'group': ['A','B','C','D'],
'var1': [38, 1.5, 30, 4],
'var2': [29, 10, 9, 34],
'var3': [8, 39, 23, 24],
'var4': [7, 31, 33, 14],
'var5': [28, 15, 32, 14]
})
 
# number of variable
categories=list(df)[1:]
N = len(categories)
 
# We are going to plot the first line of the data frame.
# But we need to repeat the first value to close the circular graph:
values=df.loc[0].drop('group').values.flatten().tolist()
values += values[:1]
values
 
# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
 
# Initialise the spider plot
ax = plt.subplot(111, polar=True)
 
# Draw one axe per variable + add labels labels yet
plt.xticks(angles[:-1], categories, color='grey', size=8)
 
# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([10,20,30], ["10","20","30"], color="grey", size=7)
plt.ylim(0,40)
 
# Plot data
ax.plot(angles, values, linewidth=1, linestyle='solid')
 
# Fill area
ax.fill(angles, values, 'b', alpha=0.1)

import plotly.express as px
wind = px.data.wind()
fig = px.line_polar(wind, r="frequency", theta="direction",
                   color="strength")
fig.show()
plt.figure(figsize=(14,12))
df = pd.DataFrame({
'group': ['A','B','C','D'],
'var1': [38, 1.5, 30, 4],
'var2': [29, 10, 9, 34],
'var3': [8, 39, 23, 24],
'var4': [7, 31, 33, 14],
'var5': [28, 15, 32, 14]
})
 
 
 
# ------- PART 1: Create background
 
# number of variable
categories=list(df)[1:]
N = len(categories)
 
# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
 
# Initialise the spider plot
ax = plt.subplot(111, polar=True)
 
# If you want the first axis to be on top:
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)
 
# Draw one axe per variable + add labels labels yet
plt.xticks(angles[:-1], categories)
 
# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([10,20,30], ["10","20","30"], color="grey", size=7)
plt.ylim(0,40)
 
 
# ------- PART 2: Add plots
 
# Plot each individual = each line of the data
# I don't do a loop, because plotting more than 3 groups makes the chart unreadable
 
# Ind1
values=df.loc[0].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="group A")
ax.fill(angles, values, 'b', alpha=0.1)
 
# Ind2
values=df.loc[1].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="group B")
ax.fill(angles, values, 'r', alpha=0.1)
 
# Add legend
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))


plt.figure(figsize=(14,14))
r = np.arange(0, 2, 0.01)
theta = 2 * np.pi * r

ax = plt.subplot(111, projection='polar')
ax.plot(theta, r)
ax.set_rmax(2)
ax.set_rticks([0.5, 1, 1.5, 2])  # less radial ticks
ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
ax.grid(True)

ax.set_title("A line plot on a polar axis", va='bottom')
plt.show()
plt.figure(figsize=(14,14))
ax = plt.axes([0.025, 0.025, 0.95, 0.95], polar=True)

N = 20
theta = np.arange(0.0, 2 * np.pi, 2 * np.pi / N)
radii = 10 * np.random.rand(N)
width = np.pi / 4 * np.random.rand(N)
bars = plt.bar(theta, radii, width=width, bottom=0.0)

for r,bar in zip(radii, bars):
    bar.set_facecolor(plt.cm.jet(r/10.))
    bar.set_alpha(0.5)

ax.set_xticklabels([])
ax.set_yticklabels([])
plt.show()
import plotly.express as px
wind = px.data.wind()
fig = px.scatter_polar(wind, r="frequency", theta="direction",
                   color="strength")
fig.show()
!pip install mpl_finance
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc
import pandas as pd
import matplotlib.dates as mpl_dates

plt.style.use('ggplot')

# Extracting Data for plotting
data = pd.read_csv('../input/candlestick-python-datacsv/candlestick_python_data.csv')
ohlc = data.loc[:, ['Date', 'Open', 'High', 'Low', 'Close']]
ohlc['Date'] = pd.to_datetime(ohlc['Date'])
ohlc['Date'] = ohlc['Date'].apply(mpl_dates.date2num)
ohlc = ohlc.astype(float)

# Creating Subplots
fig, ax = plt.subplots()
plt.figure(figsize=(14,14))
candlestick_ohlc(ax, ohlc.values, width=0.6, colorup='green', colordown='red', alpha=0.8)

# Setting labels & titles
ax.set_xlabel('Date')
ax.set_ylabel('Price')
fig.suptitle('Daily Candlestick Chart of NIFTY50')

# Formatting Date
date_format = mpl_dates.DateFormatter('%d-%m-%Y')
ax.xaxis.set_major_formatter(date_format)
fig.autofmt_xdate()

fig.tight_layout()

plt.show()
import warnings
warnings.filterwarnings("ignore")
import plotly.graph_objects as go
from datetime import datetime
Tesla = pd.read_csv('../input/tesla-stock-data-from-2010-to-2020/TSLA.csv')
fig = go.Figure(data=[go.Candlestick(x=Tesla['Date'],
                open=Tesla['Open'],
                high=Tesla['High'],
                low=Tesla['Low'],
                close=Tesla['Close'])])

fig.show()
import seaborn as sns
plt.figure(figsize=(15,6))
h=pd.pivot_table(Place,columns='sl_no',values=["salary"])
sns.heatmap(h,cmap=['yellow','red','green'],linewidths=0.05)
import folium 
import webbrowser
m = folium.Map(location=[45.5236, -122.6750])

m

import os
import json
import requests


url = 'https://raw.githubusercontent.com/python-visualization/folium/master/examples/data'
vis1 = json.loads(requests.get(f'{url}/vis1.json').text)
vis2 = json.loads(requests.get(f'{url}/vis2.json').text)
vis3 = json.loads(requests.get(f'{url}/vis3.json').text)
m = folium.Map(
    location=[46.3014, -123.7390],
    zoom_start=7,
    tiles='Stamen Terrain'
)

folium.Marker(
    location=[47.3489, -124.708],
    popup=folium.Popup(max_width=450).add_child(
        folium.Vega(vis1, width=450, height=250))
).add_to(m)

folium.Marker(
    location=[44.639, -124.5339],
    popup=folium.Popup(max_width=450).add_child(
        folium.Vega(vis2, width=450, height=250))
).add_to(m)

folium.Marker(
    location=[46.216, -124.1280],
    popup=folium.Popup(max_width=450).add_child(
        folium.Vega(vis3, width=450, height=250))
).add_to(m)


m
covid = pd.read_csv("../input/global-hospital-beds-capacity-for-covid19/hospital_beds_USA_v1.csv") #COVID 19 DATA

import folium
map = folium.Map(location=[37.0902,-95.7129 ], zoom_start=4,tiles='cartodbpositron')

for lat, lon,state,type in zip(covid['lat'], covid['lng'],covid['state'],covid['type']):
    folium.CircleMarker([lat, lon],
                        radius=5,
                        color='green',
                      popup =(
                    'State: ' + str(state) + '<br>'),

                        fill_color='green',
                        fill_opacity=0.7 ).add_to(map)
map
import pandas as pd


url = 'https://raw.githubusercontent.com/python-visualization/folium/master/examples/data'
state_geo = f'{url}/us-states.json'
state_unemployment = f'{url}/US_Unemployment_Oct2012.csv'
state_data = pd.read_csv(state_unemployment)

m = folium.Map(location=[48, -102], zoom_start=3)

folium.Choropleth(
    geo_data=state_geo,
    name='choropleth',
    data=state_data,
    columns=['State', 'Unemployment'],
    key_on='feature.id',
    fill_color='YlGn',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Unemployment Rate (%)'
).add_to(m)

folium.LayerControl().add_to(m)

m
Covid19 = pd.read_csv("../input/coronavirus-2019ncov/covid-19-all.csv")
grp = Covid19.groupby(['Country/Region'])['Confirmed', 'Deaths', 'Recovered'].max()
grp = grp.reset_index()
grp['Country'] =  grp['Country/Region']
fig = px.choropleth(grp, locations="Country", locationmode='country names', 
                     color="Confirmed", hover_name="Country/Region",hover_data = [grp.Recovered,grp.Deaths],projection="orthographic",
                     color_continuous_scale='Plasma')
fig.update(layout_coloraxis_showscale=True)

fig.show()#open browser tab to view
cases = pd.read_csv("../input/coronavirus-2019ncov/covid-19-all.csv")
grp = cases.groupby(['Country/Region'])['Confirmed', 'Deaths', 'Recovered'].max()
grp = grp.reset_index()
grp['Country'] =  grp['Country/Region']
fig = px.choropleth(grp, locations="Country", locationmode='country names', 
                     color="Confirmed", hover_name="Country/Region",hover_data = [grp.Recovered,grp.Deaths],projection="mercator",
                     color_continuous_scale='greens')
fig.update(layout_coloraxis_showscale=True)

fig.show()#open browser tab to view
import pandas as pd 
Covid19 = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")
import plotly.express as px
import plotly.graph_objects as go




grp = Covid19.groupby(['ObservationDate', 'Country/Region'])['Confirmed', 'Deaths', 'Recovered'].max()
grp = grp.reset_index()
grp['Date'] = pd.to_datetime(grp['ObservationDate'])
grp['Date'] = grp['Date'].dt.strftime('%m/%d/%Y')
grp['Active'] = grp['Confirmed'] - grp['Recovered'] - grp['Deaths']
grp['Country'] =  grp['Country/Region']

fig = px.choropleth(grp, locations="Country", locationmode='country names', 
                     color="Active", hover_name="Country/Region",hover_data = [grp.Recovered,grp.Deaths,grp.Active],projection="natural earth",
                     animation_frame="Date",width=800, height=500,
                     color_continuous_scale='Blues',
                     range_color=[1000,100000])
fig.update(layout_coloraxis_showscale=True)

fig.show()#open browser tab to view

hsp =  pd.read_csv("../input/global-hospital-beds-capacity-for-covid19/hospital_beds_global_regional_v1.csv")
import pandas as pd

import plotly.graph_objects as go
fig = go.Figure(go.Densitymapbox(lat=hsp.lat, lon=hsp.lng,
                                 radius=10))
fig.update_layout(mapbox_style="stamen-terrain", mapbox_center_lon=0)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()#open browser tab to view
k = covid.sort_values(by=['year'])
k= k.head(200)
fig=px.bar(k,x='state', y="population", animation_frame="year", 
           animation_group="state", color="state", hover_name="state")
fig.update_layout(title='Deaths vs Region')

fig.show()#open browser tab to view
fig = px.line_3d(Covid19, x="Country/Region", y="Confirmed", z="Deaths",color = "Country/Region")

fig.show()#open browser tab to view
fig = px.line_3d(covid, x="state", y="population", z="beds",color = "state")

fig.show()#open browser tab to view
fig = px.scatter_3d(Covid19, x="Country/Region", y="Confirmed", z="Deaths",color = "Country/Region")

fig.show()#open browser tab to view
fig = px.scatter_3d(covid, x="state", y="population", z="beds",color = "state")

fig.show()#open browser tab to view
cds = covid
cds = cds.sort_values(by=['year'])

map1  = cds.groupby(['year','state' ,'lat','lng'])['beds'].max()


map1 = map1.reset_index()
map1['size'] = map1['beds']*90000000
fig = px.scatter_mapbox(map1, lat="lat", lon="lng",
                     color="beds", size='size',hover_data=['beds'],
                     color_continuous_scale='Blues',
                     animation_frame="year")
fig.update(layout_coloraxis_showscale=True)
fig.update_layout(mapbox_style="carto-positron",
                  mapbox_zoom=3)

fig.show()#open browser tab to view

cdf = covid
cdf = cdf.sort_values(by=['year'])

cdf["year"] = cdf["year"].astype(str)

fig = px.choropleth(covid, locations=cdf["state"],       

 color=cdf["beds"],
                    locationmode="USA-states",
                    scope="usa",
                    animation_frame=cdf["year"],

                    color_continuous_scale='Greens',
                   )

fig.show()#open browser tab to view
import plotly.express as px
df = px.data.gapminder()
BUBBLE = px.scatter(df, x="gdpPercap", y="lifeExp", animation_frame="year", animation_group="country",
           size="pop", color="continent", hover_name="country",
           log_x=True, size_max=55, range_x=[100,100000], range_y=[25,90])

BUBBLE.show()#opens browser tab to view
fig = go.Figure(data=[go.Table(
    header=dict(values=list(Place.columns),
                align='left'),
    cells=dict(values=[Place.sl_no, Place.gender, Place.ssc_p,Place.ssc_b, Place.hsc_p, Place.hsc_b, Place.hsc_b, Place.hsc_s, Place.degree_p,Place.degree_t, Place.workex, Place.etest_p, Place.specialisation, Place.status, Place.salary],
               align='left'))
])

fig.show()#opens browser tab to view
import plotly.express as px
df = Place
df["e"] = df["hsc_p"]/50
fig = px.scatter(df, x="hsc_s", y="hsc_p", color="hsc_b",
                 error_x="e", error_y="e")

fig.show()
from pandas_profiling import ProfileReport 
report = ProfileReport(train)
report
