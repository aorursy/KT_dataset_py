import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
x = np.array([1,2,3,4,5,6,7])

y = x



plt.figure()

plt.scatter(x,y);
colors = ['green']*(len(x))

colors[-1] = 'red'



plt.figure()

plt.scatter(x,y,c=colors,s=100)
colors = ['green']*(len(x))

colors[-1] = 'red'



plt.figure()

plt.scatter(x,y,c=colors,s=100)

plt.xlabel('X Label')

plt.ylabel('Y Label')

plt.legend()

plt.title('Sample Scatter Plot');
plt.figure()

plt.scatter(x[:-1], y[:-1], c='green', s=100, label='Positive')

plt.scatter(x[-1], y[-1], c='red', s=100, label='Negative')

plt.xlabel('X Label')

plt.ylabel('Y Label')

plt.legend(loc = "upper left", fontsize = 13)

plt.title('Sample Scatter Plot');
df1 = pd.read_csv('../input/youtube-video-likes-by-corey-schafer-github/vidlikes.csv')

print(df1.head())



view_count = df1['view_count']

likes = df1['likes']

ratio = df1['ratio']
plt.figure()

plt.scatter(view_count, likes, c=ratio, cmap='summer', edgecolor='black', linewidth=1, alpha=0.75)

plt.title('Trending YouTube Videos')

plt.xlabel('View Count')

plt.ylabel('Total Likes');
plt.figure(figsize=(7,5))

plt.scatter(view_count, likes, c=ratio, cmap='summer', edgecolor='black', linewidth=1, alpha=0.75)



plt.xscale('log')

plt.yscale('log')



cbar = plt.colorbar()

cbar.set_label('Like Dislike Ratio')



plt.title('Trending YouTube Videos')

plt.xlabel('View Count')

plt.ylabel('Total Likes');
linear_data = np.array([1,2,3,4,5,6,7,8])

exponential_data = linear_data**2



# plot the linear data and the exponential data

plt.plot(linear_data, '-o', exponential_data, '-o')
plt.plot(linear_data, '-o', exponential_data, '-o')



plt.xlabel('X-axis')

plt.ylabel('y-axis')

plt.title('Linear vs Exponential')

# add a legend with legend entries (because we didn't have labels when we plotted the data series)

plt.legend(['Linear', 'Exponential'])



plt.fill_between(range(len(linear_data)), 

                       linear_data, exponential_data, 

                       facecolor='blue', 

                       alpha=0.25);
ages_x = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]

python = [45372, 48876, 53850, 57287, 63016, 65998, 70003, 70000, 71496, 75370, 83640]

java = [37810, 43515, 46823, 49293, 53437, 56373, 62375, 66674, 68745, 68746, 74583]

Others = [38496, 42000, 46752, 49320, 53200, 56000, 62316, 64928, 67317, 68748, 73752]



plt.plot(ages_x, python, 'b', linewidth=3, label='Python')   

plt.plot(ages_x, java, color='#adad3b', linewidth=3, label='Java')

plt.plot(ages_x, Others, color='black', linestyle='--', label='All Devs')



plt.title('Median Salary by Age')

plt.xlabel('Ages')

plt.ylabel('Median Salary')

plt.legend(fontsize=12);   #necessary to give labels to the line/ You can also pass label as argument inside legend
linear_data = np.array([1,2,3,4,5,6,7,8])

exponential_data = linear_data**2

xvals = range(len(linear_data))

xvals1 = []

for item in xvals:

    xvals1.append(item+0.3) #Because width of first bar is 0.3

    

plt.bar(xvals, linear_data, width = 0.3, color='red', label='Linear')

plt.bar(xvals1, exponential_data, width=0.3, color='blue', label='Exponential')

plt.title('Linear vs Exponential')

plt.legend(fontsize=13);
plt.bar(xvals, linear_data, width = 0.3, color='b', label='Linear')

plt.bar(xvals, exponential_data, width = 0.3, bottom=linear_data, color='r', label='Exponential')

plt.title('Linear vs Exponential')

plt.legend(fontsize=13);
languages =['Python', 'SQL', 'Java', 'C++', 'JavaScript']

pos = np.arange(len(languages))

popularity = [56, 39, 34, 34, 29]



plt.bar(pos, popularity, align='center')

plt.title('Top 5 Languages for Math & Data \nby % popularity on Stack Overflow', alpha=0.8);
# change the bar colors to be less bright blue

bars = plt.bar(pos, popularity, align='center', linewidth=0, color='lightslategrey')

# make one bar, the python bar, a contrasting color

bars[0].set_color('#1F77B4')



# soften all labels by turning grey

plt.xticks(pos, languages, alpha=0.8)

plt.title('Top 5 Languages for Math & Data \nby % popularity on Stack Overflow', alpha=0.8)



#remove ytick labels

plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=True)



# remove the frame of the chart

for spine in plt.gca().spines.values():

    spine.set_visible(False)



# direct label each bar with Y axis values

for bar in bars:

    plt.gca().text(bar.get_x() + bar.get_width()/2, bar.get_height() - 5, str(int(bar.get_height())) + '%', 

                 ha='center', color='w', fontsize=11)
plt.figure(figsize=(8,4))



plt.subplot(1,2,1) #number of rows, number of columns, current position

plt.bar(pos, popularity, align='center')

plt.title('Top 5 Languages for Math & Data \nby % popularity on Stack Overflow', alpha=0.8)



plt.subplot(1,2,2) #number of rows, number of columns, current position

bars = plt.bar(pos, popularity, align='center', linewidth=0, color='lightslategrey')

bars[0].set_color('#1F77B4')

plt.xticks(pos, languages, alpha=0.8)

plt.title('Top 5 Languages for Math & Data \nby % popularity on Stack Overflow', alpha=0.8)

plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=True)

for spine in plt.gca().spines.values():

    spine.set_visible(False)

for bar in bars:

    plt.gca().text(bar.get_x() + bar.get_width()/2, bar.get_height() - 5, str(int(bar.get_height())) + '%', 

                 ha='center', color='w', fontsize=11)

    

plt.tight_layout(pad=3)
from sklearn.datasets import load_iris



iris = load_iris()

df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],

                     columns= iris['feature_names'] + ['target'])

df.head()
plt.figure(figsize=(11,7))



for i in range(1, len(df.columns)):

    plt.subplot(2, 2, i)

    plt.scatter(df.iloc[:,i-1], df['target'])

    plt.title(df.columns[i-1])

    plt.ylabel('Target')

    

plt.tight_layout(pad=3)
import seaborn as sns



sns.pairplot(data=df, x_vars='sepal length (cm)', y_vars='sepal width (cm)', hue='target');
g = sns.pairplot(data=df, x_vars=df.columns[:-1], y_vars=df.columns[:-1], hue='target')

g.fig.suptitle('Relationship Among Features', y=1.08, fontsize=20);
sns.boxplot(data=df.drop(columns='target'), orient="h", palette="Set2")
sns.swarmplot(data=df.drop(columns='target'), orient="h", palette="Set2")
sns.violinplot(data=df.drop(columns='target'), orient="h", palette="Set2")
plt.figure(figsize=(10,5))



plt.subplot(1,2,1)

sns.regplot(df['petal length (cm)'], df['petal width (cm)'])



plt.subplot(1,2,2)

sns.regplot(df['sepal length (cm)'], df['sepal width (cm)'])



plt.tight_layout(pad=3)
sns.countplot(x='target', data=df);
list1 = np.random.randint(2, size=50)

test = pd.DataFrame(list1, columns=['int'])



sns.countplot(x='int', data=test);
speed = [0.1, 17.5, 40, 48, 52, 69, 88]

lifespan = [2, 8, 70, 1.5, 25, 12, 28]

index = ['snail', 'pig', 'elephant', 'rabbit', 'giraffe', 'coyote', 'horse']

df1 = pd.DataFrame({'speed': speed, 'lifespan': lifespan}, index=index)

ax = df1.plot.bar(rot=0)
sns.heatmap(df.corr(), annot=True);
sns.heatmap(df1.isnull(), yticklabels=False, cmap='plasma')