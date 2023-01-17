# Import necessary libraries
import pandas as pd
import numpy as np
# Libraries related to matplot lib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
%matplotlib inline
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Reading the Iris data set into a data frame
df = pd.read_csv("../input/iris-dataset/Iris.csv")
df.head()
df.info()
#Change the column Species to category type
df['Species'] = df['Species'].astype('category')
df.info()
# Using plot 
df.plot(x = 'Id', y = 'SepalLengthCm', figsize = (25,5), title = "SepalLength",  linestyle='--', marker='o', color='#185357')
plt.show()
# Using plt and . notation 
fig = plt.figure(figsize=(25,5))
x = df['Id']; y = df['SepalLengthCm']
plt.plot(x,y,  linestyle='--', marker='o', color='#15848a')
plt.xlabel("Id")
plt.ylabel("Value")
plt.title("Sepal Length") # This is not working , do not know the reason
plt.show()
plt.figure(figsize=(25,8))
x = df['Id']
y1 = df['SepalLengthCm']
# plotting the line 1 points 
plt.plot(x, y1, label = "SepalLength", marker = "*")
# line 2 points
y2 = df['PetalLengthCm']
# plotting the line 2 points 
plt.plot(x, y2, label = "PetalLength", marker = "o")
plt.xlabel('ID')
# Set the y axis label of the current axis.
plt.ylabel('Length in Cm')
# Set a title of the current axes.
plt.title('SepalLengh, Petal Length ')
# show a legend on the plot
plt.legend()
# Display a figure.
plt.show()
category = df['Species'].unique()
MeanSepalLength = df[['Species', 'SepalLengthCm']].groupby('Species'). mean().reset_index()
fig = plt.figure(figsize=(15,5))
# creating the bar plot 
plt.bar(x = 'Species', height = 'SepalLengthCm', data = MeanSepalLength, color ='maroon', width = 0.4) 
plt.xlabel("Species") 
plt.ylabel("MeanLength-Cm") 
plt.title("SepalLengthCm-Mean") 
plt.show() 
fig = plt.figure(figsize=(15,5))
MeanSepal = df[['Species', 'SepalLengthCm', 'SepalWidthCm']].groupby('Species'). mean().reset_index()
plt.bar(x = 'Species', height = 'SepalLengthCm', data = MeanSepal, width = 0.4)
plt.bar(x = 'Species', height = 'SepalWidthCm', data = MeanSepal, width = 0.4) 
plt.xlabel("Species") 
plt.xlabel("Species") 
plt.ylabel("MeanLength-Cm") 
plt.title("Sepal-Mean bars") 
plt.xticks('Species', rotation = '65')
plt.show() 
fig = plt.figure(figsize=(20,5))
N = 3
ind = np.arange(N); width = 0.4  
MeanSepal = df[['Species', 'SepalLengthCm', 'SepalWidthCm']].groupby('Species'). mean().reset_index()
category = tuple(MeanSepal.Species)
plt.bar(x = ind, height = 'SepalLengthCm', data = MeanSepal, width = width, label = 'Mean-SepalLength')
plt.bar(x = ind+width , height = 'SepalWidthCm', data = MeanSepal, width = width, label = 'Mean-SepalWidth') 
plt.xlabel("Species") 
plt.ylabel("MeanLength-Cm") 
plt.title("Sepal-Means bars")
#plt.xticks(ind + width / 2, ('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'))
plt.xticks(ind + width / 2, category)
plt.legend(loc='best')
plt.show() 
x1 =df['SepalLengthCm']
x2 =df['PetalLengthCm']
y1 =df['SepalWidthCm']
y2 = df['PetalWidthCm']
#colurs for each category of the species
colors = {'Iris-setosa':'red', 'Iris-versicolor':'blue', 'Iris-virginica':'green'}
# colour patches for the legend prepration
red_patch = mpatches.Patch(color='red', label='Iris-setosa')
blue_patch = mpatches.Patch(color='blue', label='Iris-Versicolor')
green_patch = mpatches.Patch(color='green', label='Iris-Virginica')
f = plt.figure(figsize=(10,5))
#Assign subplots
ax1 = f.add_subplot(121, title = "Sepal Scatter")
ax2 = f.add_subplot(122, title = "Petal Scatter")
# Add a plot for each ax
#Plot1
ax1.scatter(x1, y1, c=df['Species'].apply(lambda x: colors[x]))
ax1.set_xlabel('Sepal Length'); ax1.set_ylabel('Sepal Width')
ax1.legend(handles=[red_patch, blue_patch,green_patch], loc = 'upper left')
#Plot2
ax2.scatter(x2, y2, c=df['Species'].apply(lambda x: colors[x]))
ax2.set_xlabel('Petal Length');ax2.set_ylabel('Petal Width')
ax2.legend(handles=[red_patch, blue_patch,green_patch], loc = 'upper left')
#wrap up to show
plt.tight_layout()
plt.show()
petalwidthmean = df[['Species', 'PetalWidthCm']].groupby("Species").mean().reset_index()
petalwidthmean.columns = ['Species', 'PetalWidthAvg']
petalwidthmean.plot(kind = 'barh', figsize = (10,5), color = '#3d8eba', title = " PetalWidthAverage")
species = list(petalwidthmean['Species'])
y_pos = np.arange(len(petalwidthmean))
plt.yticks(y_pos,species)
plt.xlabel('Values')
plt.show()
plt.figure(figsize=(20,5))
x1 =df['SepalLengthCm']
category = list(df['Species'].unique())
plt.hist(x = x1, bins = 10, range = (4,8), width = .3, label= category, color='#2db350', edgecolor = 'black', align = 'mid')
plt.xlabel('SepalLengthCm')
plt.ylabel('Count')
plt.title('Histogram of SepalLegth')
plt.show()
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize = (20,5))
ax1.hist(df['SepalLengthCm'], edgecolor = 'black', align = 'mid', color = '#bad1c0')
ax1.set_xlabel('Sepal Length')
ax1.set_ylabel('Count')
ax2.hist(df['SepalWidthCm'], edgecolor = 'black', align = 'mid', color = '#91bf9d')
ax2.set_xlabel('Sepal Width')
ax3.hist(df['PetalLengthCm'], edgecolor = 'black', align = 'mid', color = '#adc7b4')
ax3.set_xlabel('Petal Length')
ax4.hist(df['PetalWidthCm'], edgecolor = 'black', align = 'mid', color = '#6bcf86')
ax4.set_xlabel('Petal Width')
plt.tight_layout()
plt.show()
# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = df['Species'].unique()
sizes1 = df.groupby('Species').mean()['SepalLengthCm']
sizes2 = df.groupby('Species').mean()['SepalWidthCm']
sizes3 = df.groupby('Species').mean()['PetalLengthCm']
sizes4 = df.groupby('Species').mean()['PetalWidthCm']
explode = (0, 0.1, 0)  # only "explode" the 2nd slice ()

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize = (25,5))
#plot1
ax1.pie(sizes1, explode=explode, labels=labels, autopct='%1.1f%%',shadow=True, startangle=45)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax1.set_title("Mean-SepalLength")
#Plot2
ax2.pie(sizes2, explode=explode, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)
ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax2.set_title("Mean-SepalWidth")
#plot3
ax3.pie(sizes3, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
ax3.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax3.set_title("Mean-PetalLength")

#plot4
ax4.pie(sizes4, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
ax4.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax4.set_title("Mean-PetalWidth")

plt.show()
plt.figure(figsize = (25,8))
pd.plotting.andrews_curves(df, 'Species')
plt.title("Andrew's Plot")
plt.show()
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize = (20,5))
ax1.boxplot(df['SepalLengthCm'])
ax1.set_xlabel('Sepal Length')
ax1.set_ylabel('Length in Cm')
ax2.boxplot(df['SepalWidthCm'])
ax2.set_xlabel('Sepal Width')
ax3.boxplot(df['PetalLengthCm'])
ax3.set_xlabel('Petal Length')
ax4.boxplot(df['PetalWidthCm'])
ax4.set_xlabel('Petal Width')
plt.tight_layout()
plt.show()

