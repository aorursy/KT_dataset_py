import numpy as np
#In this kernel, we will use numpy library for arrays
import pandas as pd 
#In this kernel, we will use pandas library for open to csv files
import matplotlib.pyplot as plt


import os
print(os.listdir("../input"))
#Sometimes we want to see out datas with plots
#This things too easy with matplotlib
#We can make 2d or 3d plots with matplotlib
#Note for readers :
#Firstly i am too new in matplotlib and also i did this kernel for my education,
#Secondly my english is not very good.
#So thanks you for your understanding!
data = pd.read_csv('../input/battles.csv')
#now we are reading out dataset
data.info()
#This will be give us a information about dataset
#This is our first plot,
#Simple Plot

#Firstly we take datas;
year = data.year
battlenumber = data.battle_number

fig, ax = plt.subplots() #This makes x axis and y axis
ax.plot(year, battlenumber) #And this one makes our datas line

#xlabel=x axis name, ylabel=y axis name, title=title of plots
ax.set(xlabel='Year', ylabel='Battle Number',
       title='Year vs Battle Number')
ax.grid()  #this makes background lines

plt.show()
#bar chart

year = data.year
battlenumber = data.battle_number

#In here we are saying that, x axis for year, y axis for battle number
left = np.array(year)
height = np.array(battlenumber)

plt.bar(left, height, color="pink", width =0.5 ) #0.5 for column width
plt.grid(True) #You can say plt.grid() or plt.grid(true), its depends to you (same meaning)

plt.xlabel('Year')
plt.ylabel('Battle Number')
plt.title('Year vs Battle Number - Bar Chart')

plt.show()
#Pie chart - 1

labels = ['Test1', 'Test2', 'Test3', 'Test4']
#lets give random objects
sizes = [40.0, 40.6, 20.7, 10.3]

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.0f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  

plt.show()
# Pie chart - 2
labels = ['Test1', 'Test2', 'Test3', 'Test4']
sizes = [38.4, 40.6, 20.7, 10.3]
 
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
  
ax1.axis('equal')
plt.tight_layout()
plt.show()
#Pie chart - 3

labels = ['Test1', 'Test2', 'Test3', 'Test4']
sizes = [38.4, 40.6, 20.7, 10.3]

patches, texts = plt.pie(sizes, shadow=True, startangle=90)

plt.legend(patches, labels, loc="best")

plt.axis('equal')
plt.tight_layout()
plt.show()
#Line Plot

data.attacker_size.plot(kind = 'line' , color = 'g' , label = 'Attacker Size' , linewidth = 1, alpha = 0.5 , grid=True , linestyle = ':')
data.defender_size.plot(kind = 'line' , color = 'r' , label = 'defender size' , linewidth = 1, alpha = 0.5 , grid=True , linestyle = '-.')

plt.legend(loc = 'upper right')

plt.title('Attacker Size vs Defender Size')

plt.show()
#Scatter Plot

N = 38 #number of battle
x = data.year
y = data.attacker_size

colors = np.random.rand(N)
#if we dont say this, every bubble will be same color

area = np.pi * (15 * np.random.rand(N))**2  # 0 to 15 point radii
#if we dont say this, every bubble has same size

plt.scatter(x, y,s=area,  c=colors , alpha=0.5)
plt.show()
#Fill Plot

x = data.year
y = data.major_death

fig, ax = plt.subplots()
ax.fill(x, y)

plt.show()