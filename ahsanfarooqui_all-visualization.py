import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('../input/data_titanic.csv')
df.head()
#plotting multiple graphs

plt.plot(df.PassengerId,df.Age,color='red',label='Age',alpha=0.8)
plt.plot(df.Fare,color='blue',label='Fare',alpha=0.2)
plt.legend(loc='best')
plt.xlabel("Passenger ID")
plt.show()
#Plotting using subplots
plt.subplot(2,1,1)
plt.title('Using subplots')
plt.plot(df.PassengerId,df.Age,color='red',marker='o')
plt.xlabel('Passenger ID')
plt.ylabel('Age')
plt.show()

plt.subplot(2,1,2)
plt.title('Using subplots 2')

plt.plot(df.PassengerId,df.Fare,color='green',marker='*')
plt.xlabel('Passenger ID')
plt.ylabel('Fare')

plt.show()
#using axes to specify [xlo, ylo, width, height] instead of subplot.

plt.axes([0.05, 0.5, 0.5, 0.9])
plt.plot(df.Age,c='orange')
plt.xlabel('passengers')
plt.ylabel('Age')
plt.title('Age Plot')


plt.axes([0.625, 0.5, 0.5, 0.9])
plt.plot(df.Fare,c='magenta')
plt.xlabel('passengers')
plt.ylabel('Fare')
plt.title('Passenger plot')
plt.show()

#Using custom limits of x and y axes

plt.plot(df.PassengerId,df.Age,c='blue',marker='*')
plt.xlim([0,200])
plt.show()
#If we wish to set both x and y limits, we can also do that using axis. 

plt.plot(df.PassengerId,df.Age,color='blue')
plt.axis((75,90,0,40))
plt.show()
#Placing an arrow at the maximum point of the age

#plotting the two arrays as usual
plt.plot(df.PassengerId,df.Age,color='Green')
plt.xlabel('Passenger ID')
plt.ylabel('Age')
plt.title('Checking Arrow')

#setting y limit a bit higher to place arrow properly

plt.ylim(0,100)

#Getting max age to see which is the highest age. 
age = df['Age']
max_age = df['Age'].max()

# argmax() get argument corresponding to the max value

passengerid = df['PassengerId']
pidmax = passengerid[age.argmax()]

plt.annotate('Maximum', xy=(pidmax, max_age), xytext=(pidmax+20, max_age+20), arrowprops=dict(facecolor='black'))

plt.show()
#seeing visualization techniques 

plt.style.available
#You can use one of the above mentioned styles to plot your graphs. 

plt.style.use('classic')
plt.subplot(2, 2, 1) 
plt.plot(df.PassengerId,df.Age)
plt.subplot(2, 2, 2) 
plt.plot(df.PassengerId,df.Survived,'ro')
plt.subplot(2, 2, 3) 
plt.plot(df.PassengerId,df.Fare)
plt.tight_layout()
plt.show()
#visualizing mesh grids 

u = np.linspace(-2, 2, 41)
v = np.linspace(-1,1,21)

X,Y = np.meshgrid(u,v)

Z = np.sin(3*np.sqrt(X**2 + Y**2)) 
plt.pcolor(Z)
plt.xlim([0,41])
plt.ylim([0,21])
plt.tight_layout()
plt.show()

#Using contourplots

plt.style.use('classic')

plt.subplot(2,2,1)
plt.contourf(X,Y,Z,20, cmap='viridis')
plt.colorbar()
plt.title('Viridis')

# Create a filled contour plot with a color map of 'gray'
plt.subplot(2,2,2)
plt.contourf(X,Y,Z,20, cmap='gray')
plt.colorbar()
plt.title('Gray')

# Create a filled contour plot with a color map of 'autumn'
plt.subplot(2,2,3)
plt.contourf(X,Y,Z,20,cmap="autumn")
plt.colorbar()
plt.title('Autumn')

# Create a filled contour plot with a color map of 'winter'
plt.subplot(2,2,4)
plt.contourf(X,Y,Z,20,cmap='winter')
plt.colorbar()
plt.title('Winter')

# Improve the spacing between subplots and display them
plt.tight_layout()
plt.show()
plt.hist2d(df.Fare,df.Age,bins=(30,30),cmap='YlGnBu')

#complete list of cmaps available at https://matplotlib.org/users/colormaps.html

plt.xlabel('Fares')
plt.ylabel('Ages')
plt.title('2D histogram between Age and Fare')
plt.colorbar()
plt.show()
plt.hist2d(df.Survived,df.Age,bins=(30,30),cmap='YlGnBu')

#complete list of cmaps available at https://matplotlib.org/users/colormaps.html

plt.xlabel('Survived')
plt.ylabel('Ages')
plt.title('2D histogram between Age and Fare')
plt.colorbar()
plt.show()
#using hexbins to visualize same data. 

plt.hexbin(df.Age,df.Fare,gridsize=(15,12))

plt.show()
df.head()
df = df.drop('Name',axis=1)
df.head()
df = df.drop('Cabin',axis=1)
df.head()
df = df.drop('Ticket',axis=1)
df.head()
df = pd.get_dummies(df)
df.head()
df = df.drop('Sex_female',axis=1)
df = df.drop('Embarked_C',axis=1)
df = df.drop('Initial_Master',axis=1)
df.head()
#LMPlot
sns.lmplot(x='Age',y='Fare',data=df)
plt.show()
#Residual Plot
sns.residplot(x='Age', y='Fare', data=df, color='green')
#plt.show()
#Higher order regression

plt.scatter(df['Age'], df['Fare'], label='data', color='red', marker='o')
sns.regplot(x='Age', y='Fare', data=df, scatter=None, color='blue', label='order 1')
sns.regplot(x='Age', y='Fare', data=df, scatter=None, color='green', label='order 2',order=2)
sns.regplot(x='Age', y='Fare', data=df, scatter=None, color='green', label='order 3',order=3)
sns.regplot(x='Age', y='Fare', data=df, scatter=None, color='green', label='order 4',order=4)

plt.legend(loc='best')

plt.show()

#Regression by hue

sns.lmplot(x='Age',y='Fare',hue='Survived',data=df,palette='Set1')

plt.show()
sns.lmplot(x='Age',y='Fare',hue='Survived',row='Pclass',data=df)
plt.show()
#visualizing univariate data

sns.stripplot(x='Survived', y='Fare', data=df)
plt.show()
sns.stripplot(x='Survived', y='Fare', data=df,jitter=True,size=2)
plt.show() 
#Swarm Plot works same as strip plot in jitter

sns.swarmplot(x='Survived',y='Age',hue='Sex_male',data=df)
plt.show()
#Violin plots

sns.violinplot(x='Survived', y='Age', data=df)
plt.show()
#violin plots with strip plots.
#inner = None removes the inner boxplot. coloring it light gray makes it a light background for overlaying the strip plot
sns.violinplot(x='Survived', y='Age', data=df,inner=None,color='lightgray')
sns.stripplot(x='Survived',y='Age',data=df,jitter=True,size=1.5)
plt.show()
#bivariate distributions

sns.jointplot('Fare','Age',data=df,kind='hex')
plt.show()

#pairplot
sns.pairplot(df)
plt.show()
# Plot the pairwise joint distributions grouped by 'origin' along with regression lines
sns.pairplot(df,kind='reg',hue='Survived')
plt.show()

df.corr()
sns.heatmap(df.corr(),annot=True,annot_kws={"size": 5})
plt.show()
import tensorflow as tf
