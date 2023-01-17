import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline


df = pd.DataFrame({
    'name':['Ram','mary','Drik','John','Arav','lisa','Gud'],
    'age':[23,78,22,19,45,33,20],
    'gender':['M','F','M','M','M','F','M'],
    'state':['california','texas','california','dc','california','dc','texas'],
    'num_children':[2,0,4,3,2,1,1],
    'num_pets':[3,1,0,5,2,2,1]})
print(df)
##Scatter Plot
df.plot(kind='scatter',x='num_children',y='num_pets',color='red')
plt.show()
##Barplot
df.plot(kind='bar',x='name',y='age')
##BarPlot
df.groupby('state')['name'].nunique().plot(kind='bar')
plt.show()
##StackedBarPlot
df.groupby(['state','gender']).size().unstack().plot(kind='bar',stacked=True)
plt.show()
#Line PLot
df.plot(kind='line',x='name',y='num_children')
plt.show()
#Histogram
df.hist()
#Histogram
df[['age']].plot(kind='hist',bins=[0,20,40,60,80,100],rwidth=0.8)
plt.show()
##Area Plot
df.plot.area()
##BoxPlot
df.boxplot('age')
pd.plotting.scatter_matrix(df)
plt.show()
##LinePlot
time = [0, 1, 2, 3]
position = [0, 10, 20, 30]
##Normal X & Y axis plot
plt.plot(time, position)
plt.xlabel('Time (hr)') ##labelling the axis
plt.ylabel('Position (km)')
##Scatter Plot
x=[1,2,3,4,5,6,7,8,9,10]
y=[11,23,46,56,76,87,56,22,99,76]
plt.scatter(x,y,label='Scatter',color='g',marker='o',s=100) #k represents black color
plt.xlabel('X Value')
plt.ylabel('Y Value')
plt.show()
plt.ioff() 
#Pie Plot
hours=[8,2,8,2,4]
activity=['Sleep','Eat','Work','Play','Others']
col=['c','b','m','r','y']
plt.figure(figsize=(9,9))
plt.pie(hours,labels=activity,colors=col,shadow=True,startangle=0,explode=(0,0,0,0.2,0),autopct='%1.1f%%')
plt.title('How time is in a Day')
plt.ioff()

##BarPlot
car = ['Audi', 'BMW', 'Honda', 'Maruti', 'Toyota']
count = [20,45,65,60,35]
plt.bar(car,count)
plt.show()
##Grouped BarPlot

men_means = (20, 35, 30, 35, 27)
women_means = (25, 32, 34, 20, 25)

ind = np.arange(5) 
width = 0.35       
plt.bar(ind, men_means, width, label='Men')
plt.bar(ind + width, women_means, width,
    label='Women')

plt.ylabel('Scores')
plt.title('Scores by group and gender')

plt.xticks(ind + width / 2, ('G1', 'G2', 'G3', 'G4', 'G5'))
plt.legend(loc='best')
plt.show()
##Stacked bar plot
p1 = plt.bar(ind, men_means, width)
p2 = plt.bar(ind, women_means, width,
             bottom=men_means)

plt.ylabel('Scores')
plt.title('Scores by group and gender')
plt.xticks(ind, ('G1', 'G2', 'G3', 'G4', 'G5'))
plt.yticks(np.arange(0, 81, 10))
plt.legend((p1[0], p2[0]), ('Men', 'Women'))

plt.show()
tips = sns.load_dataset("tips")## default dataset present in seaborn package
tips.head()## will display top 5 rows

sns.relplot(x="total_bill", y="tip", data=tips) ##Relating variables with scatter plots
sns.relplot(x="total_bill", y="tip", hue="smoker", data=tips);
sns.relplot(x="total_bill", y="tip", hue="smoker", style="smoker",data=tips);
sns.relplot(x="total_bill", y="tip", hue="smoker", style="time", data=tips);
sns.relplot(x="total_bill", y="tip", hue="size", data=tips);
sns.relplot(x="total_bill", y="tip", size="size", sizes=(15, 200), data=tips);

sns.catplot(x="day", y="total_bill", data=tips);
sns.catplot(x="day", y="total_bill", jitter=False, data=tips);
sns.catplot(x="day", y="total_bill", kind="swarm", data=tips);
sns.catplot(x="total_bill", y="day", kind="swarm", data=tips);
sns.catplot(x="day", y="total_bill", hue="sex", kind="swarm", data=tips);
sns.catplot(x="size", y="total_bill", kind="swarm",data=tips.query("size != 3"));
sns.catplot(x="smoker", y="tip", order=["No", "Yes"], data=tips);
##Box Plot
sns.catplot(x="day", y="total_bill", kind="box", data=tips);
sns.catplot(x="day", y="total_bill", hue="smoker", kind="box", data=tips);
tips["weekend"] = tips["day"].isin(["Sat", "Sun"])
sns.catplot(x="day", y="total_bill", hue="weekend",kind="box", dodge=False, data=tips);
sns.catplot(x="day", y="total_bill", hue="weekend",kind="boxen", dodge=False, data=tips);
#Violinplots
sns.catplot(x="total_bill", y="day", hue="sex",kind="violin", data=tips);
sns.catplot(x="day", y="total_bill", hue="sex",kind="violin", split=True, data=tips);
sns.catplot(x="day", y="total_bill", hue="sex",kind="violin", inner="stick", split=True,
            palette="pastel", data=tips);
g = sns.catplot(x="day", y="total_bill", kind="violin", inner=None, data=tips)
sns.swarmplot(x="day", y="total_bill", color="k", size=3, data=tips, ax=g.ax);

sns.catplot(x="day", y="total_bill",kind="bar", data=tips);
sns.catplot(x="day", kind="count", palette="ch:.25", data=tips)
sns.catplot(x="day", hue="sex", kind="count",palette="pastel", edgecolor=".6",data=tips);
#Point plots
sns.catplot(x="day", y="total_bill", hue="time", kind="point", data=tips);

sns.regplot(x="total_bill", y="tip", data=tips);
sns.lmplot(x="total_bill", y="tip", data=tips); ##More linear relation with facetgrid
g = sns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips,markers=["o", "x"])
g = sns.lmplot(x="total_bill", y="tip", col="smoker", data=tips)
g = sns.lmplot(x="total_bill", y="tip", col="day", hue="day",data=tips, col_wrap=2, height=3)
g = sns.lmplot(x="total_bill", y="tip", row="sex", col="time",data=tips, height=3)
sns.set_style('ticks',{"xtick.major.size": 12, "ytick.major.size": 12})
sns.distplot(tips['total_bill'], kde = False, color ='red', bins = 30)
sns.kdeplot(tips['total_bill'], color ='green',shade=True)
g = sns.jointplot(x="total_bill", y="tip", data=tips)
sns.jointplot(x ='total_bill', y ='tip', data = tips, kind ='kde')
g = sns.jointplot(x="total_bill", y="tip", data=tips, kind="reg")
g = sns.jointplot(x="total_bill", y="tip", data=tips, kind="hex")
iris = sns.load_dataset("iris")
g = sns.pairplot(iris)
g = sns.pairplot(iris, hue="species")
g = sns.pairplot(iris, hue="species", palette="husl")
g = sns.pairplot(iris, corner=True) #Plot only the lower triangle of bivariate axes
g = sns.pairplot(iris, diag_kind="kde")
