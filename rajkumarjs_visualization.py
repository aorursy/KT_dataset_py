import matplotlib.pyplot as plt
#Data
dataset = [['Name1','Name2','Name3','Name4','Name5'],[8,15,20,25,20],[10,30,20,50,40]]
Name = dataset[0]
Age = dataset[1]
Rank = dataset[2]
#Line plot - Avoid using this. Best practice is to use Figure - add_subplot()
plt.plot(Name,Age,label='Name Vs Age',color='blue',linewidth=3)
plt.plot(Name,Rank,label='Name Vs Rank',color='green')
plt.scatter(Name,Rank,label='Name Vs Rank',color='green',marker='^',linewidth=3)
plt.legend()
plt.show()
#Line plot - Figure - add_axes() - Use this only when the axes positioning matters
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(Name,Age,label='Name Vs Age',color='blue',linewidth=3)
ax.plot(Name,Rank,label='Name Vs Rank',color='green')
ax.scatter(Name,Rank,label='Name Vs Rank',color='green',marker='^',linewidth=3)
ax.set_xlim(-1.0, 5.0)
ax.set_ylim(0.0, 55.0)
plt.legend()
plt.show()
#Line plot - Figure - add_subplot() - One axes
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(121) #(#Rows,#Columns,#Plot number)
ax.plot(Name,Age,label='Name Vs Age',color='blue',linewidth=3)
ax.plot(Name,Rank,label='Name Vs Rank',color='green')
ax.scatter(Name,Rank,label='Name Vs Rank',color='green',marker='^',linewidth=3)
ax.set_xlim(-1.0, 5.0)
ax.set_ylim(0.0, 55.0)
plt.legend()
plt.show()
#Plot - Figure - add_subplot() - Multiple axes

fig = plt.figure(figsize=(20,10))
fig.suptitle('All charts',color='blue')

ax1 = fig.add_subplot(331) #(#Rows,#Columns,#Plot number)
ax2 = fig.add_subplot(332) #(#Rows,#Columns,#Plot number)
ax3 = fig.add_subplot(333) #(#Rows,#Columns,#Plot number)
ax4 = fig.add_subplot(334) #(#Rows,#Columns,#Plot number)
ax5 = fig.add_subplot(335) #(#Rows,#Columns,#Plot number)
ax6 = fig.add_subplot(336) #(#Rows,#Columns,#Plot number)
ax7 = fig.add_subplot(337) #(#Rows,#Columns,#Plot number)
ax8 = fig.add_subplot(338) #(#Rows,#Columns,#Plot number)
ax9 = fig.add_subplot(339) #(#Rows,#Columns,#Plot number)
#Line Plot
ax1.plot(Name,Age,label='Name Vs Age',color='blue',linewidth=3)
ax1.set(title="Line Plot", xlabel="Name", ylabel="Age") 
ax1.legend(loc='upper right')
ax1.set_xlim(-1.0, 5.0)
ax1.set_ylim(0.0, 55.0)
#Scatter Plot
ax2.scatter(Name,Rank,label='Name Vs Rank',color='green',marker='^',linewidth=3)
ax2.set(title="Scatter Plot", xlabel="Name", ylabel="Rank") 
ax2.legend(loc='lower right')
#Line and Scatter Plot
ax3.plot(Name,Rank,label='Name Vs Rank',color='lightgreen')
ax3.scatter(Name,Rank,label='Name Vs Rank',color='green',marker='^',linewidth=3)
ax3.set(title="Scatter & Line Plot", xlabel="Name", ylabel="Rank") 
ax3.legend(bbox_to_anchor=(1,1))
#Vertical Bar chart
ax4.bar(Name,Age,color='blue')
ax4.set(title="Vertical Bar chart", xlabel="Name", ylabel="Age") 
ax4.axhline(15,color='red',linewidth=3)
#Horizontal Bar Chart
ax5.barh(Name,Age,color='green')
ax5.set(title="Horizontal Bar chart", xlabel="Name", ylabel="Age") 
ax5.axvline(15,color='red',linewidth=3)
#Histogram
ax6.hist(Age)
ax6.set(title="Histogram Bar chart", xlabel="Age") 
#Box plot
ax7.boxplot(Age)
#Violinplot
ax8.violinplot(Age)
#Pie chart
ax9.pie(Age)
ax9.set(title= 'Pie')

plt.subplots_adjust()

#To fit plots nicely in Figure
plt.tight_layout()

#To show the Plot
plt.show()

#To save the Figure to an image file. This piece of code should appear in place of plt.show()
plt.savefig("AllCharts.png", Transparent=True)

#To clear an axis
plt.cla() 
#To clear the entire figure
plt.clf() 
#To close a window that has popped up to show you your plot
plt.close() 
import seaborn as sns
import pandas as pd
df = pd.read_csv("../input/dataset.csv")
df
#Scatterplot
sns.pairplot(df, hue='Name',plot_kws = {'alpha': 0.6, 's': 80, 'edgecolor': 'k'},size = 4)
plt.show()
sns.pairplot(df, hue = 'Name', diag_kind = 'kde',plot_kws = {'alpha': 0.6, 's': 80, 'edgecolor': 'k'},size = 4)
plt.show()
# Create an instance of the PairGrid class.
grid = sns.PairGrid(data= df, size = 4)
# Map a scatter plot to the upper triangle
grid = grid.map_upper(plt.scatter, color = 'darkred')
# Map a histogram to the diagonal
grid = grid.map_diag(plt.hist, bins = 10, color = 'darkred', 
                     edgecolor = 'k')
# Map a density plot to the lower triangle
grid = grid.map_lower(sns.kdeplot, cmap = 'Reds')
plt.show()
#correlation matrix

fig = plt.figure(figsize=(5,5))
#Heatmap using numbers
#saleprice correlation matrix
k = 10 #number of variables for heatmap
sns.set(font_scale=1.25)
sns.heatmap(df.corr(),cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10})
plt.show()
