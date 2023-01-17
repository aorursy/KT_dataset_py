# library & dataset
import matplotlib.pyplot as plt
import seaborn as sns
df = sns.load_dataset('iris')
 
# left
sns.pairplot(df, kind="scatter", hue="species", markers=["o", "s", "D"], palette="Set2")
plt.show()
!pip install wikipedia

# Libraries
import numpy as np
import pandas as pd
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns
from math import pi
import squarify
from matplotlib_venn import venn3
from mpl_toolkits.mplot3d import Axes3D
import wikipedia
from wordcloud import WordCloud, STOPWORDS


# Call once to configure Bokeh to display plots inline in the notebook.
output_notebook()
df1 = wikipedia.page('Michael Jordan')
df1_content = df1.content
df1.content
plt.figure(figsize = (20,20)) 
wc = WordCloud(max_words = 500 , width = 1600 , height = 800 , stopwords = STOPWORDS)
wc.generate(df1_content) # Generating WordCloud
plt.imshow(wc , interpolation = 'bilinear')
df2 = pd.read_csv('../input/us-elections-dataset/2012_US_elect_county.csv')
df2.head(5)
plt.style.use('seaborn')
x  = [(i+1) for i in range(10)]
y1 = df2['Obama vote'][1:11]
y2 = df2['Romney vote'][1:11]

plt.plot(x, y1, label="Obama vote", color = '#FF3600')
plt.plot(x, y2, label="Romney vote", color = '#0082FF')
plt.plot()

plt.xlabel("x axis")
plt.ylabel("y axis")
plt.title("Line Graph Example")
plt.legend()
plt.show()
import matplotlib.pyplot as plt
import numpy as np

ys = 200 + np.random.randn(100)
x = [x for x in range(len(ys))]

plt.plot(x, ys, '-')
plt.fill_between(x, ys, 195, where=(ys > 195), facecolor='#00FF86', alpha=0.6)

plt.title("Fills and Alpha Example")
plt.show()
df3 = pd.read_csv("../input/mobile-price-classification/train.csv")
df3.head()
# Look at index 4 and 6, which demonstrate overlapping cases.
x1  = [(i+1) for i in range(5)]
y1 = df3['sc_w'][:5]

x2  = [(i+1) for i in range(5, 10)]
y2 = df3['sc_w'][5:10]

plt.bar(x1, y1, label="Blue Bar", color='#00B8FF')
plt.bar(x2, y2, label="Green Bar", color='#00FF86')
plt.plot()

plt.xlabel("bar number")
plt.ylabel("bar height")
plt.title("Bar Chart Example")
plt.legend()
plt.show()
df4 = pd.read_csv("../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv")
df4.head()
# Use numpy to generate a bunch of random data in a bell curve around 5.
n = 5 + df4['AMBIENT_TEMPERATURE'][:300]

m = [m for m in range(len(n))]
plt.bar(m, n, color = '#FFA600')
plt.title("Raw Data")
plt.show()

plt.hist(n, bins=20, color='#00FF86')
plt.title("Histogram")
plt.show()
# Import library and dataset
import seaborn as sns
df = sns.load_dataset('iris')
 
# Method 1: on the same Axis
sns.distplot( df["sepal_length"] , color="skyblue", label="Sepal Length")
sns.distplot( df["sepal_width"] , color="red", label="Sepal Width")
 
#sns.plt.show()
df5 = pd.read_csv("../input/av-healthcare-analytics-ii/healthcare/train_data.csv")
df5.head()
x1 = df5['City_Code_Hospital'][:10]
y1 = df5['Hospital_code'][:10]

x2 = df5['case_id'][:10]
y2 = df5['City_Code_Hospital'][:10]

# Markers: https://matplotlib.org/api/markers_api.html

plt.scatter(x1, y1, marker='v', color='#0082FF')
plt.scatter(x2, y2, marker='o', color='#FF3600')
plt.title('Scatter Plot Example')
plt.show()
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

x1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y1 = np.random.randint(10, size=10)
z1 = np.random.randint(10, size=10)

x2 = [-1, -2, -3, -4, -5, -6, -7, -8, -9, -10]
y2 = np.random.randint(-10, 0, size=10)
z2 = np.random.randint(10, size=10)

ax.scatter(x1, y1, z1, c='b', marker='o', label='blue')
ax.scatter(x2, y2, z2, c='g', marker='D', label='green')

ax.set_xlabel('x axis')
ax.set_ylabel('y axis')
ax.set_zlabel('z axis')
plt.title("3D Scatter Plot Example")
plt.legend()
plt.tight_layout()
plt.show()
df6 = pd.read_csv('../input/summer-products-and-sales-in-ecommerce-wish/summer-products-with-rating-and-performance_2020-08.csv')
df6.head()
import matplotlib.pyplot as plt

idxes = [(i+1) for i in range(10)]
arr1  = df6['uses_ad_boosts'][:10]
arr2  = df6['rating'][:10]
arr3  = df6['price'][:10]

# Adding legend for stack plots is tricky.
plt.plot([], [], color='#FF4500', label = 'D 1')
plt.plot([], [], color='#00EB42', label = 'D 2')
plt.plot([], [], color='#00D3FF', label = 'D 3')

plt.stackplot(idxes, arr1, arr2, arr3, colors= ['#FF4500', '#00EB42', '#00D3FF'])
plt.title('Stack Plot Example')
plt.legend()
plt.show()
df5['Department'].value_counts()
labels = 'S1', 'S2', 'S3', 'S4'
sections = df5['Department'].value_counts()[1:]
colors = ['#FFE300', '#00ACFF', '#00FF8D', '#FF4900']

plt.pie(sections, labels=labels, colors=colors,
        startangle=90,
        explode = (0, 0, 0.1, 0),
        autopct = '%1.2f%%')
        
plt.axis('equal') # Try commenting this out.
plt.title('Pie Chart Example')
plt.show()
# create data
names='groupA', 'groupB', 'groupC', 'groupD',
size=[12,11,3,30]
 
# Create a circle for the center of the plot
my_circle=plt.Circle( (0,0), 0.7, color='white')

# Give color names
plt.pie(size, labels=names, colors=['red','green','blue','skyblue'])
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.show()
def random_plots():
  xs = []
  ys = []
  
  for i in range(20):
    x = i
    y = np.random.randint(10)
    
    xs.append(x)
    ys.append(y)
  
  return xs, ys

fig = plt.figure()
ax1 = plt.subplot2grid((5, 2), (0, 0), rowspan=1, colspan=2)
ax2 = plt.subplot2grid((5, 2), (1, 0), rowspan=3, colspan=2)
ax3 = plt.subplot2grid((5, 2), (4, 0), rowspan=1, colspan=1)
ax4 = plt.subplot2grid((5, 2), (4, 1), rowspan=1, colspan=1)

x, y = random_plots()
ax1.plot(x, y)

x, y = random_plots()
ax2.plot(x, y)

x, y = random_plots()
ax3.plot(x, y)

x, y = random_plots()
ax4.plot(x, y)

plt.tight_layout()
plt.show()
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

x, y, z = axes3d.get_test_data()

ax.plot_wireframe(x, y, z, rstride = 3, cstride = 3, color='#000000')

plt.title("Wireframe Plot Example")
plt.tight_layout()
plt.show()
# Generate some random data
num_points = 20
# x will be 5, 6, 7... but also twiddled randomly
x = 5 + np.arange(num_points) + np.random.randn(num_points)
# y will be 10, 11, 12... but twiddled even more randomly
y = 10 + np.arange(num_points) + 5 * np.random.randn(num_points)
sns.regplot(x, y, color='#FE4C84')
plt.show()
# Make a 10 x 10 heatmap of some random data
side_length = 10
# Start with a 10 x 10 matrix with values randomized around 5
data = 5 + np.random.randn(side_length, side_length)
# The next two lines make the values larger as we get closer to (9, 9)
data += np.arange(side_length)
data += np.reshape(np.arange(side_length), (side_length, 1))
# Generate the heatmap
sns.heatmap(data)
plt.show()
df = sns.load_dataset('iris')
 
# Custom the inside plot: options are: “scatter” | “reg” | “resid” | “kde” | “hex”
sns.jointplot(x=df["sepal_length"], y=df["sepal_width"], kind='hex')
sns.jointplot(x=df["sepal_length"], y=df["sepal_width"], kind='kde')
df = sns.load_dataset('iris')
 
# plot
sns.violinplot(x='species', y='sepal_length', data=df, order=[ "versicolor", "virginica", "setosa"])
df = sns.load_dataset('iris')
 
# Change line width
sns.boxplot( x=df["species"], y=df["sepal_length"], linewidth=5)
#sns.plt.show()

# create data
x = np.random.rand(15)
y = x+np.random.rand(15)
z = x+np.random.rand(15)
z=z*z
 
# Change color with c and alpha. I map the color to the X axis value.
plt.scatter(x, y, s=z*2000, c=x, cmap="Blues", alpha=0.4, edgecolors="grey", linewidth=2)
 
# Add titles (main and on axis)
plt.xlabel("the X axis")
plt.ylabel("the Y axis")
plt.title("A colored bubble plot")
 
plt.show()
N = 4000
x = np.random.random(size=N) * 100
y = np.random.random(size=N) * 100
radii = np.random.random(size=N) * 1.5
colors = ["#%02x%02x%02x" % (r, g, 150) for r, g in zip(np.floor(50+2*x).astype(int), np.floor(30+2*y).astype(int))]

p = figure()
p.circle(x, y, radius=radii, fill_color=colors, fill_alpha=0.6, line_color=None)
show(p)
# Set data
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

# Change color
squarify.plot(sizes=[13,22,35,5], label=["group A", "group B", "group C", "group D"], color=["red","green","blue", "grey"], alpha=.4 )
plt.axis('off')
plt.show()
# Line style: can be 'dashed' or 'dotted' for example
v=venn3(subsets = (10, 8, 22, 6,9,4,2), set_labels = ('Group A', 'Group B', 'Group C'))
plt.show()

# Color
v.get_patch_by_id('100').set_alpha(1.0)
v.get_patch_by_id('100').set_color('white')
plt.show()
# Get the data (csv file is hosted on the web)
url = 'https://python-graph-gallery.com/wp-content/uploads/volcano.csv'
data = pd.read_csv(url)
 
# Transform it to a long format
df=data.unstack().reset_index()
df.columns=["X","Y","Z"]
 
# And transform the old column name in something numeric
df['X']=pd.Categorical(df['X'])
df['X']=df['X'].cat.codes
 
# Make the plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=plt.cm.viridis, linewidth=0.2)
plt.show()
 
# to Add a color bar which maps values to colors.
surf=ax.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=plt.cm.viridis, linewidth=0.2)
fig.colorbar( surf, shrink=0.5, aspect=5)
plt.show()
 
# Rotate it
ax.view_init(30, 45)
plt.show()
 
# Other palette
ax.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=plt.cm.jet, linewidth=0.01)
plt.show()