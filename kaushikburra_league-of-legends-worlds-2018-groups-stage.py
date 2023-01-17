import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from bokeh.plotting import figure, output_file, show
from bokeh.io import output_notebook
output_notebook()
%matplotlib inline
#importing data#
df= pd.read_csv('../input/LOL Worlds 2018 Groups stage - Player Ratings.csv')
df.head()
df.columns
#printing the features/variables#
#converting string to float#

df['KDA Ratio'] = df['KDA Ratio'].astype(float)
df['Kills Total'] = df['Kills Total'].astype(float)
df['Deaths'] = df['Deaths'].astype(float)
df['Assists'] = df['Assists'].astype(float)
df['CS Total'] = df['CS Total'].astype(float)
df['CS Per Minute'] = df['CS Per Minute'].astype(float)
df['Kill Participation'] = df['Kill Participation'].astype(float)
df['Games Played'] = df['Games Played'].astype(float)
print('KDA Ratio Max:')
print(df['KDA Ratio'].max())

print('Games Played:')
print(df['Games Played'].max())
# plotting scatterplot between sqft_living and price using "bokeh"#

from bokeh.plotting import figure
from bokeh.io import output_notebook, push_notebook, show
from bokeh.models import Range1d

x = df['Games Played']
y = df['KDA Ratio']

TOOLS="hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select,"

p = figure(tools=TOOLS)
p = figure(plot_width=800, plot_height=600, x_range=(0,15))
p.y_range = Range1d(0,10)
p.scatter(x, y,
          fill_color="red", fill_alpha=0.6,
          line_color=None)
print('KDA Ratio Max:')
print(df['KDA Ratio'].max())

print('Games Played:')
print(df['Games Played'].max())

output_file("scatter.html", title="df_scatter.py")
show(p)
output_notebook()


#Grade vs.Price(Sqft_living)
Position = df['Position'].value_counts()
print('Position count:')
print(Position)

sns.countplot(x = 'Position',data = df,palette = "bright")
plt.title('Position count')
plt.show()

sns.boxplot(x = 'Position',y = 'KDA Ratio',data = df,orient='v',width= 0.5,palette = "bright")
plt.title('Position Vs. KDA Ratio')
plt.show()

sns.FacetGrid(data = df,hue = 'Position', aspect=4,palette = "bright").map(plt.scatter,'Games Played','KDA Ratio')
                                                                           
                                                                        
plt.title('KDA Ratio Vs. Games Played under differnt Position')
plt.show()
#Correlation matrix
corr = df[['KDA Ratio','Kills Total','Deaths','Assists','Games Played','CS Total']]
 

plt.figure(figsize=(10,10))
plt.title('Correlation of variables')
sns.heatmap(corr.astype(float).corr(),vmax=1.0)
plt.show()

