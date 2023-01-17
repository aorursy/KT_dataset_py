# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')
df=pd.read_csv('../input/pollution_wide.csv')

df.head()
df.city.unique()
df.month.value_counts()
df.describe(include='all')
pd.plotting.scatter_matrix(df[['CO','NO2','O3','SO2']], alpha=0.1)

plt.show()
cinci= df[df.city=='Cincinnati']

cinci_col= ['orangered' if day==38 else 'steelblue'

            for day in cinci.day]

c= sns.regplot(x='NO2', y='SO2', data= cinci, fit_reg= False,

              scatter_kws={'facecolors': cinci_col, 'alpha':0.7})
houston_pollution = df[df.city  ==  'Houston']



# Make array orangred for day 330 of year 2014, otherwise lightgray

houston_colors = ['orangered' if (day  ==  330) & (year  ==  2014) else 'lightgray' 

                  for day,year in zip(houston_pollution.day, houston_pollution.year)]



sns.regplot(x = 'NO2',

            y = 'SO2',

            data = houston_pollution,

            fit_reg = True, 

            # Send scatterplot argument to color points 

            scatter_kws = {'facecolors': houston_colors, 'alpha': 0.7})

plt.show()
houston_pollution = df[df.city  ==  'Houston'].copy()



# Find the highest observed O3 value

max_O3 = houston_pollution.O3.max()



# Make a column that denotes which day had highest O3

houston_pollution['point type'] = ['Highest O3 Day' if O3  ==  max_O3 else 'Others' for O3 in houston_pollution.O3]



# Encode the hue of the points with the O3 generated column

sns.scatterplot(x = 'NO2',

                y = 'SO2',

                hue = 'point type',

                data = houston_pollution)

plt.show()
pollution_nov = df[df.month == 10]

sns.distplot(pollution_nov[pollution_nov.city =='Denver'].O3, hist=True,

color ='red')

sns.distplot(pollution_nov[pollution_nov.city !='Denver'].O3, hist=True)
sns.distplot(pollution_nov[pollution_nov.city =='Denver'].O3, hist=True,

color ='red', rug= True)

sns.distplot(pollution_nov[pollution_nov.city !='Denver'].O3, hist=True)
sns.swarmplot(y='city', x='O3', data= pollution_nov, size= 5)
# Filter dataset to the year 2012

sns.kdeplot(df[df.year == 2012].O3, 

            # Shade under kde and add a helpful label

            shade = True,

            label = '2012')



# Filter dataset to everything except the year 2012

sns.kdeplot(df[df.year != 2012].O3, 

            # Again, shade under kde and add a helpful label

            shade = True,

            label = 'other years')

plt.show()
sns.distplot(df[df.city == 'Vandenberg Air Force Base'].O3, 

             label = 'Vandenberg', 

             # Turn of the histogram and color blue to stand out

             hist = False,

             color = 'steelblue', 

             # Turn on rugplot

             rug = True)



sns.distplot(df[df.city != 'Vandenberg Air Force Base'].O3, 

             label = 'Other cities',

             # Turn off histogram and color gray

             hist = False,  

             color = 'gray')

plt.show()
# Filter data to just March

pollution_mar = df[df.month == 3]



# Plot beeswarm with x as O3

sns.swarmplot(y = "city",

              x = 'O3', 

              data = pollution_mar, 

              # Decrease the size of the points to avoid crowding 

              size = 3)



# Give a descriptive title

plt.title('March Ozone levels by city')

plt.show()
# Query and filter to New Years in Long Beach

jan_pollution = df.query("(month  ==  1) & (year  ==  2012)")

lb_newyears = jan_pollution.query("(day  ==  1) & (city  ==  'Long Beach')")



sns.scatterplot(x = 'CO', y = 'NO2',

                data = jan_pollution)



# Point arrow to lb_newyears & place text in lower left 

plt.annotate('Long Beach New Years',

             xy = (lb_newyears.CO, lb_newyears.NO2),

             xytext = (2, 15), 

             # Shrink the arrow to avoid occlusion

             arrowprops = {'facecolor':'gray', 'width': 3, 'shrink': 0.03},

             backgroundcolor = 'white')

plt.show()
is_lb = ['orangered' if city  ==  'Long Beach' else 'lightgray' for city in df['city']]



# Map facecolors to the list is_lb and set alpha to 0.3

sns.regplot(x = 'CO',

            y = 'O3',

            data = df,

            fit_reg = False, 

            scatter_kws = {'facecolors':is_lb, 'alpha':0.3})

plt.show() 
y=sns.scatterplot('CO', 'NO2',

                alpha = 0.2,

                hue = 'city',

                data = df)

plt.show()
g = sns.FacetGrid(data = df,

                  col = 'city',

                  col_wrap = 4)



# Map sns.scatterplot to create separate city scatter plots

g.map(sns.scatterplot, 'CO', 'NO2', alpha = 0.2)

plt.show()
import numpy as np



sns.barplot(y = 'city', x = 'CO', 

              estimator = np.mean,

            ci = False,

              data = df,

              # Add a border to the bars

            edgecolor = 'black')

plt.show()
import numpy as np



sns.barplot(y = 'city', x = 'CO', 

              estimator = np.mean,

            ci = False,

              data = df,

              # Replace border with bar colors

            color = 'cadetblue')

plt.show()
blue_scale = sns.light_palette("steelblue")

sns.palplot(blue_scale)
red_scale = sns.dark_palette("orangered")

sns.palplot(red_scale)
indy_oct = df.query("year == 2015 & city =='Indianapolis'")

blue_scale = sns.light_palette("steelblue", as_cmap = True)

sns.heatmap(indy_oct[['O3']], cmap = blue_scale)
indy_oct = df.query("year == 2015 & city =='Indianapolis'")

jet_scale = palette = sns.dark_palette('red',as_cmap = True)

sns.heatmap(indy_oct[['O3']], cmap = jet_scale)
pal_light = sns.diverging_palette(250, 0)

pal_dark = sns.diverging_palette(250, 0, center ='dark')

sns.heatmap(indy_oct[['O3']], cmap = pal_light)
plt.style.use('seaborn-white')

sns.scatterplot(x =

'CO'

, y =

'NO2'

, hue =

'O3'

, data = df)
plt.style.use('dark_background')

sns.scatterplot(x =

'CO'

, y =

'NO2'

, hue =

'O3'

, data = df)
plt.style.use('ggplot')

# Filter the data

cinci_2014 = df.query("city  ==  'Cincinnati' & year  ==  2014")



# Define a custom continuous color palette

color_palette = sns.light_palette('orangered',

                         as_cmap = True)



# Plot mapping the color of the points with custom palette

sns.scatterplot(x = 'CO',

                y = 'NO2',

                hue = 'O3', 

                data = cinci_2014,

                palette = color_palette)

plt.show()
# nov_2015= df.query("year  ==  2015 & month==11")

# nov_2015_CO= nov_2015[['day','city']]

# nov_2015_CO=nov_2015_CO.dropna()

# # Define a custom palette

# color_palette = sns.diverging_palette(250, 0, as_cmap = True)



# # Pass palette to plot and set axis ranges

# sns.heatmap(nov_2015_CO,

#             cmap = color_palette,

#             center = 0,

#             vmin = -4,

#             vmax = 4)

# plt.yticks(rotation = 0)

# plt.show()
sns.palplot(sns.color_palette('Set2'

, 11))
colorbrewer_palettes = ['Set1'

,

'Set2'

,

'Set3'

,

'Accent'

,

'Paired'

,

'Pastel1'

,

'Pastel2'

,

'Dark2']

for pal in colorbrewer_palettes:

    sns.palplot(pal=sns.color_palette(pal))

    plt.title(pal, loc ='left')
colorbrewer_palettes = ['Reds'

,

'Blues'

,

'YlOrBr'

,

'PuBuGn'

,

'GnBu'

,

'Greys']

for i, pal in enumerate(colorbrewer_palettes):

    sns.palplot(pal=sns.color_palette(pal, n_colors=i+4))
df['interesting cities'] = [x if x in ['Long Beach'

,

'Cincinnati']

else 'other' for x in df['city'] ]

sns.scatterplot(x=

"NO2"

, y=

"SO2"

, hue =

'interesting cities'

, palette=

'Set2'

,

data=df.query('year == 2014 & month == 12'))
# Make a tertials column using qcut()

df['NO2 Tertial'] = pd.qcut(df['NO2'], 3, labels = False)

# Plot colored by the computer tertials

sns.scatterplot(x=

"CO"

, y=

"SO2"

, hue=

'NO2 Tertial'

, palette=

"OrRd"

,

data=df.query("city =='Long Beach' & year == 2014"))
# Filter our data to Jan 2013

pollution_jan13 = df.query('year  ==  2013 & month  ==  1')



# Color lines by the city and use custom ColorBrewer palette

sns.lineplot(x = "day", 

             y = "CO", 

             hue = "city",

             palette = "Set2", 

             linewidth = 3,

             data = pollution_jan13)

plt.show()
# Divide CO into quartiles

df['CO quartile'] = pd.qcut(df['CO'], q = 4, labels = False)



# Filter to just Des Moines

des_moines = df.query("city  ==  'Des Moines'")



# Color points with by quartile and use ColorBrewer palette

sns.scatterplot(x = 'SO2',

                y = 'NO2',

                hue = 'CO quartile', 

                  data = des_moines,

                palette = 'GnBu')

plt.show()
def bootstrap(data, n_boots):

    return [np.mean(np.random.choice(data,len(data)))

    for _ in range(n_boots) ]
cinci_may_NO2 = df.query("city  ==  'Cincinnati' & month  ==  5").NO2



# Generate bootstrap samples

boot_means = bootstrap(cinci_may_NO2, 1000)



# Get lower and upper 95% interval bounds

lower, upper = np.percentile(boot_means, [2.5, 97.5])



# Plot shaded area for interval

plt.axvspan(lower, upper, color = 'gray', alpha = 0.2)



# Draw histogram of bootstrap samples

sns.distplot(boot_means, bins = 100, kde = True)



plt.show()
sns.lmplot('NO2', 'SO2', data = df,

           # Tell seaborn to a regression line for each sample

           hue = 'month', 

           # Make lines blue and transparent

           line_kws = {'color': 'steelblue', 'alpha': 0.2},

           # Disable built-in confidence intervals

           ci = None, legend = False, scatter = False)



# Draw scatter of all points

plt.scatter('NO2', 'SO2', data = df)



plt.show()

# Initialize a holder DataFrame for bootstrap results

city_boots = pd.DataFrame()



for city in ['Cincinnati', 'Des Moines', 'Indianapolis', 'Houston']:

    # Filter to city

    city_NO2 = df[df.city  ==  city].NO2

    # Bootstrap city data & put in DataFrame

    cur_boot = pd.DataFrame({'NO2_avg': bootstrap(city_NO2, 100), 'city': city})

    # Append to other city's bootstraps

    city_boots = pd.concat([city_boots,cur_boot])



# Beeswarm plot of averages with citys on y axis

sns.swarmplot(y = "city", x = "NO2_avg", data = city_boots, color = 'coral')



plt.show()