# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
print('This is a short notebook about an intersting insight I found when visualizing FiveThirtyEight\'s Comic Characters Dataset. Just some backround on the dataset, it is from Marvel Wikia and DC Wikia. Characters were scraped on August 24. Appearance counts were scraped on September 2. The month and year of the first issue each character appeared in was pulled on October 6. The chart is below with number of characters introduced on the x-axis, the year on the y-axis, and the character\'s alignment in the legend.')

print('I wanted to see how characters introduced over the years changed. I also wanted to see how their Alignment changed, which is why i used a grouped bar chart.')

import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

#print('Setup complete')
file = '../input/fivethirtyeight-comic-characters-dataset/dc-wikia-data.csv' #loading the file in 

df = pd.read_csv(file)
#pd.set_option("display.max_rows", None, "display.max_columns", None) #grouping the columns I want 

group = df.groupby(['YEAR', 'ALIGN'], as_index=False)

dfGroup = group.count()



dfGroup #showing the dataframe 
sns.set_style("darkgrid", {"xtick.color": ".5"}) #creating and styling the chart

sns.set_style("ticks")



plt.figure(figsize=(20,6))

bplot = sns.barplot(x = 'YEAR', y = 'urlslug', hue = 'ALIGN', data = dfGroup)

bplot.set_xticklabels(bplot.get_xticklabels(), rotation=55, horizontalalignment='center', fontweight='light', fontsize='small')





bplot.set_title('Number of DC Characters Introduced from 1936 to 2013')

bplot.set(xlabel="Year", ylabel = "Character Count")



print('Though this is a simple looking graph, it does show some pretty intersting things. We see that in 1989 there is a peak, that from 1936 to 1980 there were very low numbers of new characters introduced, and we also can see that in 2012 there were very few characters introduced.')

print('I wondered what caused these peaks and valleys, so i turneed to reddit to see what I could find, and found this article that put some sense to a lot of the things we saw in this chart.')

print('Here is a link to the article: comicbooked.com/revisiting-90s-speculative-boom-nearly-ended-comic-book-industry')

      
print('Interstingly enough this article describes a bubble market that popped about in the early nineties for comic book stores. People saw potnetial to make profit in this industry and as a result, characters were bing introduced at very high rates. A side effect of these introductions was a sacrifice in the story, this is something that comic book collectors and experts saw as a dissapointment because now there niche market was being flooded with what they may describe as trash narratives. The chart above illistrates these different eras in the comic book market. The article from comicooked.com goes in depth on these trends, as well as give explanations for what exactly a bubble market is and it\'s specualtive boom.')