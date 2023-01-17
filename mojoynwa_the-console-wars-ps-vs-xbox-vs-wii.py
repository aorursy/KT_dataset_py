# Import the relevant libaries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from matplotlib import pyplot as plt

%matplotlib inline
video = pd.read_csv('../input/Video_Games_Sales_as_at_22_Dec_2016.csv')

video.head()
print(video.shape)
video.isnull().any()
video = video.dropna(axis=0)
video.info()
# Conversion of "User_Score" Column from Str to Numeric

video['User_Score'] = video.User_Score.convert_objects(convert_numeric=True)

# SIDE-NOTE: use pd.to_numeric. I was copying what I had written locally on my machine and it seems I have an old pandas version and therefore no pd.to_numeric
from wordcloud import WordCloud, STOPWORDS

# Generating the wordcloud with the values under the Platform feature

platform = WordCloud(

                          stopwords=STOPWORDS,

                          background_color='black',

                          width=1200,

                          height=800

                         ).generate(" ".join(video['Platform'].values))

plt.imshow(platform, alpha=1)

plt.axis('off')

plt.show()
#sns.jointplot(x='Critic_Score',y='User_Score',data=video,kind='hex', color='goldenrod', size=6)

video.plot(y= 'Critic_Score', x ='User_Score',kind='hexbin',gridsize=50, sharex=False, colormap='afmhot', title='Hexbin of Critic_Score and User_Score')
#sns.jointplot('Critic_Score','Critic_Count',data=video,kind='hex', color='goldenrod', size=6)

video.plot(y= 'Critic_Score', x ='Critic_Count',kind='hexbin',gridsize=45, sharex=False, colormap='afmhot', title='Hexbin of Critic_Score and Critic_Count')
str_list = [] # empty list to contain columns with strings (words)

for colname, colvalue in video.iteritems():

    if type(colvalue[2]) == str:

         str_list.append(colname)

# Get to the numeric columns by inversion            

num_list = video.columns.difference(str_list) 

# Create Dataframe containing only numerical features

video_num = video[num_list]

f, ax = plt.subplots(figsize=(7, 7))

plt.title('Pearson Correlation of Video Game Numerical Features')

# Draw the heatmap using seaborn

sns.heatmap(video_num.astype(float).corr(),linewidths=0.25,vmax=1.0, square=True, cmap="PuBuGn", linecolor='k', annot=True)
# Dataframe contain info only on the 7th Gen consoles

video7th = video[(video['Platform'] == 'Wii') | (video['Platform'] == 'PS3') | (video['Platform'] == 'X360')]

video7th.shape
yearlySales = video7th.groupby(['Year_of_Release','Platform']).Global_Sales.sum()

yearlySales.unstack().plot(kind='bar',stacked=True, colormap= 'Blues',  grid=False)

plt.title('Stacked Barplot of Global Yearly Sales of the 7th Gen Consoles')

plt.ylabel('Global Sales')
ratingSales = video7th.groupby(['Rating','Platform']).Global_Sales.sum()

ratingSales.unstack().plot(kind='bar',stacked=True,  colormap= 'Greens', grid=False)

plt.title('Stacked Barplot of Sales per Rating type of the 7th Gen Consoles')

plt.ylabel('Sales')
genreSales = video7th.groupby(['Genre','Platform']).Global_Sales.sum()

genreSales.unstack().plot(kind='bar',stacked=True,  colormap= 'Greys', grid=False)

plt.title('Stacked Barplot of Sales per Game Genre')

plt.ylabel('Sales')
# Plotting our pie charts

# Create a list of colors 

colors = ['#008DB8','#00AAAA','#00C69C']

plt.subplot(121)

plt.pie(

   video7th.groupby('Platform').Global_Sales.sum(),

    # with the labels being platform

    labels=video7th.groupby('Platform').Global_Sales.sum().index,

    # with no shadows

    shadow=False,

    # stating our colors

    colors=colors,

    explode=(0.05, 0.05, 0.05),

    # with the start angle at 90%

    startangle=90,

    # with the percent listed as a fraction

    autopct='%1.1f%%'

    )

plt.axis('equal')

plt.title('Pie Chart of Global Sales')

plt.subplot(122)

plt.pie(

   video7th.groupby('Platform').User_Count.sum(),

    labels=video7th.groupby('Platform').User_Count.sum().index,

    shadow=False,

    colors=colors,

    explode=(0.05, 0.05, 0.05),

    startangle=90,

    autopct='%1.1f%%'

    )

plt.axis('equal')

plt.title('Pie Chart of User Base')

plt.tight_layout()

plt.show()
video8th = video[(video['Platform'] == 'WiiU') | (video['Platform'] == 'PS4') | (video['Platform'] == 'XOne')]

video8th.shape
yearlySales = video8th.groupby(['Year_of_Release','Platform']).Global_Sales.sum()

yearlySales.unstack().plot(kind='bar',stacked=True, colormap= 'Blues',  grid=False)

plt.title('Stacked Barplot of Global Yearly Sales of the 8th Gen Consoles')

plt.ylabel('Global Sales')
ratingSales = video8th.groupby(['Rating','Platform']).Global_Sales.sum()

ratingSales.unstack().plot(kind='bar',stacked=True,  colormap= 'Greens', grid=False)

plt.title('Stacked Barplot of Sales per Rating type of the 8th Gen Consoles')

plt.ylabel('Sales')
genreSales = video8th.groupby(['Genre','Platform']).Global_Sales.sum()

genreSales.unstack().plot(kind='bar',stacked=True,  colormap= 'Greys', grid=False)

plt.title('Stacked Barplot of Sales per Game Genre')

plt.ylabel('Sales')
# Plotting our pie charts

# Create a list of colors 

colors = ['#008DB8','#00AAAA','#00C69C']

plt.subplot(121)

plt.pie(

   video8th.groupby('Platform').Global_Sales.sum(),

    # with the labels being platform

    labels=video8th.groupby('Platform').Global_Sales.sum().index,

    # with no shadows

    shadow=False,

    # stating our colors

    colors=colors,

    explode=(0.05, 0.05, 0.05),

    # with the start angle at 90%

    startangle=90,

    # with the percent listed as a fraction

    autopct='%1.1f%%'

    )

plt.axis('equal')

plt.title('Pie Chart of 8th Gen Global Sales')

plt.subplot(122)

plt.pie(

   video8th.groupby('Platform').User_Count.sum(),

    labels=video8th.groupby('Platform').User_Count.sum().index,

    shadow=False,

    colors=colors,

    explode=(0.05, 0.05, 0.05),

    startangle=90,

    autopct='%1.1f%%'

    )

plt.axis('equal')

plt.title('Pie Chart of 8th Gen User Base')

plt.tight_layout()

plt.show()