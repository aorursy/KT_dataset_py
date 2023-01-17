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
video
video.info()
video.Platform.unique()
video['User_Score'] = pd.to_numeric(video['User_Score'])
sns.jointplot(x='Critic_Score',y='User_Score',data=video,kind='hex')

# video.plot(y= 'Critic_Score', x ='User_Score',kind='hexbin',gridsize=35, sharex=False, colormap='afmhot_r', title='Hexbin of Critic_Score and User_Score')
sns.jointplot('Critic_Score','Critic_Count',data=video,kind='hex')

#video.plot(y= 'Critic_Score', x ='Critic_Count',kind='hexbin',gridsize=40, sharex=False, colormap='cubehelix', title='Hexbin of Critic_Score and Critic_Count')
video.corr()
f, ax = plt.subplots(figsize=(7, 7))

plt.title('Pearson Correlation of Video Game Numerical Features')

# Draw the heatmap using seaborn

sns.heatmap(video.corr(), square=True, annot=True)
video.select_dtypes(include=[np.number]).head()
# Dataframe contain info only on the 7th Gen consoles

video7th = video[video.Platform.isin(['Wii', 'PS3', 'X360'])]

video7th.shape
yearlySales = video7th.groupby(['Year_of_Release','Platform']).Global_Sales.sum()

yearlySales.unstack().plot(kind='bar')

plt.title('Stacked Barplot of Global Yearly Sales of the 7th Gen Consoles')

plt.ylabel('Global Sales')
yearlySales = video7th.groupby(['Year_of_Release','Platform']).Global_Sales.sum()

yearlySales.unstack().head()
ratingSales = video7th.groupby(['Rating','Platform']).Global_Sales.sum()

ratingSales.unstack().plot(kind='bar')

plt.title('Stacked Barplot of Sales per Rating type of the 7th Gen Consoles')

plt.ylabel('Sales')
genreSales = video7th.groupby(['Genre','Platform']).Global_Sales.sum()

genreSales.unstack(level=0).plot(kind='bar', figsize=(16,9))

plt.title('Stacked Barplot of Sales per Game Genre')

plt.ylabel('Sales')
# Plotting our pie charts

plt.figure(figsize=(8,4))

plt.subplot(121)

sales_by_platform = video7th.groupby('Platform').Global_Sales.sum()

plt.pie(

    sales_by_platform,

    # with the labels being platform

    labels=sales_by_platform.index,

    # with the start angle at 90%

    startangle=90,

    # with the percent listed as a fraction

    autopct='%.1f%%'

)

plt.axis('equal')

plt.title('Global Sales')



plt.subplot(122)

user_count_by_platform = video7th.groupby('Platform').User_Count.sum()

plt.pie(

    user_count_by_platform,

    labels=user_count_by_platform.index,

    startangle=90,

    autopct='%.1f%%'

)

plt.axis('equal')

plt.title('User Base')

plt.tight_layout()

plt.show()
video8th = video[video.Platform.isin(['WiiU', 'PS4', 'XOne'])]

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