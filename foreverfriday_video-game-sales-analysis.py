# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
video = pd.read_csv('../input/Video_Games_Sales_as_at_22_Dec_2016.csv')
video.head(10)
videogroup = video[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']].groupby(video['Genre']).sum()



videogroup
videofull = video[video.Year_of_Release.notnull()]

videoyears = videofull.groupby(['Year_of_Release', 'Genre']).Global_Sales.sum()



videoyears.unstack().plot(kind='area', stacked=True, colormap= 'rainbow', figsize=(13, 6) )
#regions = video[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']].groupby(video['Publisher']).sum()

#regions.unstack().



regionsPub = [video.groupby('Publisher').sum().unstack().NA_Sales.sort_values(ascending=False).head(10), 

                      video.groupby('Publisher').sum().unstack().EU_Sales.sort_values(ascending=False).head(10),

                      video.groupby('Publisher').sum().unstack().Other_Sales.sort_values(ascending=False).head(10),

                      video.groupby('Publisher').sum().unstack().JP_Sales.sort_values(ascending=False).head(10)

                      ]



regions = ['NA_Sales', 'EU_Sales', 'Other_Sales', 'JP_Sales']



NAlist = video.groupby('Publisher').sum().unstack().NA_Sales.sort_values(ascending=False).head(10)

EUlist = video.groupby('Publisher').sum().unstack().EU_Sales.sort_values(ascending=False).head(10)

Otherlist = video.groupby('Publisher').sum().unstack().Other_Sales.sort_values(ascending=False).head(10)

JPlist = video.groupby('Publisher').sum().unstack().JP_Sales.sort_values(ascending=False).head(10)



#print (regions, '\n', regionsPub)

regionsPub
platforms = video['Platform'].unique()

platforms
video[video['Platform'].str.contains('GG')]
hands = video[(video.Platform == 'GB') | (video.Platform == 'DS') | (video.Platform == 'GBA') | (video.Platform == '3DS') |

             (video.Platform == 'PSP') | (video.Platform == 'GC') | (video.Platform == 'GC') | (video.Platform == 'PSV') | 

             (video.Platform == 'SCD') | (video.Platform == 'WS') | (video.Platform == 'GG')]



homes = video[(video.Platform == 'Wii') | (video.Platform == 'NES') | (video.Platform == 'X360') | (video.Platform == 'PS3') |

             (video.Platform == 'PS2') | (video.Platform == 'SNES') | (video.Platform == 'PS4') | (video.Platform == 'N64') |

             (video.Platform == 'PS') | (video.Platform == 'XB') | (video.Platform == '2600') | (video.Platform == 'XOne') |

             (video.Platform == 'WiiU') | (video.Platform == 'GEN') | (video.Platform == 'DC') | (video.Platform == 'SAT') |

             (video.Platform == 'NG') | (video.Platform == 'TG16') | (video.Platform == '3DO') | (video.Platform == 'PCFX')]



#j = 0

#while j in range(len(handnames)):

#    if video[(video.Platform == handnames[j])]:

  #      hands = video[(video.Platform == handnames[j])]

 #       j += 1



homesum = homes['Global_Sales'].sum()

handsum = hands['Global_Sales'].sum()



sns.barplot(x=['Handheld Consoles', 'Home Consoles'], y=[handsum, homesum])
homesyear = homes.groupby('Year_of_Release').Global_Sales.sum()

handsyear = hands.groupby('Year_of_Release').Global_Sales.sum()



homesyear.plot(kind='line', stacked=True, colormap= 'rainbow', figsize=(13, 6) )

handsyear.plot(kind='line', stacked=True, colormap= 'Dark2', figsize=(13, 6) )