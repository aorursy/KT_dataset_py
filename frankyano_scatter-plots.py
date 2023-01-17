import pandas as pd

from matplotlib import pyplot as plt



plt.style.use('seaborn')



x = [5, 7, 8, 5, 6, 7, 9, 2, 3, 4, 4, 4, 2, 6, 3, 6, 8, 6, 4, 1]

y = [7, 4, 3, 9, 1, 3, 2, 5, 2, 4, 8, 7, 1, 6, 4, 9, 7, 7, 5, 1]



plt.scatter(x,y, s = 100, color = 'green',edgecolor = 'black', linewidth = 1, alpha = 0.75) 

#for more on the marker styles, we can always refer to the documentation

plt.tight_layout()

plt.show()
#say our data has more than one type of value. Different colored plots would be nice.



import pandas as pd

from matplotlib import pyplot as plt



plt.style.use('seaborn')



x = [5, 7, 8, 5, 6, 7, 9, 2, 3, 4, 4, 4, 2, 6, 3, 6, 8, 6, 4, 1]

y = [7, 4, 3, 9, 1, 3, 2, 5, 2, 4, 8, 7, 1, 6, 4, 9, 7, 7, 5, 1]



colors = [7, 5, 9, 7, 5, 7, 2, 5, 3, 7, 1, 2, 8, 1, 9, 2, 5, 6, 7, 5] #additional data

sizes = [209, 486, 381, 255, 191, 315, 185, 228, 174,

         538, 239, 394, 399, 153, 273, 293, 436, 501, 397, 539]



plt.scatter(x,y, s = sizes, c = colors, cmap = 'Greens', edgecolor = 'black',

            linewidth = 1, alpha = 0.75)

cbar = plt.colorbar() #this is a method

cbar.set_label('Satisfaction')





plt.tight_layout()



plt.show()
#working with real world data



import pandas as pd

from matplotlib import pyplot as plt



plt.style.use('seaborn')



data = pd.read_csv('../input/vidlikes.csv')

view_count = data['view_count']

likes = data['likes']

ratio = data['ratio']



plt.scatter(view_count, likes, edgecolor = 'black', linewidth = 1, alpha = 0.75)



plt.title('Trending YouTube Videos')

plt.xlabel('View Count')

plt.ylabel('Total Likes')



plt.tight_layout()



plt.show()

#this plot is not so accurate because there is an out liar, has far much more views and likes than others

import pandas as pd

from matplotlib import pyplot as plt



plt.style.use('seaborn')



data = pd.read_csv('../input/vidlikes.csv')

view_count = data['view_count']

likes = data['likes']

ratio = data['ratio']



plt.scatter(view_count, likes, edgecolor = 'black', linewidth = 1, alpha = 0.75)



#applying log scale on both axes to correct the plot choke

plt.xscale('log')

plt.yscale('log')



plt.title('Trending YouTube Videos')

plt.xlabel('View Count')

plt.ylabel('Total Likes')



plt.tight_layout()



plt.show()



#Applying color map

import pandas as pd

from matplotlib import pyplot as plt



plt.style.use('seaborn')



data = pd.read_csv('../input/vidlikes.csv')

view_count = data['view_count']

likes = data['likes']

ratio = data['ratio']



plt.scatter(view_count, likes,c = ratio,

            #summer is just one of the several cmap styles. Reffer to documentation for more.

            cmap = 'summer',

            edgecolor = 'black', linewidth = 1, alpha = 0.75)

cbar = plt.colorbar()

cbar.set_label = ('Like/Dislike Ratio')

#applying log scale on both axes to correct the plot choke

plt.xscale('log')

plt.yscale('log')



plt.title('Trending YouTube Videos')

plt.xlabel('View Count')

plt.ylabel('Total Likes')



plt.tight_layout()



plt.show()


