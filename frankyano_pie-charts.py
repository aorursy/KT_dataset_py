#starting with a really simple pie chart

import matplotlib.pyplot as plt

plt.style.use('seaborn')



slices = [10, 30, 60, 150]

labels = ['Ten', 'Thirty', "Sixty", 'Big'] #adding labels to our pie chart

#colors = ['blue', 'red', 'yellow', 'green'] #to customize the colors

plt.pie(slices, labels = labels, #colors = colors,

        #wedgeprops is to put a boundary where the two colors meet.

        wedgeprops = {'edgecolor': 'black'})



#more on the matplotlib wedge documentation for more customisations.

plt.title('My Awesome Piechart')

plt.tight_layout()

plt.show()

#Real world data

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')



# Language Popularity

#pie charts is not good with a lot of data so we shall modify our list to top 5 languages.

slices = [59219, 55466, 47544, 36443, 35917]

labels = ['JavaScript', 'HTML/CSS', 'SQL', 'Python', 'Java']



plt.pie(slices, labels = labels, wedgeprops = {'edgecolor': 'black'})



#wedgeprops is to put a boundary where the two colors meet.

#more on the matplotlib wedge documentation for more customisations.



plt.title('Top 5 programming languages')

plt.tight_layout()

plt.show()
#More customizations

#Real world data

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')



# Language Popularity

#python is not good with a lot of data so we shall modify our list to top 5 languages.

slices = [59219, 55466, 47544, 36443, 35917]

labels = ['JavaScript', 'HTML/CSS', 'SQL', 'Python', 'Java']

explode = [0,0,0, 0.1, 0] #this puts an emphasis on python



plt.pie(slices, labels = labels, explode = explode, 

        #shadow = True for asthetics to add a shadow to our plot

        shadow = True,

        #startangle = 90, rotates the original chart by 90 degress

        startangle = 90,

        #autopct = '%1.1f%%' for adding the % value for each element. Refer to documentation to confirm

        autopct = '%1.1f%%',

        wedgeprops = {'edgecolor': 'black'})



#wedgeprops is to put a boundary where the two colors meet.

#more on the matplotlib wedge documentation for more customisations.



plt.title('Top 5 programming languages')

plt.tight_layout()

plt.show()