# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import csv

import pandas as pd

data = pd.read_csv("../input/food.csv")



import numpy as np

import matplotlib.pyplot as plt
data.head(1)
#One way of creating pie charts is looking at the distribution of the variable that you need

#to create the pie chart of and find the frequencies. Then create the actual pie based on those

#numbers. 

gend = data['Gender'].value_counts()

print(gend)
#and here we create the actual pie chart of the gender distribution

labels = ['Female', 'Male']

sizes = [76, 49]

colors = ['yellow', 'gray']

explode = [0,0]

plt.pie(sizes, labels=labels,colors = colors, autopct = '%1.1f%%', shadow = True,startangle=60)

#plt.axis('equal')

plt.show()
#same as above for breakfast option. Participants are shown two types of pictures: one of oatmeal and

#one of donuts. Let's see the frequency of their options

breakfast = data['breakfast'].value_counts()

print(breakfast)
#we know that 1 is the cereal option and 2 is the donut option

# the cereal option is selected 88.8% of the time and 11.2% selected the donut

labels = ['Oatmeal', 'Donut']

sizes = [111, 14]

colors = ['orange', 'gray']

patches, texts = plt.pie(sizes, labels=labels, colors = colors, shadow = True,startangle=90)

plt.legend(patches, labels, loc="best")

plt.axis('equal')

plt.tight_layout()

plt.show()
#the importance of calories per day

calories = data['calories_day'].value_counts()

print(calories)
import matplotlib.pyplot as plt; plt.rcdefaults()

import numpy as np

import matplotlib.pyplot as plt

 

objects = ('Very important', 'Moderately important', 'Not important')

y_pos = np.arange(len(objects))

performance = [23,63,20]

 

plt.bar(y_pos, performance, align='center', alpha=0.5)

plt.xticks(y_pos, objects)

plt.ylabel('Frequency')

plt.title('Importance of consuming the right amount of calories per day')

 

plt.show()
#frapuccino or espresso?

coffee = data['coffee'].value_counts()

print(coffee)
#we know that 1 is the frapuccino option and 2 is the espresso option

#it looks that participants agree on what is called 'coffee'

labels = ['Frapuccino', 'Espresso']

sizes = [31, 94]

colors = ['royalblue', 'aliceblue']

patches, texts = plt.pie(sizes, labels=labels, colors = colors, shadow = True,startangle=90)

plt.legend(patches, labels, loc="best")

plt.axis('equal')

plt.tight_layout()

plt.show()
#last one for this kernel. how often do these 18-22 year olds cook?

cook = data['cook'].value_counts()

print(cook)
import matplotlib.pyplot as plt; plt.rcdefaults()

import numpy as np

import matplotlib.pyplot as plt

 

objects = ('Daily', '2-3 times/w', 'Not often', 'On holidays', 'Never')

y_pos = np.arange(len(objects))

performance = [13, 34, 49, 18, 8]

 

plt.bar(y_pos, performance, align='center', alpha=0.5)

plt.xticks(y_pos, objects)

plt.ylabel('Frequency')

plt.title('How often do you cook?')

 

plt.show()
