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
#import another library and we don't need all of matplotlib, and give it a shorthand name

import matplotlib.pyplot as plt

#state the plot style to be used throughout

plt.style.use('seaborn')



#create a variable to read the dataframe

df = pd.read_csv('../input/acnhvillagers.csv')
#grab only the columns I want to use, assign those to a variable

aboutvillagers = df[['Name', 'Species', 'Gender', 'Personality', 'Birthday', 'Birthmonth', 'Style 1', 'Style 2']]

#run this new variable

aboutvillagers
#count and graph the number of times different values appear for gender

#genders aren't distributed equally... pretty interesting

aboutvillagers['Gender'].value_counts().plot(kind='bar')
#count and graph the number of times different values appear for species

#kind of interesting to see that there's a few more cat types of villagers

aboutvillagers['Species'].value_counts().plot(kind='bar')
#it's interesting that these are all quite similar except for Big Sister or Smug personality types

aboutvillagers['Personality'].value_counts().plot(kind='bar')
#here I'm just trying to see what style types there are and their frequency of occurrence

aboutvillagers['Style 1'].value_counts().plot(kind='bar')
#set the x axis to represent birthDAY data

x = aboutvillagers[['Birthday']]

#set the y axis to represent birthMONTH data

y = aboutvillagers[['Birthmonth']]

#overlapping this data in a bar graph shows which birthdays are shared by more than one villager

plt.scatter(x, y, alpha=0.2, color='indigo')
#define what terms to group under either of these variables, to be used below to evaluate likability

positive_traits = ['Cool', 'Peppy', 'Big Sister', 'Lazy', 'Normal']

negative_traits = ['Cranky', 'Snooty', 'Smug', 'Jock']



def personality_evaluator(personality_type):

    #if the personality type contains elements from the positive_traits variable

    if personality_type in positive_traits:

        #i'm going to categorize these as likable

        return "Likable"

    #if it contains elements from the negative_traits variable, they must be unlikable

    elif personality_type in negative_traits:

        return "Unlikable"

    #otherwise, return an error message

    else:

        return "Something has gone terribly wrong"

    

#plot the data that appears after applying the function from above    

aboutvillagers['Personality'].apply(personality_evaluator).value_counts().plot(kind='pie')
#used the code on this next line to identify all the different types of species listed so that I could add them to their own categories

#aboutvillagers['Species'].value_counts()



#i want to find a way to assign values to species types and then rank the villagers based on that

#i created variables to hold lists of animals that I manually assigned, based on observation and subjective analysis of how users interact with villagers

#it would be really cool to do something like this based on a Twitter analysis or some other sentimental data set

docile_animal = ['Cat', 'Rabbit', 'Dog', 'Mouse', 'Horse', 'Pig', 'Bird', 'Chicken', 'Hamster', 'Cow']

cute_animal = ['Squirrel', 'Duck', 'Cub', 'Deer', 'Koala', 'Bear', 'Sheep', 'Penguin', 'Hippo', 'Rhino', 'Octopus']

weird_animal = ['Frog', 'Wolf', 'Elephant', 'Ostrich', 'Eagle', 'Gorilla', 'Kangaroo', 'Goat', 'Alligator', 'Anteater', 'Tiger', 'Lion', 'Bull', 'Monkey']

                

def animal_type(species):

    #if the species contains elements from the docile_animal variable

    if species in docile_animal:

        #i'm going to categorize these as likable

        return "These are often kept as pets"

    #if it contains elements from the cute_animal variable (these are often the more popular characters among users)

    elif species in cute_animal:

        return "These ones are adorable and we love them"

    #if it doesn't fall into household pet or cute animal territory, it's weird

    elif species in weird_animal:

        return "These ones are WEIRD"

    #otherwise, return an error message

    else:

        return "Something has gone terribly wrong"

    

#create a variable to refer to all the villager species names

words = ['Cat', 'Rabbit', 'Dog', 'Mouse', 'Horse', 'Pig', 'Bird', 'Chicken', 'Hamster', 'Cow', 'Squirrel', 'Duck', 'Cub', 'Deer', 'Koala', 'Bear', 'Sheep', 'Penguin', 'Hippo', 'Rhino', 'Octopus', 'Frog', 'Wolf', 'Elephant', 'Ostrich', 'Eagle', 'Gorilla', 'Kangaroo', 'Goat', 'Alligator', 'Anteater', 'Tiger', 'Lion', 'Bull', 'Monkey']



#create a place to house the first letter of each villagers name

#i will refer back to this list within the "borrowed code" section

first_char_list = []

#for every word

for everyword in words:

    #add the first character of that 

    first_char_list.append(everyword[0])



#make data: I have a few groups and a bunch of subgroups but I don't want to manually list them, so I'm converting them to lists automatically

group_names = ['These are often kept as pets', 'These ones are adorable and we love them', 'These ones are WEIRD']

#this grabs the information I categorized with the animal_type function, counts it, and converts it to a list

group_size = aboutvillagers['Species'].apply(animal_type).value_counts(sort=True).tolist()

subgroup_names = ['Cat', 'Rabbit', 'Dog', 'Mouse', 'Horse', 'Pig', 'Bird', 'Chicken', 'Hamster', 'Cow', 'Squirrel', 'Duck', 'Cub', 'Deer', 'Koala', 'Bear', 'Sheep', 'Penguin', 'Hippo', 'Rhino', 'Octopus', 'Frog', 'Wolf', 'Elephant', 'Ostrich', 'Eagle', 'Gorilla', 'Kangaroo', 'Goat', 'Alligator', 'Anteater', 'Tiger', 'Lion', 'Bull', 'Monkey']

#these are the coresponding numbers for each subgroup size

subgroup_size = [23, 20, 16, 15, 15, 15, 13, 9, 8, 4, 18, 17, 16, 10, 9 , 15, 13, 13, 7, 6, 3, 18, 11, 11, 10, 9, 9, 8, 8, 7, 7, 7, 7, 6, 8]









#START - BORROWED CODE

#almost all of the code below (modified in some places) this is borrowed from https://python-graph-gallery.com/163-donut-plot-with-subgroups

#my own work is above this section



#create colors

cmap = plt.get_cmap("tab20c")

outer_colors = cmap(np.arange(3)*4)

inner_colors = cmap(np.array([1, 2, 5, 6, 9, 10]))

 

#first ring (outside)

fig, ax = plt.subplots()

ax.axis('equal')

mypie, _ = ax.pie(group_size, radius=1.3, labels=group_names, colors=outer_colors)

plt.setp(mypie, width=0.3, edgecolor='white')

 

#second ring (inside)

#the labels for this will be shortened versions, since the regular names showed up messy

mypie2, _ = ax.pie(subgroup_size, radius=1.3-0.3, labels=first_char_list, labeldistance=0.7, colors=inner_colors)

plt.setp(mypie2, width=0.4, edgecolor='white')

plt.margins(0,0)



#added a legend 

plt.legend()

#this creates a connection between colors in the chart and the legend

handles, labels = ax.get_legend_handles_labels()



#the legend will show proper subgroup names

#aligned location slightly further to the right of the pie chart

ax.set_axis_off()

ax.legend(handles[3:], words, loc=(1.2, -0.3))

#END - BORROWED CODE