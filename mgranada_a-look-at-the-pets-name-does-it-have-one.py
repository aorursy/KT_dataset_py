

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt



from wordcloud import WordCloud

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))
data = pd.read_csv('../input/train/train.csv')

#color_data = pd.read_csv('../input/train/color_labels.csv')

data.head(10)

# data=data[data['Gender'] ]

# type(data['Name'][8])
names = data.Name.unique()

no_names=[]

for name in names:

    if type(name) is float:

        continue

    if 'name' in name.lower() or 'kitt' in name.lower() or 'pupp' in name.lower() or 'cats' in name.lower() or 'dogs' in name.lower():

        no_names+=[name]

# no_names
data['Is_Nameless'] = (pd.isnull(data['Name'])).astype(int)



for index in range(len(data['Name'])):

    if type(data['Name'].iloc[index]) == float:

        continue

    if data['Name'].iloc[index] in no_names:

        data.Is_Nameless[index] = 1
Named = data[data['Is_Nameless'] == 0]

NotNamed = data[data['Is_Nameless'] == 1]

NamedTotal =[]

NotNamedTotal = []

for i in range(5):

    curr_=sum(sum([Named.AdoptionSpeed == i]))

    NamedTotal+=[curr_/len(Named)*100]

    curr_=sum(sum([NotNamed.AdoptionSpeed == i]))

    NotNamedTotal+=[curr_/len(NotNamed)*100]

    

# Finding the probability of a pet getting adopted up to a period of time

cont_named=[NamedTotal[0]]

cont_unnamed=[NotNamedTotal[0]]



for i in range(1,5):

    cont_named+=[cont_named[i-1]+NamedTotal[i]]

    cont_unnamed+=[cont_unnamed[i-1]+NotNamedTotal[i]]
fig, ax = plt.subplots(figsize = (18, 6))

plt.subplot(1, 2, 1)

plt.plot(range(5),NamedTotal,range(5),NotNamedTotal)

plt.legend(['Named','NotNamed'])

plt.xlabel('Adoption Speed')

plt.ylabel('Probability of adoption in %')

plt.title('Probability that a pet gets adopted on a given period of time')



plt.subplot(1, 2, 2)

plt.plot(range(5),cont_named,range(5),cont_unnamed)

plt.legend(['Named','NotNamed'])

plt.xlabel('Adoption Speed')

plt.ylabel('Probability of adoption in %')

plt.title('Probability that a pet gets adopted up to a given period of time')

plt.show()
# lets try to get the percentage of group animals and singular pets with our without a name



num_named = len(Named['Name'])

num_named_group = len(Named[Named['Gender']==3])

print("\t\t# animals\t # groups\t % of groups")



print("named animals:    %d\t\t%d\t\t%.2f"%(num_named,num_named_group,num_named_group/num_named*100))



num_not_named = len(NotNamed['Name'])

num_not_named_group = len(NotNamed[NotNamed['Gender']==3])

print("unnamed animals:   %d\t\t%d\t\t%.2f"%(num_not_named,num_not_named_group,num_not_named_group/num_not_named*100))

Names_Named_group=Named[Named['Gender']==3]

Names_NotNamed_group=NotNamed[NotNamed['Gender']==3]

#Names_Named_group['Name']
fig, ax = plt.subplots(figsize = (16, 12))

plt.subplot(1, 2, 1)

text_cat = ' '.join(Named['Name'].fillna('').values)

wordcloud = WordCloud(max_font_size=None, background_color='white',

                      width=1200, height=1000).generate(text_cat)

plt.imshow(wordcloud)

plt.title('Pets Actual Names')

plt.axis("off")



plt.subplot(1, 2, 2)

text_dog = ' '.join(NotNamed['Name'].fillna('').values)

wordcloud = WordCloud(max_font_size=None, background_color='white',

                      width=1200, height=1000).generate(text_dog)

plt.imshow(wordcloud)

plt.title('Words of Unnamed Pets')

plt.axis("off")



plt.show()

i=0

for Name in Named.Name.unique():

    print(Name)

    if i == 10: # during proper study this value is increased

        break

    i+=1
i=0

for Name in NotNamed.Name.unique():

    print(Name)

    if i == 10: # during proper study this value is increased

        break

    i+=1