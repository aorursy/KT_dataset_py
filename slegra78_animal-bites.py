import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt

import matplotlib.patches as patches





from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

data = pd.read_csv('../input/Health_AnimalBites.csv')
data.columns
data.head()
species = data.SpeciesIDDesc

species = species.dropna() 

speciesOfAnimal = species.unique()

print(speciesOfAnimal)


animal_list = []

for  i in speciesOfAnimal:

    animal_list.append(len(species[species==i]))

ax = sns.barplot(x=speciesOfAnimal, y =animal_list)

plt.title('Number of Species Bite')

plt.xticks(rotation=90)

print(animal_list)
def animal_month(animal,data):

    month_list= ['01','02','03','04','05','06','07','08','09','10','11','12']

    numberOfAnimal = []

    for i in month_list:

        x = data.loc[(data['SpeciesIDDesc']==animal)&(data['bite_date'].str.split('-').str[1]==i)]

        numberOfAnimal.append(len(x))

    ax = sns.barplot(x=month_list,y=numberOfAnimal,palette  = "Greens")

    plt.title(animal + ' bite for 12 month')
animal_month('DOG',data)
animal_month('CAT',data)
animal_month('BAT',data)
count = data.BreedIDDesc.value_counts()

plt.figure(figsize=(15,8))

ax = sns.barplot(x=count[0:10].index,y=count[0:10])

plt.xticks(rotation=40)

plt.ylabel("Number of Bite")

print(count[0:10].index)
def bite_place(animal,data):

    bitePlaces = data.WhereBittenIDDesc.unique()

    head = data.loc[(data['SpeciesIDDesc']==animal)&(data['WhereBittenIDDesc']=='HEAD')]

    body = data.loc[(data['SpeciesIDDesc']==animal)&(data['WhereBittenIDDesc']=='BODY')]

    numberOfHead = len(head)

    numberOfBody = len(body)

    total = numberOfHead+numberOfBody

    fig1=plt.figure()

    ax1=fig1.add_subplot(111,aspect='equal')

    ax1.add_patch(

        patches.Rectangle((0.3,0.1),0.4,0.5,alpha=numberOfBody/float(total),color='c')

    )

    circle = plt.Circle((0.5,0.7),0.1,color='m',alpha=numberOfHead/float(total))

    ax1.add_artist(circle)

    plt.text(0.45,0.7,round(numberOfHead/float(total),2))

    plt.text(0.45,0.4,round(numberOfBody/float(total),2))

    plt.title(str(animal)+' Bite Probability of Head and Body')
bite_place('DOG',data)
bite_place('CAT',data)
bite_place('BAT',data)


gender = ['MALE','FEMALE']

count_gender = data.GenderIDDesc.value_counts()

plt.figure(figsize= (4,4))

x = sns.barplot(x=gender, y= count_gender[0:2])

plt.ylabel('Number of Bite ')

plt.xticks(rotation = 10)

plt.title('MALE VS FEMALE')

print(count_gender[0:2])
def rabid_prob(animal,data):

    labels = ['POSITIVE','NEGATIVE']

    colors = ['cyan','green']

    explode = [0.1,0]

    p = data.loc[(data['SpeciesIDDesc']==animal)&(data['ResultsIDDesc']=='POSITIVE')]

    n = data.loc[(data['SpeciesIDDesc']==animal)&(data['ResultsIDDesc']=='NEGATIVE')]

    sizes = [len(p),len(n)]

    print(sizes)

    if len(p)==0:

        labels = ['','NEGATIVE']

    plt.pie(sizes,explode=explode,labels=labels,colors=colors,autopct ='%1.1f&&')

    plt.axis('equal')

    plt.title(animal + ' Rabid Probability')

    plt.show()
rabid_prob('DOG',data)
rabid_prob('BAT',data)
a = data.loc[(data['ResultsIDDesc']=='POSITIVE')]

a = a.loc[:,['bite_date','SpeciesIDDesc','BreedIDDesc','GenderIDDesc','color','ResultsIDDesc']]

print(a)