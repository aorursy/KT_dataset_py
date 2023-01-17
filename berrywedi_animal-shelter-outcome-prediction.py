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
import pandas as pd 
import numpy as np
import seaborn as snb
import matplotlib.pyplot as plt
%matplotlib inline

all_animals=pd.read_csv('../input/train.csv')
#our next important step is to identify the type of all_animals we have in our dataset
snb.countplot(all_animals.AnimalType, palette='Set1')

#So from the barchart distribution we can see that the number of dogs are 4500 more than the number of cats. 
# we can also see the sex of the all_animals in our dataset
#to have a better undestanding of our animals groups.
snb.countplot(all_animals.SexuponOutcome, palette='Set1')

#the output shows most of the animals are Neutered Male and very few are Unknown
# Furthermore we can also see the distribution of the age outcome of the all_animals in our dataset 
snb.countplot(all_animals.AgeuponOutcome, palette='Set1')
# so in the above 2 bar charts we observe two important informations that we need to separate for clear visualization
# first we have the information whether the animal is Male or Female
#And second we have whether the animal is neureted, spayed or intact.
#So sepatating these information is helpful for visualization and manupulation
all_animals.head(5)
#ela = all_animals.drop("Name")
all_animals.tail(5)

def new_sex(m):
    m = str(m)
    if m.find('Male') >= 0:
        return 'male'
    if m.find('Female') >= 0: 
        return 'female'
    return 'unknown'
def new_neutered(m):
    m = str(m)
    if m.find('Spayed') >= 0: 
        return 'neutered'
    if m.find('Neutered') >= 0: 
        return 'neutered'
    if m.find('Intact') >= 0: 
        return 'intact'
    return 'unknown'
all_animals['Sex'] = all_animals.SexuponOutcome.apply(new_sex)
all_animals['Neutered'] = all_animals.SexuponOutcome.apply(new_neutered)
f, (am1, am2) = plt.subplots(1, 2, figsize=(15, 4))
snb.countplot(all_animals.Sex, palette='Set1', ax=am1)
snb.countplot(all_animals.Neutered, palette='Set1', ax=am2)
all_animals.Sex
#Now we have a good picture of the the male and female distribution as well as the neutered and intact. thus the number of Male and Female animals are almost equal
#However the nauereted animals are more than twice to intact animals. 

#But beside the above independent variables in our dataset, we have also another variable Breed. And we want to see if this vatiable have also significant influence on the outcome of the animals.
def Newcol_Breed(m):
    m=str(m)
    if m.find('Mix')>=0:
        return 'mixed_animal'
    else:
        return 'not_Mixed_animal'
all_animals['Mix']=all_animals.Breed.apply(Newcol_Breed)
snb.countplot(all_animals.Mix, palette='Set1')
#So what we can observe is that most of the animals are mixed but only 5000 animals are not mixed

#lets see the influence of different independent variables on the final outcome
f, (ax1, ax2)=plt.subplots(1, 2, figsize=(15, 4))
snb.countplot(data=all_animals, x='OutcomeType', hue='Sex', palette='Set1', ax=ax1)
snb.countplot(data=all_animals, x='Sex', hue='OutcomeType', palette='Set1', ax=ax2)

#Here we can clearly see that most of the male and female animals have higher adoption rate.
f, (ax1, ax2)=plt.subplots(1, 2, figsize=(15, 4))
snb.countplot(data=all_animals, x='OutcomeType', hue='AnimalType', palette='Set1', ax=ax1)
snb.countplot(data=all_animals, x='AnimalType', hue='OutcomeType', palette='Set1', ax=ax2)

#But here interestingly enough most of the dogs in our dataset  have the highest probablity of return to their owners
#However the number of cats transfer is higher than the dogs, enevthough they have relatively lower adoption rate than their counterpart dogs. 
f, (ax1, ax2)=plt.subplots(1, 2, figsize=(15,4))
snb.countplot(data=all_animals, x='OutcomeType', hue='Neutered', palette='Set1',ax=ax1)
snb.countplot(data=all_animals, x='Neutered', hue='OutcomeType', palette='Set1', ax=ax2)

#Another interesting outcome is almost all of the neutered animals have the highest probablity if adoption
#on the otherhand intact animals have higher transfer rate than the neutered animals. 
#And what about the Breed animals, lets see how does mixed animals affect the ourcome of the animals

f, (ax1, ax2)=plt.subplots(1, 2, figsize=(15,4))
snb.countplot(data=all_animals, x='OutcomeType', hue='Mix', palette='Set1', ax=ax1)
snb.countplot(data=all_animals, x='Mix', hue='OutcomeType', palette='Set1', ax=ax2)

#the graph below shows that Mixed animlas have the highest chance of adoption and transfer, 
#while non mixed animals have have the lowest chance of adoption and transfer
#Age is another independent variable that we have in our dataset. Age may also have some significance on the outcome and lets see it
# Here age is given in different in months and years, so our first step should be converting them all to the same measure
def years_of_age(y):
    y = str(y)
    if y == 'nan': 
        return 0
    AgeOfAnimals = int(y.split()[0])
    if y.find('year') > -1: 
        return AgeOfAnimals 
    if y.find('month')> -1: 
        return AgeOfAnimals / 12.
    if y.find('week')> -1: 
        return AgeOfAnimals / 52.
    if y.find('day')> -1: 
        return AgeOfAnimals / 365.
    else: 
        return 0
all_animals['AnimalAge'] = all_animals.AgeuponOutcome.apply(years_of_age)
#print(all_animals['AnimalAge'])
print(all_animals['AnimalAge'].head(5))
all_animals['AnimalAge'].tail(5)
all_animals['AnimalAge'] = all_animals.AgeuponOutcome.apply(years_of_age)
snb.distplot(all_animals.AnimalAge, bins = 20, kde=False)
#Hence the aobe graph shows that most of the animals in the shelter have 0 and 1.5 yrs

#But does this have an effect on the outcome? lets see it.
def age_division(y):
    if y <=2.5:
        return 'Young'
    elif y>2.5 and y<8:
        return 'Young Adult'
    else:
        return 'Old'
all_animals['CatagoryAge'] = all_animals.AnimalAge.apply(age_division)
print(all_animals['CatagoryAge'])
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))
snb.countplot(data=all_animals, x='OutcomeType', hue='CatagoryAge', ax=ax1)
snb.countplot(data=all_animals, x = 'CatagoryAge', hue='OutcomeType', ax=ax2)
# Finally in the above two output chats indicate that young animals of dogs and cats have higher probablity of aadoption that the other types of animals
#on the other hand older dogs and cats have the lowest probablity of adoption or transfer
