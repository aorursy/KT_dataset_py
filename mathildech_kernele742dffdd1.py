import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import math as math
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score 
from sklearn.metrics import log_loss
from sklearn import datasets
from sklearn.model_selection import train_test_split 
from sklearn import tree
from sklearn import metrics 
from scipy.stats import itemfreq
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
print("INFO TRAIN")
train.info()
print()
print('*'*50)
print()
print("INFO TEST")
test.info()
train.head()
test.head()
del train['OutcomeSubtype']
data=pd.concat([train,test], ignore_index=True)
data.info()
data['Sex']=data.SexuponOutcome.str.extract('. ([A-Za-z]+)', expand=False)
data['Neutered']=data.SexuponOutcome.str.extract('([A-Za-z]+) .', expand=False)
data.loc[(data.Sex.isnull()==1), 'Sex']='Unknow'
data.loc[(data.Neutered.isnull()==1), 'Neutered']='Unknow'
pd.value_counts(data.SexuponOutcome)
del data['SexuponOutcome']
data['Neutered']=data['Neutered'].replace(['Neutered','Spayed'],1)
data['Neutered']=data['Neutered'].replace('Intact',0)
data['Neutered']=data['Neutered'].replace('Unknow',2)
pd.value_counts(data.Neutered)
data['Sex']=data['Sex'].replace('Male',1)
data['Sex']=data['Sex'].replace('Female',0)
data['Sex']=data['Sex'].replace('Unknow',2)
pd.value_counts(data.Sex)
data.head()
data['Year']=data.DateTime.str.extract('([0-9]+)-', expand=False)
data['Month']=data.DateTime.str.extract('.-([0-9]+)-', expand=False)
data['Day']=data.DateTime.str.extract('.-([0-9]+) ', expand=False)
data['Hour']=data.DateTime.str.extract('. ([0-9]+):',expand=False)
del data['DateTime']
data.head()
pd.value_counts(data['AnimalType'])
data['AnimalType']=data['AnimalType'].replace('Cat',0)

data['AnimalType']=data['AnimalType'].replace('Dog',1)
data['valeurs']=data.AgeuponOutcome.str.extract('([0-9]+) ', expand=False)
data['unités']=data.AgeuponOutcome.str.extract('. ([A-Za-z]+)', expand=False)
pd.value_counts(data.unités)
data['unités']=data['unités'].replace(['months','month'],0.0833) #1/12=0.083333
data['unités']=data['unités'].replace(['years','year'],1)
data['unités']=data['unités'].replace(['weeks','week'],0.0192)#1/52=0.0192
data['unités']=data['unités'].replace(['days','day'],0.00274) #1/365=0.00273972
data['valeurs']=data.valeurs.astype(float)
data['AgeInYears']=data['valeurs']*data['unités']
del data['AgeuponOutcome']
del data['unités']
del data['valeurs']
data.head()
data['temp']=data.AnimalID.str.extract('([A-z]+)',expand=False)

data['temp2']=data.AnimalID.str.extract('.([0-9]+)')
pd.value_counts(data.temp)
#del data['AnimalID']
#del data['ID']
del data['temp']
del data['temp2']
data.head(40)
data.loc[(data['Name'].isnull()==0), 'Name']=1
data.loc[(data['Name'].isnull()==1), 'Name']=0
data.head()
cats=data[data.AnimalType==0]
dogs=data[data.AnimalType==1]
pd.value_counts(cats.Breed)
cats['Hair']=cats.Breed.str.extract('. ([A-Za-z]+)hair')
pd.value_counts(cats.Hair)
print(cats[cats.Hair=='Wire'])
cats['Mediumtemp']=cats.Breed.str.extract('(Medium) Hair')
pd.value_counts(cats.Mediumtemp)
cats.loc[(cats['Mediumtemp']=='Medium'), 'Hair']='Medium'
pd.value_counts(cats.Hair)
del cats['Mediumtemp']
cats['temp']=cats.Breed.str.extract(' (Mix)$',expand=False)
cats['temp2']=cats.Breed.str.extract('.(/).',expand=False)
cats.loc[(cats['temp'].isnull()==0) | (cats['temp2'].isnull()==0), 'Mix']=1
cats.loc[(cats['Mix'].isnull()==1), 'Mix']=0
# First, if we have '/' in breed, (so if temp2 is not null) we will put all info after it into Race2
cats.loc[(cats['temp2'].isnull()==0), 'Race2']=cats.Breed.str.extract('/([A-Z-a-z]+)', expand=False)
# Then, if the cat is not Mix, have no '/' and have not info on Hair then we put Breed into Race1 
#(this way we nom we xon't have to change info into Race1 to remove 'Hair')
cats.loc[(cats['Mix']==0) & (cats['temp2'].isnull()==1) & (cats['Hair'].isnull()==1), 'Race1']=cats.Breed.str.extract('(.+)',expand=False)
# Next if the cat is not mix but has Hair info, we will put Breed minus hair info into Race1
cats.loc[(cats['temp2'].isnull()==1) & (cats['Hair'].isnull()==0), 'Race1']=cats.Breed.str.extract('(.+) .hair',expand=False)
# After, if the cat as two breeds (so if temp2 is not null), then we will put the info before '/' into Race1
cats.loc[(cats['Mix']==0) & (cats['temp2'].isnull()==0), 'Race1']=cats.Breed.str.extract('(.+)/',expand=False)
# Then if the cat is Mix, we will but Breed minus 'Mix' into Race1 
#This way we will still have the Hair info in Race1 but we will fix this later
cats.loc[(cats['Mix']==1), 'Race1']=cats.Breed.str.extract('(.+) Mix',expand=False)
#cats['Race1']=cats.Breed.str.extract('([A-Za-z]+)/',expand=False)
#Finally, there were somme Race1 missing so we put avery thing before '/' into Race1
cats.loc[(cats['Race1'].isnull()==1), 'Race1']=cats.Breed.str.extract('([A-Za-z]+)/',expand=False)
#cats['Race2']=cats.Breed.str.extract('/([A-Z-a-z]+)', expand=False)
cats.head(40)
cats['Race1']=cats.Race1.str.replace('Domestic Shorthair','Domestic')
cats['Race1']=cats.Race1.str.replace('Domestic Longhair','Domestic')
cats['Race1']=cats.Race1.str.replace('DOmestic Medium Hair','Domestic')
cats.loc[(cats.Race1.isnull()==1) & (cats.Breed.str.find('Domestic')>-1), 'Race1']='Domestic'
cats.loc[(cats.Race1.isnull()==1) & (cats.Breed.str.find('British')>-1), 'Race1']='British'
#We create a temporary dataframe with only cats where Hair is null :
catsnohair=cats[cats.Hair.isnull()]
pd.value_counts(catsnohair.Breed)
cats['Siamese']=cats.Breed.str.extract('(Siamese)')
cats.loc[(cats.Siamese.notnull()==1), 'Hair']='Short'
cats.loc[(cats.Breed=='Snowshoe Mix'), 'Hair']='Short'
cats.loc[(cats.Breed=='Maine Coon Mix'), 'Hair']='Long'
cats.loc[(cats.Breed=='Manx Mix'), 'Hair']='Medium'
cats.loc[(cats.Breed=='Russian Blue Mix'), 'Hair']='Short'
cats.loc[(cats.Breed=='Himalayan Mix'), 'Hair']='Long'
cats.loc[(cats.Breed=='Ragdoll Mix'), 'Hair']='Long'
cats.loc[(cats.Breed=='Persian Mix'), 'Hair']='Long'
cats.loc[(cats.Breed=='Angora Mix'), 'Hair']='Medium'
cats.loc[(cats.Breed=='Balinese Mix'), 'Hair']='Long'
cats.loc[(cats.Breed=='Bengal Mix'), 'Hair']='Short'
cats.loc[(cats.Breed=='Bombay Mix'), 'Hair']='Short'
cats.loc[(cats.Breed=='Cymric Mix'), 'Hair']='Long'
cats.loc[(cats.Breed=='Devon Rex Mix'), 'Hair']='Wire'
cats.loc[(cats.Breed=='Devon Rex'), 'Hair']='Wire'
cats.loc[(cats.Breed=='Abyssinian Mix'), 'Hair']='Short'
cats.loc[(cats.Breed=='Chartreux Mix '), 'Hair']='Short'
cats.loc[(cats.Breed=='Burmese'), 'Hair']='Short'
cats.loc[(cats.Breed=='Chartreux Mix'), 'Hair']='Short'
cats.loc[(cats.Breed=='Japanese Bobtail Mix'), 'Hair']='Medium'
cats.loc[(cats.Breed=='Maine Coon'), 'Hair']='Long'
cats.loc[(cats.Breed=='Cornish Rex Mix'), 'Hair']='Wire'
cats.loc[(cats.Breed=='Havana Brown Mix'), 'Hair']='Short'
cats.loc[(cats.Breed=='Bengal'), 'Hair']='Short'
cats.loc[(cats.Breed=='Tonkinese Mix'), 'Hair']='Short'
cats.loc[(cats.Breed=='Snowshoe'), 'Hair']='Short'
cats.loc[(cats.Breed=='Persian'), 'Hair']='Long'
cats.loc[(cats.Breed=='Himalayan'), 'Hair']='Long'
cats.loc[(cats.Breed=='Javanese Mix'), 'Hair']='Long'
cats.loc[(cats.Breed=='Turkish Van Mix'), 'Hair']='Medium'
cats.loc[(cats.Breed=='Norwegian Forest Cat Mix'), 'Hair']='Long'
cats.loc[(cats.Breed=='Ragdoll'), 'Hair']='Long'
cats.loc[(cats.Breed=='Sphynx'), 'Hair']='None'
cats.loc[(cats.Breed=='Scottish Fold Mix'), 'Hair']='Medium'
cats.loc[(cats.Breed=='Oriental Sh Mix'), 'Hair']='Short'
cats.loc[(cats.Breed=='Manx'), 'Hair']='Medium'
cats.loc[(cats.Breed=='Angora/Persian'), 'Hair']='Long' #Mi-long + Long = Long non ? 
cats.loc[(cats.Breed=='Snowshoe/Ragdoll'), 'Hair']='Medium' #Long+Short=Medium
cats.loc[(cats.Breed=='Turkish Angora Mix'), 'Hair']='Medium'
cats.loc[(cats.Breed=='Ocicat Mix'), 'Hair']='Short'
cats.loc[(cats.Breed=='Russian Blue'), 'Hair']='Short'
del catsnohair
cats.head()
del cats['Siamese']
del cats['temp']
del cats['temp2']
pd.value_counts(cats.Hair)
cats.info()
cats.loc[(cats.Race2.isnull()==1), 'Race2']=cats.Race1
pd.value_counts(cats.Color)
cats.loc[(cats.Color.str.find('Tabby')>-1), 'Tabby']=1
cats.loc[(cats.Color.str.find('Tabby')==-1), 'Tabby']=0
pd.value_counts((cats.Color.str.find('Point')>-1) & (cats.Color.str.find('Tabby')>-1))
pd.value_counts((cats.Color.str.find('Point')>-1))
# We extract the color before the '/' symbole as the fisrt column
cats['Color1']=cats.Color.str.extract('(.+)/')
# If there is no '/' that means there is only one color so we can put it in Color1
cats.loc[(cats.Color.str.find('/')==-1) , 'Color1']=cats['Color']
# And finally, color after '/' is the second color.
cats['Color2']=cats.Color.str.extract('/(.+)')
cats.info()
pd.value_counts(cats.Color2)
pd.value_counts(cats.Color1)
cats['Color1']=cats['Color1'].str.replace('Tabby','')
cats['Color2']=cats['Color2'].str.replace('Tabby','')
cats['Color1']=cats['Color1'].str.replace('Point','')
cats['Color2']=cats['Color2'].str.replace('Point','')
cats.loc[(cats.Color2.isnull()==1), 'Color2']=cats.Color1
del cats['Color']
cats.head()
dogs.head()
pd.value_counts(dogs.Breed)
dogs.loc[(dogs.Breed.str.find('Mix')>1) | (dogs.Breed.str.find('/')>-1), 'Mix']=1
dogs.loc[(dogs.Mix.isnull()==1), 'Mix']=0
dogs.head()
dogs['Race1']=dogs.Breed.str.extract('(.+)/')
dogs['Race2']=dogs.Breed.str.extract('/(.+)')
dogs.loc[(dogs.Breed.str.find('/')==-1), 'Race1']=dogs['Breed']
dogs['Race1']=dogs.Race1.str.replace('Mix','')
pd.value_counts(dogs.Race1)
dogs['Hair']=dogs.Breed.str.extract('([A-Za-z]+)hair')
pd.value_counts(dogs.Hair)
dogs['Race1']=dogs['Race1'].str.replace('Mix','')
dogs['Race2']=dogs['Race2'].str.replace('Mix','')
dogs.info()
dogs.loc[(dogs.Race2.isnull()==1), 'Race2']=dogs.Race1
del dogs['Hair']
dogs.head()
dogs['Color1']=dogs.Color.str.extract('(.+)/')
dogs.loc[(dogs.Color.str.find('/')==-1), 'Color1']=dogs['Color']
dogs['Color2']=dogs.Color.str.extract('/(.+)')
dogs.head()
del dogs['Color']
pd.value_counts(dogs.Color1)
dogs.loc[(dogs.Color2.isnull()==1), 'Color2']=dogs.Color1
dogs.info()
cats.info()
cats.loc[(cats.AgeInYears.isnull()==1), 'AgeInYears']=cats['AgeInYears'].fillna(method='pad')
cats_train=cats[(cats.OutcomeType.isnull()==0)]
cats_test=cats[(cats.OutcomeType.isnull()==1)]
#cats_train.to_csv('C:/Users/mathi/Downloads/all1/cats_train2.csv')
#cats_test.to_csv('C:/Users/mathi/Downloads/all1/cats_test2.csv')
dogs.info()
dogs[dogs.AgeInYears.isnull()==1]
sns.countplot(data=dogs[dogs.Breed=='Toy Poodle Mix'], x='AgeInYears')
dogs[dogs.Breed=='Toy Poodle Mix'].mean()
dogs['AgeInYears'][3875]=4.5
dogs.info()
del dogs['Breed']
dogs_train=dogs[(dogs.OutcomeType.isnull()==0)]
dogs_test=dogs[(dogs.OutcomeType.isnull()==1)]
#dogs_train.to_csv('C:/Users/mathi/Downloads/all1/dogs_train2.csv')
#dogs_test.to_csv('C:/Users/mathi/Downloads/all1/dogs_test2.csv')

data_train=pd.concat([cats_train,dogs_train])
data_test=pd.concat([cats_test,dogs_test])

data_train.info()
#Animals(Cats and Dogs) Outcomes 
OutcomeTypeTrain = data_train.OutcomeType.value_counts() / len(data_train.index)
OutcomeTypeTrain.plot(kind='barh')
#We see that the most animals are adopted or transfered 
#We remarque that the number of died or euthanized animal is less important
#Cats Future
OutcomeTypeCats = cats_train.OutcomeType.value_counts() / len(cats_train.index)
OutcomeTypeCats.plot(kind='barh')
#Cats are transfered with a large percentage of 50% and get adopted with a percentage of ~40%
#A less percentage of them are euthanized , returned_to_owner or died
#Dogs Future
OutcomeTypeDogs = dogs_train.OutcomeType.value_counts() / len(dogs_train.index)
OutcomeTypeDogs.plot(kind='barh')
#Dogs have more than 40% of chance to be adopted, 28% to be returned to their owner and 25% of chance to be transfered
#The resting percents are for dogs who are euthanized or died
#The order of popularity of the outcomes is different than the one of the cats
data_train["AgeInYears"].plot.hist(weights = np.ones_like(data_train.index) / len(data_train.index))
#Most animals (~60%) have an age between 0 and 2.5 Years 
cats_train['Animal']='cats'
dogs_train['Animal']='dogs'
cats_train['AnimalType']='0'
dogs_train['AnimalType']='1'
data_train=pd.concat([cats_train,dogs_train])
sns.boxplot(x = "Animal", y = "AgeInYears", data = data_train) 
#We remarque that as we see previously , cats in our training dataset are in majority young (age between 0 and 1 Year)
#dogs seems to have more variation in their ages, even if the majority are young 
#OutcomeType for cats in reference to the Age 
g = sns.FacetGrid(cats_train, col='OutcomeType')
g.map(plt.hist, 'AgeInYears')
#The majority of cats transfered or adopted are young ( <2.5 years)
#OutcomeType for dogs in reference to the Age 
g = sns.FacetGrid(dogs_train, col='OutcomeType')
g.map(plt.hist, 'AgeInYears')
#The majority of dogs transfered or adopted are young ( <2.52 years)
#Contraty to the cats, the biggest part of dogs returned to their owner or euthanized are not the youngest
cats_train.loc[(cats_train.AgeInYears<=0.5), 'Stage']='Baby'
cats_train.loc[(cats_train.AgeInYears>0.5) & (cats_train.AgeInYears<3), 'Stage']='Junior'
cats_train.loc[(cats_train.AgeInYears>=3) & (cats_train.AgeInYears<7), 'Stage']='Prime'
cats_train.loc[(cats_train.AgeInYears>=7) & (cats_train.AgeInYears<11), 'Stage']='Mature'
cats_train.loc[(cats_train.AgeInYears>=11) & (cats_train.AgeInYears<15), 'Stage']='Prime'
cats_train.loc[(cats_train.AgeInYears>=15), 'Stage']='Geriatric'
cats_train.count()
dogs_train.loc[(dogs_train.AgeInYears<=0.5), 'Stage']='Baby'
dogs_train.loc[(dogs_train.AgeInYears>0.5) & (dogs_train.AgeInYears<3), 'Stage']='Junior'
dogs_train.loc[(dogs_train.AgeInYears>=3) & (dogs_train.AgeInYears<7), 'Stage']='Prime'
dogs_train.loc[(dogs_train.AgeInYears>=7) & (dogs_train.AgeInYears<11), 'Stage']='Mature'
dogs_train.loc[(dogs_train.AgeInYears>=11) & (dogs_train.AgeInYears<15), 'Stage']='Prime'
dogs_train.loc[(dogs_train.AgeInYears>=15), 'Stage']='Geriatric'
#Age distribution for cats and dogs 
fig=plt.figure(figsize=(13,13))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.set_title('Age Distribution for Cats')
ax2.set_title('Age Distribution for Dogs')
sns.countplot(x="Stage" , hue="OutcomeType" , data=cats_train,ax=ax1)
sns.countplot(x="Stage" , hue="OutcomeType" , data=dogs_train,ax=ax2)
#Kittens are the more numerous, followed by Juniors.
#Juniors seem the most likely to be transfered, and kittens to die.
#the majority of adopted cats are kittens.
#Cats destiny in reference to the Sex
sns.countplot(x="Sex",hue='OutcomeType',data=cats_train)
#If the sex of the cat is known(0=Female,1=Male), they are more likely adopted or transfered
#the Number of female and male cats adopted or transfered is the same 
#The cats of unknown sex are more likely transfered or euthanized
#it does not seem to exist a relation between the sex and the outcome of the cat, if the sex is known
#Dogs destiny in reference to the Sex
sns.countplot(x="Sex",hue='OutcomeType',data=dogs_train)
#If the sex of the dog is known(0=Female,1=Male) , they are more likely to be adopted, returned to their owner or transfered
#There are no dogs of unknown sex who get adopted
#Cats selected by their Sex , we visualize their destiny 
sns.barplot(x="Sex",y="AgeInYears" , hue="OutcomeType", data=cats_train)
#the average age of a euthanized or returned cat of known sex is far superior to an adopted, transfered or dead one.
#The exception is for the cats of unknown sex.
#Dogs selected by their Sex , we visualize their destiny 
sns.barplot(x="Sex",y="AgeInYears" , hue="OutcomeType", data=dogs_train)
#the average age of a euthanized or returned dog of known sex is superior to an adopted, transfered or dead one.
#The exception is for the dogs of unknown sex
#Numerical transformation for the OutcomeType (to use it for the violon plots)
data=cats_train
cats_train.loc[(data['OutcomeType']=='Adoption'), 'Destin']=0
cats_train.loc[(data['OutcomeType']=='Died'), 'Destin']=1
cats_train.loc[(data['OutcomeType']=='Euthanasia'), 'Destin']=2
cats_train.loc[(data['OutcomeType']=='Return_to_owner'), 'Destin']=3
cats_train.loc[(data['OutcomeType']=='Transfer'), 'Destin']=4

data=dogs_train
dogs_train.loc[(data['OutcomeType']=='Adoption'), 'Destin']=0
dogs_train.loc[(data['OutcomeType']=='Died'), 'Destin']=1
dogs_train.loc[(data['OutcomeType']=='Euthanasia'), 'Destin']=2
dogs_train.loc[(data['OutcomeType']=='Return_to_owner'), 'Destin']=3
dogs_train.loc[(data['OutcomeType']=='Transfer'), 'Destin']=4
#Graphics in violon to describe the destiny of cats depending on their age and Sex 
fig=plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.set_title('Age Distribution for Cats')
ax2.set_title('Age Distribution for Dogs')
sns.violinplot(x='Sex', y='Destin', hue='Stage', data=cats_train,ax=ax1)
sns.violinplot(x='Sex', y='Destin', hue='Stage', data=dogs_train,ax=ax2)            
#As said previously, the fact that the cat is a male or a female does not seem to play a role in it outcome : the repartitions for male and female look alike. 
#If the sex is unknown, the outputs are not the same.
#The age definitely plays a role : there are variations depending on the age.
#Animals Destiny depending if they are intact(0), neutered(1) or unknown(2)
sns.countplot(x="Neutered",hue='OutcomeType',data=data_train)
#Neutered Animals are the most adopted and reutrned to teir owner
#Intact animals and animals for which it's unknown  are more likely to be transfered
#Animals Destiny depending if they have a name or not
sns.countplot(x="Name",hue='OutcomeType',data=data_train)
#It's clear that animals with a Name are the most adopted and reutrned to their owner
#Having a Name is an important feature for adoption or return to the owner
#Cats Destiny depending on their Hair (Short,Meduim,...)
sns.countplot(x="Hair",hue='OutcomeType',data=cats_train)
# The majority of the cats seems to have short hair. There seem to be more likey to be transfered.
#there does not seem to exist a difference between the outcomes of cats with long or medium hair.
data_train=pd.concat([cats_train,dogs_train])
del data_train['Tabby'] # we delete the column Tabby, present only for the cats
# we make sure to have all the numerical variables in a numeric type
data_train['Year'] = data_train['Year'].apply(pd.to_numeric, errors='coerce')
data_train['Month'] = data_train['Month'].apply(pd.to_numeric, errors='coerce')
data_train['Hour'] = data_train['Hour'].apply(pd.to_numeric, errors='coerce')
data_train['AnimalType'] = data_train['AnimalType'].apply(pd.to_numeric, errors='coerce')
data_train.info()
#heat Map for all animals : data_train is composed of the train sets of cats and dogs 

plt.figure(figsize=(10,10))
h=sns.heatmap(data_train.corr(),annot=True)
#AgeInYears, AnimalType, Hour, Name, Neuterd and Sex are the variables that are the most correlated with the animal future
#We load the data for cats and dogs
cats_train=cats[(cats.OutcomeType.isnull()==0)]
cats_test=cats[(cats.OutcomeType.isnull()==1)]

dogs_train=dogs[(dogs.OutcomeType.isnull()==0)]
dogs_test=dogs[(dogs.OutcomeType.isnull()==1)]
    
#We add AnimalType to difference between cats and dogs after putting them in the same dataframe
cats_train['AnimalType']='0'
dogs_train['AnimalType']='1'

cats_test['AnimalType']='0'
dogs_test['AnimalType']='1'

data_train=pd.concat([cats_train,dogs_train])
data_test=pd.concat([cats_test,dogs_test])
data_train.head()
#In order to use the learning methods, we need to convert object columns (Race, Color...) in int64 columns.
from sklearn import preprocessing 
le=preprocessing.LabelEncoder()

le.fit(data_train.Race1)
le.transform(data_train.Race1)
data_train['Race12']=le.transform(data_train.Race1)

le.fit(data_train.Race2)
le.transform(data_train.Race2)
data_train['Race22']=le.transform(data_train.Race2)

le.fit(data_train.Color1)
le.transform(data_train.Color1)
data_train['Col1']=le.transform(data_train.Color1)

le.fit(data_train.Color2)
le.transform(data_train.Color2)
data_train['Col2']=le.transform(data_train.Color2)

le.fit(data_test.Race1)
le.transform(data_test.Race1)
data_test['Race12']=le.transform(data_test.Race1)

le.fit(data_test.Race2)
le.transform(data_test.Race2)
data_test['Race22']=le.transform(data_test.Race2)

le.fit(data_test.Color1)
le.transform(data_test.Color1)
data_test['Col1']=le.transform(data_test.Color1)

le.fit(data_test.Color2)
le.transform(data_test.Color2)
data_test['Col2']=le.transform(data_test.Color2)

data_train.head()
#We cut the column AgeInYears in 6 categories
#we create a new column of types int64, containing the OutcomeType converted in int64
data_train.loc[(data_train.AgeInYears<=0.5), 'Stage']='0'
data_train.loc[(data_train.AgeInYears>0.5) & (data_train.AgeInYears<3), 'Stage']='1'
data_train.loc[(data_train.AgeInYears>=3) & (data_train.AgeInYears<7), 'Stage']='2'
data_train.loc[(data_train.AgeInYears>=7) & (data_train.AgeInYears<11), 'Stage']='3'
data_train.loc[(data_train.AgeInYears>=11) & (data_train.AgeInYears<15), 'Stage']='4'
data_train.loc[(data_train.AgeInYears>=15), 'Stage']='5'

data_train.loc[(data_train['OutcomeType']=='Adoption'), 'Destin']='0'
data_train.loc[(data_train['OutcomeType']=='Died'), 'Destin']='1'
data_train.loc[(data_train['OutcomeType']=='Euthanasia'), 'Destin']='2'
data_train.loc[(data_train['OutcomeType']=='Return_to_owner'), 'Destin']='3'
data_train.loc[(data_train['OutcomeType']=='Transfer'), 'Destin']='4'

data_test.loc[(data_test.AgeInYears<=0.5), 'Stage']='0'
data_test.loc[(data_test.AgeInYears>0.5) & (data_test.AgeInYears<3), 'Stage']='1'
data_test.loc[(data_test.AgeInYears>=3) & (data_test.AgeInYears<7), 'Stage']='2'
data_test.loc[(data_test.AgeInYears>=7) & (data_test.AgeInYears<11), 'Stage']='3'
data_test.loc[(data_test.AgeInYears>=11) & (data_test.AgeInYears<15), 'Stage']='4'
data_test.loc[(data_test.AgeInYears>=15), 'Stage']='5'
#decision tree
#We create a dataframe df with the data set data_train
df=data_train[['AgeInYears','AnimalType','Race1','Race2','Color1','Color2','Day','Hour','Mix','Month','Name','Neutered','OutcomeType','Sex','Year','Race12','Race22','Col1','Col2','Stage','Destin','OutcomeType']]
#X is a dataframe with a selection of variables that are going to be our predictors variables 
#Y is a dataframe containing only the OutcomeType (the target variable)
X=df[['AnimalType','Mix','Month','Day','Hour','Year','Name','Neutered','Sex','Year','Race12','Race22','Col1','Col2','Stage']].values
Y=df['Destin'].values
X_train,X_test,Y_train,Y_test = train_test_split(X,Y)#we split the train dataset in two : X_train et Y_train to train the method
#and X_test and Y_test to test it accuracy and log loss
classifier = tree.DecisionTreeClassifier()
classifier.fit(X_train,Y_train)
Y_predict=classifier.predict(X_test)#to get the accuracy, we predict the class of the outcomes
accuracy_score(Y_test,Y_predict)# accuracy
predictions=classifier.predict_proba(X_test)#to get the log loss, we predict the probabilities of each classes of the Outcome
log_loss(Y_test,predictions)#log loss
#randomForest
rfclassifier = RandomForestClassifier(n_estimators=100)
rfclassifier.fit(X_train,Y_train)
predictions = rfclassifier.predict(X_test)
accuracy_score(Y_test, predictions)
predictions=rfclassifier.predict_proba(X_test)
log_loss(Y_test,predictions)
#naiveBayes
nbclassifier = GaussianNB()
nbclassifier.fit(X_train,Y_train)
predictions = nbclassifier.predict(X_test)
accuracy_score(Y_test, predictions)
predictions=nbclassifier.predict_proba(X_test)
log_loss(Y_test,predictions)
#we delete the column OutcomeType of the data_test, we are going fill it later.
del data_test['OutcomeType']
#we choose the method with the best result : random forest
X_missing = data_test[['AnimalType', 'Mix', 'Month','Day','Hour','Year','Name', 'Neutered', 'Sex',  'Year','Race12','Race22', 'Col1', 'Col2', 'Stage']].values
# we select the same variables for X_missing that we selected earlier for X_train
prediction = rfclassifier.predict(X_missing)

data_test['OutcomeType']=prediction#we are going to assign the prediction to the column OutcomeType
data_test.head()
#we convert the numbers resulting into the OutcomeTyoe associated
data_test.loc[(data_test['OutcomeType']=='0'), 'OutcomeType']='Adoption'
data_test.loc[(data_test['OutcomeType']=='1'), 'OutcomeType']='Died'
data_test.loc[(data_test['OutcomeType']=='2'), 'OutcomeType']='Euthanasia'
data_test.loc[(data_test['OutcomeType']=='3'), 'OutcomeType']='Return_to_owner'
data_test.loc[(data_test['OutcomeType']=='4'), 'OutcomeType']='Transfer'
data_test.head()
#We see that with the prediction model, we still have the fact that the majority of dogs are transfered or adopted.
#The difference is that the first Outcome here is transfer, and not adoption
OutcomeTypeTest= data_test.OutcomeType.value_counts() / len(data_test.index)
OutcomeTypeTest.plot(kind='barh')
topredict=data_test[['AnimalType', 'Mix', 'Month','Day','Hour','Year','Name', 'Neutered', 'Sex',  'Year','Race12','Race22', 'Col1', 'Col2', 'Stage']].values
pred=rfclassifier.predict_proba(topredict)#we create the prediction of the probabilities of each Outcomes
pred2=pd.DataFrame(pred, columns=['Adoption', 'Died', 'Euthanasia','Return_to_owner','Transfer'])
#we start the index of data_test at one, to make the concatenation easier 
data_test.reset_index(drop=True,inplace=True)
pred2['ID']=data_test['ID']

#we are then going to change the order of the columns, to have AnimalID in first
columnsTitles = ['ID','Adoption', 'Died', 'Euthanasia','Return_to_owner','Transfer']
final=pred2.reindex(columns=columnsTitles)
final.ID = final.ID.astype(int)
final.head()
#we save the dataframe in a csv
#final.to_csv('C:/Users/mathi/Downloads/all1/final.csv', index=False)
#We encode labels with Object type by LabelEncoder to get int64 data that we can use it later to apply the learning methods 
from sklearn import preprocessing 
le=preprocessing.LabelEncoder()

cats_train=cats[(cats.OutcomeType.isnull()==0)]
cats_test=cats[(cats.OutcomeType.isnull()==1)]

le.fit(cats_train.Hair)
le.transform(cats_train.Hair)
cats_train['Pelage']=le.transform(cats_train.Hair)

le.fit(cats_test.Hair)
le.transform(cats_test.Hair)
cats_test['Pelage']=le.transform(cats_test.Hair)

le.fit(cats_train.Race1)
le.transform(cats_train.Race1)
cats_train['Race12']=le.transform(cats_train.Race1)

le.fit(cats_train.Race2)
le.transform(cats_train.Race2)
cats_train['Race22']=le.transform(cats_train.Race2)

le.fit(cats_train.Color1)
le.transform(cats_train.Color1)
cats_train['Col1']=le.transform(cats_train.Color1)

le.fit(cats_train.Color2)
le.transform(cats_train.Color2)
cats_train['Col2']=le.transform(cats_train.Color2)

le.fit(cats_train.Hair)
le.transform(cats_train.Hair)
cats_train['Pelage']=le.transform(cats_train.Hair)

le.fit(cats_test.Race1)
le.transform(cats_test.Race1)
cats_test['Race12']=le.transform(cats_test.Race1)

le.fit(cats_test.Race2)
le.transform(cats_test.Race2)
cats_test['Race22']=le.transform(cats_test.Race2)

le.fit(cats_test.Color1)
le.transform(cats_test.Color1)
cats_test['Col1']=le.transform(cats_test.Color1)

le.fit(cats_test.Color2)
le.transform(cats_test.Color2)
cats_test['Col2']=le.transform(cats_test.Color2)
#We resume the varibale AgeInYears in a variable Stage describing the differents life stages of a cat and converting them into numerical values(int64)
#Destin a variable to have the 5 cases of Destiny  according them numerical values(int64)
cats_train.loc[(cats_train.AgeInYears<=0.5), 'Stage']='0'
cats_train.loc[(cats_train.AgeInYears>0.5) & (cats_train.AgeInYears<3), 'Stage']='1'
cats_train.loc[(cats_train.AgeInYears>=3) & (cats_train.AgeInYears<7), 'Stage']='2'
cats_train.loc[(cats_train.AgeInYears>=7) & (cats_train.AgeInYears<11), 'Stage']='3'
cats_train.loc[(cats_train.AgeInYears>=11) & (cats_train.AgeInYears<15), 'Stage']='4'
cats_train.loc[(cats_train.AgeInYears>=15), 'Stage']='5'

cats_train.loc[(cats_train['OutcomeType']=='Adoption'), 'Destin']='0'
cats_train.loc[(cats_train['OutcomeType']=='Died'), 'Destin']='1'
cats_train.loc[(cats_train['OutcomeType']=='Euthanasia'), 'Destin']='2'
cats_train.loc[(cats_train['OutcomeType']=='Return_to_owner'), 'Destin']='3'
cats_train.loc[(cats_train['OutcomeType']=='Transfer'), 'Destin']='4'

cats_test.loc[(cats_test.AgeInYears<=0.5), 'Stage']='0'
cats_test.loc[(cats_test.AgeInYears>0.5) & (cats_test.AgeInYears<3), 'Stage']='1'
cats_test.loc[(cats_test.AgeInYears>=3) & (cats_test.AgeInYears<7), 'Stage']='2'
cats_test.loc[(cats_test.AgeInYears>=7) & (cats_test.AgeInYears<11), 'Stage']='3'
cats_test.loc[(cats_test.AgeInYears>=11) & (cats_test.AgeInYears<15), 'Stage']='4'
cats_test.loc[(cats_test.AgeInYears>=15), 'Stage']='5'
#We create a dataframe df with the data set cats_train
df=cats_train[['AgeInYears','Race1', 'Race2', 'Color1', 'Color2','Day', 'Hair', 'Hour', 'Mix', 'Month', 'Name', 'Neutered', 'OutcomeType', 'Sex', 'Year',  'Race12','Race22', 'Col1', 'Col2', 'Stage','Tabby', 'Pelage','Destin']]
#X is a dataframe with a selection of variables that are going to be our predictors variables 
#Y is a dataframe containing only the OutcomeType (the target variable) 
X=df[[ 'Mix', 'Month','Day','Hour','Year','Pelage','Tabby','Name', 'Neutered', 'Sex',  'Year',  'Race12', 'Race22', 'Col1', 'Col2', 'Stage']].values
Y=df['Destin'].values
#We split X and Y in 2 train subsets and 2 test subsets 
X_train,X_test,Y_train,Y_test = train_test_split(X,Y)
#decision trees
classifier = tree.DecisionTreeClassifier()
classifier.fit(X_train,Y_train)
Y_predict=classifier.predict(X_test)
accuracy_score(Y_test,Y_predict)
predictions=classifier.predict_proba(X_test)
log_loss(Y_test,predictions)
#randomForest
rfclassifier = RandomForestClassifier(n_estimators=100)
rfclassifier.fit(X_train,Y_train)
predictions = rfclassifier.predict(X_test)
accuracy_score(Y_test, predictions)
predictions=rfclassifier.predict_proba(X_test)
log_loss(Y_test,predictions)
#naiveBayes
nbclassifier = GaussianNB()
nbclassifier.fit(X_train,Y_train)
predictions = nbclassifier.predict(X_test)
accuracy_score(Y_test, predictions)
predictions=nbclassifier.predict_proba(X_test)
log_loss(Y_test,predictions)
#We delete the column OutcomeType of the cats_test,it will be completed later
del cats_test['OutcomeType']
#we choose the classifier of the RandomForest method to work with 
X_missing = cats_test[['Mix', 'Month','Day','Hour','Year','Pelage','Tabby','Name', 'Neutered', 'Sex',  'Year',  'Race12', 'Race22', 'Col1', 'Col2', 'Stage']].values
X_missing
prediction = rfclassifier.predict(X_missing)

cats_test['OutcomeType']=prediction
cats_test.head()
#The numbers are converted to the corresponding OutcomeType
cats_test.loc[(cats_test['OutcomeType']=='0'), 'OutcomeType']='Adoption'
cats_test.loc[(cats_test['OutcomeType']=='1'), 'OutcomeType']='Died'
cats_test.loc[(cats_test['OutcomeType']=='2'), 'OutcomeType']='Euthanasia'
cats_test.loc[(cats_test['OutcomeType']=='3'), 'OutcomeType']='Return_to_owner'
cats_test.loc[(cats_test['OutcomeType']=='4'), 'OutcomeType']='Transfer'
cats_test.head()
#We see that with the prediction model, we still have the fact that the majority of cats are transfered or adopted 
OutcomeTypeCatsTest= cats_test.OutcomeType.value_counts() / len(cats_test.index)
OutcomeTypeCatsTest.plot(kind='barh')
#We encode labels with Object type by LabelEncoder to get int64 data that we can use it later to apply the learning methods 
dogs_train=dogs[(dogs.OutcomeType.isnull()==0)]
dogs_test=dogs[(dogs.OutcomeType.isnull()==1)]

le.fit(dogs_train.Race1)
le.transform(dogs_train.Race1)
dogs_train['Race12']=le.transform(dogs_train.Race1)

le.fit(dogs_train.Race2)
le.transform(dogs_train.Race2)
dogs_train['Race22']=le.transform(dogs_train.Race2)

le.fit(dogs_train.Color1)
le.transform(dogs_train.Color1)
dogs_train['Col1']=le.transform(dogs_train.Color1)

le.fit(dogs_train.Color2)
le.transform(dogs_train.Color2)
dogs_train['Col2']=le.transform(dogs_train.Color2)

le.fit(dogs_test.Race1)
le.transform(dogs_test.Race1)
dogs_test['Race12']=le.transform(dogs_test.Race1)

le.fit(dogs_test.Race2)
le.transform(dogs_test.Race2)
dogs_test['Race22']=le.transform(dogs_test.Race2)

le.fit(dogs_test.Color1)
le.transform(dogs_test.Color1)
dogs_test['Col1']=le.transform(dogs_test.Color1)

le.fit(dogs_test.Color2)
le.transform(dogs_test.Color2)
dogs_test['Col2']=le.transform(dogs_test.Color2)
dogs_train.loc[(dogs_train.AgeInYears<=0.5), 'Stage']='0'
dogs_train.loc[(dogs_train.AgeInYears>0.5) & (dogs_train.AgeInYears<3), 'Stage']='1'
dogs_train.loc[(dogs_train.AgeInYears>=3) & (dogs_train.AgeInYears<7), 'Stage']='2'
dogs_train.loc[(dogs_train.AgeInYears>=7) & (dogs_train.AgeInYears<11), 'Stage']='3'
dogs_train.loc[(dogs_train.AgeInYears>=11) & (dogs_train.AgeInYears<15), 'Stage']='4'
dogs_train.loc[(dogs_train.AgeInYears>=15), 'Stage']='5'

dogs_train.loc[(dogs_train['OutcomeType']=='Adoption'), 'Destin']='0'
dogs_train.loc[(dogs_train['OutcomeType']=='Died'), 'Destin']='1'
dogs_train.loc[(dogs_train['OutcomeType']=='Euthanasia'), 'Destin']='2'
dogs_train.loc[(dogs_train['OutcomeType']=='Return_to_owner'), 'Destin']='3'
dogs_train.loc[(dogs_train['OutcomeType']=='Transfer'), 'Destin']='4'

dogs_test.loc[(dogs_test.AgeInYears<=0.5), 'Stage']='0'
dogs_test.loc[(dogs_test.AgeInYears>0.5) & (dogs_test.AgeInYears<3), 'Stage']='1'
dogs_test.loc[(dogs_test.AgeInYears>=3) & (dogs_test.AgeInYears<7), 'Stage']='2'
dogs_test.loc[(dogs_test.AgeInYears>=7) & (dogs_test.AgeInYears<11), 'Stage']='3'
dogs_test.loc[(dogs_test.AgeInYears>=11) & (dogs_test.AgeInYears<15), 'Stage']='4'
dogs_test.loc[(dogs_test.AgeInYears>=15), 'Stage']='5'
#We create df a dataframe with the dataset dogs_train
df=dogs_train[['AgeInYears','Race1', 'Race2', 'Color1', 'Color2','Day', 'Hour', 'Mix', 'Month', 'Name', 'Neutered', 'OutcomeType', 'Sex', 'Year',  'Race12','Race22', 'Col1', 'Col2', 'Stage', 'Destin']]
X=df[[ 'Mix', 'Month','Day','Hour','Year','Name', 'Neutered', 'Sex',  'Year',  'Race12', 'Race22', 'Col1', 'Col2', 'Stage']].values
Y=df['Destin'].values
X_train,X_test,Y_train,Y_test = train_test_split(X,Y)
#decision trees
classifier = tree.DecisionTreeClassifier()
classifier.fit(X_train,Y_train)
Y_predict=classifier.predict(X_test)
accuracy_score(Y_test,Y_predict)
predictions=classifier.predict_proba(X_test)
log_loss(Y_test,predictions)
#randomForest
rfclassifier = RandomForestClassifier(n_estimators=100)
rfclassifier.fit(X_train,Y_train)
predictions = rfclassifier.predict(X_test)
accuracy_score(Y_test, predictions)
predictions=rfclassifier.predict_proba(X_test)
log_loss(Y_test,predictions)
#naiveBayes
nbclassifier = GaussianNB()
nbclassifier.fit(X_train,Y_train)
predictions = nbclassifier.predict(X_test)
accuracy_score(Y_test, predictions)
predictions=nbclassifier.predict_proba(X_test)
log_loss(Y_test,predictions)
#We delete the column OutcomeType of the dogs_test,it will be completed later
del dogs_test['OutcomeType']
#we choose the classifier of the RandomForest method to work with 
X_missing = dogs_test[[ 'Mix', 'Month','Day','Hour','Year','Name', 'Neutered', 'Sex',  'Year','Race12','Race22', 'Col1', 'Col2', 'Stage']].values
X_missing
prediction = rfclassifier.predict(X_missing)

dogs_test['OutcomeType']=prediction
dogs_test.head()
#The numbers will be converted to the corresponding OutcomeType
dogs_test.loc[(dogs_test['OutcomeType']=='0'), 'OutcomeType']='Adoption'
dogs_test.loc[(dogs_test['OutcomeType']=='1'), 'OutcomeType']='Died'
dogs_test.loc[(dogs_test['OutcomeType']=='2'), 'OutcomeType']='Euthanasia'
dogs_test.loc[(dogs_test['OutcomeType']=='3'), 'OutcomeType']='Return_to_owner'
dogs_test.loc[(dogs_test['OutcomeType']=='4'), 'OutcomeType']='Transfer'
dogs_test
#We see that with the prediction model, we still have the fact that the majority of dogs are adopted or returned to owner
OutcomeTypeDogsTest= dogs_test.OutcomeType.value_counts() / len(dogs_test.index)
OutcomeTypeDogsTest.plot(kind='barh')
#csv result file required by the Kaggle :

#final.to_csv('F:/final.csv', index=False)

