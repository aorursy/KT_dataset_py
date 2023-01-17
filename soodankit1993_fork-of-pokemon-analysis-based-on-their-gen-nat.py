import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv("../input/Pokemon.csv")
df.head()
#Generation 1 pokemon 

df.Generation.value_counts()
#print(df.Generation.count())

df.describe()
#Classifying pokemon with Single or Dual Type Nature

def getNumberOfTypes(x):

    numberOfTypes = 'Dual'

    if(pd.isnull(x[3])):

        numberOfTypes = 'Single'

    

    return numberOfTypes



df['Types'] = df.apply(getNumberOfTypes,axis=1)
df.head()
#splitting the pokemon generationwise

genOnePokemon = df[df.Generation==1]

genTwoPokemon = df[df.Generation==2]

genThreePokemon = df[df.Generation==3]

genFourPokemon = df[df.Generation==4]

genFivePokemon = df[df.Generation==5]

genSixPokemon = df[df.Generation==6]

#Spliting the generation one pokemons on the basis of Single and Dual Nature

genOneDual = genOnePokemon[genOnePokemon.Types=='Dual']

genOneSingle = genOnePokemon[genOnePokemon.Types=='Single']
genOneSinglePer = (genOneDual['#'].count()/genOnePokemon['#'].count())*100
genOneDualPer = (genOneSingle['#'].count()/genOnePokemon['#'].count())*100
# Pie chart, where the slices will be ordered and plotted counter-clockwise:

labels = 'Single', 'Dual'

sizes = [genOneSinglePer,genOneDualPer]

explode = (0, 0.1)  # only "explode" the 2nd slice (i.e. 'Dual')



fig1, ax1 = plt.subplots()

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()
typeOfPokemonsSeriesDf = genOnePokemon['Type 1'].value_counts().reset_index()

typeOfPokemonsSeriesDf['typePercentage'] =  (typeOfPokemonsSeriesDf['Type 1'] / typeOfPokemonsSeriesDf['Type 1'].sum())*100

def classifyPokemon(temp):

    pokeType = temp[0]

    #print(temp[0] + temp[1])

    if temp[2] < 4.0:

        pokeType = 'Others'

    return pokeType

typeOfPokemonsSeriesDf['type'] =  typeOfPokemonsSeriesDf.apply(classifyPokemon,axis=1)

typeOfPokemonsSeriesDf
newTypeOfPokemonsSeriesDf = typeOfPokemonsSeriesDf.groupby('type').sum()

labelsForTypes = newTypeOfPokemonsSeriesDf.reset_index().type.tolist()

sizes = newTypeOfPokemonsSeriesDf['typePercentage'].tolist()

                

explode = (0.0,0,0,0,0,0,0,0,0,0,0,0.1)  # only "explode" the 1st slice (i.e. 'Dual')



fig1, ax1 = plt.subplots()

ax1.pie(sizes, explode=explode, labels=labelsForTypes, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()
#Let's check the total stats of dual natured, water type pokemon

waterPlusSomeTypePokemons = genOneDual[genOneDual['Type 1']=='Water'].groupby(['Type 1','Type 2']) 

waterPlusSomeTypePokemons['Total'].mean().plot(kind='barh')

plt.subplots_adjust(hspace=10)

plt.ylabel("<----- Natures of water pokemon ----->")

plt.xlabel("<----- No of observations ----->")
waterPlusSomeTypePokemons['Total'].count().plot(kind='bar')

plt.xlabel("<----- Natures of water pokemon ----->")

plt.ylabel("<----- No of observations ----->")
#Dual natured water type vs Single natured water type?? Which ones are better?

#Line graph for the avg value of all the attributes.

waterTypePokemons = genOneSingle[genOneSingle['Type 1']=='Water'].mean()

waterTypePokemons.drop(labels=['#','Total','Generation','Legendary','Type 2'],inplace=True)



waterDualTypePokemons = genOneDual[genOneDual['Type 1']=='Water'].mean()

waterDualTypePokemons.drop(labels=['#','Total','Generation','Legendary'],inplace=True)



waterTypePokemons.plot(kind='line',label='Single Type')

waterDualTypePokemons.plot(kind='line',label='Dual Type')

plt.legend()
#Does all water type pokemons in each generation have the defence points as their best stat?? Let's check.

#Line graph for the avg value of all the attributes.

waterTypePokemons1 = genOnePokemon[genOnePokemon['Type 1']=='Water'].mean()

waterTypePokemons1.drop(labels=['#','Total','Generation','Legendary'],inplace=True)



waterTypePokemons2 = genTwoPokemon[genTwoPokemon['Type 1']=='Water'].mean()

waterTypePokemons2.drop(labels=['#','Total','Generation','Legendary'],inplace=True)



waterTypePokemons3 = genThreePokemon[genThreePokemon['Type 1']=='Water'].mean()

waterTypePokemons3.drop(labels=['#','Total','Generation','Legendary'],inplace=True)



waterTypePokemons4 = genFourPokemon[genFourPokemon['Type 1']=='Water'].mean()

waterTypePokemons4.drop(labels=['#','Total','Generation','Legendary'],inplace=True)



waterTypePokemons5 = genFivePokemon[genFivePokemon['Type 1']=='Water'].mean()

waterTypePokemons5.drop(labels=['#','Total','Generation','Legendary'],inplace=True)



waterTypePokemons6 = genSixPokemon[genSixPokemon['Type 1']=='Water'].mean()

waterTypePokemons6.drop(labels=['#','Total','Generation','Legendary'],inplace=True)



fig = plt.figure(figsize=(16,10))

ax1 = fig.add_subplot(321)

waterTypePokemons1.plot(kind='line',label='gen1 Pokemon')

plt.legend()



ax2 = fig.add_subplot(322, sharey=ax1)

waterTypePokemons2.plot(kind='line',label='gen2 Pokemon')

plt.legend()



ax3 = fig.add_subplot(323,sharey=ax1)

waterTypePokemons3.plot(kind='line',label='gen3 Pokemon')

plt.legend()



ax4 = fig.add_subplot(324,sharey=ax1)

waterTypePokemons4.plot(kind='line',label='gen4 Pokemon')

plt.legend()



ax5 = fig.add_subplot(325,sharey=ax1)

waterTypePokemons5.plot(kind='line',label='gen5 Pokemon')

plt.legend()



ax6 = fig.add_subplot(326,sharey=ax1)

waterTypePokemons6.plot(kind='line',label='gen6 Pokemon')

plt.legend()
typesOfPokemon = genOnePokemon['Type 1'].value_counts().index

typeDict= {}



def getBestAttributeForType(typesOfPokemon):

    for pokemonType in typesOfPokemon:

        pokemonTypeDf = genOnePokemon[genOnePokemon['Type 1']==pokemonType].mean();

        pokemonTypeDf.drop(labels=['#','Total','Generation','Legendary'],inplace=True);

        pokemonTypeDf.sort_values(ascending=False,inplace=True);

        typeDict[pokemonType] = pokemonTypeDf.index[0];

    return typeDict;



typeDict = getBestAttributeForType(typesOfPokemon)

#print(typeDict)



def returnBestAttribute(type):

    return typeDict.get(type)



genOnePokemon['BestAttribute'] = genOnePokemon['Type 1'].apply(returnBestAttribute)
genOnePokemon.head()
temp = genOnePokemon.groupby(['Type 1'])['BestAttribute'].value_counts()

print(temp)
for pokemonType in typesOfPokemon:

    meanSpeed = genOnePokemon[genOnePokemon['Type 1']==pokemonType]['Speed'].mean();

    

typeDict[pokemonType] = meanSpeed;

print(typeDict)