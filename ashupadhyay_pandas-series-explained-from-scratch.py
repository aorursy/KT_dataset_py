import pandas as pd
dry_fruits = ['Almonds','Raisins','Cashew Nut','Walnuts']

dry_fruits
pd.Series(dry_fruits)
some_nums = [4,7,9,1,45]

pd.Series(some_nums)
some_booleans = [True,False,True,False,True,True]

pd.Series(some_booleans)
nouns = {"Banana":"Fruit",

        "Cow":"Animal",

        "Magenta":"Colour"}

pd.Series(nouns)



about_me = ['Smart','Intelligent','Humble','Focused']

pd.Series(about_me) # this series is not stored, so we need to store it in a var

s = pd.Series(about_me) # whatever happens at the right side of the equals sign, happens first :-)

# to view the series now, we have to put s on the line

s
s.values #this is an attribute!
s.index
s.dtype # internal pandas lingo for string : 'O'
prices = [44.5,22,64.6]

s = pd.Series(prices)

s
s.sum # it says something to us you see! We need parenthese
s.sum()
s.product()
s.mean()
s.min()
s.max()
fruits = ['Apple','Orange','Plum','Grape','Blueberry']

weekdays = ['Monday','Tuesday','Wednesday','Thursday','Friday']



pd.Series(data=fruits,index = weekdays) # supplying the parameters!
pd.Series(fruits,weekdays) # without supplying the name of parameters! 

# Make sure you have a look at the documentation!
pd.Series(fruits, index=weekdays)
fruits = ['Apple','Orange','Plum','Grape','Blueberry','Strawberry']

weekdays = ['Monday','Tuesday','Wednesday','Thursday','Friday','Monday']
pd.Series(fruits,weekdays)
pokemon = pd.read_csv("../input/pokemon.csv") # this will create a dataframe
my_col = pd.read_csv("../input/pokemon.csv", usecols= ['Pokemon']) # this will create a dataframe
pd.read_csv("../input/pokemon.csv", usecols= ['Pokemon'], squeeze = True) # this will create a Series
google = pd.read_csv("../input/google_stock_price.csv", squeeze = True)

google
pokemon.head() # returns a new Dataframe with only 5 values ;)
google.head() # returns a new Series! ;) # you can also pass a number as a parameter!
pokemon.tail() # returns last 5 rows! you can modify it to print any number of rows from the end
pokemon = pd.read_csv("../input/pokemon.csv", squeeze = True, usecols = ['Pokemon'])

google = pd.read_csv("../input/google_stock_price.csv", squeeze = True)
len(pokemon)
type(pokemon)
dir(pokemon) # this gives us an idea of available functions corresponding to that object!
sorted(pokemon)
list(pokemon)
dict(pokemon)
max(pokemon)
min(pokemon)
pokemon.values
google.index
pokemon.dtype
pokemon.is_unique # checks if all the values are unique
google.is_unique # if all the values are unique
pokemon.ndim
pokemon.shape #(rows,cols)
pokemon.size # rows
pokemon.name
pokemon.head()
pokemon.name = 'Pokemon Monsters'
pokemon.head()
pokemon = pd.read_csv("../input/pokemon.csv", squeeze = True, usecols = ['Pokemon'])

google = pd.read_csv("../input/google_stock_price.csv", squeeze = True)
pokemon.sort_values()
pokemon.sort_values().head()
pokemon.sort_values(ascending=False) # default - True
google.sort_values().head()
google.sort_values(ascending=False).tail()
pokemon = pd.read_csv("../input/pokemon.csv", squeeze = True, usecols = ['Pokemon'])

google = pd.read_csv("../input/google_stock_price.csv", squeeze = True)
google.head()
# this overwrite all the values

google = google.sort_values()
google.head()
google.sort_values(ascending=False).head()
google.head()
google.sort_values(ascending=False, inplace=True)
google.head()
pokemon = pd.read_csv("../input/pokemon.csv", squeeze = True, usecols = ['Pokemon'])

google = pd.read_csv("../input/google_stock_price.csv", squeeze = True)
pokemon.sort_values(inplace= True)
pokemon.head()
pokemon.sort_index(inplace=True)
pokemon.head(3)
pokemon = pd.read_csv("../input/pokemon.csv", squeeze = True, usecols = ['Pokemon'])

google = pd.read_csv("../input/google_stock_price.csv", squeeze = True)
3 in [1,2,3]
200 in [1,2,3]
pokemon.head()
"Bulbasaur" in pokemon
720 in pokemon

721 in pokemon
pokemon.index
100 in pokemon # when writing this, it is equivalent to the next line

100 in pokemon.index
"Bulbasaur" in pokemon.values # now we get it, why we were not able to get a True!!!
"Pikachu" in pokemon.values
pokemon = pd.read_csv("../input/pokemon.csv", squeeze = True, usecols = ['Pokemon'])

google = pd.read_csv("../input/google_stock_price.csv", squeeze = True)
pokemon.head()
pokemon[500]
pokemon[[10,20,30,40]]
pokemon[1:5]
pokemon[1:9:2]

pokemon = pd.read_csv("../input/pokemon.csv",index_col = "Pokemon")
pokemon.head() # our values of the column has now become the index
pokemon = pd.read_csv("../input/pokemon.csv",index_col = "Pokemon", squeeze=True)
pokemon.head()
pokemon[0] # we can continue using this
pokemon['Bulbasaur':'Pikachu']
pokemon = pd.read_csv("../input/pokemon.csv", squeeze = True, index_col = ['Pokemon'])

pokemon.sort_index(inplace = True)

pokemon.head(3)
pokemon.get("Moltres")
pokemon.get(['Moltres','Meowth'])
pokemon.get(key = "aishwarya", default = 'This is not a pokemon')
pokemon.get(key = ["aishwarya","Charizard"], default = 'This is not a pokemon')
google = pd.read_csv("../input/google_stock_price.csv", squeeze = True)

google.head(3)
google.count()
len(google)
google.sum()
google.mean()
google.sum()/google.count()


google.std()
google.min()
google.max()
google.median()
google.mode()


google.describe()
google.max()
google.min()
google.idxmax() #index of the max value
google[3011]
google.idxmin()
google[11]
google[google.idxmin()]
pokemon = pd.read_csv("../input/pokemon.csv", index_col = "Pokemon", squeeze = True)

pokemon.head(3)
pokemon.value_counts().sum()
pokemon.count()
pokemon.value_counts(ascending=True)
google = pd.read_csv("../input/google_stock_price.csv", squeeze = True)

google.head(6)
def classify_performance(number):

    if number < 300:

        return "OK"

    elif number >= 300 and number < 650:

        return "Satisfactory"

    else:

        return "Incredible!"
google.apply(classify_performance).tail()
google.head()
google.apply(lambda stock_price: stock_price + 1) # wont reflect anything until you don't assign it back
google.head()
pokemon_names = pd.read_csv("../input/pokemon.csv", usecols = ["Pokemon"], squeeze = True)

pokemon_names.head(3)
pokemon_types = pd.read_csv("../input/pokemon.csv", index_col = "Pokemon", squeeze = True)

pokemon_types.head(3)
pokemon_names.map(pokemon_types).head(10) # Mapping a Series to Series
pokemon_names = pd.read_csv("../input/pokemon.csv", usecols = ["Pokemon"], squeeze = True)

pokemon_types = pd.read_csv("../input/pokemon.csv", index_col = "Pokemon", squeeze = True).to_dict()
pokemon_names.head()
pokemon_types # this is a dictionary!
pokemon_names.map(pokemon_types) # Mapping a Series to a dictionary!