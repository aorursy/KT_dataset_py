#I have a book on ML with sk-learn and TF, however, I've never gone out and done something on my own until
#now! If you see anything I can improve on PLEASE let me know! I want to be a ML/AI Researcher and I've had 
#so much fun doing this!! 

'''Problem: Given some data about a pokemon (if they're a legendary, their generation, health stats, attack stats,etc.)
can we predict that pokemons type/types? '''

import pandas as pd
import numpy as np

#import data
pok = pd.read_csv('../input/Pokemon.csv')
pok.head()

#Given how some pokemon, such as charmander, don't have secondary types we will have to fill those in with either None or with
#their type 1 value

data = pok.copy()
labels = {'Type 1': data['Type 1'], 'Type 2':data['Type 2'] }
pok_lab = pd.DataFrame(data=labels)
data.drop(columns=['Type 1', 'Type 2'])
#For now, let's split the dataset 
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y =train_test_split(data,pok_lab ,test_size=0.25, random_state=42)

#Filters out pokemon possessing the water attribute

water_types = train_x.where(train_x['Type 1'] == 'Water')
water_types_2 = train_x.where(train_x['Type 2'] == 'Water')
        
water_types = water_types.dropna(how='all')    
water_types_2 = water_types_2.dropna(how='all')

water_types.append(water_types_2)

#However, for now we want to see how having ONLY the water type could impact your stats, so we will observe water_types first!
%matplotlib inline
import matplotlib.pyplot as plt
import plotly
import plotly.offline as py
import plotly.graph_objs as go


wt = water_types.dropna(how='all') #All water type pokemon, with the possibility of another type (i.e. Fighting | Poliwrath)
water_type_only = wt.where(wt['Type 2'].isnull() == True).dropna(how='all') #Pokemon having their ONLY type be water


data = [go.Scatter(x=water_type_only['Name'], y=water_type_only['HP'], mode='markers')]
x_ax = dict( autorange=True,
        showgrid=False,
        zeroline=False,
        showline=False,
        ticks='',
        showticklabels=False)
plotly.offline.init_notebook_mode(connected=True)
py.offline.iplot({"data":data, "layout":go.Layout(title="HP Value of Water Types", xaxis=x_ax)})

#To save repeated time, let's create a function that can plot this for us given the attribute we want to see
#We also want a function to gather x type only from the dataset

def plot_type_vs_attribute(type_data, attribute, p_type):
    data = [go.Scatter(x=type_data['Name'], y=type_data[attribute], mode='markers')]
    x_ax = dict( autorange=True,
        showgrid=False,
        zeroline=False,
        showline=False,
        ticks='',
        showticklabels=False)
    plotly.offline.init_notebook_mode(connected=True)
    t = attribute + " Values of " + p_type + " Types"
    print(t)
    py.offline.iplot({"data":data, "layout":go.Layout(title=t, xaxis=x_ax)})

#Returns pokemon with type matching pokemon_type (could be possible that they have different types as well)
def extract_type(pokemon_data,pokemon_type):
    data = pokemon_data.where(pokemon_data['Type 1'] == pokemon_type)
    data2 = pokemon_data.where(pokemon_data['Type 2'] == pokemon_type)
        
    data.append(data2)
    return data.dropna(how='all')

#Returns the pokemon with ONLY one type
def extract_mono_type(pokemon_data):
    return pokemon_data.where(pokemon_data['Type 2'].isnull() == True).dropna(how='all')


    
#Now let's check out the HP values of fire types to make sure our function works
fire_types = extract_mono_type(extract_type(train_x, 'Fire'))
plot_type_vs_attribute(fire_types, 'HP', 'Fire')

#Seems the functions work alright, now what if we plotted the water types against the fire types? 
water_x = water_type_only.shape[0]
fire_x = fire_types.shape[0]

wx=np.arange(0,water_x,1)
fx = np.arange(0,fire_x,1)

data = go.Scatter(x=wx, y=water_type_only['HP'], mode='markers', name='Water Type', marker={'color':'blue'}, text=water_type_only['Name'].astype(str))
x_ax = dict( autorange=True,
        showgrid=False,
        zeroline=False,
        showline=False,
        ticks='',
        showticklabels=False)
data2 = go.Scatter(x=fx, y=fire_types['HP'], mode='markers', name='Fire Type', marker={'color':'red'}, text=fire_types['Name'].astype(str))
d = [data, data2]
py.offline.iplot({"data":d, "layout":go.Layout(title="HP: Fire vs Water", xaxis=x_ax)})

print(water_type_only['HP'].mean())
print(fire_types['HP'].mean())


#Obviously, water types have more pokemon and a few more outliers with higher HP. Other than that, it's kind of hard to tell who's beating who
#Now let's generalize this plot for 1) all type comparison and 2) attribute based comparison

def plot_all_types(pokemon_types, attribute):
    type_color_map = {'Fire':'red', 'Water':'blue', 'Grass':'green', 'Poison':'purple','Fairy':'pink','Ice':'lightcyan',
                     'Bug':'lightgreen','Normal':'grey', 'Fighting':'brown','Flying':'orange','Dark':'black', 'Ghost':'darkgrey',
                     'Ground':'chocolate', 'Steel':'silver', 'Rock':'bisque', 'Psychic':'deeppink','Electric':'yellow', 'Dragon':'navy'}
    
    x_ax = dict( autorange=True,
            showgrid=False,
            zeroline=False,
            showline=False,
            ticks='',
            showticklabels=False)
    
    final_data = []
    for p_type in pokemon_types:
        num_entries = p_type.shape[0]
        x_axis=np.arange(0,num_entries,1)
        poke_type = p_type.iloc[0]['Type 1']
        
        data = go.Scatter(x=x_axis, y=p_type[attribute], mode='markers', name=poke_type + ' Type', marker={'color':type_color_map[poke_type]}, text=p_type['Name'].astype(str))
        final_data.append(data)
    py.offline.iplot({"data":final_data, "layout":go.Layout(title= attribute + " comparison of all types ", xaxis=x_ax)})
    
    for p_type in pokemon_types:
        print('Mean of ' + p_type.iloc[0]['Type 1']  + " types" + ": " + str(p_type[attribute].mean()))

grass_types = extract_mono_type(extract_type(train_x, 'Grass'))
poison_types = extract_mono_type(extract_type(train_x, 'Poison'))
electric_types = extract_mono_type(extract_type(train_x, 'Electric'))
fairy_types = extract_mono_type(extract_type(train_x, 'Fairy'))
normal_types = extract_mono_type(extract_type(train_x, 'Normal'))
bug_types = extract_mono_type(extract_type(train_x, 'Bug'))
ghost_types = extract_mono_type(extract_type(train_x, 'Ghost'))
ice_types = extract_mono_type(extract_type(train_x, 'Ice'))
psychic_types = extract_mono_type(extract_type(train_x, 'Psychic'))
rock_types = extract_mono_type(extract_type(train_x, 'Rock'))
ground_types = extract_mono_type(extract_type(train_x, 'Ground'))
steel_types = extract_mono_type(extract_type(train_x, 'Steel'))
dark_types = extract_mono_type(extract_type(train_x, 'Dark'))
dragon_types = extract_mono_type(extract_type(train_x, 'Dragon'))
flying_types = extract_mono_type(extract_type(train_x, 'Flying'))

all_types = [fire_types, water_type_only, grass_types, poison_types, electric_types, fairy_types, normal_types, bug_types, ghost_types, 
            ice_types, psychic_types, rock_types, ground_types, steel_types, dark_types, dragon_types, flying_types]

plot_all_types(all_types, 'HP')


    
#Works as intended, and by observing the mean we can see that Normal and flying types tend to have higher HP while poison types
#have the latter

#let's create this plot for the other attributes now! 
plot_all_types(all_types, 'Attack')
plot_all_types(all_types, 'Defense')
plot_all_types(all_types, 'Sp. Atk')
plot_all_types(all_types, 'Sp. Def')
plot_all_types(all_types, 'Speed')
'''Final Results: Flying types on average seem to have a higher HP, Sp. Atk, and speed
while rock tops have a higher defense and attack stat on average, and lastly, steel types have a higher Sp. Def on avg

You may ask, why run this analysis? Well, it's so we can get a sense of what our model may be using for its classification
as well as knowing how types may impact stats

Now, at last, we look for a promising model after doing some data cleanup'''


#Clean up Null type 2's by replacing them with their type 1's
#unfortunately, imputer can't do that for us so we'll have to create our own custom transformer
#Also need to drop ID's, and names

from sklearn.base import TransformerMixin
class cleanTypeTwo(TransformerMixin):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def transform(self, X, y):
        new_x = X['Type 2'].replace(np.nan, X['Type 1'])
        new_y = y['Type 2'].replace(np.nan, y['Type 1'])
        return new_x, new_y
        
    def fit(self):
        return self
    
class cleanStrings(TransformerMixin):
    def __init__(self, X):
        self.X = X
    def transform(self, X):
        return X.drop(columns=['#', 'Name', 'Type 1', 'Type 2'])
    def fit():
        return self

clean = cleanTypeTwo(train_x, train_y)
clean2 = cleanStrings(train_x)

x,y = clean.transform(train_x, train_y)

train_x.update(x)
train_y.update(y)

new_train_x = clean2.transform(train_x)

t_x,t_y = clean.transform(test_x, test_y)
test_x.update(t_x)
test_y.update(t_y)
n_t_x = clean2.transform(test_x)
test_y
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

#forest = RandomForestClassifier(n_estimators=100, random_state=1)
#multi_target_forest = MultiOutputClassifier(forest, n_jobs=-1)
#multi_target_forest.fit(new_train_x, train_y).predict(new_train_x)



for i in range(1,101):
    knn_clf = KNeighborsClassifier(n_neighbors=i)
    knn_clf.fit(new_train_x, train_y)
    y_pred = knn_clf.predict(n_t_x)
    print('NeigghborNumber: ' + "value of i is: " + str(i))
    print(metrics.accuracy_score(test_y,y_pred))
    
#train_y = train_y.drop(columns=['Type 2'])
#test_y = test_y.drop(columns=['Type 2'])

#Unfortunately, muulticlass-multioutput is not supported...before we do metrics and hyperparameter tweaking, let's test it on
#a new instance
#This instance will be Stakataka: a dual rock/steel type 
knn_clf = KNeighborsClassifier(n_neighbors=8)
knn_clf.fit(new_train_x, train_y)
y_pred = knn_clf.predict(n_t_x)
knn_clf.predict(np.array([570,61, 131, 211, 53, 101, 13, 7, 0]).reshape(1,-1))
#Good golly it predicted it correctly! Let's make sure it wasn't in the dataset
pok.tail()
#Phew, it wasn't! So now all that's left is to attempt to tune our hyperparameters and do a true evaluation of our model
#I will think of how I could achieve that, but for now I'll leave it here!