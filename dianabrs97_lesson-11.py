###Python Training - Class 11



######Dictionaries



'''

A dictionary is a collection which is unordered, changeable and indexed. 

In Python dictionaries are written with curly brackets, and they have 

keys and values. While the values of a dict can be any Python object, 

the keys generally have to be

immutable objects like scalar types (int, float, string) or tuples 

(all the objects in the tuple need to be immutable, too) 

'''



metal_dict = {'name': 'Mustaine',

              'band': 'Megadeth',

              'album': 'Rust In Piece',

            'year': 1990}
metal_dict
metal_dict['year']
metal_dict['band']
#Adding an element

metal_dict['instrument'] = 'guitar'
metal_dict
#Removing an element

metal_dict.pop('album')
metal_dict
del metal_dict['year']
metal_dict
l = [1, 2, 3, 4]
l[0]
l[3]
metal_dict2 = {

    'name': ['Mustaine', 'Friedman', 'Burton'],

    'band': ['Megadeth', 'Megadeth', 'Metallica'],

    'instrument': ['Guitar', 'Guitar', 'Bass']

}
metal_dict2
#Accesing elements

metal_dict2['name'][0]
metal_dict2
#Copying a dictionary

'''

You cannot copy a dictionary simply by typing dict2 = dict1, 

because: dict2 will only be a reference to dict1, 

and changes made in dict1 will automatically also be made in dict2.

There are ways to make a copy, one way is to use the built-in Dictionary method 

copy()

'''



metal_dict2_copy = metal_dict2.copy()
metal_dict2_copy
#Looping through a dictionary

#Copying a dictionary

'''

You cannot copy a dictionary simply by typing dict2 = dict1, because: dict2 will only be a reference to dict1, 

and changes made in dict1 will automatically also be made in dict2.

There are ways to make a copy, one way is to use the built-in Dictionary method copy()

'''
for key, value in metal_dict.items():

    print(key, value)
for key, value in metal_dict2.items():

    print(key, value)







    

    

    
###Creating a disctionary with lists

music = {'name': ['Frankenstein', 'Waste', 'Make Light'],

         'band':['TPC', 'FTP', 'PP'],

         'album': ['Champ', 'Torches', 'Manners']

         }







###Looping though the dictionary with lists



#By key

for key, value in music.items():

        print(key)

        

        

#By value       

for key, value in music.items():

    for element in value:

        print(value)

        

#By element in list

for key, value in music.items():

    for element in value:

        print(element)

        





###Check if value exists in a given dictionary with lists

word = input('Type the value: ')       

    

for key, value in music.items():

    for element in value:

        if word in element:

            print('{} is in the list'.format(word))

        else:

            print('{} in not in the list'.format(word))
###Exercise 1 - Write a Python program to check if a key already exists in a 

#given dictionary



d = {'1': 100, '2': 200, '3': 300, '4': 400, '5':500, '6':600}



key = input('Type the key: ')



if key in d:

    print('Key is present')

else:

    print('Key is not present')
###Bonus - Check if a value exists in a given dictionary



value = int(input('Type the Value: '))



if value in d.values():

    print('Value is present')

else:

    print('Value is not present')
###Exercise 2 - Write a Python program to sum all items in a dictionary



countries = {'US': 100, 'Canada': 54, 'Mexico': 247}

s = (sum(countries.values()))

print(s)
###Exercise 3 - Write a Python program to remove a key from a dictionary

myDict = {'a': 1, 'b': 2, 'c': 3, 'd': 4}

print (myDict)



if 'b' in myDict:

    del myDict['b']

    print(myDict)

    
###Exercise 4 - Write a Python program to merge two dictionaries



d1 = {'a': 100, 'b': 200}

d2 = {'a': 300, 'y': 400}



d = d1.copy()

d.update(d2)

print(d)
metal_copy = metal_dict2.copy()
metal_dict2
metal_copy
del metal_dict2['band']
metal_dict2
metal_copy
###Tuples

'''

A tuple is a collection which is ordered and unchangeable

In Python tuples are written with round brackets

'''
###Creating a tuple

bands = ('Megadeth', 'Metallica', 'Slayer')

print(bands)
bands[0]
for x in bands:

    print(x)
#Unpack them

tup = (4, 5, 6)

a, b, c = tup
a
b
c
tup2 = (4, 5, (6, 7))

a, b, (c, d) = tup2
###Check if element is in a tuple

if "slayer" in bands:

    print('Yes, it is in the tuple')

else:

    print('No, it is not in the tuple')
#Tuple length

print(len(bands))
#Adding items

bands[3] = "Anthrax"
del bands
print(bands)
#tuple constructor

bands = tuple(('Megadeth', "Metallica", "Slayer"))
bands
import pandas as pd



metal_df = pd.DataFrame(metal_dict2)
metal_df
metal_df['name']
metal_df.iloc[0]
df = pd.read_csv('C:/Python/performance_raw.csv')
df.head(10)
df.columns
df2 = df.rename(columns = {'period': 'Period23',

                          'country': 'Country',

                          'segment': 'Segment',

                          'outstandings': 'Outstandings'})
df2.head(5)