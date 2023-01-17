# syntex of range() --> range(start_number, end_number)

# here we are creating list from 1 to 10. 

# PS: last element wount count in list so if i want list till n number then, you have to write n+1 in range function.



num = list(range(1, 11))

print(num)



# output

# [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


programming_languages = ['python', 'java', 'c']



for programming_language in programming_languages: 

	print("I know how to code in " + programming_language.title() + " programming language.")

  

# output

# I know how to code in Python programming language.

# I know how to code in Java programming language.

# I know how to code in C programming language.
for i in range(1,11):

	print(i)



# output

# 1

# 2

# 3

# 4

# 5

# 6

# 7

# 8

# 9

# 10
num = list(range(1,11))



print(num)

# output

# [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]



print(min(num))

# output

# 1



print(max(num))

# output

# 10



print(sum(num))

# output

# 55
# without list comprehension

cubes = []



for i in range(1,11):

	cube = i ** 3

	cubes.append(cube)



print(cubes)



# output

# [1, 8, 27, 64, 125, 216, 343, 512, 729, 1000]


# with list comprehension



cubes = [cube ** 3 for cube in range(1,11)]

print(cubes)



# output

# [1, 8, 27, 64, 125, 216, 343, 512, 729, 1000]
languages = ['English', 'Hindi', 'Chinese', 'Spanish', 'Bengali', 'Russian', 'Arabic', 'Portuguese']



# print list containing - english, hindi, chinese, spanish and bengali

print(languages[0:5])

# output

# ['English', 'Hindi', 'Chinese', 'Spanish', 'Bengali']



# print list contains bengali,russian and arabic

print(languages[4:7])

# output

# ['Bengali', 'Russian', 'Arabic']



# if you want to slice list from starting then no need to mention index because python automatically start from the beginning.

print(languages[:3])

# output

# ['English', 'Hindi', 'Chinese']



# similarly if you want to slice till the end then ignore mentioning index for last element

print(languages[4:])

# output

# ['Bengali', 'Russian', 'Arabic', 'Portuguese']
languages = ['English', 'Hindi', 'Chinese', 'Spanish', 'Bengali', 'Russian', 'Arabic', 'Portuguese']



for language in languages[:5]:

  print(language)



# output

# English

# Hindi

# Chinese

# Spanish

# Bengali
languages = ['English', 'Hindi', 'Chinese', 'Spanish', 'Bengali', 'Russian', 'Arabic', 'Portuguese']



copy_languages = languages[:]



print(languages)

# output

# ['English', 'Hindi', 'Chinese', 'Spanish', 'Bengali', 'Russian', 'Arabic', 'Portuguese']



print(copy_languages)

# output

# ['English', 'Hindi', 'Chinese', 'Spanish', 'Bengali', 'Russian', 'Arabic', 'Portuguese']



my_car = ['audi', 'bmw', 'tesla']

neighbour_car = my_car[:]



del my_car[2]

neighbour_car.append('toyota')



print('My cars are : ')

print(my_car)

# output

# My cars are :

# ['audi', 'bmw']



print('\nMy neighbour\'s cars are : ')

print(neighbour_car)

# output

# My neighbour's cars are :

# ['audi', 'bmw', 'tesla', 'toyota']
dimensions = (100, 20)



print(dimensions[0])

# output

# 100



print(dimensions[1])

# output

# 20
dimensions = (100, 20)

print(dimensions)

# output

# (100, 20)



# assign new values to tuple

dimensions = (500, 100)

print(dimensions)

# output

# (500, 100)


dimensions = (100, 20)



for dimension in dimensions:

  print(dimension)



# output

# 100

# 20
my_information = { 'name': 'Durgesh', 'age': 28 }



print(my_information['name'])

# output

# Durgesh



print(my_information['age'])

# output

# 28


personal = { 'fname' : 'Durgesh', 'lname' : 'Samariya', 'age' : 28 }



print(personal)

# output

# {'fname': 'Durgesh', 'lname': 'Samariya', 'age': 28}



# change fname and lname



personal['fname'] = 'Hello'

personal['lname'] = 'World'



# output

# {'fname': 'Hello', 'lname': 'World', 'age': 28}



# add city as key and Melbourne as value

personal['city'] = 'Melbourne'



# output

# {'fname': 'Hello', 'lname': 'World', 'age': 28, 'city': 'Melbourne'}
personal = { 'fname' : 'Durgesh', 'lname' : 'Samariya', 'age' : 28, 'city':'Melbourne'}



print(personal)

# output

# {'fname': 'Durgesh', 'lname': 'Samariya', 'age': 28, 'city':'Melbourne'} 



# remove city information

del personal['city']



# output

# {'fname': 'Durgesh', 'lname': 'Samariya', 'age': 28} 


personal = { 'fname' : 'Durgesh', 'lname' : 'Samariya', 'age' : 28 }



for key,value in personal.items():

  print(key)

  print(value)

  

# output

# fname

# Durgesh

# lname

# Samariya

# age

# 28
personal = { 'fname' : 'Durgesh', 'lname' : 'Samariya', 'age' : 28 }



for key in personal.keys():

  print(key)

  

# output

# fname

# lname

# age



for value in personal.values():

  print(value)

  

# output

# Durgesh

# Samariya

# 28


person1 = {'name':'Person1', 'age':28}

person2 = {'name':'Person2', 'age':15}

person3 = {'name':'Person3', 'age':40}



persons = [person1, person2, person3]



for person in persons:

  print(person)

  

# output

# {'name': 'Person1', 'age': 28}

# {'name': 'Person2', 'age': 15}

# {'name': 'Person3', 'age': 40}
person = {'name':['Durgesh', 'Samariya'], 'age': 27}



for key,value in person.items():

  print(key)

  print(value)



# output

# name

# ['Durgesh', 'Samariya']

# age

# 27
movies = {

  'avatar': {

    'year': 2009,

    'rating': 5

  },

  'inception' :

  {

    'year': 2010,

    'rating': 5

  },

  'joker' :

  {

    'year': 2019,

    'rating': 4.5

  }

}



print(movies['avatar'])

# output

# {'year': 2009, 'rating': 5}



print(movies['avatar']['year'])

# output

# 2009