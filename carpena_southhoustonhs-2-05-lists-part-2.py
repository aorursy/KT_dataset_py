#Recall index notation

number_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#              0  1  2  3  4  5  6  7  8  9
#            -10 -9 -8 -7 -6 -5 -4 -3 -2 -1 
#Let's recall our number_list
number_list
number_list[0:3]
number_list[0:4:2]
number_list[1:5]
number_list
number_list[-7:-2]
#Above negative index slice is also the same as this:
number_list[3:8]
number_list[0:0]
#Display end to end
number_list[:]
number_list[1:]
number_list[:-5]
number_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
number_list[2:-1:2]
number_list[::-1]
Avengers = ['Thor', 'Groot', 'Starlord', 'Capt Rogers', 'Hawkeye', 'Ironman', "Thanos", "Gamora", 'Black Panther', 'Hulk']
#Get the first word of the sentence
com_sentence = "add buy orange juice"
firstword = com_sentence.split()[0] #default separator is a white space

print (firstword)
#Get the first word of the sentence
com_sentence = "add+buy orange juice"






firstword = com_sentence.split('+')[0] #default separator is a white space

print (firstword)
MH = Avengers.copy()
MH = Avengers[:]


print(len(Avengers))
Avengers[:]
Avengers[0:6]
Avengers[2:-3]
print(Avengers[:-1])
_random_places = ['HEB', 'Walmart', "Chick-fil-A", 'Discount Tire', "FR Depot", 'Taco Bell', 'Kohls', 'McDonalds', "HEB", 'Arbys']
_random_places.count('HEB')
_random_places[0:1]
_random_places[0:-1]
print(_random_places[0:4])
print(_random_places[0:4:2])
_random_places[3:-1]
_random_places[:]
MH = Avengers[:]
# MH is short for Marvel Heroes

MH[:]
MH.reverse()
MH
MH.sort()
MH
MH.append('Doctor Strange')
MH
MH.insert(0, 'Odin')
MH
MH.remove('Starlord')
MH
MH.pop()
MH.pop(3)
MH
MH.append('Thor')
MH
MH.count('Thor')


MH
MH.clear()
Avengers[:]
MH = Avengers[:]
print(MH[:])
#Snapped by Thanos
a_list = ['a', 'b', 'c', 'd', 'e']

print(len(a_list))

#print(a_list[:2])
print(a_list[-2:])
amazon_cart = ["Perforating gun", "drill bits", "water tanks", "H2S detectors", "FID"]
# Print first 3 items in cart
print(amazon_cart[:3])
# Print all items in list
print(amazon_cart[:])
import random

winning_number = random.choice(number_list)
winning_number
