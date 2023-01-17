#Lesson 2.01 - Boolean

age = int(input("What's your age? "))
resident = input ("Are you a resident (y/n)? ")
print (str(age) + ' ' + resident)


print("Can Mauricio be President:", age >= 35 and resident == "y")
mycart = ["drill bit", "light bulb", "apple", "bread", 8]

finditem = 'apple'

if finditem in mycart:
    print ("FOUND IT! You have " + str(finditem) + "  in your cart.")
else:
    print ("MISSING! You don't have " + str(finditem) + " in your cart.")
foods = ["Ramen", "Steak", "Fish Curry", "Stir Fry Vegetables", "Ice Cream", 'Pizza']
#          0         1         2                3                4             5
foods_rank = [0,0,0,0,0,0]

q1 = input("Do you like noodles? (y/n) ")
if q1 == "y":
    foods_rank[0] = foods_rank[0] + 2

q2 = input("Do you fancy seafood? (y/n) ")
if q1 == "y":
    foods_rank[2] = foods_rank[2] + 1
    
q3 = input("Meat lover? (y/n) ")
if q3 == "y":
    foods_rank[1] = foods_rank[1] + 1
    foods_rank[0] = foods_rank[0] + 1

q4 = input("Good with veggies? (y/n) ")
if q4 == "y":
    foods_rank[3] = foods_rank[3] + 1
    
q5 = input("Do you like Japanese cuisine? (y/n) ")
if q5 == "y":
    foods_rank[0] = foods_rank[0] + 2
    foods_rank[2] = foods_rank[2] + 1
    foods_rank[3] = foods_rank[3] + 1
    
q6 = input("Sweet tooth? (y/n) ")
if q6 == "y":
    foods_rank[4] = foods_rank[4] + 1

q7 = input("Fancy Domino's? (y/n) " )
if q7 == "y":
    foods_rank[len(foods_rank) -1] = foods_rank[len(foods_rank)-1] + 1

q8 = input("Veggies over meat? (y/n) ")
if q8 == "y":
    foods_rank[3] = foods_rank[3] + 1
    foods_rank[0] = foods_rank[0] + 1

foods_rank[:]


#max(enumerate(foods_rank),key=lambda x: x[1])[0]



#1 : Get the highest ranked food item
foods_sorted_list = foods_rank[:]
foods_sorted_list.sort(reverse=True)
foods_sorted_list
#Intro to min() and max()
#1 : Get the highest ranked food item
max(foods_rank)
#min(foods_rank)
ground = ['oil', 'gas', 'sand', 'water']
ground.index('sand')
foods_sorted_list[0] #GET RANK of highest, 1st place fave food

foods_sorted_list[1] #GET RANK of second, 2nd place fave food
#Get the index of food based on the rank

#foods_rank.index(foods_sorted_list[0], 0, len(foods_rank)-1) #DETERMINE what is the actual food ranked as highest?
foods_rank.index(foods_sorted_list[0])
#foods_rank.index(foods_sorted_list[1], 0, len(foods_rank)-1) #DETERMINE what is the actual food ranked as 2nd to the highest?
foods_rank.index(foods_sorted_list[1])
print("Your favorite food is ", foods[foods_rank.index(foods_sorted_list[0])])
print("Your second favorite food is ", foods[foods_rank.index(foods_sorted_list[1])])