#Your Code Goes Here

#Looping through characters of a string



s2 = 'Toronto'



for x in s2:

    print(x)

    
#Your Code Goes Here

#Looping through a range of numbers using the range() function

#The range function - specify the number of times you loop through

#It returns a sequence of numbers, starting from 0, by default, and increments

# by 1, default, and ends at a specified number



for x in range(0, 7):

    print(x)
#Your Code Goes Here

for x in range(3, 100, 5):

    print(x)
### Your Code Goes Here

### With the while loop, we can execute a set of statements as long as a condition is true



i = 1

while (i < 6):

    print(i)

    i +=1

    #i = i+1
#Your Code Goes Here



i = 0

while(i < 6):

    print(i)

    if (i == 4):

        break

    else:

        i+=1
i = 0

while (i < 6):

    i+= 1

    if(i == 3):

        continue

    print(i)
x = 'Toronto'

for x in s2:

    print(x)

    if (x == 'o'):

        break
#Your Code Goes here

# Translate this code to do the same thing but in a while loop

# Hint run this code first to understand what it is doing for each iteration



for i in range(0, 6):

    

    if (i == 3 or i == 6):

        continue

    else:

        print(i)

        

##Answer

i = -1

while(i < 6):

    i +=1

    if (i == 3 or i == 6):

        continue

    else:

        print(i)