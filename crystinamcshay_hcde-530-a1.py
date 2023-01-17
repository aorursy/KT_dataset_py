lst = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1] 

otherlst = ['a','b','c','d','e','f','g']

s = "This is a test string for HCDE 530"



#Exercise 1 (working with a list):

#a.	Print the first element of lst (this one has been completed for you)

print(lst[0])



#b.	Print the last element of otherlst

print(otherlst[-1]) 

#prints 7th (last) letter in list



#c.	Print the first five elements of lst

print(lst[0:6])

#prints 1st-5th element in list but not the 6th one



#d.	Print the fifth element of otherlst

print(otherlst[4])

#prints the 5th element of the list, but coding starts counting with 0



#e.	Print the number of items in lst

print(len(lst))

#prints the lenth of this list



#Exercise 2 (working with a string):

#a.	Print the first four characters of s

print(s[0:5])

#prints the first through 4th character of the string, but not the 5th



#b.	Using indexing, print the substring "test" from s

print(s[10:14])

#prints the 10th-14th character, it also counts the spaces



#c.	Print the contents of s starting from the 27th character (H)

print(s[27:])

#Printed from 27 to the end of the string 



#d.	Print the last three characters of s

print(s[-3:])

#Printed starting 3 from the end, to end of the string



#e.	Print the number of characters in s

print(len(s))
# Factorial 1 * 2 * 3 * 4 ... * 13 = 6227020800



n = 13   

Product = 1



for x in range(1,n + 1):   # I had to get some youtube help for this one. Used range(start,stop). This increases the multiplier by one until it hits the factorial of 13 

    Product = Product * x   # Mulitply product by a value that increases by one for each loop 

    

print(Product)

a = "Happy "

b = "New "

c = "Year!"



# Defines each variable as a string, I added spaces for legibility



print(a + b + c)



#printed each string together
def word1():

    return "Happy"

def word2():

    return "New"

def word3():

    return "Year!"

# returns each function with a word   

    

print(word1())

print(word2())

print(word3())



#prints each function
def sentence():

    print(word1(),word2(),word3())  #prints each function in a sentence format 



sentence() #calls the function so it shows below
def total(x, y):     #two parameters within total, x and y

    print(x, "+", y, "=", x + y)    #This organizes them in the correct format. I kept forgetting the commas here before.

    

total(3,4)    #this calls the function with the numbers of my choice