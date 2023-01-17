st = 'Print only the words that start with s in this sentence'



for word in st.split():

    if word[0] == 's':

        print(word)



print('='*50)



"""create a list of all numbers between 1 and 50 that are divisible by 3."""

y=[]

for x in range(1,51):

    if x % 3 ==0:

        y.append(x)

print(y)



print('='*50)





"""Use List comprehension to create a list of all numbers between 1 and 50 that are divisible by 3."""

print([x for x in range(1,51) if x%3 == 0])



print('='*50)

"""Go through the string below and if the length of a word is even print "even!"



"""

st = 'Print every word in this sentence that has an even number of letters'



for word in st.split():

    if len(word)%2 == 0: \

            print(word+" <-- has an even length!")



print('='*50)



"""Write a program that prints the integers from 1 to 100. But for multiples of three print "Fizz" instead of the number, and for the multiples of five print "Buzz". For numbers which are multiples of both three and five print "FizzBuzz"."""



for num in range(1,101):

    if num % 3 == 0 and num % 5 == 0:

        print("FizzBuzz")

    elif num % 3 == 0:

        print("Fizz")

    elif num % 5 == 0:

        print("Buzz")

    else:

        print(num)



print('='*50)



"""Use a List Comprehension to create a list of the first letters of every word in the string below:



"""

st = 'Create a list of the first letters of every word in this string'

print([word[0] for word in st.split()])


