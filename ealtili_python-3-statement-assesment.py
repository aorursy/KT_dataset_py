st = 'Print only the words that start with s in this sentence'
#Code here
st = st.lower()
for word in st.split():
    if word[0] == "s":
        print(word)
        
#Code Here
for num in range(10):
    if num % 2 == 0:
        print(num)

print(list(range(0,11,2)))
#Code in this cell
[num for num in range(1,51) if num % 3 == 0]
print(list(range(0,51,3)))
st = 'Print every word in this sentence that has an even number of letters'
#Code in this cell
st = st.lower()
for word in st.split():
    if len(word) % 2 == 0:
        print(word)
        print("even")
import random
for num in range(10):
    print(random.randint(0,10))
    
[random.randint(0,100) for n in range(0,10)]
#Code in this cell
for num in range(1,101):
    if num % 3 == 0 and num % 5 == 0:
        print("FizzBuzz")
    elif num % 3 == 0:
        print("Fizz")
    elif num % 5 == 0:
        print("Buzz")
    else:
        print(num)
        
st = 'Create a list of the first letters of every word in this string'
#Code in this cell
print([word[0] for word in st.split()])
