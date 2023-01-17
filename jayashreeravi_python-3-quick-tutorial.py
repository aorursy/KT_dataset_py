for x in range(0, 3):
    print ("Hip hip hurray!  %d" % (x))

try: # try statement encloses statements which could throw an exception
    deposit_amount = float(input("Enter deposit amount:\t\t")) # input function chained with float function
    interest_earned = float(input('Enter total interest earned:\t'))
    interest_rate = (interest_earned/deposit_amount) * 100
    print("Rate of interest: \t\t"+ str(round(interest_rate,2))) # round function rounds the result to 2 decimal places
except: # catch all exceptions if there is no name of the exception. Can catch just 'ValueError' in this case
    print("You entered an invalid integer. Please type a valid value")
finally:
    print("this will be executed no matter what") 

americas = set(['Canada', 'United States', 'Mexico', 'Mexico'])
print(americas)
# function definition with one argument, limit 
def generate_even_numbers(limit):  
    '''
    This method generates even numbers from 0 to the value 
    passed in the limit, excluding the limit
    '''
    even_number_list = []  # create an empty list 
    for i in range(limit):
        if(i % 2 == 0):  # remainder 0 indicates even
            even_number_list.append(i) # add to the list
    return even_number_list # return statement and function def ends

   
print(generate_even_numbers(10))   # call the defined function
help(generate_even_numbers) 
# while loop
x = 1.456
while True:
    print ("Integer value of x {:1.0f}".format(x))
    print ("Value of x rounded to 2 decimal points {:.2f}".format(round(x,2)))
    x += 1
    if(x > 3):
        break;


#define function
name = 'My name'
def myfunc(name):
    print("hello there! " +  name) # name from outside the block is replaced with the local name. 

myfunc("Python")
# not keyword
game_over = False
i = 1;
while not game_over:
    print("playing " + str(i) + " time(s)")
    i+=1
    if(i>2):
        break
# sending list to function
topics = ["numpy",'pandas',"seaborn"]
def displayTopics(topics):
    for topic in topics:
        print(topic)
        
displayTopics(topics)
    
# list of lists - 2d 
topics = [['numpy', 1],
         ['pandas',2],
         ['seaborn',3]]

print(topics)
print(topics[1][0])

#tuples are similar to lists except they are immutable so you can't add, remove or set items into a tuple
t_tuples = ('numpy',"pandas","seaborn")
print(t_tuples[0])
a,b,c = t_tuples  #tuples can be unpacked into multiple assignment statements
print(b)
# tuples are more efficnet than lists so for readonly structures use tuples
# string search
message = "Congratulations!  you have are all set to move on to learning numpy and pandas!"
print('you' in message)
print('congratulations' in message) # case sensitive
for char in "study":
    print(char)
print(message.startswith("Congratulations"))
    
# string manipulations
numbers = "12345"
print(numbers.isdigit()) # other functions islower(), isupper(), isalpha()

message = "this is really easy!"
print(message.title())

phoneNumber = "123 234 3489    "
print(phoneNumber.strip() + ".")
print(phoneNumber.replace(" ", "-")) # find returnes the index of the first occurance or -1 if not found
print("(" + phoneNumber[:3] + ")"+ phoneNumber[4:7] + "-"+ phoneNumber[8:13])
print(phoneNumber.split(" "))

print("book".ljust(14), "$9.99".rjust(10)) # justifies to the given length by adding spaces to fill the gap

print(message.upper())

# join() method adds the first string to every part of the second string and is more efficient than + or += with strings and lists
a = " "
b = "355"
print(a.join(b))

# date time
from datetime import date
from datetime import datetime
print(date.today())
print(datetime.now())
peace_day = datetime(1981,9,21, 17, 30)
print(peace_day)
print(peace_day.strftime("%Y/%m/%d"))
#Dictionaries. lists are ordered but dictionaries are unordered collection. Keys are indexed. Key can be any type. 
# Value can be any type including complex types
countries = {'CA': "Canada",
            "US":"United States",
            "MX": "Mexico",
            3:10}
print (countries)
print(countries['CA'])
code = 'US'
if code in countries:
    print(countries[code])
    
print(countries.get("mx")) # case sensitive
print(countries.get("MX"))

countries['IN'] = "India"
print(countries['IN'])
countries['IN'] = 'Bharath'
print(countries['IN']) 
del countries['MX']
print(countries)
countries.pop("IN")  # you can use del, pop methods to remove an item from dictionary. clear() removes all items
print(countries)
print(countries.keys())
print(countries.values())
for name in countries.values():
    print (name)
    
for code,name in countries.items():  # unpack tuples
    print(code , name)
# convert dictionary to list
codes = list(countries.keys())
print(type(codes))
del codes[2] # remove the integer so sort can work

codes.sort() 
print(codes)