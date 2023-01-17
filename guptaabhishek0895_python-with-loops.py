

## Basic Calculator

num1 = float(input("Enter your num1 : "))
op = input("Enter operator : ")
num2 = float(input("Enter another num2 : "))

if op == "+":
    print(num1+num2)
elif op == "-":
    print(num1-num2)
elif op == "*":
    print(num1*num2)
elif op == "/":
    print(num1/num2)
else:
    print("\n Invalid operator")


# While Loop

i = 1
while i <= 12:
    print(i)
    i += 2
    
print("\nWhile loops works fine")
# Building a guessing game

secret_word = "kaggle"
guess = ""

while guess != secret_word:
    guess = input("Enter input : ")
    
print("\nYou win..!!")
# Building a guessing game with condition

secret_word = "kaggle"
guess = ""
guess_count = 0
guess_limit = 3
out_of_guesses = False


while guess != secret_word and not(out_of_guesses):
    if guess_count < guess_limit:
        guess = input("Enter your word : ")
        guess_count += 1
    else:
        out_of_guesses = True
   
if out_of_guesses:
    print("out of guesses, You Lose")
else:
    print("You Win..!!")
# Getting exponent using user defined function

def pow_num(num1, power):
        print(pow(num1, power))
        

number = float(input("Enter number : "))
power  = float(input("Enter power : "))

pow_num(number, power)
# Getting exponent function using FOR LOOPS 

def raise_to_power(base_num, pow_num):
    result = 1
    for index in range(pow_num):
        result = result*base_num
    return result
print(raise_to_power(3,2))
# Using if statement with or condition
requested_toppings = (('mushrooms'), ('onions'), ('pineapple'))

for toppings in requested_toppings:
    if input("Enter your toppings : ") in toppings:
         print("yes " + toppings + " is there")
    else:
         print("Enter correct toppings")
# using if-elif-else condition with input function 
age = float(input("Enter your age : "))

if age  < 5:
    price = 0
elif age < 10:
    price = 10
else:
    price = 12
    
print("Your admissinon fee is $" + str(price)+ ".")
# Building a color game. 
# If you shot a green colored alien you will earned 5 points:
# else if it is yellow you will get 2 rewards point:
# else you will earned 1 points

alien_color = input("Enter your color : ")

if alien_color == "green":
    earned = 5
elif alien_color == "yellow":
    earned = 2
else:
    earned = 1
    
print("You earned " + str(earned) + " points")
age = float(input("Enter age to know the life stage : "))

if age < 2:
    stage = "a baby"
elif age >= 2 and age < 13:
    stage = "a kid"
elif age >= 13 and age < 20:
    stage = "a teenager"
elif age >= 20 and age < 65:
    stage = "an adult"
elif age >= 65:
    stage = "an elder"
        
print("\nYou are " + str(stage) + "..!")