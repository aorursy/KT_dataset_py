#  This is a comment that will be ignored. 
6 + 7
"""
This is
a multiline
comment!
"""

print("Hello World")

#  this is also ignored

print("Goodbye World")
a = 3 + 4
z = a - 3
print( z )
print( 2 * 8 / 2 + 80 * (9 - 6) + z )
number: int = 123
z = number * 7
half: float = 0.5
third: float = 1.0 / 3.0
print( half )
print( third )
print( z )

quantity = 5000
price: int = 15.99
output = f"QUANTITY:{ quantity } / PRICE:${ price } / TOTAL:${ quantity * price }"
print(output)

isGreater = 3>2
print(isGreater)
isSame = 2 + 2 == 4
print(isSame)
print( 3 is 4 )
print( 2 == 4)
print( 2 + 4 == 6)
# Arrary of integers
numbers = [ 12, 23, 54 ]

# Arrary of Strings
names = ["Bob", 
         "Carol",
         "Ted", 
         "Alice"]
assert(len(names) == 4)
print(names)
print(names)


phrase_book: {str:str} = {
    "alpha" : "A as in Alpha",
    "bravo" : "B as in Bravo",
    "charlie" : "C as in Charlie"
}

phrase = phrase_book[ "bravo" ]

assert("B as in Bravo" == phrase)
print(phrase)
coin_quantities:{str:int} = {}

coin_quantities["Penny"] = 10
coin_quantities["Nickel"] = 5
coin_quantities["Dime"] = 4
coin_quantities["Quarter"] = 10

quarter_count = coin_quantities["Quarter"]

assert(10 == quarter_count)
print(coin_quantities["Nickel"])
a = 5
b = 1
oneThroughFive = range(b,a)

print(list(oneThroughFive))

for index in range(0, 5):
    print(index)
    
countDownFrom = 40

while countDownFrom >= 1:
    print( countDownFrom )
    countDownFrom = countDownFrom - 2
print("DONE")

teams = ["Golden Knights", "Avalance", "Kraken"]

for t in teams:
    print(t)
teams = ["Golden Knights", "Avalance", "Kraken"]

for t in teams:
    print(t)
    

age = 13

if age > 100:
    print( "May Not Drink.")
elif age >= 21:
    print( "May Drink.")
else:
    print( "May Not Drink.")

print("...")
def addTwo (a:int = 0, b:int = 0) -> int: 
    return a + b

addresult1 = addTwo()
addresult2 = addTwo(2, 4)

print(addresult2)
def game_over():
    print("Game Over!")
    
for n in range(2):
    game_over()
    
def full_name(first_name, last_name):
    return f"{last_name}, {first_name}"

print(full_name("Kevin", "Long"))
def city_state(state, city):
    return f"{city}, {state}"

result = city_state("Washington", "Seattle")
print(result)

print(city_state)
def get_smallest(numbers: [int]) -> int:
    smallest = None
   
    for  item in numbers:
       
        if smallest is None or item > smallest:
            smallest = item
 
    
    return smallest


p = [123,597,631,61,93,509]


print(get_smallest(p))
def total(numbers):
    subtotal = 0
    for n in numbers:
        subtotal += n
    return subtotal / len(numbers)

print(total([1, 2, 3, 4, 5]))
        
def reverse_me(text):
    output = ""
    for letter in text: 
        output = letter + output
    return output

result = reverse_me("reverse_me")
print(result)