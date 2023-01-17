a = 3
b = 2
print(a * b)
print(6 * 7)

print(2 ** 65)
text = "Hello World"
print(text)
text = 3.14
print(text)
number: int = 123
print(number)
print( 1 // 3)
# print( 25 > 21)
# print( 42 == 6 * 7)
# print( 42 == 6 * 7)
age = 21
limit = 21
result =  not age == limit
print(result)
text = "apple"
print(text)
print(text[0])
print(text[-1])
print(text[2])
# SLICE :
print(text[1:4])
print(text[2:])
print(text[:4])
coin_quantity = {
    1:3,
    5:6,
    10: 8,
    50: 3
}

coin_quantity[25] = 2 # I have 2 quarters

print(coin_quantity[25]) # how many quarters do i have?

print(coin_quantity[1]) # how many pennies do i have?
print(coin_quantity[5]) # how many nickels do i have?

print(coin_quantity[10]) # how many dimes do i have?

key = 50
print(f"I have { coin_quantity[key] } of { key } which value is { key * coin_quantity[key] } ")

quantity = coin_quantity[key]
denomination = key
total = quantity * denomination
print(f"I have { quantity } of { denomination } which value is { total } ")

oneThroughFive = range(1,6)
# up to but not including the number on the right.
# [1,2,3,4,5]

zeroToFour = range(0,5) 
# Up to but not including the number on the right.
#  [0,1,2,3]
print(list(oneThroughFive))
print(list(zeroToFour))

# Loop through a range 0 to 4:
for index in range(0, 5):
    print(index)

# Loop through a list of names:
people = [ "Bob", "Carol", "Ted", "Alice", "Kevin" ]
for person in people:
    print( person )
print("DONE")
# Loop until we reach zero:

countDownFrom = 10 # Set the initial value

# Enter the loop only if the expression is true.
while countDownFrom >= 1:
    print( countDownFrom )
    # Increment the count
    countDownFrom = countDownFrom - 1 
print("BLAST OFF")

age = 20

if age >= 21:
    print( "May Drink." )
else:
    print( "May Not Drink." )

command = "sit"

if command == "speak":
    print("woof")
elif command == "sit":
    print("thump")
else:
    print("???")
    


command = "speak"

if command == "speak":
    print("woof")

if command == "sit":
    print("thump")

if command not in ["sit","speak"]:
    print("???")

