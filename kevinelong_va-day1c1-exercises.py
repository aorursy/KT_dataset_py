data = [123, 456, 789]

quantity = len(data) 
print(quantity)

# total = data[0] + data[1] + data[2]
# print(total)

total = 0
for item in data:
#     total = total + item
    total += item # same as the long hand line above
print(total)
# get average
data = [123, 456, 789,999]

# add them up get total
# then
# then divide the total by the quanity 
quantity = len(data)
total = 0
for item in data:
    total += item
output = total / quantity
print(output)

# get average
data = [22, 33, 77,123]
 
# add them up get total
# then
# then divide the total by the quanity
def get_average(data): #function recieves arguments passed as named parameters
    quantity = len(data)
    total = 0
    for item in data:
        total += item
    output = total / quantity
    return output

a = [22, 33, 77, 45, 45]
b = [11, 33, 45]
c = [55, 45]

r1 = get_average(a)
r2 = get_average(b)
r3 = get_average(c)

print(r1)
print(r2)
print(r3)


data = [123, 456, 789, 999]
print(sum(data)/len(data))
def addTwo(a:int = 0, b:int = 0) -> int:
        return a + b

addResult1 = addTwo()
addResult2 = addTwo(2, 4)

print(addResult1)
print(addResult2)

def twice(text):
    return f"{text}{text}"

twiceResult = twice("abc")
print(twiceResult)

external = "QWE"
print(twice(external))
# TODO - write and call your own function.
def show_name(index=0):
    print(index)
    print("KEVIN")
    print("LONG")
    

show_name()
show_name()

for index in range(100):
    show_name(index)
name = "KEVIN LONG"

def show_codes(text):
    for letter in text:
        number = ord(letter)
        b = bin(number)
        print(letter, number, b)
        
show_codes(name)

#CONVERT CODE ABOVE INTO A FUNCTION TAKES an argument named "text" as a parameter AND USE IT
#Print as above no need to return a value.
# call mutiple times

#L2 extra call once for each item in a list of strings.

def show_numbers(text): #note different variable/identifier (text)
    for letter in text: #note different variable/identifier (text) again
        number = ord(letter)
        b = bin(number)
        print(letter, number, b)

show_numbers("ABC")
show_numbers("abc")
show_numbers("KEVIN")

print(0b1000000)

def show_as_numbers(text):
    for letter in text:
        number = ord(letter)
        b =  bin(number)
        print(letter, number, b)
        
#L2 EXTRA CREDIT LOOP        
data = ["Dallas", "BOB", "GEORGE"]
        
for word in data: 
    show_as_numbers(word)
name = "Tyler McMinn"
def text(name):
    for letter in name:
        number = ord(letter)
        b = bin(number)
        print(letter, number, b)
text(name)
print (letter, number ,b)
print(0b1000)
print(chr(65 + 25))
print(chr(97 - 32 ))

data = [546, 123, 567, 64, 129]

# lowest number in list?


# largest number in list?

data = [546, 123, 567, 978, 129]
# lowest number in list?
#print(min(data))
zero = 0
for lowest_num in data:
    if lowest_num < zero:
        zero = lowest_num
print(lowest_num)
print(zero)
# largest number in list?
#print(max(data))
zero1 = 0
for highest_num in data:
    if highest_num > zero1:
        zero1 = highest_num
print(highest_num)
def minimum(data):
    counter = 0
    for value in data:
        if counter == 0:
            lowest = value
        elif value < lowest:
            lowest = value
        counter += 1
        
    return lowest
print(minimum([441,31,52]))

def maximum(data):
    if len(data) > 0:
        biggest = data[0]
    else:
        biggest = 0
        
    for value in data:
        if value > biggest:
            biggest = value
    return biggest

print(maximum([441,1131,952]))