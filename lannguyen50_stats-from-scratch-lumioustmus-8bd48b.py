data = [ 23, 45, 12, 55, 187, 98, 33]
# Get Total
def get_total(data):
    total = 0
    for value in data:
        total += value
    return total
print(get_total(data))

# Get Minimum Value
def get_Min_Value(data):
    minValue=data[0]
    for value in data:
        if value < minValue:
            minValue = value
    return minValue
print(get_Min_Value(data))

# Get Maximum Value
def get_Max_Value(data):
    maxValue=data[0]
    for value in data:
        if value > maxValue:
            maxValue = value
    return maxValue
print(get_Max_Value(data))

# Get Mean/Average
data = [ 23, 45, 12, 55, 187, 98, 33]
def get_Average_Value(data):
    number=0
    total=0
    for value in data:
        total += value
        number += 1
    print(total)
    print(number)
    return total/number
print(get_Average_Value(data))

def get_Average_Value2(data):
    number=0
    total=get_total(data)
    for value in data:
        number += 1
    print(total)
    print(number)
    return total/number
print(get_Average_Value2(data))



# Get Mode
# Get Median

#Extra get standard deviation (from scratch)

import random
def random_number(sides=6):
    n = random.randint(1,sides)
    return n
print(random_number())
    
def roll_many(quantity=3):
    total=0
    for roll in range(0,quantity):
        total += random_number(6)
    return total
print(roll_many())
def generate(epochs=1000):
    epochsHash={}
    for eachEpochs in range(0,epochs):
        index = roll_many(3)
        if index not in epochsHash:
             epochsHash[index]=0
        epochsHash[index] =  epochsHash[index] + 1
    return epochsHash
print(generate())
        
from operator import itemgetter 
def chart(data):
    print(data)
    data.sort(key=itemgtter())
    for item in data:
        print(item)
data=generate()
chart(data)
    