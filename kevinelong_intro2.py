# ADD UP A LIST OF NUMBERS

# 1. input parameters
numbers = [9,6,3,1]

# 2. initialize temporary value
total = 0

# 3. work the list one by one
for n in numbers:
    total = total + n

# 4. return the total value
print(total)

# MORE OF THE SAME, DO IT ALL OVER?

more_numbers = [59,36,83,91,123,456,789]
total = 0
for n in more_numbers:
    total = total + n
print(total)


def get_total(list_of_numbers):
    total = 0
    for n in list_of_numbers:
        total += n    
    return total

def get_total_weird():
    total = 0
    for n in data2:
        total += n    
    return total

print(get_total([9,6,3,1]))
print(get_total([59,36,83,91,123,456,789]))

# print(get_total_weird())

data2 = [9,6,3,1]
result = get_total(data2)
print(result)

print(get_total_weird())

def addTwo(a:int = 0, b:int = 0) -> int:
        return a + b

addResult1 = addTwo()
addResult2 = addTwo(2, 4)

print(addResult1)
print(addResult2)

TAX = 0.08 #OUTER SCOPE

def get_taxed_amount(price):
    tt = price * (1 + TAX) # WE CAN SEE TAX IN OUTER SCOPE
    return tt

print(get_taxed_amount(10.00))

# print(tt) # WE CAN'T SEE THE INNER SCOPE INSIDE THE FUNCTION
def twice(text: str) -> str:
#     return text + " " + text #CONCATENATION #CONCATENATE
    return f"{text} {text}"

twiceResult = twice("abc")
print(twiceResult)
