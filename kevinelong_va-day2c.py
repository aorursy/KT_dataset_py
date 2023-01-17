# ranges can have steps
for n in range(0,100,10):
    print(n)
# ranges can have negative steps
for n in range(10,0,-1):
    print(n)
#letters in strings can be accessed by index
text = "ABC"
index = len(text) - 1
print(text[index])
letter_list = ["C", "B", "A"]
result = "".join(letter_list)
print(result)
#use len get the length =f the text
# using either ranges and a for loop
# or a counter, and a while loop

#create a function that reverses a string (don't use the pre-made on built into python)
def reverse_text(text):
    output = []

    start = len(text) - 1
    
    #TODO - loop through letters backwards and append them to the output list.
    
    for index in range(start, -1, -1):
        letter = text[index]
        output.append(letter)
        print(output)
        
    return "".join(output)

test_data = "ABC"
expected_result = "CBA"
result = reverse_text(test_data)
print(result)
# assert(result == expected_result)
# or a counter, and a while loop

#create a function that reverses a string (don't use the pre-made on built into python)
def reverse_text(text):
    output = []

    start = len(text) - 1
    
    #TODO - loop through letters backwards and append them to the output list.
    while start >= 0:
        letter = text[start]
        output.append(letter)
        print(output)
        start -= 1
        
    return "".join(output)

test_data = "ABC"
expected_result = "CBA"
result = reverse_text(test_data)
print(result)
print(result == expected_result)
for i in range(10,0,-1): #step must be negative to get from a high starting point to a low ending point.
    print(i)
# 10, 11, 12, 13,