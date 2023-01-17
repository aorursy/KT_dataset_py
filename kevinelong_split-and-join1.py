address = "192.168.1.1"
print(address)

parts_list = address.split(".") # CONVERT STRING TO LIST WITH SPLIT

print(parts_list)

#change last part
parts_list[-1] = "128"

print(parts_list)

output = ".".join(parts_list) #CONVERY LIST TO STRING WITH JOIN
print(output)
x = "123"
y = "456"
print( x + y) #concatenate - cat - chain in latin

x = int(x) #convert to int
y = int(y)
print( x + y) # add integers


# L1 - split and print each key=value pair on a separate line using split anf a for loop.
# L2 - total all numbers - using int(text) a second split and access by index left and right side 0 and 1
# L3 - create a function to convert any querystring into a dict
query_string = "abc=123&def=456&hij=789"

#L1
query_string = "abc=123&def=456&hij=789"

parts = query_string.split("&")
print(parts)

for pair in parts:
    print(pair)
# L2 - total all numbers - using int(text) a second split and access by index left and right side 0 and 1
query_string = "abc=123&def=456&hij=789"

parts = query_string.split("&")
total = 0
for pair in parts:
    key_and_value_list = pair.split("=")
#     text_key = key_and_value_list[0]
    text_value = key_and_value_list[1]
    integer_value = int(text_value)
    print(integer_value)
    total += integer_value
print(total)
# L3 - create a function to convert any querystring into a dict
def querystring_to_dict(query_string):
    output = {}
    
    parts = query_string.split("&")
    for pair in parts:
        key_and_value_list = pair.split("=")
        text_key = key_and_value_list[0]
        text_value = key_and_value_list[1]
        
        output[text_key] = text_value
        
    return output

result_dict = querystring_to_dict("abc=123&def=456&hij=789")

print(result_dict)

#how can we add the values now?
total = 0
for key in result_dict:
    value = result_dict[key]
    total += int(value)
print(total)
    