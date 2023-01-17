raw_text = "abc=123&def=456&hij=789&xyz=999"
print(raw_text)

# convert to dictionary
output = {} # empty dictionary

pairs = raw_text.split("&")
print(pairs)

key_and_value_dict = {}

# one at a time through the list
for p in pairs:
    print(p)
    key_and_value = p.split("=")
    print(key_and_value)
    key = key_and_value[0]
    value = key_and_value[1]
    print(key)
    print(value)
    key_and_value_dict[key] = value
    print(key_and_value_dict)    

print(key_and_value_dict)    
# bonus: add up values in dictionary?
values = key_and_value_dict.values()
value_list = list(values)
print(values)
print(value_list)
number_list = []
for text in value_list:
    number_list.append(int(text))
print(sum(number_list))
