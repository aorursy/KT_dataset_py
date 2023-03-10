# REQUIREMENT: total all numbers found in this string
data = "abc=123&def=456&hij=789"

# how can we use split to access and then add the values together

# list
# split - str.split(sep) return a list (will be called twice)
# sum
# for

# TODO add code

# print(123+456+789)
# 1368

# print(sum([123,456,789]))
pairs = data.split("&")
numbers = []
for p in pairs:
    key_and_value = p.split("=")
    string_value = key_and_value[1]
    int_value = int(string_value)
    numbers.append(int_value)
total = sum(numbers)
print(total)

data2 = """k1:1
k2:"2"
k3:'3'
"""

def string_to_dict(data, line_separator="&", pair_separator="="):
    pass # TODO WRITE FUNCTION HERE
# TEST
output = string_to_dict(data2,"\n",":")
expected_output = {'k1': '1', '2': '2', 'k3': '3'}
assert(output == expected_output)

def string_to_dict(data, line_separator="&", pair_separator="="):
    pairs = data.split(line_separator)
    output_dict = {}
    for p in pairs:
        if p != "":
            key_and_value = p.split(pair_separator)
            string_value = key_and_value[1]
            key = key_and_value[0]
            output_dict[key] = string_value
    return output_dict

data2 = """k1:1
k2:2
k3:3
"""

print(string_to_dict(data2,"\n",":"))

output = string_to_dict(data2,"\n",":")
expected_output = {'k1': '1', 'k2': '2', 'k3': '3'}
assert(output == expected_output)
# sum the original data using the new function
d = string_to_dict(data)
total = 0
for k in d:
    v = d[k]
    i = int(v)
    total += i
print(total)
