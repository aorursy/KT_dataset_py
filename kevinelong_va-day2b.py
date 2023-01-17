
ip = "127.0.0.1"

separator = "." # delimiter
parts = ip.split(separator) # parse a string into a list/array

print(parts)

first = parts[0]
last = parts[-1]

print(first)
print(last)

#loop through and show all
for item in parts:
    print(item)
fragments = ["255","255","255","0"]
glue = "."
text = glue.join(fragments)
print(text)
def break_in_two(text):
    first_half = text[0] + text[1]
    second_half = text[2] + text[3]
    return [first_half, second_half]

print(break_in_two("abcd"))

def user_mac_to_switch_mac(source):
    output = []
    parts = source.split(".")
    for p in parts:
        subparts = break_in_two(p)
        print(subparts)
        for s in subparts:
            output.append(s)
    print(output)
    return ":".join(output)

test_data = "abcd.efgh.ijkl"
result = user_mac_to_switch_mac(test_data)
print(result)

expected_output = "ab:cd:ef:gh:ij:kl"
assert(expected_output == result) # break
print( "ab-cd-ef-gh-ij-kl".replace("-", ":") )
print(":".join("ab-cd-ef-gh-ij-kl".split("-")))

def windows_mac_to_switch_mac(text):
    separator = "-"
    parts = text.split(separator)
    glue = ":"
    result = glue.join(parts)
    return result

print(windows_mac_to_switch_mac("ab-cd-ef-gh-ij-kl"))
print(windows_mac_to_switch_mac("zz-cd-ef-gh-ij-xx"))

query_string = "ab\"'c=123&def=456&ghi=789" # three key value pairs separated by ampersand an then internally equal-sign.
output_dict = {}

#TODO put parsing code here
parts = query_string.split("&")
# print(parts)

for pair in parts:
#     print(pair)
    key_and_value_list = pair.split("=")
#     print(key_and_value_list)
    key = key_and_value_list[0] # zero is the left side
    value = key_and_value_list[1] # 1 is the second item i.e. the right side.
#     print(key)
#     print(value)
    output_dict[key] = value
    
# we will split twice first on & and then later on =
# one for loop
# two place access parts by index 0 and 1 repectively
# key = "zzz"
# value = 999
# output_dict[key] = value

print(output_dict)

expected = {
    "abc": "123",
    "def": "456",
    "ghi": "789",
}
