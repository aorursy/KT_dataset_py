input_data = {
    'abc': 123, 
    'def': 456, 
    'hij': 789, 
    'xyz': 999
}

lines = []
for key in input_data:
    value = input_data[key]
#     print(key, value)
    text = f"{key}={value}"
#     text = key + "=" + str(value)
    print(text)
    lines.append(text)
print(lines)

# new_line_glue = "\n"
new_line_glue = "&"
# new_line_glue = chr(10)

# CORE FUNCTION IS JOIN
output = new_line_glue.join(lines)

print(output)
expected_ouput = "abc=123&def=456&hij=789&xyz=999"

# expected_ouput = """abc=123
# def=456
# hij=789
# xyz=999"""

assert(output == expected_ouput)


# print(input_data)

