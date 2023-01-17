import json

dict1 = {
    'a': [100,123,345],
    'b': [567,234,176],
    'c': [345,555,567]
}

print(type(dict1))
print(dict1)

text = json.dumps(dict1, indent=8) #dehydrate to text string

print(type(text))
print(text)

output = json.loads(text)

print(type(output))
print(output)
# same thing but also wrting and reading from file

import json

dict1 = {
    'a': [100,123,345],
    'b': [567,234,176],
    'c': [345,555,567]
}

print(type(dict1))
print(dict1)

text = json.dumps(dict1, indent=8) #dehydrate to text string

print(type(text))
print(text)

#WRITE
json_file = open('sample.json', 'w')
json.dump(dict1, json_file)
json_file.close()

#READ
infile = open("sample.json", "r")  
output = json.load(infile) 
infile.close()

print(type(output))
print(output)


