# Make a dictionary with {} and : to signify a key and a value
my_dict = {'key1':'value1','key2':'value2'}
# Call values by their key
my_dict['key2']
my_dict = {'key1':123,'key2':[12,23,33],'key3':['item0','item1','item2']}
# Let's call items from the dictionary
my_dict['key3']
# Can call an index on that value
my_dict['key3'][0]
# Can then even call methods on that value
my_dict['key3'][0].upper()
my_dict['key1']
# Subtract 123 from the value
my_dict['key1'] = my_dict['key1'] - 123
#Check
my_dict['key1']
# Set the object equal to itself minus 123 
my_dict['key1'] -= 123
my_dict['key1']
# Create a new dictionary
d = {}
# Create a new key through assignment
d['animal'] = 'Dog'
# Can do this with any object
d['answer'] = 42
#Show
d
# Dictionary nested inside a dictionary nested inside a dictionary
d = {'key1':{'nestkey':{'subnestkey':'value'}}}
# Keep calling the keys
d['key1']['nestkey']['subnestkey']
# Create a typical dictionary
d = {'key1':1,'key2':2,'key3':3}
# Method to return a list of all keys 
d.keys()
# Method to grab all values
d.values()
# Method to return tuples of all items  (we'll learn about tuples soon)
d.items()