author = {

   "first_name": "Jonathan",

   "last_name": "Hsu",

   "username": "jhsu98"

}

print(author['username']) # jhsu98

print(author['middle_initial']) # KeyError: 'middle_initial'
print(author.get('username')) # jhsu98

print(author.get('middle_initial',)) # None is by default 
print(author.get('username')) # jhsu98 - get method does the same when the key exists

print(author.get('middle_initial','Oops Mistake!')) # User can define their own in the second parameter
print(author.setdefault('username')) # jhsu98 - it does the same when they key exists 

print(author.setdefault('middle_initial', None)) # it adds the things when there is an error!
print (author)