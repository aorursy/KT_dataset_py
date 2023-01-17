#Creating a dictionay

thisdict = {

  "brand": "Ford",

  "model": "Mustang",

  "year": 1964

}

print(thisdict)
#Replacing the value of year to 2020

thisdict["year"]=2020

print(thisdict)
#Adding Item

thisdict["color"] = "red"

print(thisdict)
#Removing an item

thisdict.pop("model")

print(thisdict)
#Searching the value using key

if "year" in thisdict:

  print("Yes, 'year' is one of the keys in the thisdict dictionary")