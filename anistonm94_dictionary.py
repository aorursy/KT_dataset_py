#Dictionary
#use of: Dictionary name={"key1":"value1","key2":"value2",...}

#1.other use of (with list and one value (object)): Dictionary naem={"key1":["values1"],"key2":"values"}
dictionary_1={"Norway":"Stavanger","Finland":"Helsinki","France":["Nice","Montpeiller","Lyon","Paris"]}
print("1.",dictionary_1)
#2.to find the keys to the dictionary;
print("2.",dictionary_1.keys())
#3.to find the values to the dictionary;
print("3.",dictionary_1.values())
#4.If we want to change the city of Norway as an example;
dictionary_1["Norway"]="Harstad"
print("4.",dictionary_1)
#5.If we want to add another country and city;
dictionary_1["Denmark"]="Copenhagen"
print("5.",dictionary_1)
#6.If you are writing the value of its key to look for any value.
print("6.",dictionary_1.pop("Finland"))
#7.to remove any key-value pair randomly;
print("7.",dictionary_1.popitem())
print("result",dictionary_1)
#8.completely to erase;
print("8.",dictionary_1.clear())   
#dictionary example about numbers
dictionary_2={81:9,49:7,25:5,9:3,1:1}
print("1.",dictionary_2)
dictionary_2[121]=11
print("2.",dictionary_2)
print("3.",dictionary_2.popitem())
print("4.",dictionary_2.pop(25))
print("5.",dictionary_2.clear())