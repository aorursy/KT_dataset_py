name = "Clara"
print(name)
my_list = list(name)
print(my_list)
my_list[-1] = "i"
my_list
my_list.append("f")
my_list.append("y")
print(my_list)
my_list[0] = my_list[0].lower()
print(my_list)
new_list = my_list.copy()
new_list.insert(0, "I")
print(new_list)
new_list.insert(1, " ")
print(new_list)
new_list.pop(1)
print(new_list)
new_list[0] = new_list[0].lower()
print(new_list)
new_list.remove("i")
print(new_list)
"a" in new_list
vowels_list = ["e", "y", "u", "i", "o", "a"]
print(vowels_list)
for k in range(0,len(new_list)):
    if new_list[k] in vowels_list:
        my_letter = new_list[k]
        letter_type = "vowel"
    else:
        my_letter = new_list[k]
        letter_type = "consonant"
    print("The letter {one} is a {two}.".format(one=my_letter, two=letter_type))
