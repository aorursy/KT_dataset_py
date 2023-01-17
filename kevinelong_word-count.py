phrase = "Now is the time for all good people to come to the aid of their planet!"


# 1. How many letters? (use len, no need for split)

# 2. How many words are there in this phrase? (use split on space and then len on the result)

# 3. How many times does each word appear? (use dict of word keys and integer values)

# 4. How many times does each letter appear? (use dict of letter keys and integer values - no need for split)

# 5. EXTRA ADVANCED Print the letters from most common to least common.
# 1. How many letters? (use len, no need for split)
print(len(phrase))

# def letter_count(phrase):
#     return len(phrase)
# print(letter_count(phrase))
# 2. How many words are there in this phrase? (use split on space and then len on the result)
def word_count(phrase):
    word_list = phrase.split(" ")
    print(word_list)
    return len(word_list)
print(word_count(phrase))

# count = 1
# for c in phrase:
#     if c == " ":
#         count += 1
# print(count)

# 3. How many times does each word appear? (use dict of word keys and integer values)
def word_frequency(phrase):
    output_dict = {}
    word_list = phrase.split(" ")
    
    for word in word_list:
        if word not in output_dict:
            output_dict[word] = 1
        else:
            output_dict[word] += 1
            
    return output_dict
print(word_frequency(phrase))
# 4. How many times does each letter appear? (use dict of letter keys and integer values - no need for split)
def letter_frequency(phrase):
    output_dict = {}
    for letter in phrase:
        letter = letter.upper() # or .lower()
        if letter not in output_dict:
            output_dict[letter] = 1
        else:
            output_dict[letter] += 1
    return output_dict
print(letter_frequency(phrase))
#  5. EXTRA ADVANCED
lf = letter_frequency(phrase)

# print(lf)
# print(lf.items())

list_of_lists = lf.items() # convert dict to list of lists

sorted_letters = sorted(list_of_lists, key=lambda pair: pair[1], reverse=True)

print(sorted_letters)

for outer_index in range(len(sorted_letters)):
    pair = sorted_letters[outer_index]

    print(pair[0], pair[1])
    
#     for inner_item in pair:
#         print(inner_item, end=" ")
#     print("")
    #TODO use .upper() to keep avoid distinguishing captial and lower case letters.
    