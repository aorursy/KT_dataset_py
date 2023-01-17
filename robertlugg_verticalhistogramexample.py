string = "XXY YY ZZZ123ZZZ AAA BB C"

string = sorted(string)

letters = {}

for char in string:

    if char.isupper() and char.isalpha():

        if char in letters:

            letters[char] += 1

        else:

            letters[char] = 1

while any(value > 0 for key, value in letters.items()):

    for key, value in letters.items():

        max_len = max(letters.values())

        if value == max_len:

            # PROBLEM PRINTING HISTOGRAM VERTICALLY

            letters[key] -= 1

print(' '.join([k for k in letters.keys()]))

string = "XXY YY ZZZ123ZZZ AAA BB C"

string = sorted(string)

letters = {}

for char in string:

    if char.isupper() and char.isalpha():

        if char in letters:

            letters[char] += 1

        else:

            letters[char] = 1

print(letters)
this_row = list()

for key, value in letters.items():

    if value >=6:

        this_row.append('*')

    else:

        this_row.append(' ')

print(''.join(this_row))
def build_a_row(distribution_dict, height):

    this_row = list()

    for key, value in distribution_dict.items():

        if value >= height:

            this_row.append('*')

        else:

            this_row.append(' ')

    return ''.join(this_row)



for count in range(max(letters.values()),0,-1):

    print(build_a_row(letters, count))



print(''.join(letters.keys()))
max([x for x in letters.values()])