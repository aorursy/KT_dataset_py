import re



def most_frequent_word(input):

    """

    input: list of strings that will be used to generate the vocabulary

    """

    if input == None or len(input) == 0:

        return None

    

    majority_word = None

    counter = 1

    for match in re.finditer(r'\S+', input):

        word = match.group(0)

        if word == majority_word:

            counter += 1

        else:

            if counter == 0:

                # initalize

                majority_word = word

                counter = 1

            else:

                counter -= 1

    return majority_word
print(most_frequent_word(""))
print(most_frequent_word(None))
print(most_frequent_word("A blue shirt cost is twenty-four dollars but a white shirt is only twenty so I bought the white shirt"))
inputs = [

    None,

    "",

    "A blue shirt cost is twenty-four dollars but a white shirt is only twenty so I bought the white shirt",

]



expected_outputs = [

    None,

    None,

    'shirt',

]
for x in zip(inputs, expected_outputs):

    actual_output = most_frequent_word(x[0])

    if actual_output == x[1]:

        print("Match")

    else:

        print("Failed input = " + str(x[0]) + " expected = " + str(x[1]) + " actual = " + str(actual_output))

    