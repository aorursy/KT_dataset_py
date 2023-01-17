'''

This is the second homework of NLP course.



The aim of the homework is building a pos tagger based on hidden markov model.



Hidden markov model is based on statistics. It looks for the probability of each word's tag 

and probability of previous words' tags of given "n" number(like window size). 



For this project n, assumed 3 according to book.

'''

import nltk

import itertools

from collections import Counter

import pandas as pd
my_sentences = "Thank you for everything"           # Your input

hold = nltk.word_tokenize(my_sentences)         

len_sent = len(hold)                            # Length of your input(word by word)

tags = nltk.corpus.brown.tagged_words()      # Take tagged_words from brown corpus
df = pd.DataFrame(tags, columns=['Word', 'Tag']) # df contains words and their tags taken from brown corpus

previous_degree = pd.DataFrame(columns=["2Dist", "1Dist", "Itself"])

itself = pd.DataFrame(columns=["Tag"])

possible_order = pd.DataFrame(columns= ["Word", "Tag"])

display(df)
i = 0

first_tag = []

second_tag = []

total = 0

probability = {'tags'}

num_of_tag = []

freq_of_tag = []

x = 0
while i < len_sent:                     # Loop for each word

    input_info = df[df['Word'] == hold[i]] # Take the word name and tag from brown corpus where the given "word" is same

    print("\n\nsearching for word --", hold[i], " --")

    if len(input_info) > 10000:    # If brown corpus has the given "word" more than 10000 times

        print("The word -- ", hold[i], " -- found ", len(input_info), "times in brown corpus.")

        input_info = input_info.head(5000)  # Then just take top 10000 of times of the word

    for row in input_info.index:  # Look at the previous 2 words and their tags of the given "word"

        if row == 0:

            previous_degree.loc[row] = ["empty"] + ["empty"]

            itself.loc[row] = df.loc[row]['Tag']

        else:

            previous_degree.loc[row] = [df.loc[row - 2]['Tag']] + [df.loc[row - 1]['Tag']] + [df.loc[row]['Tag']]

            itself.loc[row] = df.loc[row]['Tag']



    freq_of_tag.append(len(previous_degree["2Dist"]))  # Take the length of 2 previous word 

    

    

    probability = dict(Counter(itself.Tag))       # Store the tags of the given "word"

    if len(probability.keys()) >= 1:  # if a tag found in the corpus of the given "word"

        num_of_tag.append(len(probability.keys())) # Take the length of tags

        possible_order = pd.DataFrame.from_dict(probability, orient='index').reset_index()

        possible_order.rename(columns={possible_order.columns[1]: "Frequency", possible_order.columns[0]: "Tag"}, inplace=True)

        # Then store them in possible_order dict

#     print(probability.items())

    print(possible_order)  # possible order includes each of the words' tag, not specific one

    i = i +1



a = []

temp = pd.DataFrame(columns=["Tag", "Probability"])

comb = pd.DataFrame(columns=["Tag", "Probability"])

s = 0
for i in range(len(num_of_tag)):

    if i == 0:   # For first word

        add = possible_order[0 : num_of_tag[i]]

        temp["Tag"] = add["Tag"]

        total = sum(add["Frequency"])

        temp["Probability"] = add['Frequency'].div(total)  # Compute the probability of the tag for word

        temp["Word"] = hold[s]  # Put the given "word" in to dataframe with its possible tag

        for n in range(len(temp["Probability"])):

            if temp["Probability"][n] <= 0.2:  # Drop the tags which have probability less than 0.2. Because if a tag is occur less, than it mean it will be occur less with the other words

                temp = temp.drop([n])

        temp.dropna(inplace=True)

        comb = comb.append(temp)   # Take the remain tags

        add = temp["Tag"]

        s += 1

    else:   # For the other words, do the same like above

        temp.drop(temp.index, inplace=True)

        add = possible_order[num_of_tag[i-1]: num_of_tag[i]]

        add.drop(temp.index, inplace=True)

        temp["Tag"] = add["Tag"]

        total = sum(add["Frequency"])

        temp["Probability"] = add['Frequency'].div(total)

        temp["Word"] = hold[s]

        e = temp.first_valid_index()

        for p in range(e, e + len(temp["Probability"])):

            if temp["Probability"][p] <= 0.2:

                temp = temp.drop([p])

        comb = comb.append(temp)

        add = temp["Tag"] 

        s += 1



    a.append(add)

    comb.append(temp)



display(comb)   # Show each words with all possible tags and their probabilities.
combinations = list(itertools.product(*a))  # Display the combinations of pos tagging

combinations = pd.DataFrame(combinations)



display(combinations)



combinations = combinations.values.tolist()

t = 0

g = 1

multiply = 1

tag_prob = 1

to_multiply = pd.DataFrame(columns=["Tag_prob", "Tags"])

to_pass = pd.DataFrame(columns=["Final"])

se = pd.Series(combinations)

to_pass['Combinations'] = se.values

probabilities = []

while t<len(combinations):  # This loop compares the previous words probabilities

    if len(combinations) == 1:  # If there is only one combination, it means that is the result

        print("The possible pos of", my_sentences, "is", combinations[0])

        t = 1

    else:  # Else, if it has more than 1 combinations

        for x in range(len(combinations)):

            for y in range(len(hold)):  # Look for every previous word, and calculate probabilies

                if y == 0:

                    new_df = previous_degree[0: freq_of_tag[y]]

                    new_df = new_df[new_df['Itself'] == combinations[x][y]]

                    new_df = new_df.reset_index(drop=True)

                    for z in range(len(new_df)):

                        if new_df["2Dist"][z] == "" or new_df["1Dist"][z] == "":

                            g += 1

                    if g == 0:

                        g = 1

                        reach_by_word = comb.loc[comb['Word'] == hold[y]]

                        reach_by_tag = reach_by_word.loc[reach_by_word['Tag'] == combinations[x][y]]

                        tag_prob = reach_by_tag["Probability"]

                        multiply = g * tag_prob

                    else:

                        multiply = multiply * g

                        reach_by_word = comb.loc[comb['Word'] == hold[y]]

                        reach_by_tag = reach_by_word.loc[reach_by_word['Tag'] == combinations[x][y]]

                        tag_prob = reach_by_tag["Probability"]

                        multiply = g * tag_prob

                elif y == 1:

                    new_df = previous_degree[freq_of_tag[y - 1]: freq_of_tag[y]]

                    new_df = new_df[new_df['Itself'] == combinations[x][y]]

                    new_df = new_df.reset_index(drop=True)

                    for z in range(len(new_df)):

                        if new_df["1Dist"][y - 1] == combinations[x][y - 1]:

                            g += 1

                    g = g / len(new_df)

                    reach_by_word = comb.loc[comb['Word'] == hold[y]]

                    reach_by_tag = reach_by_word.loc[reach_by_word['Tag'] == combinations[x][y]]

                    tag_prob = reach_by_tag["Probability"]

                    if g >= 1:

                        multiply = multiply * g * tag_prob

                    elif g == 1:

                        multiply = multiply * g * tag_prob



                elif y >= 2:

                    new_df = previous_degree[freq_of_tag[y - 1]: freq_of_tag[y]]

                    new_df = new_df[new_df['Itself'] == combinations[x][y]]

                    new_df = new_df.reset_index(drop=True)

                    for z in range(len(new_df)):

                        if new_df["2Dist"][z] == combinations[x][y - 2] or new_df["1Dist"][z] == combinations[x][y - 1]:

                            g += 1

                    g = g / len(new_df)

                    reach_by_word = comb.loc[comb['Word'] == hold[y]]

                    reach_by_tag = reach_by_word.loc[reach_by_word['Tag'] == combinations[x][y]]

                    tag_prob = reach_by_tag["Probability"]

                    if g >= 1:

                        multiply = multiply * g * tag_prob # This calculation will be the hmm result

                    elif g == 0:

                        multiply = multiply * g * tag_prob # This calculation will be the hmm result



                g = 1

            probabilities.append(multiply.iloc[0])

            to_pass.loc[x].Final = probabilities[x]

            multiply = 1

        t = t + 1
if len(combinations) > 1:  # The final tagging

    to_pass.sort_values(by='Final', inplace=True)

    print("The possible pos of", my_sentences, "is", to_pass["Combinations"][0])





# You can find the tags of brown corpus from here: http://www.helsinki.fi/varieng/CoRD/corpora/BROWN/tags.html

# You can find a pos tagger from here to see the accuracy: https://parts-of-speech.info/
