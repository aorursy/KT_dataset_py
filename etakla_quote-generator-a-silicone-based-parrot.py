import pandas as pd

import numpy as np

import nltk

from math import log

from collections import defaultdict

import random
quotes = pd.read_csv('../input/All-seasons.csv')
quotes_by_character = quotes.groupby('Character')

quotes_by_character.count()[quotes_by_character.count().Line > 1000]
kyle_quotes = quotes[quotes.Character == "Kyle"].Line
kyle_quotes.head()
kyle_quotes_lower = kyle_quotes.apply(str.lower).apply(str.rstrip, '\n')
kyle_tokens = kyle_quotes_lower.apply(nltk.word_tokenize)
kyle_tokens.head()
kyle_tokens_list =  [ word for inner_list in list(kyle_tokens) for word in inner_list]
kyle_lexical_diversity = len(set(kyle_tokens_list)) / len(kyle_tokens_list)

print(kyle_lexical_diversity)
len(kyle_tokens_list)/len(kyle_tokens)
top_characters = quotes_by_character.count()[quotes_by_character.count().Line > 100].index
#This function will redo all the computation explained above to compute the lexical diversity and the average sentence length

def get_character_params(data, character):

    character_quotes = data[data.Character == character].Line

    character_quotes_lower = character_quotes.apply(str.lower).apply(str.rstrip, '\n')

    character_tokens = character_quotes_lower.apply(nltk.word_tokenize)

    character_tokens_list =  [ word for inner_list in list(character_tokens) for word in inner_list]

    number_of_unique_words = len(set(character_tokens_list))

    character_lexical_diversity = number_of_unique_words / len(character_tokens_list)

    character_avg_sentence_length = len(character_tokens_list)/len(character_tokens)

    

    return [len(character_tokens), len(character_tokens_list),  character_avg_sentence_length, number_of_unique_words, character_lexical_diversity]
top_characters_tokens = []



columns = ['Name', 'Number of Lines','Total Word Count', "Average Sentence Length", 'Unique Words', "Lexical Diversity"]

character_quotes_parameters_df = pd.DataFrame(columns=columns)



for speaker in top_characters:

    temp_entry_dict = {'Name':"", 'Number of Lines':"",'Total Word Count':"", 

                       "Average Sentence Length":"", 'Unique Words':"", "Lexical Diversity":""}

    

    character_params = get_character_params(quotes, speaker)

    

    temp_entry_dict['Name'] = speaker

    temp_entry_dict['Number of Lines'] = character_params[0]

    temp_entry_dict['Total Word Count'] = character_params[1]

    temp_entry_dict['Average Sentence Length'] = character_params[2]

    temp_entry_dict['Unique Words'] = character_params[3]

    temp_entry_dict['Lexical Diversity'] = character_params[4]

    

    character_quotes_parameters_df = character_quotes_parameters_df.append(temp_entry_dict, ignore_index=True)

    

character_quotes_parameters_df.head()
import matplotlib.pyplot  as plt



plt.plot(character_quotes_parameters_df['Total Word Count'], character_quotes_parameters_df['Lexical Diversity'],'ro')



plt.xlabel('Total Word Count', fontsize=16)

plt.ylabel('Lexical Diversity', fontsize=16)



plt.show()
import matplotlib.pyplot  as plt

import math

plt.plot(np.log(character_quotes_parameters_df['Total Word Count'].astype(float)), np.log(character_quotes_parameters_df['Lexical Diversity'].astype(float)),'ro')



plt.xlabel('Ln Total Word Count', fontsize=16)

plt.ylabel('Ln Lexical Diversity', fontsize=16)



plt.show()
import matplotlib.pyplot  as plt



plt.plot(character_quotes_parameters_df['Total Word Count'], character_quotes_parameters_df['Unique Words'],'ro')



plt.xlabel('Total Word Count', fontsize=16)

plt.ylabel('Unique Word Count', fontsize=16)



plt.show()
import matplotlib.pyplot  as plt



plt.plot(np.log(character_quotes_parameters_df['Total Word Count'].astype(float)), np.log(character_quotes_parameters_df['Unique Words'].astype(float)),'ro')



plt.xlabel('Ln Total Word Count', fontsize=16)

plt.ylabel('Ln Unique Word Count', fontsize=16)



plt.show()
kyle_word_freq = nltk.FreqDist(kyle_tokens_list)
kyle_word_log_probability = [ {word: -log(float(count)/ len(kyle_tokens_list))} for word, count in kyle_word_freq.items() ]
kyle_word_log_probability[:10]
#Build n-gram builder

def build_ngram(text, n):

    #sanity check, but generally speaking we want the text to be much much longer than the sentence length to get

    #interesting\funny results

    if len(text) < n:

        print( "Text length is less than n")

        return text

    index = 0

    tokenized_text = nltk.word_tokenize(text)#Try it with lower case, i.e. text.lower()

    

    ngram = defaultdict()

    

    #Loop over all text, except the last n words, since they cannot have n words after

    for index in range(len(tokenized_text) - n):

        #Get current word from the corpus

        current_word = tokenized_text[index]

        

        #Get the next n words, so that we can push them into the current word's entry in the ngram dictionary

        ngram_tail = " ".join(tokenized_text[index + 1 : index+n])

        

        #The general structure of an entry is as follows: the beginning of the ngram is the key. Its contents is a dictionary

        #that contains the total number of grams that are started by this first word, plus another dictionary of all the grams

        #and their counts. To save a little space, only the tail is stored in that last dictionary. That way, we can compute 

        #easily the probability since everything needed is already stored inside

        

        #If this is a new entry, create a new one

        if current_word not in ngram.keys():

            ngram[current_word] = {

                                     'total_grams_start' : 1, 

                                     'grams':  { ngram_tail : 1  } 

                                   }

        else:

            #increase the total count of grams starting with this word

            ngram[current_word]['total_grams_start'] += 1

            #If this ngram tail is new, create a new sub-entry with this ngram

            if ngram_tail not in ngram[current_word]['grams'].keys():

                ngram[current_word]['grams'][ngram_tail] = 1

            #else, increment the entry count by one

            else:

                ngram[current_word]['grams'][ngram_tail] += 1



    return ngram
def Generate_quote(grammed_input, gram_size, start_word, quote_length):

    

    output_str = start_word

    

    #This is like the seed based on which we will pick the next word.

    current_word = start_word.lower()

    

    next_word = ""

    

    #iterate length by gram size times + 1. We want to iterate as much as needed to build a sentence of n size

    for i in range(quote_length//gram_size + 1):

        #We want some randomness in picking the next word, not just pick the highest probable next word. So we are going to

        #set a minimum probability under which the gram is not going to get picked.

        random_num = random.random()

        

        #cumulative probability

        cum_prob = 0

        for potential_next_word, count in grammed_input[current_word]['grams'].items():

            #The cumulative probability is the count of this gram-tail divided by how many time the see word appeared

            cum_prob += float(count)/grammed_input[current_word]['total_grams_start']

            #print cum_prob, random_num

            #If the cumulative probability has reached the minimum probability threshold, then this is the gram to use

            if cum_prob > random_num:

                output_str += ( " " + potential_next_word) 

                current_word = potential_next_word.split()[-1]

                break

            #else, i.e. this gram's probability is lower than our random threshold, get the next gram

            else:

                continue

    # finish with an end of sentence. For now, a sentence ends with a full stop, no question\exclamation marks.

    # The code will continue to generate text until we encounter a gram that ends with a full stop.

    if output_str[-1] != '.':

        #eos = end of sentence

        no_eos = True

        while no_eos:

            cum_prob = 0

            random_num = random.random()

            

            for potential_next_word, count in grammed_input[current_word]['grams'].items():

                cum_prob += float(count)/grammed_input[current_word]['total_grams_start']

                #print cum_prob, random_num

                if cum_prob > random_num:

                    if '.' in potential_next_word:

                        potential_next_word = potential_next_word.split('.')[0]

                        output_str += ( " " + potential_next_word + ".") 

                        no_eos = False

                    else:

                        output_str += ( " " + potential_next_word) 

                        current_word = potential_next_word.split()[-1]

                    break

                else:

                    continue

        

    return output_str
kyle_bigram = build_ngram(' '.join(kyle_tokens_list), 2)
Generate_quote(kyle_bigram, 2, 'i', 12)
kyle_trigram = build_ngram(' '.join(kyle_tokens_list), 3)
Generate_quote(kyle_trigram, 3, 'i', 12)
kyle_quadgram = build_ngram(' '.join(kyle_tokens_list), 4)
Generate_quote(kyle_quadgram, 4, 'i', 12)
kyle_pentgram = build_ngram(' '.join(kyle_tokens_list), 5)
Generate_quote(kyle_pentgram, 5, 'i', 12)