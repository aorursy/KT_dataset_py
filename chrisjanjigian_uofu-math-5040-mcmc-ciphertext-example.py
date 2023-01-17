##################################################################################################################

# This is an example for University of Utah MATH 5040 Stochastic Processes and Simulation I (001), Fall 2019.

# It works through using Markov Chain Monte Carlo to decode a simple substitution cipher.

# This is a workthrough of Persi Diaconis' "The Markov Chain Monte Carlo Revolution."

##################################################################################################################



##################################################################################################################

# The basic idea: We will have a text which someone has scrambled using a substitution cipher. Our

# goal will be to recover the original text. We will simplify the problem a bit and assume that all letters are

# upper case and we do not include numbers. A substitution cipher works by replacing one letter with another and

# we will store ciphers dictionaries which maps 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' to some other string.

# For example, the cipher which does not do anything is {A:A, B:B, C:C, ... Z:Z}, while the cipher which

# reverses the order of the alphabet is {A: Z, B: Y, C: W, ... Z:A}. There are 26! ~ 2^88 such ciphers, which

# is to say a lot more than we could reasonably check one at a time.

#

# We will model English writing as a Markov chain on letters, where the next letter only depends on the current

# one. We use a reference text to capture the frequencies with which different pairs of letters appear together

# in the actual English language. We store these, along with transitions to and from all other characters (we replace

# all punctuation with spaces), into a 27x27 transition matrix, which we call M. Given an encrypted string which we 

# call text = 'a_1a_2a_3...' and a proposal cipher, which we call cipher, we can define a likelihood using the transition 

# matrix:

#

#       f_reference(text|cipher) = \Prod_{i=1}^{len(text)-1} M(cipher(a_i), cipher(a_{i+1}))

#

# We follow a Bayesian framework for this problem and will assume that the cipher is drawn uniformly at

# random from the 26! possible ciphers. That is, our prior distribution is f_{prior}(cipher) = 1/26! for any

# cipher. Our question is then to find the mode of the posterior distribution (that is, the cipher

# which is most likely to have given the observed text under this assumption) or at least something close to it,

# where

#

#           f_posterior(cipher|text) = f_reference(text|cipher) f_{prior}(cipher)/f_true(text)

#

# We can then use Metropolis-Hastings to sample approximately from this distribution.

# Our underlying Markov Chain will be what is called a "random walk on the symmetric group."

# All this means is that we will start with a cipher, say the order-reversing cipher which we call old_cipher

# {A:Z, B:Y, C:W, ...},and then we pick uniformly at random two letters and then propose that they swap where

# they map. For example, we might pick b and z, in which case our new cipher, new_cipher would be

# {A:Z, B:A, C:W, ..., Z:Y}. Notice that for any ciphers which differ by one step of this random walk,

# call them cipher_1 and cipher_2 the transition probabilities q(cipher_1,cipher_2) satisfy

# q(cipher_1,cipher_2) = q(cipher_2,cipher_1).We will then sample a Uniform[0,1] random variable and

# accept this swap if

#

#                 f_posterior(new_cipher|text)q(old_cipher,new_cipher)      f_reference(text|new_cipher)

#    Unif[0,1] <  ---------------------------------------------------- =    ----------------------------

#                 f_posterior(old_cipher|text)q(new_cipher,old_cipher)      f_reference(text|old_cipher)

#

# where the equality is true because f_true(text), f_prior(cipher), and the two q probabilities all cancel

# in the numerator and denominator.

#

##################################################################################################################





##################################################################################################################

#

# Run this cell first to see the encrypted text and load all of the functions that are used in the decryption.

#

##################################################################################################################



##################################################################################################################

# We start off by importing the packages we need and defining a few useful functions.

##################################################################################################################



#Loading packages

import numpy as np

import random

from random import shuffle

import string

import math



#Don't forget to seed your random number generator with a good number

random.seed(12345)



#Next we load our training data as well as our hidden text

with open('../input/warpeace/warandpeace.txt', 'r') as reference:

   reference_text=reference.read().replace('\n', '')



with open('../input/hidden-text/hidden.txt', 'r') as hidden:

   hidden_text=hidden.read().replace('\n', '')



with open('../input/hiddentext3/hidden.txt', 'r') as hidden:

   hidden_text_2=hidden.read().replace('\n', '')





hidden_text = hidden_text[0:5000]

hidden_text_2 = hidden_text_2[0:5000]



alphabet = string.ascii_uppercase

list_alphabet = list(alphabet)

alphabet_list = list_alphabet



#The next two functions allow us to switch between thinking of ciphers as dictionaries and thinking of them as strings.

def cipher_string(cipher):

    cipher_st = ''

    for key in alphabet:

        if key in cipher:

            cipher_st = cipher_st + cipher[key]

    return cipher_st



def string_cipher(in_string):

    cipher = {}

    for i in range(len(in_string)):

        cipher[list_alphabet[i]] = in_string[i]

    return cipher



#It will be helpful to be able to generate a random cipher

def random_cipher():

    cipher = {}

    random_index = [[i] for i in range(len(alphabet))]

    shuffle(random_index)

    for i in range(len(alphabet)-1):

        cipher[list_alphabet[i]] = list_alphabet[random_index[i][0]]

    return cipher





# This function takes a text and applies the cipher/key on the text and returns text.

def apply_cipher(text,ci):

    text = list(text)

    new_text = ''

    for char in text:

        if char.upper() in ci:

            new_text +=ci[char.upper()]

        else:

            new_text += char

    return new_text



def create_single_count_dict(text):

    single_count = {}

    data = list(text.strip())

    for i in range(len(data) - 1):

        char = data[i].upper()

        if char not in alphabet_list and char != " ":

            char = " "

        if char in single_count:

            single_count[char] += 1

        else:

            single_count[char] = 1

    return single_count



def create_pair_count_dict(text):

    pair_count = {}

    data = list(text.strip())

    for i in range(len(data) - 1):

        char_1 = data[i].upper()

        char_2 = data[i + 1].upper()

        key = char_1 + char_2

        if char_1 not in alphabet_list and char_1 != " ":

            char_1 = " "

        if char_2 not in alphabet_list and char_2 != " ":

            alpha_j = " "

        if key in pair_count:

            pair_count[key] += 1

        else:

            pair_count[key] = 1

    return pair_count





#The following function gives the transition probabilities and their logs in a text using the previous two functions.

def create_pair_frequency_dict(text):

    frequency_dict = {}

    text_pair = create_pair_count_dict(text)

    text_single = create_single_count_dict(text)

    for i in range(len(list(text_pair.keys())) - 1):

        key = list(text_pair.keys())[i]

        if key[0] in text_single:

            frequency_dict[key] = text_pair[key]/text_single[key[0]]

    return frequency_dict



#The following function gives the log frequencies, which we will use for computation.

def create_pair_log_frequency_dict(text):

    frequency_dict = {}

    text_pair = create_pair_count_dict(text)

    text_single = create_single_count_dict(text)

    for i in range(len(list(text_pair.keys())) - 1):

        key = list(text_pair.keys())[i]

        if key[0] in text_single:

            frequency_dict[key] = math.log(text_pair[key]) - math.log(text_single[key[0]])

    return frequency_dict



#Call this function to create the reference pair frequencies and likelihoods:

reference_pair = create_pair_frequency_dict(reference_text)

likelihood_table = create_pair_log_frequency_dict(reference_text)





#Define a function which gives the log likelihood of a text given a cipher

def get_cipher_log_likelihood(text, in_cipher):

    decrypted_text = apply_cipher(text, in_cipher)

    likelihood = 0

    for i in range(len(decrypted_text)-1):

        char_1 = decrypted_text[i]

        char_2 = decrypted_text[i + 1]

        key = char_1 + char_2

        if key in likelihood_table:

            likelihood = likelihood + likelihood_table[key]

        else: #This is so that things which do not appear in our reference text are strongly penalized.

            likelihood = likelihood - 25

    return likelihood



def get_cipher_score(text, cipher):

    decrypted_text = apply_cipher(text, cipher)

    scored_f = create_pair_count_dict(decrypted_text)

    cipher_score = 0

    for key, value in scored_f.items():

        if key in reference_pair:

            cipher_score += value * math.log(reference_pair[key])

    return cipher_score



#Define a function to generate a proposed swap in our Markov chain.

def generate_swap(cipher): 

    pos1 = random.randint(0, len(list(cipher)) - 1)

    pos2 = random.randint(0, len(list(cipher)) - 1)

    if pos1 == pos2:

        return generate_swap(cipher)

    else:

        cipher = list(cipher)

        pos1_alpha = cipher[pos1]

        pos2_alpha = cipher[pos2]

        cipher[pos1] = pos2_alpha

        cipher[pos2] = pos1_alpha

        return "".join(cipher)



def MCMC_sample_cipher(text, steps):

    current_cipher_st = string.ascii_uppercase

    state_keeper = set()

    best_state = ''

    switched = 0

    score = -1000000

    for i in range(steps):

        proposed_cipher_st = generate_swap(current_cipher_st)

        current_cipher = string_cipher(current_cipher_st)

        proposed_cipher = string_cipher(proposed_cipher_st)

        score_current_cipher = get_cipher_log_likelihood(text, current_cipher)

        score_proposed_cipher = get_cipher_log_likelihood(text, proposed_cipher)

        if math.log(np.random.uniform(low=0,high=1,size=1)) <  score_proposed_cipher - score_current_cipher:

            current_cipher_st = proposed_cipher_st

            switched +=1

        if i % 500 == 0:

            print('Iteration ' + str(i) + ', step ' + str(switched) + ' of the chain : ' + apply_cipher(text, current_cipher)[0:100] + '...')

        if score_current_cipher > score:

            best_state = current_cipher_st

            score = score_current_cipher

    return best_state





#We now pick a random cipher and encrypt our hidden files with it

random_cipher_list = list(string.ascii_uppercase)

random.shuffle(random_cipher_list)

test_cipher_st = "".join(random_cipher_list)

test_cipher = string_cipher(test_cipher_st)

inverse_test_cipher = {v: k for k, v in test_cipher.items()}

encrypted_text = apply_cipher(hidden_text,test_cipher)

encrypted_text_2 = apply_cipher(hidden_text_2,test_cipher)



print('The encrypted texts are: ')

print('')

print(encrypted_text)

print('')

print('and')

print('')

print(encrypted_text_2)
runs = 20000

MCMC = MCMC_sample_cipher(encrypted_text,runs)

print('The best cipher we have found is:')

print(MCMC + '.')

print('The true inverse is:')

print(cipher_string(inverse_test_cipher) + '.')

print('The full unencrypted text is:')

print(apply_cipher(encrypted_text,string_cipher(MCMC)))
runs = 20000

MCMC = MCMC_sample_cipher(encrypted_text_2,runs)

print('The best cipher we have found is:')

print(MCMC + '.')

print('The true inverse is:')

print(cipher_string(inverse_test_cipher) + '.')

print('The full unencrypted text is:')

print(apply_cipher(encrypted_text_2,string_cipher(MCMC)))