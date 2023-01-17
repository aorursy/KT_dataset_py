# Libraires
import random
import numpy as np
# Get list of words 
word_list = ['toilet','virus','hospital','netflix','home','news','president','brick','computer''news','media','python']
# Have computer choose a random word
def choose_word(word_list):
    rand_number = random.randint(0,len(word_list)-1)
    return word_list[rand_number]

# Choose word 
word = choose_word(word_list)
print(word)
# Make the hangman stages (taken from https://trinket.io/python/02c914e46c)
Hangman = (
    """
 ------
|     |
|
|
|
|
|
|
----------
"""
    ,
"""
 ------
|     |
|     0
|
|
|
|
|
----------
"""
   ,
"""
 ------
|     |
|     0
|     +
|
|
|
|
----------
"""
    ,
"""
 ------
|     |
|     0
|    -+
|
|
|
|
----------
"""
    ,
"""
 ------
|     |
|     0
|    -+-
|
|
|
|
----------
"""
    ,
"""
 ------
|     |
|     0
|   /-+-
|
|
|
|
----------
"""
    ,
"""
 ------
|     |
|     0
|   /-+-/
|
|
|
|
----------
"""
    ,
"""
 ------
|     |
|     0
|   /-+-/
|     |
|
|
|
----------
"""
    ,
"""
 ------
|     |
|     0
|   /-+-/
|     |
|     |
|
|
----------
"""
    ,
"""
 ------
|     |
|     0
|   /-+-/
|     |
|     |
|    |
|    |
----------
"""
    ,
"""
 ------
|     |
|     0
|   /-+-/
|     |
|     |
|    | |
|    | |
----------
"""
)
# How long is this word?
word_length = len(word)

# Create blanks for the word
guess_word = '_'*word_length

# Create blank letters guessed bank
letters_guessed = []
def check_for_leter_fnc(word,letter_guess):
    length_word = len(word)
    check_for_letter = np.zeros(length_word)
    for i in range(0,length_word):
        if word[i] == letter_guess:
            check_for_letter[i] = 1
            
    return check_for_letter
x = check_for_leter_fnc('hello','l')
print(x)
def find_letters(check_for_letter):
    letter_is_here = []
    for i in range(0,len(check_for_letter)):
        if check_for_letter[i] == 1:
            letter_is_here.append(i)
            
    return letter_is_here
find_letters(x)
def did_you_win(guess_word):
    if '_' in guess_word:
        print('still more letters to find!')
        status = 'playing'
    else: 
        print('YOU WIN!')
        status = 'won'
    return status
turn = 0 
while turn < len(Hangman):
    #print(word)
    print('WORD:')
    print(guess_word)
    print('Word has ' + str(word_length) + ' letters')
    print("you've guessed:")
    print(letters_guessed)
    print(Hangman[turn])
    letter_guess = input('Enter a guess: ')
    check_for_letter = np.zeros(word_length)
    any_letter = check_for_leter_fnc(word,letter_guess)
    
    if 1 in any_letter:
        
        letter_is_here = find_letters(any_letter)
        print(letter_is_here)
        for letter_order in letter_is_here:
            list_guess_word = list(guess_word)
            list_guess_word[letter_order] = letter_guess
            guess_word = ''.join(list_guess_word)
        print(guess_word)
        print('hi')
        status = did_you_win(guess_word)
        if status == 'won':
            print('Congrats!')
            break
    else:
        print('nope wrong guess')
        turn = turn + 1 
        letters_guessed.append(letter_guess)

if turn > len(Hangman)-1:
    print('YOU LOSE!')

    
word
