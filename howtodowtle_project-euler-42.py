from math import sqrt
from string import ascii_uppercase
import pandas as pd
def word_to_number(word):
    """Takes a word, matches its single letters to a dictionary
    value in LETTERS (which matches "A":1 ... "Z":26) and sums
    up the individual numbers to create the sum of letter numbers.
    """
    word_sum = sum([LETTERS[letter] for letter in word])
    return word_sum
def check_if_triangle(x):
    """Checks if a number is a triangle number by above definition.
    Finds the candidate for n (which has to lie between sqrt(2*x)-1 
    and sqrt(2*x) mathematically if the number in fact is a triangle
    number), then checks if it in fact is.
    """
    n_cand = int(sqrt(2*x))  # integer smaller than or equal to sqrt(2*number
    is_triangle = (n_cand*(n_cand+1)/2 == x)
#     if is_triangle:
#         print(f"{x} is the {n_cand}th triangle number!")
#     else:
#         print(f"{x} is not a triangle number. Sorry! :(")
    return is_triangle
def check_word(word):
    """Checks if a word is a triangle word
    by combining the first two functions.
    """
    is_triangle = check_if_triangle(word_to_number(word))
    return is_triangle
x = 50
n = int(sqrt(2*x))

print(f"If {x} is a triangle number, then n is {n}.")
check_if_triangle(55)
ascii_uppercase
LETTERS = {letter : index+1 for index, letter in enumerate(ascii_uppercase)} 

LETTERS
word = "JULE"
word_sum = sum([LETTERS[letter] for letter in word])
word_sum
word = "ALEX"
check_if_triangle(word_to_number(word))
words = pd.read_csv("../input/p042_words.txt", header=None, dtype=str).values[0]
triangles = sum([check_word(word) for word in words])
triangles