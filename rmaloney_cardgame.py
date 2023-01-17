# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import itertools

import random

import os

from collections import Counter, defaultdict

import more_itertools as mit



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Need a list of players

players = ['Ryan', 'Dave', 'Jim', 'Alphonse', 'Patrick', 'Steph', 'Yan', 'Adam']
suits = ['s', 'c', 'd', 'h']

ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']



# lets build a deck of cards   

deck = []

for rank in ranks:

    for suit in suits:

       card = rank+suit

       deck.append(card)

       

print(deck)  

print(len(deck))     
#shuffle the cards - we can use random.shuffle() for this 

print("Shuffle Up and Deal...")

random.shuffle(deck)

print(deck)
# give each player $1000 to play with

chip_counts = {}

for player in players:

    chip_counts[player] = 1000

    

print(chip_counts)
# each player antes $10 into the pot

pot = len(chip_counts.keys()) * 10

print("pot is: ", pot)



# adjust chip stacks after antes

for player in players:

    chip_counts[player] -= 10

    

print(chip_counts)

print ("\n")
# For each player, deal them their 2 hole cards:

print("Dealing Hole Cards...")

hole_cards = {}



for player in players:

    card1 = deck.pop(0)

    hole_cards[player] = [card1]

 

for player in players:

    card2 = deck.pop(0)

    hole_cards[player].append(card2)



for key, value in hole_cards.items():

    print( key, ' => ', value)
# Deal the Flop

print("Dealing Flop...")



community_cards = []  #create an empty list for community cards



for card in range(1,4):

    community_cards.append(deck.pop(0))

    

print(community_cards)



print( "--------------------------------------")
print("Dealing Turn...")



community_cards.append(deck.pop(0))



print(community_cards)

print( "--------------------------------------" )



# Deal the River Card

print("Dealing River...")



community_cards.append(deck.pop(0))



print(community_cards)
for player in hole_cards.keys():

    hole_cards[player] += community_cards

    

    

print(hole_cards)
print ("HANDS FOR PLAYERS:")

for key, value in hole_cards.items():

    value.sort(key = lambda x: x.split()[0])

    print((key + " => "), value)
# Start evaluating hands

# Need a map to map card ranks to values:

card_order_map = {"2":2, "3":3, "4":4, "5":5, "6":6, "7":7, "8":8, "9":9, "T":10,"J":11, "Q":12, "K":13, "A":14}



# We also need a ranking lookup table for the hands:

hand_rank_map = {

        9:"a straight-flush", 

        8:"four of a kind", 

        7:"a full house", 

        6:"a flush", 

        5:"a straight", 

        4:"three of a kind", 

        3:"two pair", 

        2:"one pair", 

        1:"high card"

 }
# Helper functions to check for existence of specific hands



def check_straight_flush(hand):

    if check_flush(hand) and check_straight(hand):

        return True

    else:

        return False

    

def check_four_of_a_kind(hand):

    values = [i[0] for i in hand]

    counts = Counter(values)

    for count in counts.values():

        if count == 4:

            return True  

    return False



def check_full_house(hand):

    values = [i[0] for i in hand]

    value_counts = Counter(values)

    if set([3,2]).issubset(value_counts.values()):

        return True

    return False



#check_full_house(['Ac','As','Kh','Qd','Kc','9h','3d'])     





def check_flush(hand):

    suits = [h[1] for h in hand]

    counts = Counter(suits)

    for count in counts.values():

        if count >= 5:

            return True  

    return False



#check_flush(['Ac','Ks','4s','7c','Tc','9h','3d'])

        

def check_straight(hand):

    values = [i[0] for i in hand]

    rank_values = [card_order_map[i] for i in values]

    for group in mit.consecutive_groups(rank_values):

        if len(list(group)) >= 5:

            return True

    else: 

        #check straight with low Ace

        if set(values) == set(["A", "2", "3", "4", "5"]):

            return True

        return False

 



def check_three_of_a_kind(hand):

    values = [i[0] for i in hand]

    counts = Counter(values)

    for count in counts.values():

        if count == 3:

            return True

    return False



#check_three_of_a_kind(['Ac','As','Kh','Ad','Tc','9h','3d'])



def check_two_pair(hand):

    values = [i[0] for i in hand]

    value_counts = Counter(values)

    pairs = 0

    for count in value_counts.values():

        if count == 2:

            pairs += 1

    if pairs == 2:

        return True

    return False

  

#check_two_pair(['Ac','As','Kh','Kc','Tc','9h','3d'])



def check_one_pair(hand):

    values = [i[0] for i in hand]

    value_counts = Counter(values)

    if 2 in value_counts.values():

        return True

    else:

        return False
def check_hand(hand):

    if check_straight_flush(hand):

        return 9

    if check_four_of_a_kind(hand):

        return 8

    if check_full_house(hand):

        return 7

    if check_flush(hand):

        return 6

    if check_straight(hand):

        return 5

    if check_three_of_a_kind(hand):

        return 4

    if check_two_pair(hand):

        return 3

    if check_one_pair(hand):

        return 2

    return 1



for player, cards in hole_cards.items():

    print(player, "has", hand_rank_map[check_hand(cards)])

       