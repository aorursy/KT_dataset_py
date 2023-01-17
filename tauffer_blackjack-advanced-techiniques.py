# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
deck = 4 * ([str(i) for i in range(2, 11)] + ['J', 'Q', 'K', 'A'])



print(deck)
table_soft_totals = [ # use when player hand has at least one ace

    # 0   1   2   3   4   5   6   7   8   9   10  A

    [' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '], #0

    [' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '], #1

    [' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '], #2

    [' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '], #3

    [' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '], #4

    [' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '], #5

    [' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '], #6

    [' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '], #7

    [' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '], #8

    [' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '], #9

    [' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '], #10

    [' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '], #11

    [' ',' ','H','H','H','H','H','H','H','H','H','H'], #12

    [' ',' ','H','H','H','D','D','H','H','H','H','H'], #13

    [' ',' ','H','H','H','D','D','H','H','H','H','H'], #14

    [' ',' ','H','H','D','D','D','H','H','H','H','H'], #15

    [' ',' ','H','H','D','D','D','H','H','H','H','H'], #16

    [' ',' ','H','D','D','D','D','H','H','H','H','H'], #17

    [' ',' ','D','D','D','D','D','S','S','H','H','H'], #18

    [' ',' ','S','S','S','S','D','S','S','S','S','S'], #19

    [' ',' ','S','S','S','S','S','S','S','S','S','S'], #20

    [' ',' ','S','S','S','S','S','S','S','S','S','S'], #21

]    



table_hard_totals = [ # use when there are no aces

    # 0   1   2   3   4   5   6   7   8   9   10  A

    [' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '], #0

    [' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '], #1

    [' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '], #2

    [' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '], #3

    [' ',' ','H','H','H','H','H','H','H','H','H','H'], #4

    [' ',' ','H','H','H','H','H','H','H','H','H','H'], #5

    [' ',' ','H','H','H','H','H','H','H','H','H','H'], #6

    [' ',' ','H','H','H','H','H','H','H','H','H','H'], #7

    [' ',' ','H','H','H','H','H','H','H','H','H','H'], #8

    [' ',' ','H','D','D','D','D','H','H','H','H','H'], #9

    [' ',' ','D','D','D','D','D','D','D','D','H','H'], #10

    [' ',' ','D','D','D','D','D','D','D','D','D','D'], #11

    [' ',' ','H','H','S','S','S','H','H','H','H','H'], #12

    [' ',' ','S','S','S','S','S','H','H','H','H','H'], #13

    [' ',' ','S','S','S','S','S','H','H','H','H','H'], #14

    [' ',' ','S','S','S','S','S','H','H','H','H','H'], #15

    [' ',' ','S','S','S','S','S','H','H','H','H','H'], #16

    [' ',' ','S','S','S','S','S','S','S','S','S','S'], #17

    [' ',' ','S','S','S','S','S','S','S','S','S','S'], #18

    [' ',' ','S','S','S','S','S','S','S','S','S','S'], #19

    [' ',' ','S','S','S','S','S','S','S','S','S','S'], #20

    [' ',' ','S','S','S','S','S','S','S','S','S','S'], #21

]    



table_splits = [ # used to decide if will split or not

    # 0   1   2   3   4   5   6   7   8   9   10  A

    [' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '], #0

    [' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '], #1

    [' ',' ','Y','Y','Y','Y','Y','Y','N','N','N','N'], #2

    [' ',' ','Y','Y','Y','Y','Y','Y','N','N','N','N'], #3

    [' ',' ','N','N','N','Y','Y','N','N','N','N','N'], #4

    [' ',' ','N','N','N','N','N','N','N','N','N','N'], #5

    [' ',' ','Y','Y','Y','Y','Y','N','N','N','N','N'], #6

    [' ',' ','Y','Y','Y','Y','Y','Y','N','N','N','N'], #7

    [' ',' ','Y','Y','Y','Y','Y','Y','Y','Y','Y','Y'], #8

    [' ',' ','Y','Y','Y','Y','Y','N','Y','Y','N','N'], #9

    [' ',' ','N','N','N','N','N','N','N','N','N','N'], #10

    [' ',' ','Y','Y','Y','Y','Y','Y','Y','Y','Y','Y'], #11 (A)

]



# I could have used the codes directly, but when I decided to use the tables I had already written

# most of the action functions. Creating a dictionary was easier than rewriting the code.

action = {'H':'HIT', 'S':'STAND', 'D':'DOUBLE'}
def shuffle_cards(shoe_size=8):

    """

    Return a 'shoe' with 'shoe_size' decks, shuffled.

    

    In casinos, the dealer uses several decks (usually 6 to 8).

    We can change this later to check how this affects the odds.

    """

    new_shoe = shoe_size * deck

    random.shuffle(new_shoe)

    

    return new_shoe





def hand_value(hand):

    value = 0    

    num_aces = sum([i == 'A' for i in hand])

    

    for card in hand:

        if card.isnumeric():

            value += int(card)

        elif card in ['J', 'Q', 'K']:

            value += 10

        else: # Aces are treated as 1 at first; later they can be "upgraded"

            value += 1

    

    if num_aces and (value <= 11):

        value += 10 # Ace value = 11

        

    return value

    
def dealer_action(player_hand, dealer_hand):

    if hand_value(dealer_hand) < 17:

        action = 'HIT'

    else:

        action = 'STAND'

        

    #print('Dealer ', action)

    return action

def player_action_stand_default(player_hand, dealer_hand):

    # This is the initial strategy presented in the Microchallenge. Always returns 'STAND'

    action = 'STAND'

    #print('player ', action)

    

    return action

def should_hit(player_total, dealer_card_val, player_aces):

    # this is the function I used in the exercise. It will be called from another function to keep same pattern

    if player_aces:

        if player_total >= 19:

            return False

        elif player_total == 18:

            return dealer_card_val >= 9

        else:

            return True

    else: # no aces

        if player_total >= 17:

            return False

        elif player_total >= 13:

            return dealer_card_val >= 7

        elif player_total == 12:

            return (dealer_card_val <= 3) or (dealer_card_val >= 7)

        else:

            return True





def player_action_exercise(player_hand, dealer_hand):

    # This is the best I could get in the exercise. I am creating the function here just to check if my simulations

    # get the same result.

    # Calls the original function without changes, just adjusting the parameters



    player_aces = sum([i == 'A' for i in player_hand])

    player_total = hand_value(player_hand)

    dealer_card_val = hand_value(dealer_hand)

    

    if should_hit(player_total, dealer_card_val, player_aces):

        result = 'HIT'

    else:

        result = 'STAND'

        

    return result
def player_action_with_double(player_hand, dealer_hand):

    # Same one from the exercise, but including the DOUBLE result

    player_aces = sum([i == 'A' for i in player_hand])



    player_total = hand_value(player_hand)

    dealer_total = hand_value(dealer_hand)

    

    #print(player_hand)

    #print(dealer_hand)

    if player_aces:

        result = action[table_soft_totals[player_total][dealer_total]]

    else:

        result = action[table_hard_totals[player_total][dealer_total]]

    

    return result



def can_split(hand):

    return (len(hand) == 2) and (hand_value([hand[0]]) == hand_value([hand[1]]))

def player_action_full(player_hand, dealer_hand):

    # Now with EVERYTHING!!!

    player_aces = sum([i == 'A' for i in player_hand])



    player_total = hand_value(player_hand)

    dealer_total = hand_value(dealer_hand)



    if can_split(player_hand):

        card = hand_value([player_hand[0]])

        if table_splits[card][dealer_total] == 'Y':

            return 'SPLIT'

    

    if player_aces:

        result = action[table_soft_totals[player_total][dealer_total]]

    else:

        result = action[table_hard_totals[player_total][dealer_total]]

    return result

def simulate_one_game (shoe, bet, action_function):

    # since we will consider splits and double-downs the concept of wins/losses does not apply anymore.

    # instead, we will return the final balance.

    

    player_initial_hand = []

    dealer_hand = []

    

    # deal initial cards

    player_initial_hand.append(shoe.pop())

    player_initial_hand.append(shoe.pop())

        

    dealer_hand.append(shoe.pop())



    # to handle splits we will have to consider a list of player hands.

    # everytime a SPLIT happens this list increases

    hands_list = [player_initial_hand]

    

    # dealer´s actions will only happen once, so we need to keep the final hand states to compare

    final_hands = []

    bets = []

    

    # choose which function to use. Will allow us to compare different strategies

    player_action = action_function



    while len(hands_list) > 0:

        #print(hands_list, dealer_hand)

        current_bet = bet

        player_hand = hands_list.pop()

        

        # run player actions

        # first test: should I split?

        if player_action(player_hand, dealer_hand) == 'SPLIT':

            #print('SPLIT')

            

            # create 2 new hands and append to hands list

            hand_1 = [player_hand[0]]

            hand_1.append(shoe.pop())

            

            hand_2 = [player_hand[1]]

            hand_2.append(shoe.pop())

            

            hands_list.append(hand_1)

            hands_list.append(hand_2)

            

            continue # start handling hands list again

            

        # no split here, so let´s process until player stands or bust

        while (hand_value(player_hand) <= 21) and (player_action(player_hand, dealer_hand) != 'STAND'):

            action = player_action(player_hand, dealer_hand)

            #print(action)

            

            if (action == 'HIT'):

                player_hand.append(shoe.pop())

                #print(player_hand, hand_value(player_hand))

            elif (action == 'DOUBLE'):

                # doubles bet and draws last card

                current_bet = bet * 2

                player_hand.append(shoe.pop())

                break

        

        # ended the loop, so include hand and bet in final lists

        final_hands.append(player_hand)

        bets.append(current_bet)

                

    # run dealer actions

    while (dealer_action(player_hand, dealer_hand) != 'STAND'):

        dealer_hand.append(shoe.pop())

        #print(dealer_hand, hand_value(dealer_hand))



    #print(final_hands, dealer_hand, bets)

        

    # evaluate results

    final_balance = 0

    

    for i, hand in enumerate(final_hands):

        current_bet = bets[i]

        

        if hand_value(hand) > 21: # player busted

            #print('Player busted!')

            final_balance -= current_bet

        elif hand_value(dealer_hand) > 21:

            #print('Dealer busted!')

            final_balance += current_bet

        elif hand_value(dealer_hand) < hand_value(hand):

            #print('Player won!')

            final_balance += current_bet

        else:

            #print('Player lost!')

            final_balance -= current_bet 

    

    return final_balance    
def simulate_games(n_games, action_function):

    n_decks = 8

    shoe = shuffle_cards(n_decks)



    reshuffle_rate = 0.20 # when shoe hits this percentage dealer will reshuffle.

    reshuffle_point = len(shoe) * reshuffle_rate 

    



    bet_value = 1

    balance = 0



    for i in range(n_games):

        #print('GAME ', i)

        balance += simulate_one_game(shoe, bet_value, action_function)



        if len(shoe) <= reshuffle_point:

            #print('\n\nShuffling cards...\n\n')

            shoe = shuffle_cards(n_decks)

    

    return balance
def calc_win_rate(balance, n_games):

    # calculates the equivalent win rate from the final balance. Will be useful to compare with results in exercise

    win_ratio = (balance + n_games) / 2 / n_games

    

    return win_ratio
# testing

n_games = 1000000



# Make simulations with different action functions

value_1 = simulate_games(n_games, player_action_stand_default)

value_2 = simulate_games(n_games, player_action_exercise)

value_3 = simulate_games(n_games, player_action_with_double)

value_4 = simulate_games(n_games, player_action_full)



print('Comparing results in different strategies')

print('STAND by default:\t{}%'.format(calc_win_rate(value_1, n_games)*100))

print('Exercise:\t\t{}%'.format(calc_win_rate(value_2, n_games)*100))

print('Exercise w/ doubles:\t{}%'.format(calc_win_rate(value_3, n_games)*100))

print('FULL STRATEGY:\t\t{}%'.format(calc_win_rate(value_4, n_games)*100))