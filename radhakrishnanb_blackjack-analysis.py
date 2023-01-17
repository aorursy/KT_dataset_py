# First, I am importing some packages that I think I may need later on

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
table = list(range(1,10)) + [10]*4

table
draw_count = {str(card_value):table.count(card_value) for card_value in set(table)}

print('Card count in each draw',draw_count)

total_cards = sum(card_count for card_count in draw_count.values())

print('Net cards', total_cards)

draw = {k:(v/total_cards) for k,v in draw_count.items()}

print('Probability of each card:', draw)
def bj_probability(player_count, dealer_count):

    '''Returns a list as follows: [probability player wins if he opts to stay, probability player wins if he opts to hit]'''

    if player_count > 21: return [0,0] # Player busts

    if dealer_count > 21: return [1,1] # Dealer busts

    if dealer_count >=17: 

        prob_win = 1*(player_count > dealer_count) # Player only wins if his count is higher than dealer's 

                                                   # once dealer hits 17 or more

        return [prob_win, prob_win]

    # Here is for other undecided scenarios

    # Stay prob = probability of drawing each card * winning after dealer draws each card and player opts to stay again

    stay_prob = sum(draw[card]  * bj_probability(player_count, dealer_count + int(card))[0]  for card in draw)

    # Hit prob = probability of drawing each card * winning after dealer draws each card 

    # and player decides to play hit or stay depending on max prob of winning 

    hit_prob = sum(draw[card] * max(bj_probability(player_count + int(card), dealer_count)) for card in draw)

    return [stay_prob, hit_prob]
arr = [[None for i in range(23)] for j in range(23)]



def bj_probability_with_dp(player_count, dealer_count):

    '''Returns a list as follows: [probability player wins if he opts to stay, probability player wins if he opts to hit]'''

    if player_count > 21: return [0,0] # Player busts

    if dealer_count > 21: return [1,1] # Dealer busts

    if arr[player_count][dealer_count] is None:

        if dealer_count >=17: 

            prob_win = 1*(player_count > dealer_count) # Player only wins if his count is higher than dealer's 

                                                       # once dealer hits 17 or more

            result =  [prob_win, prob_win]

        else:

            # Here is for other undecided scenarios

            # Stay prob = probability of drawing each card * winning after dealer draws each card and player opts to stay again

            stay_prob = sum(draw[card]  * bj_probability_with_dp(player_count, dealer_count + int(card))[0]  for card in draw)

            # Hit prob = probability of drawing each card * winning after dealer draws each card 

            # and player decides to play hit or stay depending on max prob of winning 

            hit_prob = sum(draw[card] * max(bj_probability_with_dp(player_count + int(card), dealer_count)) for card in draw)

            result = [stay_prob, hit_prob]

        arr[player_count][dealer_count] = result

    return arr[player_count][dealer_count]
%%time 

print(bj_probability_with_dp(0,0))
df = pd.DataFrame(data=arr, columns=list(range(23)))

df.head(10)
for i in range(12,18):

    print(i, 'normal prob', bj_probability(i,i), 'dp prob', bj_probability_with_dp(i,i))
df
df_1 =df.rename_axis('Player_count', axis='rows').rename_axis('Dealer_count', axis='columns').iloc[:-1,:-1]

df_1
def stay_prob(x): return x[0]

df_stay = df_1.applymap(lambda x:stay_prob(x))

df_stay
df_hit = df_1.applymap(lambda x:x[1])

df_hit
plt.figure(figsize=(7,7))

plt.imshow(df_stay.values)

plt.xlabel('Dealer count')

plt.ylabel('Player count')

plt.title('Stay probabilities colormap')

plt.colorbar()

plt.show()
plt.figure(figsize=(7,7))

plt.imshow(df_hit.values)

plt.xlabel('Dealer count')

plt.ylabel('Player count')

plt.title('Hit probabilities colormap')

plt.colorbar()

plt.show()