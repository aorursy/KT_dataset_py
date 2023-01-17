from learntools.core import binder; binder.bind(globals())

from learntools.python.ex3 import q7 as blackjack
def should_hit(player_total, dealer_card_val, player_aces):

    return False
blackjack.simulate_one_game()
blackjack.simulate(n_games=90000)
def hard(player_aces):

    if player_aces >= 1:

        return True

    else: return False







def should_hit(player_total, dealer_card_val, player_aces):

    """Return True if the player should hit (request another card) given the current game

    state, or False if the player should stay. player_aces is the number of aces the player has.

    """

    if hard(player_aces)== True and player_total <= 11: return True

    elif hard(player_aces)== True and player_total >= 18: return False

    elif hard(player_aces)== False and player_total <= 15: return True

    elif hard(player_aces)== False and player_total >= 17: return False

    elif hard(player_aces)== True and (4 <= dealer_card_val <= 6) and player_total == 12: return False

    elif hard(player_aces)== True and (4 <= dealer_card_val <= 6) and (13 <= player_total <= 16): return False

    elif hard(player_aces)== False and [(dealer_card_val != 9) or (dealer_card_val != 10) or (dealer_card_val != 11) or (dealer_card_val != 1)]: return False

    else: return True
blackjack.simulate_one_game()
blackjack.simulate(n_games=90000)
def count(player_total):

    if player_total <= 8: return 2

    elif 8< player_total < 13: return 1

    elif player_total >= 17: return -2

    else: return 0

 



def count_d(dealer_card_val):

    if dealer_card_val >= 9: return -1

    elif dealer_card_val < 7: return 1

    else: return 0

    

    

def tot(player_total,dealer_card_val):

    return count_d(dealer_card_val) + count(player_total)
def should_hit(player_total, dealer_card_val, player_aces):

    """Return True if the player should hit (request another card) given the current game

    state, or False if the player should stay. player_aces is the number of aces the player has.

    """

    if tot(player_total,dealer_card_val) > 0: return True

    if tot(player_total,dealer_card_val) < 0: return False

    elif tot(player_total,dealer_card_val) == 0: return True
blackjack.simulate_one_game()
blackjack.simulate(n_games=90000)
def should_hit(player_total, dealer_card_val, player_aces):

    """Return True if the player should hit (request another card) given the current game

    state, or False if the player should stay. player_aces is the number of aces the player has.

    """

    if hard(player_aces)== True and player_total <= 11: return True

    elif hard(player_aces)== True and player_total >= 18: return False

    elif hard(player_aces)== False and player_total <= 15: return True

    elif hard(player_aces)== False and player_total >= 17: return False

    elif hard(player_aces)== True and (4 <= dealer_card_val <= 6) and player_total == 12: return False

    elif hard(player_aces)== True and (4 <= dealer_card_val <= 6) and (13 <= player_total <= 16): return False

    elif hard(player_aces)== False and [(dealer_card_val != 9) or (dealer_card_val != 10) or (dealer_card_val != 11) or (dealer_card_val != 1)]: return False

    elif tot(player_total,dealer_card_val) > 0: return True

    elif tot(player_total,dealer_card_val) < 0: return False

    elif tot(player_total,dealer_card_val) ==0: return True

    else: return False
blackjack.simulate_one_game()
blackjack.simulate(n_games=90000)