from learntools.core import binder; binder.bind(globals())
from learntools.python.ex3 import q7 as blackjack

def should_hit(player_total, dealer_total, player_aces):
    """Return True if the player should hit (request another card) given the current game
    state, or False if the player should stay. player_aces is the number of aces the player has.
    """
    if player_total <=11 and dealer_total >0 and player_aces>0:
        return True
    elif player_total <=0 and dealer_total >0 and player_aces==1:
        return True
    elif player_total <= 14 and (dealer_total < 3 or dealer_total > 7):
        return True
    else:
        return False

blackjack.simulate(n_games=1000000)
    