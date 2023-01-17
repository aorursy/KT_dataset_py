import numpy as np 

1
# outcome probabilities from https://wizardofodds.com/games/blackjack/appendix/2a/

np.random.choice(

                 [17, 18, 19, 20, 'Blackjack', 'Bust'], #outcomes

                 p=[0.14583, 0.138063, 0.13482, 0.175806, 0.121896,0.283585]) #outcome probabilities