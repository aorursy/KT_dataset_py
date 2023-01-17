import random

import pandas as pd

#import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
def gift_cards(trials):

    remaining_uses=[]

    for t in range(trials):

        first_card,second_card=50,50

        while first_card>=0 and second_card>=0:

            if random.random()<0.5:

                first_card-=1

            else:

                second_card-=1

            #print(f"first {first_card} second {second_card}")

        #print("all done")

        if first_card<0:

            other_card=second_card

        else:

            other_card=first_card

        remaining_uses.append(other_card)

    return(remaining_uses)

ru = gift_cards(1000000)



rus=pd.Series(ru)





both_empty = len(rus[rus==0])

be_pct= both_empty/len(rus)



print(f"In {both_empty} cases of {len(rus)} trials, both cards were empty at the same time.  {be_pct*100}% chance of no covfefe for you.")



print(f"Average free drinks remaining on the other card: {rus.mean()}")
sns.distplot(rus,kde=False)