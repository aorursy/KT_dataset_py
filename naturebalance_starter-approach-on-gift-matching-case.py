import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input/"))
child=pd.read_csv('../input/child_wishlist_v2.csv',header=None,index_col=0)

childArray=child.values

#childArray=np.fliplr(childArray)
gift=pd.read_csv('../input/gift_goodkids_v2.csv',header=None,index_col=0)

giftArray=gift.values

#giftArray=np.fliplr(giftArray)
submitReader=pd.read_csv('../input/sample_submission_random_v2.csv')

#submitReader=pd.read_csv('../input/santa-gift-matching/sample_submission_random_v2.csv')

submit=submitReader
def childListPosition(whichChild,whichGift):

    temp=np.argwhere(childArray[whichChild]==whichGift)

    if temp.size>0:

        return temp[0][0]+1

    else:

        return 0



def giftListPosition(whichGift,whichChild):

    temp=np.argwhere(giftArray[whichGift]==whichChild)

    if temp.size>0:

        return temp[0][0]+1

    else:

        return 0

    

    

    

def childScore(whichChild,whichGift):

    position=childListPosition(whichChild,whichGift)

    if position>0:

        return (100-position-1)*2

    else:

        return -1



def giftScore(whichGift,whichChild):

    position=childListPosition(whichGift,whichChild)

    if position>0:

        return (1000-position-1)*2

    else:

        return -1    





def score(whichChild,whichGift):

    tempa=childScore(whichChild,whichGift)

    tempb=giftScore(whichGift,whichChild)

    return tempa*1+tempb*1 #replace 1 to weights
def changePosition(positionA,positionB):

    tempa=submit.iloc[positionA][1]

    tempb=submit.iloc[positionB][1]

    submit.iloc[positionA][1]=tempb

    submit.iloc[positionB][1]=tempa

    return     
def currentScore(childA,childB):

    giftA=submit.iloc[childA][1]

    giftB=submit.iloc[childB][1]

    scoreA=score(childA,giftA)

    scoreB=score(childB,giftB)

    return scoreA+scoreB



def predictScore(childA,childB):

    giftA=submit.iloc[childA][1]

    giftB=submit.iloc[childB][1]

    scoreA=score(childA,giftB)

    scoreB=score(childB,giftA)

    return scoreA+scoreB   
def agent(childA,childB):

    tempA=currentScore(childA,childB)

    tempB=predictScore(childA,childB)

    if tempA>tempB:

        return 0

    elif tempA==tempB:

        changePosition(childA,childB)

    elif tempA<tempB:

        changePosition(childA,childB)

        print('childA:'+str(childA)

              +' childB:'+str(childB)

              +' currentScore:'+str(tempA)

              +' predictScore:'+str(tempB)

              +' improved:'+str(tempB-tempA)

             )

    else:

        return 1

        
for i in range(1000000):  # make the number biger 

    dice=np.random.randint(45001,1000000,2)

    agent(dice[0],dice[1])
submit.to_csv('submit.csv',index=None)