import random
def is_set(a,b,c):

    a1,a2,a3,a4=a

    b1,b2,b3,b4=b

    c1,c2,c3,c4=c

    same,diff=False,False

    if a1==b1 and b1==c1:same=True

    if a1!=b1 and a1!=c1 and b1!=c1:diff=True

    check1=same or diff

    same,diff=False,False

    if a2==b2 and b2==c2:same=True

    if a2!=b2 and a2!=c2 and b2!=c2:diff=True

    check2=same or diff

    same,diff=False,False

    if a3==b3 and b3==c3:same=True

    if a3!=b3 and a3!=c3 and b3!=c3:diff=True

    check3=same or diff

    same,diff=False,False

    if a4==b4 and b4==c4:same=True

    if a4!=b4 and a4!=c4 and b4!=c4:diff=True

    check4=same or diff

    return(check1 and check2 and check3 and check4)



def find_3rd(a,b):

    a1,a2,a3,a4=a

    b1,b2,b3,b4=b

    if a1==b1:c1=a1

    else:c1=list({1,2,3}-{a1,b1})[0]

    if a2==b2:c2=a2

    else:c2=list({1,2,3}-{a2,b2})[0]

    if a3==b3:c3=a3

    else:c3=list({1,2,3}-{a3,b3})[0]

    if a4==b4:c4=a4

    else:c4=list({1,2,3}-{a4,b4})[0]

    return((c1,c2,c3,c4))

    

card1=(1,1,1,1)

card2=(2,2,3,1)

card3=find_3rd(card1,card2)

print(card3)

is_set(card1,card2,card3)
def create_deck():

    deck=[]

    for a in range(3):

        for b in range(3):

            for c in range(3):

                for d in range(3):

                    card=(a+1,b+1,c+1,d+1)

                    deck.append(card)

    return(deck)
deck=create_deck()

deck
include=[]

exclude=[]

for card in deck:

    if card not in exclude:

        print(f"Including {card}")

        if len(include)>0:

            for i_card in include:

                e_card=find_3rd(card,i_card)

                if e_card in exclude:

                    print(f"{e_card} is already excluded, but creates a set with {card} and {i_card}.")

                else:

                    exclude.append(e_card)

                    print(f"Excluding {e_card}, which creates a set with {card} and {i_card}.")

        include.append(card)



    
master_deck=deck.copy()

trials=1000000

successes=0

for _ in range(trials):

    trial_deck=master_deck.copy()

    random.shuffle(trial_deck)

    cards=trial_deck[0:12]

    setfound=False

    for first in range(10):

        for second in range(10-first):

            #check if their setmatch is in the 12

            if find_3rd(cards[first],cards[second+first+1]) in cards:

                successes+=1

                setfound=True

                break

        if setfound:break

print(f"{successes} deals of 12 cards contained a set out of {trials} trials, for a {100*successes/trials}% success rate.")    



    
