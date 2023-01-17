# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



import random
def translate_position(position):

    if position in ((1,3),(3,1),(1,7),(7,1),(3,9),(9,3),(7,9),(9,7)):return((1,3))

    if position in ((1,5),(5,1),(3,5),(5,3),(5,7),(7,5),(5,9),(9,5)):return((1,5))

    if position in ((1,9),(9,1),(3,7),(7,3)):return((1,9))

    if position in ((2,4),(4,2),(2,6),(6,2),(8,6),(6,8),(4,8),(8,4)):return((2,4))

    if position in ((2,8),(8,2),(4,6),(6,4)):return((2,8))

    if position[0]==position[1]:return('winner')

    print("error")

    print(position)



def moveduck(old_position):

    if old_position==1:new_position=random.choice([2,4])

    if old_position==3:new_position=random.choice([2,6])

    if old_position==5:new_position=random.choice([2,4,6,8])

    if old_position==9:new_position=random.choice([6,8])

    if old_position==2:new_position=random.choice([1,3,5])

    if old_position==4:new_position=random.choice([1,5,7])

    if old_position==8:new_position=random.choice([5,7,9])

    return(new_position)



def twoduckmove(old_position):

    a,b=old_position

    c=moveduck(a)

    d=moveduck(b)

    return(translate_position((c,d)))



def twoduckgame(position=(5,5)):

    turns=0

    while (position!='winner'):

        turns+=1

        #print(f'{turns} {position}')

        position=twoduckmove(position)

    return(turns)

trials=1000000

totalturns=[]

for x in range(trials):

    totalturns.append(twoduckgame((5,5)))

print(f"After {trials} trials, average {sum(totalturns)/len(totalturns)}, max {max(totalturns)}")

print(363/74)
def moveduck_notranslate(old_position):

    if old_position==1:new_position=random.choice([2,4])

    if old_position==3:new_position=random.choice([2,6])

    if old_position==5:new_position=random.choice([2,4,6,8])

    if old_position==7:new_position=random.choice([4,6])

    if old_position==9:new_position=random.choice([6,8])

    if old_position==2:new_position=random.choice([1,3,5])

    if old_position==4:new_position=random.choice([1,5,7])

    if old_position==6:new_position=random.choice([3,5,9])

    if old_position==8:new_position=random.choice([5,7,9])

    return(new_position)



def threeduckmove(old_position):

    a,b,c=old_position

    d=moveduck_notranslate(a)

    e=moveduck_notranslate(b)

    f=moveduck_notranslate(c)

    new_position=(d,e,f)

    if (d==e):

        if (e==f):

            new_position='winner'

    return(new_position)



def threeduckgame(position=(5,5,5)):

    turns=0

    while (position!='winner'):

        turns+=1

        #print(f'{turns} {position}')

        position=threeduckmove(position)

    return(turns)
trials=1000000

totalturns=[]

for x in range(trials):

    totalturns.append(threeduckgame((5,5,5)))

print(f"After {trials} trials, average {sum(totalturns)/len(totalturns)}, max {max(totalturns)}")
