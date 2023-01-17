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
def recurse_list(length, ones):

    newlist=[]

    #base case

    if length==1:

        if ones==1:newlist=[[1]]

        else:newlist=[[0]]

        return(newlist)

    else:

        newlist=[]

        if ones>0:

            applist=recurse_list(length-1,ones-1)

            for l in applist:

                newlist.append([1]+l)

        if ones<length:

            applist=recurse_list(length-1,ones)

            for l in applist:

                newlist.append([0]+l)

    return newlist

    
recurse_list(9,5)
def check_ttt_winners(list_xo):

    x_win, o_win=False,False

    xstring, ostring="loses","loses"

    windices=[[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]]

    for windex in windices:

        if sum([list_xo[i] for i in windex])==3:

            x_win=True

            xstring="wins"

        if sum([list_xo[i] for i in windex])==0:

            o_win=True

            ostring="wins"

    print(f"{list_xo} X {xstring}, O {ostring}")

    return(x_win,o_win)

        
check_ttt_winners([0,0,0,0,0,0,0,1,1])

x_total,o_total,double_win_total=0,0,0

for board in recurse_list(9,5):

    x,o=check_ttt_winners(board)

    x_total+=x

    o_total+=o

    double_win_total+=(x*o)

print(f'{len(recurse_list(9,5))} total possibilities, {x_total} X winners, {o_total} O winners, {double_win_total} double winners')
(98-36)/126