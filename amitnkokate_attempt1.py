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
import numpy as np

import pandas as pd



import os



print(os.listdir("../input/icc-test-cricket-runs/"))



a = pd.read_excel('../input/icc-test-cricket-runs/ICC Test Bat 3001.xlsx')



s = a.loc[:,['Player']]



df = ((s.Player.str.split('(').str[1]).str.split(')').str[0])

country =[["India"],["Aus"],["Eng"],["Sl"],["Pak"],["Sa"],["Wi"],["Nz"]]

for player in df:

    if player =="INDIA"or player =="ICC/INDIA":

    	(country[0]).append(player)

    elif player =="AUS"or player =="ICC/AUS":

    	(country[1]).append(player)

    elif player =="ENG"or player =="ICC/ENG":

    	(country[2]).append(player)

    elif player =="SL"or player =="ICC/SL":

    	(country[3]).append(player)

    elif player =="PAK"or player =="ICC/PAK":

    	(country[4]).append(player)

    elif player =="SA"or player =="ICC/SA":

    	(country[5]).append(player)

    elif player =="WI"or player =="ICC/WI":

    	(country[6]).append(player)

    elif player =="NZ"or player =="ICC/NZ":

    	(country[7]).append(player)

i=0

for i in range(0,8):

    p = len(country[i]) -1

    print(p)

    i=i+1
