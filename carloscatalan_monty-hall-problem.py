# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import random as rdm

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Monty Hall Game



smart=False

n_games=100



smart_list=[]

fool_list=[]



for k in range (2):

    

    # Set iterations

    for j in range(n_games): 



        # Set randomly items in the doors



        door_items=pd.DataFrame([rdm.random(),rdm.random(),rdm.random()],['Goat 1','Goat 2','Lambo'],['Values'])

        door_items=door_items.sort_values(by=['Values'])



        door=pd.DataFrame([door_items.index[0],door_items.index[1],door_items.index[2]],[1,2,3],['Item'])



        #  ontestant choose a door



        chosen_door=round(rdm.uniform(0.5, 3.4))



        # Host open a door with one of the goats



        if door['Item'][chosen_door] != ('Goat 1'):

            door_open=door.index[door['Item']=='Goat 1']

        elif door['Item'][chosen_door] != ('Goat 2'):

                door_open=door.index[door['Item']=='Goat 2']



 

        # Change you initial door?



        # If change the door

        if smart == True:



            for i in range (len(door)):

                if (i+1 != chosen_door) and (i+1 != door_open):

                    chosen_door=i+1

                    break

            



        # Time to open the door



        # If have a lambo in the door!!!

        if door['Item'][chosen_door] == ('Lambo'):

            if smart==True:

                smart_list.append('Lambo')

            else:

                fool_list.append('Lambo')



        # If not :(    

        else:

            if smart==True:

                smart_list.append('Goat')  

            else:

                fool_list.append('Goat')



    # Now try with change the door                

    smart=True
# Results



data={'Lambos':[smart_list.count('Lambo'),fool_list.count('Lambo')],'Iterations':n_games}

monty_hall_data=pd.DataFrame(data=data,index=['Changing the door','Keep with the first door'])

monty_hall_data
# Graph



plt.figure(figsize=(7,6))

sns.barplot(x=monty_hall_data.index,y=monty_hall_data['Lambos'])