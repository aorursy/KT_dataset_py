import os

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt



# Reading in the data

amazon_file_name = os.path.join("../input/forest-fires-in-brazil/amazon.csv")

amazon = pd.read_csv(amazon_file_name, encoding='latin1')



# Removing unhelpful columns

amazon.drop('date',axis=1, inplace=True)

amazon.head()
# Function for generating a line chart consisting of number of fires per year for a given state



def plot_line_by_state(state):

    plt.figure(figsize=(20,7))

    state_data = amazon.loc[(amazon['state']==state)]

    by_year = state_data[['year','number']].groupby('year',as_index=False).sum()

    p = plt.plot('year', 'number', data=by_year, color='red')

    plt.title('Number of Fires Per Year in {}'.format(state), fontsize=20)

    plt.xlabel('Year', fontsize=14)

    plt.ylabel('Number of Fires', fontsize=14)

    plt.show()

print(amazon.state.unique())

state = input('Enter a state from the list above: ')

plot_line_by_state(state)



    
# save graph

plt.savefig('Number of Fires Per Year in {}.pdf'.format(state))