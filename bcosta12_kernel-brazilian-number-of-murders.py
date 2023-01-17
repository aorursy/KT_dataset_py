import numpy as np

import pandas as pd

import os
# if you want to run localy, download the dataset and use the commented dataset variable

# dataset = 'brazilian-number-of-murders-1996-to-2016-ipea.csv'



dataset = os.listdir("../input")[0] # running on kaggle

print('dataset: {}'.format(dataset))
df = pd.read_csv('../input/'+dataset, sep=';')

df.head(5)
unique_states = np.unique(df['nome'])

number_of_unique_states = len(unique_states)



print('List of States: {}.'.format(unique_states))

print('There are {} unique states.'.format(number_of_unique_states))
last_year = max(np.unique(df['período']))

first_yeat = min(np.unique(df['período']))

delta_year = last_year - first_yeat + 1



df_count = df.groupby(['nome']).size().reset_index(name='contagem')

df_count['validacao'] = df_count['contagem'] == delta_year

validation_count = df_count['validacao'].sum()



print('Range of years in the dataset: {}.\nNumber of stated equaly represented: {}'.format(delta_year, validation_count))
df_negative_validation = pd.DataFrame()

df_negative_validation['negative'] = df['valor'] >= 0

len_dataset = delta_year* 27 # the number of sample should be

number_of_nonpositives = df_negative_validation.negative.sum()

print('Number of non-negatives number of murders {}. Len of dataset {}. check: {}'.format(number_of_nonpositives, len_dataset, number_of_nonpositives == len_dataset))



#print('Number of stated equaly represented: {}'.format(validation_count))
import matplotlib.pyplot as plt
def plt_murderes_per_state(df, state, color='C0'):

    df_state = df.loc[df['nome'] == state]

    

    %matplotlib inline

    

    x = list(df_state['período'])

    y = list(df_state['valor'])



    plt.figure(figsize=(15,5))

    x_pos = [i for i, _ in enumerate(x)]



    plt.bar(x_pos, y, label='State of {}'.format(state), color=color)

    plt.legend(loc='upper right')

    plt.xlabel('Year')

    plt.ylabel('Number of Murders')

    plt.title('Number of Murders Between {} and {} of {} State'.format(x[0], x[-1], state))



    plt.xticks(x_pos, x)



    plt.show()
plt_murderes_per_state(df, 'RJ') # Rio de Janeiro
plt_murderes_per_state(df, 'SP', color='C3') # Sao Paulo
def plt_murderes_side_by_side(df, state1, state2, color1='C0', color2='C3'):

    df_state1 = df.loc[df['nome'] == state1]

    df_state2 = df.loc[df['nome'] == state2]

    

    x1 = list(df_state1['período'])

    y1 = list(df_state1['valor'])

    

    x2 = list(df_state2['período'])

    y2 = list(df_state2['valor'])

    

    N = len(x1)

        

    %matplotlib inline

    

    ind = np.arange(N) 

    width = 0.35

    

    plt.figure(figsize=(15,5))

    x1_pos = [i for i, _ in enumerate(x1)]

    x2_pos = [i + width for i, _ in enumerate(x1)]

    xlabel_pos = [i + (width / 2) for i, _ in enumerate(x1)]

    

    plt.bar(x1_pos, y1, width, label='State of {}'.format(state1), color=color1)

    plt.bar(x2_pos, y2, width, label='State of {}'.format(state2), color=color2)

    

    plt.xticks(xlabel_pos, x1)

    

    plt.legend(loc='upper right')

    plt.xlabel('Year')

    plt.ylabel('Number of Murders')

    plt.title('Number of Murders Between {} and {} of States {} and {}'.format(x1[0], x1[-1], state1, state2))



    

    plt.show()
plt_murderes_side_by_side(df, 'RJ', 'SP')