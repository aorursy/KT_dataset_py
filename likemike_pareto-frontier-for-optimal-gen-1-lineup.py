# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
poke_df = pd.read_csv('../input/Pokemon.csv')
poke_gen1_raw_df = poke_df[poke_df['Generation'] == 1]

poke_gen1_df = poke_gen1_raw_df[~poke_gen1_raw_df['Name'].str.contains('Mega')]
#credit for function goes to Peter from stackoverflow

#http://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python



def is_pareto_efficient(costs):

    """

    :param costs: An (n_points, n_costs) array

    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient

    """

    is_efficient = np.ones(costs.shape[0], dtype = bool)

    for i, c in enumerate(costs):

        if is_efficient[i]:

            # Remove dominated points

            is_efficient[is_efficient] = np.any(costs[is_efficient]<=c, axis=1)  

    return is_efficient
#here are the attributes we wish to construct our pareto frontier with

stat_list = ['HP','Attack', 'Defense', 'Sp. Def', 'Sp. Atk', 'Speed']



#the above function asks for a cost vector, so we multiply our stats by -1

pareto_indexer = is_pareto_efficient(np.array(poke_gen1_df[stat_list])*-1)
pareto_frontier_df = poke_gen1_df[pareto_indexer]
pareto_frontier_df.sort_values('Total', ascending=False)