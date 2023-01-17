from __future__ import division

import numpy as np

import pandas as pd

from itertools import combinations_with_replacement

from ast import literal_eval

from scipy.optimize import linprog
gifts = pd.read_csv('../input/gifts.csv')

gifts["gift_type"] = gifts.GiftId.map(lambda x: x[:x.index("_")])

gift_types = np.sort(gifts.gift_type.unique())

gift_type_to_int = {gift_types[i]: i for i in range(len(gift_types))}

int_to_gift_type = {val: key for key, val in gift_type_to_int.items()}

gifts['gift_type_int'] = gifts.gift_type.map(gift_type_to_int.get)

gifts.head()
gifts.gift_type.value_counts()
gift_types_int = range(len(gift_types))

combinations = []

for n_gifts in range(3, 9): # all bags must have >= 3 gifts; 8 is an arbitrary upper limit on the number of items in a bag

    combinations.extend(list(combinations_with_replacement(gift_types_int, r=n_gifts)))



gift_sets = pd.DataFrame([str(elem) for elem in combinations], columns=['set'])

gift_sets.set = gift_sets.set.map(literal_eval) # back to tuples from strings; probably a better way to do this?

print("There are {:,} different gift sets.".format(len(gift_sets)))

gift_sets.head()
n_samples=100000

distributions = {7: np.maximum(0, np.random.normal(5, 2, n_samples)),

                0: np.maximum(0, np.random.normal(2, 0.3, n_samples)),

                1: np.maximum(0, np.random.normal(20, 10, n_samples)),

                8: np.maximum(0, np.random.normal(10, 5, n_samples)),

                4: 47 * np.random.beta(0.5, 0.5 ,n_samples),

                3: np.random.chisquare(2, n_samples),

                5: np.random.gamma(5 ,1, n_samples),

                2: np.random.triangular(5, 10, 20, n_samples)}

gloves1 = 3.0 + np.random.rand(n_samples)

gloves2 = np.random.rand(n_samples)

gloves3 = np.random.rand(n_samples)

distributions[6] = np.where(gloves2 < 0.3, gloves1, gloves3)



def expected_score(gift_types):

    """

    Computes the expected score of the bag with gifts specified in gift_types by taking n_samples samples from

    the bag's score distribution and returning their average.

    

    :param gift_types: tuple[int] that specifies the types of gifts in this bag

    :param n_samples: number of samples to take from the bag's score distribution

    """

    

    global n_samples

    scores = np.zeros(n_samples)



    for gift_type in gift_types:

        scores += distributions[gift_type]

    

    scores[scores > 50] = 0

    return scores.mean()
%%time

# ~30s now

scores = gift_sets.set.apply(expected_score)



gift_sets['score'] = scores

gift_sets.sort_values('score', inplace=True, ascending=False)

scores = gift_sets.score

gift_sets.head()
def count(array, val):

    count = 0

    for elem in array:

        if elem == val:

            count += 1

    return count



# constraint: # of any kind of gift can't exceed max number

bounds = gifts.gift_type_int.value_counts()

type_counts = []

for i in bounds.index:

    type_counts.append(gift_sets.set.apply(count, args=(i,)).values)

type_counts = np.array(type_counts)



# constraint: max # bags is 1K; 1*bag_count <= 1K

constraint_matrix = np.row_stack((type_counts, np.ones(type_counts.shape[1])))

constraint_bounds = np.concatenate((bounds.values, np.array(1000).reshape(-1)))



result = linprog(-scores, constraint_matrix, constraint_bounds)

bag_counts = result.x

print("Expected score: {:,}".format(-result.fun))
# now this part is a little tricky... the bag_counts are floats, but we need integer numbers of bags

# approach: floor each element (so we never use too many of a bag), use that many of each kind of bag,

# then greedy search to fill up the remaining bags



def lp_greedy_fill(gift_sets, bag_counts):

    """

    :param gift_sets: dataframe with a column 'set' that contains types of gift bags (using integer ids for gifts) 

                      must be sorted s.t. best bags come first

    :param bag_counts: numpy array containing the optimal (float) count of each type of bag;

                       result of LP optimization

    Other variables are taken from globals.

    """

    

    gift_counts = gifts.gift_type_int.value_counts()

    counts_dict = {i: 0 for i in range(len(gift_counts))}

    def get_gift_id_and_increment(gift_type):

        """

        For making gift ids (e.g. bike_1) for the submission file.

        """

        count = counts_dict[gift_type]

        counts_dict[gift_type] += 1

        return int_to_gift_type[gift_type] + "_" + str(count)



    out_file_name = "output.csv"

    with open(out_file_name, 'w') as outfile:

        outfile.write("Gifts\n")

        

        bags_filled = 0

        for gift_set_idx in range(len(bag_counts)):

            count = int(bag_counts[gift_set_idx])

            next_gift_set = gift_sets.set.iloc[gift_set_idx]

            bag_gift_counts = pd.Series(next_gift_set).value_counts()

            gift_counts.loc[bag_gift_counts.index] -= bag_gift_counts * count

            for _ in range(count):

                outfile.write(" ".join(map(get_gift_id_and_increment, next_gift_set)) + "\n")

            bags_filled += count



        # greedy search to fill the rest

        gift_set_idx = 0

        next_gift_set = gift_sets.set.iloc[gift_set_idx]

        bag_gift_counts = pd.Series(next_gift_set).value_counts()

        while bags_filled < 1000:

            

            if np.all(bag_gift_counts < gift_counts.loc[bag_gift_counts.index]):

                gift_counts.loc[bag_gift_counts.index] -= bag_gift_counts

                outfile.write(" ".join(map(get_gift_id_and_increment, next_gift_set)) + "\n")

                bags_filled += 1

            else: # can't do any more of this bag type; move to next best

                gift_set_idx += 1

                if gift_set_idx == len(gift_sets):

                    print("Ran out of possible gift sets!")

                    break

                else:

                    next_gift_set = gift_sets.set.iloc[gift_set_idx]

                    bag_gift_counts = pd.Series(next_gift_set).value_counts()



        print("Output was written to {}".format(out_file_name))
%%time

lp_greedy_fill(gift_sets, bag_counts)