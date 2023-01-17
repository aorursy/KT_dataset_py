import pandas as pd

import numpy as np

from matplotlib import pyplot as plt



%matplotlib inline
samples = 100000

z_critical = 1.036433 ## For 85% cummulative probability



horse_est_weight = 5 + z_critical*(2)

ball_est_weight = 1 + z_critical*(0.3)

bike_est_weight = 20 + z_critical*(10)

train_est_weight = 10 + z_critical*(5)

coal_est_weight = 47*0.9455  ## Source: http://keisan.casio.com/exec/system/1180573227

doll_est_weight = 7.267 ## Source: http://keisan.casio.com/exec/system/1180573218

block_est_weight = 15.834 ## Source: http://www.math.uah.edu/stat/apps/SpecialCalculator.html

gloves_est_weight = 3 + 0.85

weights = [{'name':'horse', 'weight':horse_est_weight, 'quantity':1000}

          ,{'name':'ball', 'weight':ball_est_weight, 'quantity':1100}

          ,{'name':'bike', 'weight':bike_est_weight, 'quantity':500}

          ,{'name':'train', 'weight':train_est_weight, 'quantity':1000}

          ,{'name':'coal', 'weight':coal_est_weight, 'quantity':166}

          ,{'name':'doll', 'weight':doll_est_weight, 'quantity':1000}

          ,{'name':'blocks', 'weight':block_est_weight, 'quantity':1000}

          ,{'name':'gloves', 'weight':gloves_est_weight, 'quantity':200}]



'''from https://www.kaggle.com/wcukierski/santas-uncertain-bags/plotting-example-gift-weights '''

class book:

    def __init__(self, id):

        self.weight = np.random.chisquare(2,1)[0]

        self.name = 'book'



books = np.histogram([book(x).weight for x in range(samples)], bins = 100)
def calc_rela_freq_cummulative(hist):

    cummulative = 0

    rel_freq = []

    for val in hist[0]:

        freq = cummulative + float(val)/float(samples)

        cummulative = freq

        rel_freq.append(freq)

    return np.asarray(rel_freq)



books = (calc_rela_freq_cummulative(books), books[1])
def get_85_percentile(rel_hist):

    for i, x in enumerate(rel_hist[0]):

        if x > 0.845 and x < 0.855:

            return [i, i+1]

    print ('No Element found in that range')

    return [0, 0]



book_est_weight = books[1][get_85_percentile(books)[1]]

print ('For books the value in the 85 percentile is', book_est_weight)
weights.append({'name':'book', 'weight':book_est_weight, 'quantity': 1200})
weights
Weight_per_bag = 50

id_gifts = [0,0,0,0,0,0,0,0,0]

weights = sorted(weights, key=lambda obj: obj['weight'], reverse = True)

bags = 1000

bags_contain = []

for i in range(bags):

    weight_to_fill = Weight_per_bag

    bag_str = ''

    while weight_to_fill > 0:

        if(weight_to_fill < weights[8]['weight']):

            break;

        for j, gift in enumerate(weights):

            if (gift['quantity'] >= 1) and (weight_to_fill - gift['weight'] >= 0):

                bag_str += ' ' + gift['name'] + '_' + str(id_gifts[j]);

                weight_to_fill = weight_to_fill - gift['weight'];

                id_gifts[j] += 1;

                gift['quantity'] -= 1;

    bags_contain.append(bag_str.strip())
with open("Santa_gifts.csv", 'w') as f:

        f.write("Gifts\n")

        for i in bags_contain:

            f.write( i +'\n')