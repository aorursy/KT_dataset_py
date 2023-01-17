# necessary imports

import numpy as np

import csv

from matplotlib import pyplot as plt
jokes = [] # list of all jokes

filepath = '../input/shortjokes.csv'

f = open(filepath)

reader = csv.reader(f) 

next(reader, None)  # skip the headers

for row in reader:

    jokes.append(row[1])

f.close()
len(jokes) # total number of jokes 
lengths = [len(joke) for joke in jokes]
bins = np.arange(10, 200, 2) # set range from 10 to 200 with bin size of 2

plt.figure(figsize=(12, 9))  

ax = plt.subplot(111)

ax.spines["top"].set_visible(False)

ax.spines["right"].set_visible(False)

ax.get_xaxis().tick_bottom()

ax.get_yaxis().tick_left()

plt.hist(lengths, bins=bins, color="#3F5D7D",histtype='bar') # plot histogram of lengths 

plt.title('Jokes Length Histogram',fontsize=20)

plt.ylabel('Count',fontsize=16)

plt.xlabel('Length',fontsize=16)

plt.show()