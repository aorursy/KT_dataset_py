def pretty_print_review_and_label(i):

    print(labels[i] + "\t:\t" + reviews[i][:80] + "...")



g = open('reviews.txt','r') # What we know!

reviews = list(map(lambda x:x[:-1],g.readlines()))

g.close()



g = open('labels.txt','r') # What we WANT to know!

labels = list(map(lambda x:x[:-1].upper(),g.readlines()))

g.close()
len(reviews)
reviews[0]
labels[0]
print("labels.txt \t : \t reviews.txt\n")

pretty_print_review_and_label(2137)

pretty_print_review_and_label(12816)

pretty_print_review_and_label(6267)

pretty_print_review_and_label(21934)

pretty_print_review_and_label(5297)

pretty_print_review_and_label(4998)
from collections import Counter

import numpy as np
positive_counts = Counter()

negative_counts = Counter()

total_counts = Counter()
for i in range(len(reviews)):

    if(labels[i] == 'POSITIVE'):

        for word in reviews[i].split(" "):

            positive_counts[word] += 1

            total_counts[word] += 1

    else:

        for word in reviews[i].split(" "):

            negative_counts[word] += 1

            total_counts[word] += 1
positive_counts.most_common()
pos_neg_ratios = Counter()



for term,cnt in list(total_counts.most_common()):

    if(cnt > 100):

        pos_neg_ratio = positive_counts[term] / float(negative_counts[term]+1)

        pos_neg_ratios[term] = pos_neg_ratio



for word,ratio in pos_neg_ratios.most_common():

    if(ratio > 1):

        pos_neg_ratios[word] = np.log(ratio)

    else:

        pos_neg_ratios[word] = -np.log((1 / (ratio+0.01)))
# words most frequently seen in a review with a "POSITIVE" label

pos_neg_ratios.most_common()
# words most frequently seen in a review with a "NEGATIVE" label

list(reversed(pos_neg_ratios.most_common()))[0:30]
from IPython.display import Image



review = "This was a horrible, terrible movie."



Image(filename='sentiment_network.png')
review = "The movie was excellent"



Image(filename='sentiment_network_pos.png')