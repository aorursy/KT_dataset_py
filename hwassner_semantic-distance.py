# This block just read the data .. 

# each line is a user, each word in a line is the list of word or hashtag tweeted by the user

# example :  

# 0,#uberpop,agnès saal,#fruigopoirdu,myriam el khomri ...

# => means user 0 tweeted (or re-tweeted) something about #uberpop and 'agnes saal', ...

fp = open('/kaggle/input/trending-topics/dataset.csv')

lines = fp.readlines() # get all lines ( ~5 Millions!)
# this block counts the occurence of each hashtag or words seen 

# using a python dictionary : key=word or hashtag, value= number of time the key has been seen among users



limit = 0

tags_count = {} # this will hold the counts 

for line in lines: # loop over all users in the dataset 

    tags = line.strip().split(',') # get rid of useless ' ' , and split on ','

    id = tags.pop(0) # get rid of the anonymized twiter user ID

    for tag in tags: # loop over tags from one twitter user 

        if tag in tags_count: # if this tag has been already seen

            tags_count[tag] += 1 # just increment

        else: # first this tag is seen

            tags_count[tag] = 1 # init the counter to 1

print(len(tags_count))

print(tags_count) #ouput the tags and the respective counts
# this bloc is optional, it just show a sorted version of the count dictionary 

import operator

tags_count_sorted = sorted(tags_count.items(), key=operator.itemgetter(1), reverse=True)

tags_count_sorted
# this bloc defines the 2 function needed to compute asemantic distance between words (these words must be in the dataset)



import math



def count(a,b): # count the co-occurence of both a & b in a same user (same line)

    c = 0

    for line in lines:

        if a in line and b in line:

            c += 1

    return c



def distance(a,b): # see NGD (semantic distance) definition https://homepages.cwi.nl/~paulv/papers/amdug.pdf

    x = math.log(tags_count[a])

    y = math.log(tags_count[b])

    xy = math.log(count(a,b))

    m = math.log(len(lines))

    #print(x,y,xy,m)

    return ((max(x,y)-xy)/(m-min(x,y)))
distance('hollande','neymar')
distance('hollande','ben arfa')
distance('ben arfa','neymar')
distance('#fillon','hollande')
distance('hollande','ben arfa')
distance('kanye','aya nakamura')
distance('kanye','rihanna')