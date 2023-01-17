# This notebook just read the data .. 

# each line is a user, each word in a line is the list of word or hashtag tweeted by the user

# ... and count the occurence of each hashtag or words seen in the first 1000 users (or lines).



# example :  

# 0,#uberpop,agnès saal,#fruigopoirdu,myriam el khomri ...

# => means user 0 tweeted (or re-tweeted) something about #uberpop and 'agnes saal', ...



limit = 0

tags_count = {} # this will hold the count : key=word or hashtag, value= number of time the key has been seen

with open('/kaggle/input/trending-topics/dataset.csv') as fp:  # open the dataset

    while(limit<100000):   # read until the limit is not reached (the full dataset is 4984127 lines, more that 1Go!)...

        line = fp.readline() # get one line

        #print(line)

        tags = line.strip().split(',') # get rid of useless ' ' , and split on ','

        id = tags.pop(0) # get rid of the anonymized twiter user ID

        #print('TEST',tags)

        for tag in tags: # loop over tags from one twitter user 

            if tag in tags_count: # if this tag has been already seen

                tags_count[tag] += 1 # just increment

            else: # first this tag is seen

                tags_count[tag] = 1 # init the counter to 1

        limit += 1

        

print(tags_count) #ouput the tags and the respective counts
