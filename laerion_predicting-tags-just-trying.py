# Useful modules and such

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output
# Putting all the data from various areas together

areas = ["biology", "cooking", "crypto", "diy", "robotics", "travel"]

tags_df = None

for i in areas:

    if tags_df is None:

        tags_df = pd.read_csv("../input/"+i+".csv")

    else:

        tags_df = pd.concat([tags_df, pd.read_csv("../input/"+i+".csv")])
tags_df.info()
tags_df.head()
# IDs will repeat and are unimportant anyway

tags_df.drop('id', axis=1, inplace=True)
# I believe this is helpful

tags_df = tags_df.reset_index()
# Content needs some clearing up from the tags

for pos,i in enumerate(tags_df.content):

    print (i)

    if pos > 10:

        break

        

# I'm 100% sure there's an easier way

def clear_tags(i):

    to_del = False

    for x in range(len(i)-1,-1,-1):

        let = i[x]

        if let == ">":

            # Some people use these arrows ->

            if i[x-1] != "-":

                to_del = True

        elif let == "<":

            i = i[:x] + i[x+1:]

            to_del = False

        if to_del:

            i = i[:x] + i[x+1:]

    return i



tags_df["content2"] = tags_df.content.apply(clear_tags)
for pos,i in enumerate(tags_df.content2):

    print (i)

    if pos > 10:

        break
# I believe that now I should convert the content into just words (eliminate !?.,())

# Also, convert to lowercase

# Title should go through the same thing

# Possibly tags too but I guess that's pretty much been done



# What about decimal points? Do we care about numbers?

# My best guess is that we don't, but we should analyse if there are any in the tags first



def chop_down_sentences(i):

    # Remove punctuation

    for x in range(len(i)-1,-1,-1):

        let = i[x]

        if let in "!?.,()":

            i = i[:x] + i[x+1:]

    return i.lower().split()



tags_df["content_list"] = tags_df.content2.apply(chop_down_sentences)

tags_df["title_list"] = tags_df.title.apply(chop_down_sentences)

tags_df["tags_list"] = tags_df.tags.apply(chop_down_sentences)
# Now, what proportion of the tags is in the title/content?

total_count = 0

title_count = 0

content_count = 0



def count_tag_freq(i):

    global total_count, title_count, content_count

    for j in i['tags_list']:

        total_count += 1

        if j in i['title_list']:

            title_count += 1

        if j in i['content_list']:

            content_count += 1

            

tags_df.apply(count_tag_freq, axis=1)           

            

print ("Content ratio:", float(content_count)/total_count)

print ("Title ratio:", float(title_count)/total_count)
# These numbers seem a bit low, I suppose multi-word tags are the reason

# Look at that later