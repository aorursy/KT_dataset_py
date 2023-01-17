import csv

import matplotlib.pyplot as plot

import math



excluded_words = ['the','and','a','that','I','it','not','he','as','you','this','but','his','they','her','she','or','an','will','my',

                  'one','all','would','there','their','','i','to','of','in','on',"i'm",'is',"it's","that's",'be','for','was',"you're",

                  'if','with','at','by','from','up','about','into','over','after','your','so','are','me','we',"i'll",'us','did','our',

                  'um', 'uh', 'er', 'ah', 'like','okay', 'right','have','just','yeah','do', "don't", "can't",'yes','no','who','what',

                  'when','where','why','how','get','him','her','them', "we're"]



# SEASON 1

s1e01 = []

with open(r'../input/s1e01.csv','r', encoding='utf8')as f:

    reader = csv.reader(f)

    for line in reader:

        s1e01.append(line)

s1e02 = []

with open(r'../input/s1e02.csv','r', encoding='utf8')as f:

    reader = csv.reader(f)

    for line in reader:

        s1e02.append(line)

s1e03 = []

with open(r'../input/s1e03.csv','r', encoding='utf8')as f:

    reader = csv.reader(f)

    for line in reader:

        s1e03.append(line)

s1e04 = []

with open(r'../input/s1e04.csv','r', encoding='utf8')as f:

    reader = csv.reader(f)

    for line in reader:

        s1e04.append(line)

s1e05 = []

with open(r'../input/s1e05.csv','r', encoding='utf8')as f:

    reader = csv.reader(f)

    for line in reader:

        s1e05.append(line)

s1e06 = []

with open(r'../input/s1e06.csv','r', encoding='utf8')as f:

    reader = csv.reader(f)

    for line in reader:

        s1e06.append(line)

s1e07 = []

with open(r'../input/s1e07.csv','r', encoding='utf8')as f:

    reader = csv.reader(f)

    for line in reader:

        s1e07.append(line)

s1e08 = []

with open(r'../input/s1e08.csv','r', encoding='utf8')as f:

    reader = csv.reader(f)

    for line in reader:

        s1e08.append(line)

s1e09 = []

with open(r'../input/s1e09.csv','r', encoding='utf8')as f:

    reader = csv.reader(f)

    for line in reader:

        s1e09.append(line)

s1e10 = []

with open(r'../input/s1e10.csv','r', encoding='utf8')as f:

    reader = csv.reader(f)

    for line in reader:

        s1e10.append(line)

# END SEASON 1

#S1E1

s1e1_wordlist = []

s1e1_count = []

for word, count in s1e01:

    if((word[:-1] in excluded_words) == False):

        s1e1_wordlist.append(word[:-1])

        s1e1_count.append(int(count))

s1e1_sorted = sorted(list(sorted(zip(s1e1_count, s1e1_wordlist))), reverse=True)

s1e1_20 = []

for i in range(0,20):

    s1e1_20.append(s1e1_sorted[i])

s1e1_x = []

s1e1_y = []

for count, word in s1e1_20:

    s1e1_x.append(word)

    s1e1_y.append(count)

# END S1E1

# S1E2

s1e2_wordlist = []

s1e2_count = []

for word, count in s1e02:

    if((word[:-1] in excluded_words) == False):

        s1e2_wordlist.append(word[:-1])

        s1e2_count.append(int(count))

s1e2_sorted = sorted(list(sorted(zip(s1e2_count, s1e2_wordlist))), reverse=True)

s1e2_20 = []

for i in range(0,20):

    s1e2_20.append(s1e2_sorted[i])

s1e2_x = []

s1e2_y = []

for count, word in s1e2_20:

    s1e2_x.append(word)

    s1e2_y.append(count)

# END S1E2

# S1S3

s1e3_wordlist = []

s1e3_count = []

for word, count in s1e03:

    if((word[:-1] in excluded_words) == False):

        s1e3_wordlist.append(word[:-1])

        s1e3_count.append(int(count))

s1e3_sorted = sorted(list(sorted(zip(s1e3_count, s1e3_wordlist))), reverse=True)

s1e3_20 = []

for i in range(0,20):

    s1e3_20.append(s1e3_sorted[i])

s1e3_x = []

s1e3_y = []

for count, word in s1e3_20:

    s1e3_x.append(word)

    s1e3_y.append(count)

# END S1S3

# S1E4

s1e4_wordlist = []

s1e4_count = []

for word, count in s1e04:

    if((word[:-1] in excluded_words) == False):

        s1e4_wordlist.append(word[:-1])

        s1e4_count.append(int(count))

s1e4_sorted = sorted(list(sorted(zip(s1e4_count, s1e4_wordlist))), reverse=True)

s1e4_20 = []

for i in range(0,20):

    s1e4_20.append(s1e4_sorted[i])

s1e4_x = []

s1e4_y = []

for count, word in s1e4_20:

    s1e4_x.append(word)

    s1e4_y.append(count)

# END S1E4

# S1E4

s1e5_wordlist = []

s1e5_count = []

for word, count in s1e05:

    if((word[:-1] in excluded_words) == False):

        s1e5_wordlist.append(word[:-1])

        s1e5_count.append(int(count))

s1e5_sorted = sorted(list(sorted(zip(s1e5_count, s1e5_wordlist))), reverse=True)

s1e5_20 = []

for i in range(0,20):

    s1e5_20.append(s1e5_sorted[i])

s1e5_x = []

s1e5_y = []

for count, word in s1e5_20:

    s1e5_x.append(word)

    s1e5_y.append(count)

# END S1E5

# S1E6

s1e6_wordlist = []

s1e6_count = []

for word, count in s1e06:

    if((word[:-1] in excluded_words) == False):

        s1e6_wordlist.append(word[:-1])

        s1e6_count.append(int(count))

s1e6_sorted = sorted(list(sorted(zip(s1e6_count, s1e6_wordlist))), reverse=True)

s1e6_20 = []

for i in range(0,20):

    s1e6_20.append(s1e6_sorted[i])

s1e6_x = []

s1e6_y = []

for count, word in s1e6_20:

    s1e6_x.append(word)

    s1e6_y.append(count)

# END S1E6

# S1E7

s1e7_wordlist = []

s1e7_count = []

for word, count in s1e07:

    if((word[:-1] in excluded_words) == False):

        s1e7_wordlist.append(word[:-1])

        s1e7_count.append(int(count))

s1e7_sorted = sorted(list(sorted(zip(s1e7_count, s1e7_wordlist))), reverse=True)

s1e7_20 = []

for i in range(0,20):

    s1e7_20.append(s1e7_sorted[i])

s1e7_x = []

s1e7_y = []

for count, word in s1e7_20:

    s1e7_x.append(word)

    s1e7_y.append(count)

# END S1E7

# S1E8

s1e8_wordlist = []

s1e8_count = []

for word, count in s1e08:

    if((word[:-1] in excluded_words) == False):

        s1e8_wordlist.append(word[:-1])

        s1e8_count.append(int(count))

s1e8_sorted = sorted(list(sorted(zip(s1e8_count, s1e8_wordlist))), reverse=True)

s1e8_20 = []

for i in range(0,20):

    s1e8_20.append(s1e8_sorted[i])

s1e8_x = []

s1e8_y = []

for count, word in s1e8_20:

    s1e8_x.append(word)

    s1e8_y.append(count)

# END S1E8

# S1E9

s1e9_wordlist = []

s1e9_count = []

for word, count in s1e09:

    if((word[:-1] in excluded_words) == False):

        s1e9_wordlist.append(word[:-1])

        s1e9_count.append(int(count))

s1e9_sorted = sorted(list(sorted(zip(s1e9_count, s1e9_wordlist))), reverse=True)

s1e9_20 = []

for i in range(0,20):

    s1e9_20.append(s1e9_sorted[i])

s1e9_x = []

s1e9_y = []

for count, word in s1e9_20:

    s1e9_x.append(word)

    s1e9_y.append(count)

# END S1E9

# S1E9

s1e10_wordlist = []

s1e10_count = []

for word, count in s1e10:

    if((word[:-1] in excluded_words) == False):

        s1e10_wordlist.append(word[:-1])

        s1e10_count.append(int(count))

s1e10_sorted = sorted(list(sorted(zip(s1e10_count, s1e10_wordlist))), reverse=True)

s1e10_20 = []

for i in range(0,20):

    s1e10_20.append(s1e10_sorted[i])

s1e10_x = []

s1e10_y = []

for count, word in s1e10_20:

    s1e10_x.append(word)

    s1e10_y.append(count)

# END S1E10

# PLOTTING

plot.figure(1, figsize=(20,20))



plot.subplot2grid((2,5), (0,0))

plot.title('Season1 : Episode 1')

plot.tick_params(axis='y',labelsize=8)

plot.barh(range(len(s1e1_x)), s1e1_y)

plot.yticks(range(len(s1e1_x)),s1e1_x)

plot.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)



plot.subplot2grid((2,5), (0,1))

plot.title('Season1 : Episode 2')

plot.tick_params(axis='y',labelsize=8)

plot.barh(range(len(s1e2_x)), s1e2_y)

plot.yticks(range(len(s1e2_x)),s1e2_x)

plot.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)



plot.subplot2grid((2,5), (0,2))

plot.title('Season1 : Episode 3')

plot.tick_params(axis='y',labelsize=8)

plot.barh(range(len(s1e3_x)), s1e3_y)

plot.yticks(range(len(s1e3_x)),s1e3_x)

plot.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)



plot.subplot2grid((2,5), (0,3))

plot.title('Season1 : Episode 4')

plot.tick_params(axis='y',labelsize=8)

plot.barh(range(len(s1e4_x)), s1e4_y)

plot.yticks(range(len(s1e4_x)),s1e4_x)

plot.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)



plot.subplot2grid((2,5), (0,4))

plot.title('Season1 : Episode 5')

plot.tick_params(axis='y',labelsize=8)

plot.barh(range(len(s1e5_x)), s1e5_y)

plot.yticks(range(len(s1e5_x)),s1e5_x)

plot.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)



plot.subplot2grid((2,5), (1,0))

plot.title('Season1 : Episode 6')

plot.tick_params(axis='y',labelsize=8)

plot.barh(range(len(s1e6_x)), s1e6_y)

plot.yticks(range(len(s1e6_x)),s1e6_x)

plot.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)



plot.subplot2grid((2,5), (1,1))

plot.title('Season1 : Episode 7')

plot.tick_params(axis='y',labelsize=8)

plot.barh(range(len(s1e7_x)), s1e7_y)

plot.yticks(range(len(s1e7_x)),s1e7_x)

plot.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)



plot.subplot2grid((2,5), (1,2))

plot.title('Season1 : Episode 8')

plot.tick_params(axis='y',labelsize=8)

plot.barh(range(len(s1e8_x)), s1e8_y)

plot.yticks(range(len(s1e8_x)),s1e8_x)

plot.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)



plot.subplot2grid((2,5), (1,3))

plot.title('Season1 : Episode 9')

plot.tick_params(axis='y',labelsize=8)

plot.barh(range(len(s1e9_x)), s1e9_y)

plot.yticks(range(len(s1e9_x)),s1e9_x)

plot.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)



plot.subplot2grid((2,5), (1,4))

plot.title('Season1 : Episode 10')

plot.tick_params(axis='y',labelsize=8)

plot.barh(range(len(s1e10_x)), s1e10_y)

plot.yticks(range(len(s1e10_x)),s1e10_x)

plot.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)



plot.show()
import csv

import matplotlib.pyplot as plot

import math



excluded_words = ['the','and','a','that','I','it','not','he','as','you','this','but','his','they','her','she','or','an','will','my',

                  'one','all','would','there','their','','i','to','of','in','on',"i'm",'is',"it's","that's",'be','for','was',"you're",

                  'if','with','at','by','from','up','about','into','over','after','your','so','are','me','we',"i'll",'us','did','our',

                  'um', 'uh', 'er', 'ah', 'like','okay', 'right','have','just','yeah','do', "don't", "can't",'yes','no','who','what',

                  'when','where','why','how','get','him','her','them', "we're"]



# SEASON 1

s2e01 = []

with open(r'../input/s2e01.csv','r', encoding='utf8')as f:

    reader = csv.reader(f)

    for line in reader:

        s2e01.append(line)

s2e02 = []

with open(r'../input/s2e02.csv','r', encoding='utf8')as f:

    reader = csv.reader(f)

    for line in reader:

        s2e02.append(line)

s2e03 = []

with open(r'../input/s2e03.csv','r', encoding='utf8')as f:

    reader = csv.reader(f)

    for line in reader:

        s2e03.append(line)

s2e04 = []

with open(r'../input/s2e04.csv','r', encoding='utf8')as f:

    reader = csv.reader(f)

    for line in reader:

        s2e04.append(line)

s2e05 = []

with open(r'../input/s2e05.csv','r', encoding='utf8')as f:

    reader = csv.reader(f)

    for line in reader:

        s2e05.append(line)

s2e06 = []

with open(r'../input/s2e06.csv','r', encoding='utf8')as f:

    reader = csv.reader(f)

    for line in reader:

        s2e06.append(line)

s2e07 = []

with open(r'../input/s2e07.csv','r', encoding='utf8')as f:

    reader = csv.reader(f)

    for line in reader:

        s2e07.append(line)

s2e08 = []

with open(r'../input/s2e08.csv','r', encoding='utf8')as f:

    reader = csv.reader(f)

    for line in reader:

        s2e08.append(line)

s2e09 = []

with open(r'../input/s2e09.csv','r', encoding='utf8')as f:

    reader = csv.reader(f)

    for line in reader:

        s2e09.append(line)

s2e10 = []

with open(r'../input/s2e10.csv','r', encoding='utf8')as f:

    reader = csv.reader(f)

    for line in reader:

        s2e10.append(line)

# END SEASON 2

#S2E1

s2e1_wordlist = []

s2e1_count = []

for word, count in s2e01:

    if((word[:-1] in excluded_words) == False):

        s2e1_wordlist.append(word[:-1])

        s2e1_count.append(int(count))

s2e1_sorted = sorted(list(sorted(zip(s2e1_count, s2e1_wordlist))), reverse=True)

s2e1_20 = []

for i in range(0,20):

    s2e1_20.append(s2e1_sorted[i])

s2e1_x = []

s2e1_y = []

for count, word in s2e1_20:

    s2e1_x.append(word)

    s2e1_y.append(count)

# END S2E1

# S2E2

s2e2_wordlist = []

s2e2_count = []

for word, count in s2e02:

    if((word[:-1] in excluded_words) == False):

        s2e2_wordlist.append(word[:-1])

        s2e2_count.append(int(count))

s2e2_sorted = sorted(list(sorted(zip(s2e2_count, s2e2_wordlist))), reverse=True)

s2e2_20 = []

for i in range(0,20):

    s2e2_20.append(s2e2_sorted[i])

s2e2_x = []

s2e2_y = []

for count, word in s2e2_20:

    s2e2_x.append(word)

    s2e2_y.append(count)

# END S1E2

# S2S3

s2e3_wordlist = []

s2e3_count = []

for word, count in s2e03:

    if((word[:-1] in excluded_words) == False):

        s2e3_wordlist.append(word[:-1])

        s2e3_count.append(int(count))

s2e3_sorted = sorted(list(sorted(zip(s2e3_count, s2e3_wordlist))), reverse=True)

s2e3_20 = []

for i in range(0,20):

    s2e3_20.append(s2e3_sorted[i])

s2e3_x = []

s2e3_y = []

for count, word in s2e3_20:

    s2e3_x.append(word)

    s2e3_y.append(count)

# END S2S3

# S2E4

s2e4_wordlist = []

s2e4_count = []

for word, count in s2e04:

    if((word[:-1] in excluded_words) == False):

        s2e4_wordlist.append(word[:-1])

        s2e4_count.append(int(count))

s2e4_sorted = sorted(list(sorted(zip(s2e4_count, s2e4_wordlist))), reverse=True)

s2e4_20 = []

for i in range(0,20):

    s2e4_20.append(s2e4_sorted[i])

s2e4_x = []

s2e4_y = []

for count, word in s2e4_20:

    s2e4_x.append(word)

    s2e4_y.append(count)

# END S2E4

# S2E4

s2e5_wordlist = []

s2e5_count = []

for word, count in s2e05:

    if((word[:-1] in excluded_words) == False):

        s2e5_wordlist.append(word[:-1])

        s2e5_count.append(int(count))

s2e5_sorted = sorted(list(sorted(zip(s2e5_count, s2e5_wordlist))), reverse=True)

s2e5_20 = []

for i in range(0,20):

    s2e5_20.append(s2e5_sorted[i])

s2e5_x = []

s2e5_y = []

for count, word in s2e5_20:

    s2e5_x.append(word)

    s2e5_y.append(count)

# END S2E5

# S2E6

s2e6_wordlist = []

s2e6_count = []

for word, count in s2e06:

    if((word[:-1] in excluded_words) == False):

        s2e6_wordlist.append(word[:-1])

        s2e6_count.append(int(count))

s2e6_sorted = sorted(list(sorted(zip(s2e6_count, s2e6_wordlist))), reverse=True)

s2e6_20 = []

for i in range(0,20):

    s2e6_20.append(s2e6_sorted[i])

s2e6_x = []

s2e6_y = []

for count, word in s2e6_20:

    s2e6_x.append(word)

    s2e6_y.append(count)

# END S2E6

# S2E7

s2e7_wordlist = []

s2e7_count = []

for word, count in s2e07:

    if((word[:-1] in excluded_words) == False):

        s2e7_wordlist.append(word[:-1])

        s2e7_count.append(int(count))

s2e7_sorted = sorted(list(sorted(zip(s2e7_count, s2e7_wordlist))), reverse=True)

s2e7_20 = []

for i in range(0,20):

    s2e7_20.append(s2e7_sorted[i])

s2e7_x = []

s2e7_y = []

for count, word in s2e7_20:

    s2e7_x.append(word)

    s2e7_y.append(count)

# END S2E7

# S2E8

s2e8_wordlist = []

s2e8_count = []

for word, count in s2e08:

    if((word[:-1] in excluded_words) == False):

        s2e8_wordlist.append(word[:-1])

        s2e8_count.append(int(count))

s2e8_sorted = sorted(list(sorted(zip(s2e8_count, s2e8_wordlist))), reverse=True)

s2e8_20 = []

for i in range(0,20):

    s2e8_20.append(s2e8_sorted[i])

s2e8_x = []

s2e8_y = []

for count, word in s1e8_20:

    s2e8_x.append(word)

    s2e8_y.append(count)

# END S2E8

# S2E9

s2e9_wordlist = []

s2e9_count = []

for word, count in s2e09:

    if((word[:-1] in excluded_words) == False):

        s2e9_wordlist.append(word[:-1])

        s2e9_count.append(int(count))

s2e9_sorted = sorted(list(sorted(zip(s2e9_count, s2e9_wordlist))), reverse=True)

s2e9_20 = []

for i in range(0,20):

    s2e9_20.append(s2e9_sorted[i])

s2e9_x = []

s2e9_y = []

for count, word in s2e9_20:

    s2e9_x.append(word)

    s2e9_y.append(count)

# END S2E9

# S2E10

s2e10_wordlist = []

s2e10_count = []

for word, count in s2e10:

    if((word[:-1] in excluded_words) == False):

        s2e10_wordlist.append(word[:-1])

        s2e10_count.append(int(count))

s2e10_sorted = sorted(list(sorted(zip(s2e10_count, s2e10_wordlist))), reverse=True)

s2e10_20 = []

for i in range(0,20):

    s2e10_20.append(s2e10_sorted[i])

s2e10_x = []

s2e10_y = []

for count, word in s2e10_20:

    s2e10_x.append(word)

    s2e10_y.append(count)

# END S2E10

# PLOTTING

plot.figure(1, figsize=(20,20))



plot.subplot2grid((2,5), (0,0))

plot.title('Season2 : Episode 1')

plot.tick_params(axis='y',labelsize=8)

plot.barh(range(len(s2e1_x)), s2e1_y)

plot.yticks(range(len(s2e1_x)),s2e1_x)

plot.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)



plot.subplot2grid((2,5), (0,1))

plot.title('Season2 : Episode 2')

plot.tick_params(axis='y',labelsize=8)

plot.barh(range(len(s2e2_x)), s2e2_y)

plot.yticks(range(len(s2e2_x)),s2e2_x)

plot.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)

plot.subplot2grid((2,5), (0,2))

plot.title('Season2 : Episode 3')

plot.tick_params(axis='y',labelsize=8)

plot.barh(range(len(s2e3_x)), s2e3_y)

plot.yticks(range(len(s2e3_x)),s2e3_x)

plot.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)

plot.subplot2grid((2,5), (0,3))

plot.title('Season2 : Episode 4')

plot.tick_params(axis='y',labelsize=8)

plot.barh(range(len(s2e4_x)), s2e4_y)

plot.yticks(range(len(s2e4_x)),s2e4_x)

plot.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)

plot.subplot2grid((2,5), (0,4))

plot.title('Season2 : Episode 5')

plot.tick_params(axis='y',labelsize=8)

plot.barh(range(len(s2e5_x)), s2e5_y)

plot.yticks(range(len(s2e5_x)),s2e5_x)

plot.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)

plot.subplot2grid((2,5), (1,0))

plot.title('Season2 : Episode 6')

plot.tick_params(axis='y',labelsize=8)

plot.barh(range(len(s2e6_x)), s2e6_y)

plot.yticks(range(len(s2e6_x)),s2e6_x)

plot.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)

plot.subplot2grid((2,5), (1,1))

plot.title('Season2 : Episode 7')

plot.tick_params(axis='y',labelsize=8)

plot.barh(range(len(s2e7_x)), s2e7_y)

plot.yticks(range(len(s2e7_x)),s2e7_x)

plot.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)

plot.subplot2grid((2,5), (1,2))

plot.title('Season2 : Episode 8')

plot.tick_params(axis='y',labelsize=8)

plot.barh(range(len(s2e8_x)), s2e8_y)

plot.yticks(range(len(s2e8_x)),s2e8_x)

plot.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)

plot.subplot2grid((2,5), (1,3))

plot.title('Season2 : Episode 9')

plot.tick_params(axis='y',labelsize=8)

plot.barh(range(len(s2e9_x)), s2e9_y)

plot.yticks(range(len(s2e9_x)),s2e9_x)

plot.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)

plot.subplot2grid((2,5), (1,4))

plot.title('Season2 : Episode 10')

plot.tick_params(axis='y',labelsize=8)

plot.barh(range(len(s2e10_x)), s2e10_y)

plot.yticks(range(len(s2e10_x)),s2e10_x)

plot.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)

plot.show()