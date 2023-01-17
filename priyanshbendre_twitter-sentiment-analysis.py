import time

start = time.time()
import re
fp = open("../input/positive.txt","r")

fn = open("../input/negative.txt","r")
contentpositive=fp.read().split()

contentnegative=fn.read().split()
contentposneg = list()

contentposneg = contentpositive + contentnegative
def analyze(l):

    l = re.sub('[!@#$%^&*()_+-\;–./\'’|‘]','', l)

    #clearing the wordsdict each time the fucntion is called to clear the list of the old elements

    wordsdict.clear()

    for words in l.split():

        wordsdict.append(words)



    global total_pos

    global total_neg

    

    neutral = set(wordsdict) - set(contentposneg)

    setposneg = set(wordsdict) - set(neutral)

    setpos = set(setposneg) - set(contentnegative)

    setneg = set(setposneg) - set(setpos)  

    

    if len(setpos) > len(setneg):

        total_pos = total_pos +1

    else:

        total_neg = total_neg +1



#   return(total_pos, total_neg)
total_pos = 0

total_neg = 0

words = list()

wordsdict = list()

f = open("../input/10yearchallenge.txt","r")

for line in f: #\n iteration based

    (analyze(line))



f.close()

fp.close()

fn.close()
print("Total Positive tweets", total_pos, ":: Total Negative tweets", total_neg)

if total_pos>total_neg:

    print ("\n** Sentiment is Positive! **")

else: print("\n** Sentiment is Negative! **")