shakespeare = open('../input/alllines.txt').readlines()
shakespeare
len(shakespeare)
shakespeare[1].strip()
shakespeare[1].lower().strip().split()
punctuation = ['!','?',',','.','"',"'",':']

def split_and_clean_line(mystring):

    returnstring = ''

    for mycharacter in mystring:

        if not mycharacter in punctuation:

            returnstring += mycharacter

    return alist



# def split_and_clean_line(mystring):

#     #list comprehension

#     returnstring = ''.join(mychar for mychar in mystring if not mychar in punctuation)

#     return returnstring
#test it by hand.
type(shakespeare[1].lower().strip())
print(shakespeare[1].lower().strip())

print(split_string_and_strip_punctuation(shakespeare[1].lower().strip()))
counting_words = {}

for i in range(len(shakespeare)):

    words_in_line = split_string_and_strip_punctuation(shakespeare[i])

    for word in words_in_line:

        if len(word) < 5: continue

        if not word in counting_words.keys():

            counting_words[word]  = 1

        else:

            counting_words[word] += 1
print('This many different words!',len(counting_words.keys()))
#Now lets sort!

words = numpy.sort(list(counting_words.keys()))

counts = numpy.sort(list(counting_words.values()))

indices_sorted = numpy.argsort(counts)

print(words[indices_sorted][::-1])
sortedwords = words[indices_sorted][::-1]

sortedcounts = counts[indices_sorted][::-1]
for word,count in zip(sortedcounts[:10],sortedwords[:10]):

    print('Word:',word,'Count:',count)    
#Now Functions

def func(listofstrings):

    counting_words = {}

    for i in range(len(listofstrings)):

        words_in_line = split_string_and_strip_punctuation(listofstrings[i].lower().strip()).split()

        for word in words_in_line:

            if len(word) < 5: continue

            if not word in counting_words.keys():

                counting_words[word]  = 1

            else:

                counting_words[word] += 1

    words = numpy.array(list(counting_words.keys()))

    counts = numpy.array(list(counting_words.values()))

    indices_sorted = numpy.argsort(counts)

    sortedwords = words[indices_sorted][::-1]

    sortedcounts = counts[indices_sorted][::-1]

    for word,count in zip(sortedcounts[:10],sortedwords[:10]):

        print('Word:',word,'Count:',count)    