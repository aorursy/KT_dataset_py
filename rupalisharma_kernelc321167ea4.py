!pip install pyspark
import os

print(os.listdir("../input")) # check file in the input dir
from pyspark import SparkContext

from pyspark.sql import SparkSession
context = SparkContext(appName = "shakespeare_wordcount") # create context with app name
session = SparkSession.Builder().getOrCreate() # create session -- unique -- has to be closed before opening any other session
file = context.textFile("../input/t8.shakespeare.txt") # get the input file to the context
counts = file.flatMap(lambda line: line.split(" ")).map(lambda word: (word, 1)).reduceByKey(lambda a, b: a + b)

counts # using mapReduce to count the frequency of words in the input file
counts_sorted = counts.sortBy(lambda wordCount: wordCount[1], ascending=False)

counts_sorted # sorting the counts in descending order as we have to find the 24th most common word in the file
i = 0

for word, count in counts_sorted.collect()[0:25]:

    print("{} : {} : {} ".format(i, word, count))

    i += 1 # getting the first 25 most common words to have a margin

# we see that the most common word is in fact not a word but a puctuation symbol. So the excluding that

# the 24th most common word is 'as'
session.stop() # closing the session in order to avoid conflicts when the next session is opened