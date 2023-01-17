from functools import reduce
# define square function:

def sqr(x): return x ** 2 



# the traditional way using loop:

numbers = [1, 2, 3, 4]

squared = []



print('numbers: ' + str(numbers)) # print thr input numbers



for n in numbers: # loop

    squared.append(sqr(n)) # save the results

    print (str(n) + " X " + str(n) + " = " + str(squared[-1])) # print n and the last element of squared

    

print('squared: ' + str(squared))
# then map the function onto the list

map(sqr, numbers) # note the different syntax/usage for a single input: e.g. sqr(6)
map((lambda x: x **2), numbers) # mapping with a lmbda function
numbers = range(1,5)

reduce((lambda x, y: x + y), numbers) # reduce 1,2,3,4 : 1 + 2 + 3 + 4 = 10
x = numbers[0] # grab the first

for y in numbers[1:]: # or use range(1, len(numbers) + 1)

    x = x + y # multiply by the second and update

    

print(x)
numbers = range(1,5)

reduce((lambda x, y: x+ y), map((lambda x: x **2), numbers)) # 1 + 4 + 9 + 16 = 30
!pip install pyspark
from __future__ import print_function # this version is Py2: print vs print()

from pyspark import SparkContext, SparkConf

from operator import add
sc.stop() # sometimes you have to stop previous 'context', especially if it crashed
sc = SparkContext(conf=SparkConf(), appName='PyWordCount')
inputRDD = sc.textFile("./spark.txt")

print(inputRDD)
print('The number of partitions: ',inputRDD.getNumPartitions(),

      '\nThe total number of elements: ', inputRDD.count())
inputRDD.mapPartitions(lambda m: [1]).reduce(lambda a,b: a+b)
inputRDD.map(lambda m: len(m)).reduce(lambda a,b: a+b)
! cat ./spark.txt
inputRDD.map(lambda m: 1).reduce(lambda a,b: a+b) # = inputRDD.inputRDD.count()
inputRDD.mapPartitions(lambda m: [1]).reduce(lambda a,b: a+b) # = inputRDD.getNumPartitions()
words = inputRDD.flatMap(lambda x: x.split(' '))

print(words)
wordsOne = words.map(lambda x: (x, 1))

print(wordsOne)
wordCounts = wordsOne.reduceByKey(add)

print(wordCounts)
output = wordCounts.collect() 
# Print the 10 first entries of the result

output[:10]
sorted(output, key=lambda x: x[1], reverse = True)[0:10] # top 10 
sc.stop() # stop previous 'context'

sc = SparkContext(appName="PythonWordCount") # create a new context

# the following does all the job at once:

output = sc.textFile("./spark.txt").flatMap(lambda x: x.split(' ')).map(lambda x: (x, 1)).reduceByKey(add).collect()

# and the output:

sorted(output, key=lambda x: x[1], reverse = True)[0:10] # top 10 