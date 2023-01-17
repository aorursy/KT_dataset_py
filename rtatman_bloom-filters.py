! pip install bloom_filter
# import bloom filters

from bloom_filter import BloomFilter

from nltk.util import ngrams



# bloom filter with default # of max elements and 

# acceptable false positive rate

bloom = BloomFilter(max_elements=1000, error_rate=0.1)



# sample text

text = '''The numpy sieve with trial division is actually a pretty fast Python

implementation. I've done some benchmarks in the past and saw around of 2-3x or so slower 

than a similar C++ implementation and less than an order of magnitude slower than C.'''

text = text.lower()



# split by word & add to filter

for i in text.split():

    bloom.add(i)



# check if word in filter

"sieve" in bloom 
# bloom filter to store our ngrams in 

bloom_ngram = BloomFilter(max_elements=1000, error_rate=0.1)



# get 5 grams from our text

tokens = [token for token in text.split(" ") if token != ""]

output = list(ngrams(tokens, 5))



# add each 5gram to our bloom filter

for i in output:

    bloom_ngram.add(" ".join(i))



# check if word in filter

print("check unigram:")

print("sieve" in bloom_ngram)



# check if ngram in filter

print("check 5gram:")

print("numpy sieve with trial division" in bloom_ngram)