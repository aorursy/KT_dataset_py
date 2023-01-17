!pip install git+https://github.com/jaybaird/python-bloomfilter.git
from pybloom import BloomFilter

f = BloomFilter(capacity=1000, error_rate=0.001)
f.add("One String")
"One String" in f
"1 String" in f