!pip install pyspark



from pyspark import SparkContext

sc = SparkContext("local", "Scratch")
comp = sc.parallelize (

   ["Amazon", 

   "Apple", 

   "Banjo", 

   "DJI", 

   "Facebook",

   "Google", 

   "HiSilicon",

   "IBM",

   "Intel",

   "Microsoft",

   "Nvidia",

   "OpenAI",

   "Qualcomm",

   "SenseTime",

   "Twitter"]

)

comp_cnt = comp.count()

print("Number of elements in RDD: %i" %(comp_cnt))
comp = sc.parallelize (

   ["Amazon", 

   "Apple", 

   "Banjo", 

   "DJI", 

   "Facebook",

   "Google", 

   "HiSilicon",

   "IBM",

   "Intel",

   "Microsoft",

   "Nvidia",

   "OpenAI",

   "Qualcomm",

   "SenseTime",

   "Twitter"]

)

coll = comp.collect()

print("Elements in RDD: %s" %(coll))
comp = sc.parallelize (

   ["Amazon", 

   "Apple", 

   "Banjo", 

   "DJI", 

   "Facebook",

   "Google", 

   "HiSilicon",

   "IBM",

   "Intel",

   "Microsoft",

   "Nvidia",

   "OpenAI",

   "Qualcomm",

   "SenseTime",

   "Twitter"]

)

comp_filter = comp.filter(lambda x: 'Go' in x)

filtered = comp_filter.collect()

print("Fitered RDD: %s" %(filtered))
comp = sc.parallelize (

   ["Amazon", 

   "Apple", 

   "Banjo", 

   "DJI", 

   "Facebook",

   "Google", 

   "HiSilicon",

   "IBM",

   "Intel",

   "Microsoft",

   "Nvidia",

   "OpenAI",

   "Qualcomm",

   "SenseTime",

   "Twitter"]

)

def func(x): print(x)

fore = comp.foreach(func)
comp = sc.parallelize (

   ["Amazon", 

   "Apple", 

   "Banjo", 

   "DJI", 

   "Facebook",

   "Google", 

   "HiSilicon",

   "IBM",

   "Intel",

   "Microsoft",

   "Nvidia",

   "OpenAI",

   "Qualcomm",

   "SenseTime",

   "Twitter"]

)

comp_map = comp.map(lambda x: (x, 1))

mapping = comp_map.collect()

print("Key value pair: %s" % (mapping))
comp = sc.parallelize (

   ["Amazon", 

   "Apple", 

   "Banjo", 

   "DJI", 

   "Facebook",

   "Google", 

   "HiSilicon",

   "IBM",

   "Intel",

   "Microsoft",

   "Nvidia",

   "OpenAI",

   "Qualcomm",

   "SenseTime",

   "Twitter"]

)

concat = comp.reduce(lambda x,y:  x + ', ' + y)

print("List of all companies : %s" %(concat))
x = sc.parallelize([("spark", 1), ("hadoop", 4)])

y = sc.parallelize([("spark", 2), ("hadoop", 5)])

joined = x.join(y)

final = joined.collect()

print("Joined RDD: %s" %(final))
comp = sc.parallelize (

   ["Amazon", 

   "Apple", 

   "Banjo", 

   "DJI", 

   "Facebook",

   "Google", 

   "HiSilicon",

   "IBM",

   "Intel",

   "Microsoft",

   "Nvidia",

   "OpenAI",

   "Qualcomm",

   "SenseTime",

   "Twitter"]

)

comp.cache()

caching = comp.persist().is_cached

print("Words got cached: %s" %(caching))