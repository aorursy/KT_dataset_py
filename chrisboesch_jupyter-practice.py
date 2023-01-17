print("hi chris")
student_name = "Chris"

import datetime
now = datetime.datetime.now()

message = "{} ran all at {}".format(student_name, now)
print(message)

import pandas as pd
from pyspark import SparkContext, SparkConf
