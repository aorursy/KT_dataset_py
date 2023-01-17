import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



df = pd.read_csv("../input/test-data.csv")



print("Timestamp column dtype is " + df.Timestamp.dtype.name)
%%time



print("Timestamp column dtype is " + pd.to_datetime(df.Timestamp).dtype.name)



# 20s, CPU 100% and can't parse multi-timezone-aware timestamps properly.
%%time



# I know the format, let's use it.

format = "%m/%d/%Y %H:%M:%S.%f%z"



print("Timestamp column dtype is " + pd.to_datetime(df.Timestamp, format=format).dtype.name) 



# A bit better (5s), but still slow and can't parse multi-timezone-aware timestamps properly.
format = "%m/%d/%Y %H:%M:%S.%f%z"



# Let's try converting them to UTC...



try:

    pd.to_datetime(df.Timestamp, format=format, utc=True).dtype

except ValueError as e:

    print("ValueError: "+ str(e))



# ValueError: Cannot pass a tz argument when parsing strings with timezone information.

# Why???
%%time



series = pd.to_datetime(df.Timestamp, infer_datetime_format=True, utc=True)

print("Timestamp column type is " + series.dtype.name)



# This is the only way to get proper datetime64 column, but it's still too slow.

# 10s for 50K rows.