#import pandas as our tool of choice

import pandas as pd
# import data frame

df = pd.read_csv("../input/dsm-beuth-edl-demodata-dirty.csv")

df
#drop rows with NaN only. No missing values

df = df.dropna(how = 'all')



# with out how = 'all' as an argument it would drop every row with any missing value.
#now let's drop duplicates

df = df.drop_duplicates(subset=["full_name", "email"])

df
#once again

df = df.dropna(how = 'all')

df
#Now we want to fix the column age



#first let's ensure that everything is a number

df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(0) 

#I wanted to use .filnna("") for errors instead of, but then the next step couldnt be done easily.

df
#then we want to set negative numbers to positive, asuming the "-" was an accident.

df["age"] = df["age"].map(lambda age: int(age) if int(age) > 0 else int(age)*-1)

df
#Now lets erase for the E-Mail NaN values with blank spaces to avoid missunderstanding

df["email"] = df["email"].fillna('')

df