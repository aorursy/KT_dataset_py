import pandas as pd
# don't parse dates at first; and note that initial filtering is per withdrawal, not unique day/date (will refilter later)



# df = pd.concat([pd.read_csv("Checkouts_By_Title_Data_Lens_2017.csv.gz",usecols=["BibNumber","CheckoutDateTime"]).groupby("BibNumber").filter(lambda x: len(x) > 30)

#              ,pd.read_csv("Checkouts_By_Title_Data_Lens_2016.csv.gz",usecols=["BibNumber","CheckoutDateTime"]).groupby("BibNumber").filter(lambda x: len(x) > 40)

#              ,pd.read_csv("Checkouts_By_Title_Data_Lens_2015.csv.gz",usecols=["BibNumber","CheckoutDateTime"]).groupby("BibNumber").filter(lambda x: len(x) > 40)

#              ,pd.read_csv("Checkouts_By_Title_Data_Lens_2014.csv.gz",usecols=["BibNumber","CheckoutDateTime"]).groupby("BibNumber").filter(lambda x: len(x) > 40)



#                ])



df = pd.concat([pd.read_csv("../input/Checkouts_By_Title_Data_Lens_2017.csv",usecols=["BibNumber","CheckoutDateTime"]).groupby("BibNumber").filter(lambda x: len(x) > 30)

             ,pd.read_csv("../input/Checkouts_By_Title_Data_Lens_2016.csv",usecols=["BibNumber","CheckoutDateTime"]).groupby("BibNumber").filter(lambda x: len(x) > 40)

             ,pd.read_csv("../input/Checkouts_By_Title_Data_Lens_2015.csv",usecols=["BibNumber","CheckoutDateTime"]).groupby("BibNumber").filter(lambda x: len(x) > 40)

             ,pd.read_csv("../input/Checkouts_By_Title_Data_Lens_2014.csv",usecols=["BibNumber","CheckoutDateTime"]).groupby("BibNumber").filter(lambda x: len(x) > 40)

               ])



print(df.shape)

df.tail()
# get count of uniques per col

df.apply(lambda x: len(x.unique()))
%%timeit

# df2["date"] = df2.CheckoutDateTime.str.split(" ",1,expand=True)[0]

#%%timeit

# df2["date"] = df2.CheckoutDateTime.apply(lambda x: x.split(" ")[0])

df2.CheckoutDateTime.apply(lambda x: x.split(" ")[0])
# %%timeit

# df2.CheckoutDateTime.apply(lambda x: pd.to_datetime(x.split(" ")[0],format="%m/%d/%Y"))
df.CheckoutDateTime = df.CheckoutDateTime.apply(lambda x: x.split(" ")[0])
# get count of uniques per col

df.apply(lambda x: len(x.unique()))
df.tail()
# df2 = df.head(100000).sample(7000)

# df2.CheckoutDateTime = pd.to_datetime(df2.CheckoutDateTime,infer_datetime_format=True)

# df2.CheckoutDateTime.dt.date.head()



# # 

# df2.groupby(['BibNumber',pd.Grouper(key='CheckoutDateTime', freq='W')])['BibNumber'].agg('count')



# ## round timestamps to date:

### df2.CheckoutDateTime = df2.CheckoutDateTime.dt.date



### df2['CheckoutDateTime'].dt.floor('d')



# possibly - rounds to nearest week:

### df2['CheckoutDateTime'] - pd.to_timedelta(df2['CheckoutDateTime'].dt.dayofweek, unit='d')
print(df.shape)
df.CheckoutDateTime = pd.to_datetime(df.CheckoutDateTime,format="%m/%d/%Y", infer_datetime_format=True)
df.head()
df = df.groupby(['BibNumber',pd.Grouper(key='CheckoutDateTime', freq='SMS')])['BibNumber'].agg('count')
df.shape
df.groupby("BibNumber").filter(lambda x: len(x) > 30).shape
df.groupby("BibNumber").filter(lambda x: len(x) > 24).shape
df = df.groupby("BibNumber").filter(lambda x: len(x) > 12)

print(df.shape)

df.head()
# df.to_csv("SeattleLibrary_Sum_BiMonthly_Checkouts_2017_2014.csv.gz", compression="gzip")

df.to_csv("SeattleLibrary_Sum_BiMonthly_Checkouts_2017_2014.csv")

### added columns manually to the csv with a text editor
df = pd.read_csv("SeattleLibrary_Sum_BiMonthly_Checkouts_2017_2014.csv",names=['BibNumber', 'CheckoutDateTime', 'counts'])

df.head()
df.to_csv("SeattleLibrary_Sum_BiMonthly_Checkouts_2017_2014.csv.gz",index=False,compression="gzip")
df.shape
df_meta = pd.read_csv("../input/Library_Collection_Inventory.csv").drop(["ISBN","ReportDate"],axis=1)  # report date is just 2 dates. may want ot keep ISBNs? 



print(df_meta.shape)



# Some cleanup: note that there can be multiple publication years in the col (or no year)! This is just a quick hack:

# some historical dates will fail due to pandas limitation on parsing (e.g. 1416). 

# we extract the year into another column and hack a date onto it ("01-01"+year )

df_meta["PublicationYearDate"] =  "01/01/"+ df_meta.PublicationYear.str.extract('(\d{4})')

## orig: #  df_meta.PublicationYear = pd.to_datetime(df_meta.PublicationYear.str.extract('(\d{4})'),format="%Y") # full, failed for some cases



# df_meta.PublicationYear = df_meta.PublicationYear.str.extract('(\d{4})').astype(int) # fails due to nans (can do .astype(float) instead)



# replace single stre with boolean

df_meta.FloatingItem.replace("Floating",True,inplace=True)
df_meta.head()
# df_meta.loc[df_meta.PublicationYear<1500].head()
# get count of uniques per col

df_meta.apply(lambda x: len(x.unique()))
df_meta.info()
Bibs = set(df.BibNumber)

len(Bibs)
len(df_meta.BibNum.unique())

# only ~12% of our catalogue is relevant to the "popular" items
df_meta.to_csv("Library_Collection_Inventory_lookup.csv.gz",compression="gzip",index=False)
# df['Week-Year'] = df['CheckoutDateTime'].apply(lambda x: "%d/%d" % (x.week, x.year))

# df.groupby(['Week-Year', 'BibNumber']).size()
# df.groupby(['BibNumber', pd.Grouper(key='CheckoutDateTime', freq='W')]).sum()

# df.groupby('Week-Year').BibNumber.value_counts()



# df.groupby('Week-Year').BibNumber.value_counts().to_csv("wy.csv", date_format='%Y%m%d')