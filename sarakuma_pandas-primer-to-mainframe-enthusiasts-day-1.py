# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df_VendorStyles = pd.read_fwf("../input/vendorStyles.txt",
                              [(0,1),(1,26),(26,38),(38,58),(58,65),
                              (65,85),(85,89),(89,90),(90,93),(93,103),
                              (103,106),(106,116),(116,120),(120,123),
                              (123,124),(124,126),(126,128),(128,153)],
                             names=["Action","Id","Edi_vndr_id",
                                   "Style","Vndr","Desc","Class",
                                   "Set?","RetUOM","Retail","CostUOM","Cost",
                                   "PackSize","OrderMin","Item_type","SeasonCode",
                                   "SeasonYear","TrendCode"], #based on the copybook/copycode.this translates
                                                              # as header in the resultant dataframe
                             index_col=["Id"]) #if you want declare the ID field as an index of the dataframe
print(df_VendorStyles.head())
df_VendorStyles_simple = pd.read_fwf("../input/vendorStyles.txt",
                              [(0,1),(1,26),(26,38),(38,58),(58,65),
                              (65,85),(85,89),(89,90),(90,93),(93,103),
                              (103,106),(106,116),(116,120),(120,123),
                              (123,124),(124,126),(126,128),(128,153)], #pandas is 0-indexed
                             header=None, 
                             index_col=None) # resultant dataframe doesn't have a header and an index
print(df_VendorStyles_simple.head())
df_markDowns = pd.read_csv("../input/markDowns.csv",sep=",",header=0) #for tab separated specify sep="\t"
                                                                      #for pipe delimited sep="|"
print(df_markDowns.head())
print(df_VendorStyles.info()) #inspect the data types of each series along with the overall counts
                              #based on the file contents pandas automatically infer the data types. 
                              #automatic inferring can be changed/ignored
                              #277 rows are present in the VendorStyles dataframe
print(df_VendorStyles.tail(3)) #inspect last 3 rows of data
print((df_VendorStyles.head(10)).tail(5)) #inspect rows 5-10
print(df_VendorStyles.head()) #inspect first 5 rows of data. default when no values are specified
print(df_VendorStyles.TrendCode.unique()) 
print(df_VendorStyles["SeasonYear"].unique()) #inspect one column/series. Unique method results in a list
print(df_markDowns.columns) #inspect the header of a dataframe. an attribute of a datframe
print(df_markDowns.index) #inspect the index of a dataframe. an attribute of a datframe
print(df_markDowns.shape) #inspect the # of rows and columns in a datframe. results in a tuple. 
                          #an attribute of a datframe
print(df_markDowns.empty) #check if a dataframe is empty or not. Returns boolean value
BOYS_styles = df_VendorStyles["TrendCode"] == "BOYS"
#print(BOYS_styles) # filter results in a boolean
df_boys_styles = df_VendorStyles[BOYS_styles]
print(df_boys_styles) #when filter condition is applied to the df it results in a dataframe 
                                    #with rows having trendcode = "BOYS"

print("*"*50)
print("with logical AND condition in filters - include")
print("*"*50)
BOYS_styles_season13 = (df_VendorStyles["TrendCode"] == "BOYS") & (df_VendorStyles["SeasonYear"] == 13) 
print(df_VendorStyles[BOYS_styles_season13])

print("*"*50)
print("With operator")
print("*"*50)
df_VendorStyles_morecost =  (df_VendorStyles["Cost"]> 250)
print(df_VendorStyles[df_VendorStyles_morecost].head())


print("*"*50)
print("filter equivalent to omit")
print("*"*50)
BOYS_styles_notseason13 =  ~(df_boys_styles["SeasonYear"] == 13)
print(df_boys_styles[BOYS_styles_notseason13])
#filter rows having only numeric values in 'Style' series/column which is of object type. equivalent using NUM in OMIT/INCLUDE statements

style_num = df_VendorStyles['Style'].str.isnumeric() #isalnum method for alpha numeric
df_style_num = df_VendorStyles[style_num]
print(df_style_num.head())

#filter rows not having BLACK,SOLID,/ keywords in 'Desc series/column using regex (yes! regular expressions in pandas)
filter_keys = [r'BLACK',r'SOLID',r'/']
filter_cond = ((~df_VendorStyles['Desc'].str.contains(filter_keys[0])) & (~df_VendorStyles['Desc'].str.contains(filter_keys[1])) & (~df_VendorStyles['Desc'].str.contains(filter_keys[2])))
print(df_VendorStyles[filter_cond].head())

#filter rows having trendcode BASIC NO SUBCLASS - equivalent to substring search in legacy world
filter_keys = ['BASIC', 'NO SUBCLASS']
filter_cond = (df_VendorStyles['TrendCode'].isin(filter_keys))
print(df_VendorStyles[filter_cond].head())
df_new_VS = df_VendorStyles
print(df_new_VS.head())

print(id(df_VendorStyles), id(df_new_VS))  #both the dataframes have the same object ID
df_new_VS["remarks"] = " "
print(df_VendorStyles.columns) #any changes to new dataframe changes the old dataframe

#to create a copy of an existing datafame 
df_new_new_VS = df_VendorStyles.copy()
print(id(df_VendorStyles), id(df_new_new_VS))

#to delete the newly copied dataframe
del df_new_new_VS
#print(df_new_new_VS) #will throw an error

df_new_new_VS = df_VendorStyles.copy()[5:] #copies from 5th row
print(df_new_new_VS.shape)
print(df_VendorStyles.shape)
#make 3 copies of markDowns dataframe
df_1 = df_markDowns.copy()
df_2 = df_markDowns.copy()
df_3 = df_markDowns.copy()
print(df_1.shape,df_2.shape)

df_concat1 = pd.concat([df_1,df_2,df_3]) #more dataframes can be concatenated. axis=0 is rowwise concatenation and is default
print(df_concat1.shape)

df_concat2 = df_1.append([df_2,df_3])   #concat method has more functionalities than append
rows,columns = df_concat2.shape #tuple unpacking
print(rows,columns)
#let us take a copy of VendorStyles dataframe
df_copy = df_VendorStyles.copy()
print(df_copy.shape)


#call clear method to clear the contents of a dataframe
df_copy = df_copy.iloc[0:0]
print(df_copy.shape) #all 277 rows have been removed with the column and index names intact
print(df_copy.index)
print(df_copy.columns)
#let us sort the VendorStyles dataframe in ascending order of Vndr
print(df_VendorStyles.head())
df_VendorStyles.sort_values(by=['Vndr'],ascending=True,inplace=True) #sorts and replaces the same dataframe
print(df_VendorStyles.head())

#let us now sort more fields in different orders
print(df_VendorStyles.head())
df_VendorStyles.sort_values(by=['Vndr','Class'],ascending=[False,True],inplace=True) #sorts and replaces the same dataframe
print(df_VendorStyles.head())

#let us first duplicate the markDowns dataframe
df_dups = df_markDowns.append(df_markDowns)
print(df_markDowns.shape[0],df_dups.shape[0])

#now let us remove those duplicated rows quote SORT FIELDS=NONE
df_dups.drop_duplicates(inplace=True) #please make a note of inplace KWARG
print(df_dups.shape[0])
#let us first duplicate first 5 rows of the markDowns dataframe
del df_dups #deletes the object
df_dups = df_markDowns.append(df_markDowns.head())
print(df_markDowns.shape[0],df_dups.shape[0])

#identifies only those rows that are duplicated in df_dups dataframe
df_discard = df_dups.duplicated()  
print(df_dups[df_discard]) 