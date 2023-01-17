#Import the encessary libraries
import numpy as np 
import pandas as pd 
import os
import re
import seaborn as sns
#To get the dataset file name to work with
print(os.listdir("../input"))
#Read the CSV file into a DataFrame
df_SFTrees = pd.read_csv("../input/san_francisco_street_trees.csv")
#Look at few observations
df_SFTrees.head()
#Understand the total rows, the columns, the non-null values, count of each data type and total memory
df_SFTrees.info()
#Preview Object data type columns. 'object' includes String, Unicode.. 
df_SFTrees.describe(include="object")
#Preview Integer data type columns. 'integer' includes int8, int16, int32, int64
df_SFTrees.describe(include="integer")
#Preview Floating data type columns. 'floating' includes float16, float32, float64, float128
df_SFTrees.describe(include="floating")
print("Total Rows: {0}".format (df_SFTrees.shape[0]))
print("Count of valid Permit Notes: {0:d}, {1:.0%} of the total rows in the dataset".format (df_SFTrees["permit_notes"].notna().sum(),df_SFTrees["permit_notes"].notna().sum()/df_SFTrees.shape[0]))
print("Count of missing Permit Notes: {:d}, {:.0%} of the total rows in the dataset".format (df_SFTrees["permit_notes"].isna().sum(),df_SFTrees["permit_notes"].isna().sum()/df_SFTrees.shape[0]))
#1. Let us look at missing info, like we did for "Permit Notes"
print("Total rows: {0}".format (df_SFTrees.shape[0]))
print("Count of valid \'Plant Date\' rows: {0:d}, {1:.0%} of the total rows in the dataset".format (df_SFTrees["plant_date"].notna().sum(),df_SFTrees["plant_date"].notna().mean()))
print("Count of missing \'Plant Date\' rows: {:d}, {:.0%} of the total rows in the dataset".format (df_SFTrees["plant_date"].isna().sum(),df_SFTrees["plant_date"].isna().mean()))
#2. Let us group and chart this data, for valid dates

#This function extracts the 'year' from observations with valid 'date'
def valid_year(plant_date):
    plantdate = str(plant_date)
    try:
        ts = pd.to_datetime(plantdate)
        return int(ts.year)
    except:
        return int(1900)

#Create a new column to hold the date
df_SFTrees["plantdate"] = df_SFTrees["plant_date"].apply(valid_year)
df_SFTrees.drop("plant_date", inplace=True, axis=1)

sns.set(style="white")
planting_date = sns.factorplot(x="plantdate", data=df_SFTrees[df_SFTrees["plantdate"]>1900], kind="count",palette="PiYG", size=8, aspect=1.5)
planting_date.set_xticklabels(step=3)  #The 'step' parameter values ensures the 'X' axis is not crampped
#Info on trees ploted before 1970.
df_SFTrees[(df_SFTrees["plantdate"]<1970)  & (df_SFTrees["plantdate"]>1900) ]
#1. Let us look at missing info, like we did for "Permit Notes"
print("Total rows: {0}".format (df_SFTrees.shape[0]))
print("Count of valid \'Plant Type\' rows: {0:d}, {1:.0%} of the total rows in the dataset".format (df_SFTrees["plant_type"].notna().sum(),df_SFTrees["plant_type"].notna().sum()/df_SFTrees.shape[0]))
print("Count of missing \'Plant Type\' rows: {:d}, {:.0%} of the total rows in the dataset".format (df_SFTrees["plant_type"].isna().sum(),df_SFTrees["plant_type"].isna().sum()/df_SFTrees.shape[0]))
#Verify uniqueness in 'plant_type' column
df_SFTrees.groupby(["plant_type"])["plant_type"].count()
#Cleanse the 'plant_type' column by updating the 'lower case' value, "tree"
df_SFTrees.loc[df_SFTrees["plant_type"]=="tree",["plant_type"]]="Tree"
#Verify uniqueness in 'plant_type' column now
df_SFTrees.groupby(["plant_type"])["plant_type"].count()
#Let us use One Hot Encoding to turn this 2-value only ("Landscaping and Tree") column to a interger column with values 1 and 0 only. 
#Then, drop one of the columns
df_PlantType = pd.get_dummies(df_SFTrees["plant_type"])
df_SFTrees = pd.concat ([df_SFTrees, df_PlantType], axis=1)
df_SFTrees.drop("Landscaping", axis=1, inplace=True) #Instead of this, we can use the 'drop_first' attribute of get_dummies function, to drop first column
df_SFTrees.head(4)
#Drop the original column
df_SFTrees.drop("plant_type", axis=1, inplace=True)
#1. Let us look at missing info, like we did for "Permit Notes"
print("Total rows: {0}".format (df_SFTrees.shape[0]))
print("Count of values in \'Plot Size\' rows: {0:d}, {1:.0%} of the total rows in the dataset".format (df_SFTrees["plot_size"].notna().sum(),df_SFTrees["plot_size"].notna().sum()/df_SFTrees.shape[0]))
print("Count of missing \'Plot Size\'  rows: {:d}, {:.0%} of the total rows in the dataset".format (df_SFTrees["plot_size"].isna().sum(),df_SFTrees["plot_size"].isna().sum()/df_SFTrees.shape[0]))
#A brief look into the values in "plot_size" column reveals several things"
    #1. It has text and numeric values
    #2. The numeric values are inconsistent. View the values given by the function below:
    
pl_size =df_SFTrees["plot_size"]
pl_count = df_SFTrees["plot_size"].count()
J= 0
for i  in range(pl_count):
    if str(pl_size[i])[ :5] != "Width":
           if len(str(pl_size[i]))>6:
                print(pl_size[i])
                J = J + 1
        
print (J)
#The following function tries to bring the "plot_size" column into something consistent.
#It uses Regex to look for matching text, tries to extract what is required and then tries to mutiply the values
#EG: 3X3 or 3X3' will become 9 ('X' and apostrophe removed). I am assuming all valid values are in "feet" and none in "metres". 
#    If there are meters, note "1 feet" = ".3048 metre"

def format_plot(Sizes):
    Size = str(Sizes)
    Size = Size.lower()
    st=1
    en=1
    Size = re.sub('[a-w|y-z|\' \'|\-|\`]', '', Size)  
    Size = re.sub('[/]','x', Size)
    
    if len(Size)<1:
        Size='0'
    if Size[-1]=="x":
        Size = Size[:-1]
    
    try:
        if Size.lower().find("x")>0:
            if len (Size[: Size.lower().find("x")].strip())<=0:
                st = 1
            else:
                st= float(Size[: Size.lower().find("x")].strip())
            if len(Size[Size.lower().find("x")+1:].strip())<=0:
                en = 1
            else:
                en = float(Size[Size.lower().find("x")+1:].strip())
            
            Size = st * en
            return (float(Size))
    except:
            return -1.     

#Cleanse the Plot Size column
df_SFTrees["plotsize"] = df_SFTrees["plot_size"].apply(format_plot)
#Let us look at what values were not processed
#We need to restrict the values one cane enter, for plot_size. Else, we need to write a function that can do cleansing for values like below
df_SFTrees[df_SFTrees["plotsize"] ==-1] [["plot_size","plotsize"]]
#Since there are very few "invalid" plotsize columns, let us delete them
df_SFTrees.drop(df_SFTrees[df_SFTrees["plotsize"] ==-1].index, axis=0, inplace=True )
#1. Let us look at missing info, like we did for "Permit Notes"
print("Total rows: {0}".format (df_SFTrees.shape[0]))
print("Count of values in \'Site_Info\' rows: {0:d}, {1:.0%} of the total rows in the dataset".format (df_SFTrees["site_info"].notna().sum(),df_SFTrees["site_info"].notna().sum()/df_SFTrees.shape[0]))
print("Count of missing \'Site_Info\'  rows: {:d}, {:.0%} of the total rows in the dataset".format (df_SFTrees["site_info"].isna().sum(),df_SFTrees["site_info"].isna().sum()/df_SFTrees.shape[0]))
#Let us see the possible values for this column
df_SFTrees.groupby("site_info")["site_info"].count()
#Let us see the possible values for this column
df_SFTrees.groupby(["care_assistant","species"])[["species"]].count().rename(columns= {"species": "count"}).sort_values(by="count",ascending=False).head(20).unstack(0).plot.barh()
#We could cluster the data and see how many clusters are there, what are the size of the clusters, outliers, etc...