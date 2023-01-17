# install magic module

! pip install python-magic
# import module we'll need to import our custom module

from shutil import copyfile



# copy our file into the working directory (make sure it has .py suffix)

copyfile(src = "../input/sheep-data-validation-script/virginia-lamb-data-validation-script.py", dst = "../working/script.py")



# import all our functions

from script import *
# use the script we imported for data validation 

# you can see a copy of the script here

check_lamb_data("../input/lamb-auction-data/")
import pandas as pd



# create an empty dataframe for our clean data

cleaned_data = pd.DataFrame()



# we'll use these labels in each loop, so it's more efficient

# to declare them out of the loop

column_labels = ["category","head_sold"]



# loop through every .txt file in our target directory

for file in glob.glob("../input/lamb-auction-data/*.txt"):

   

    # read in all the data for a specific file (they're small

    # so this shouldn't be a big problem)

    data = open(file).read()

   

    # find # of head sold

    sales = re.findall('(lamb|ram|ewe|wether|sheep).* (.*) head', data.lower())

    

    # name of market (always on fourth line)

    market = data.split(sep="\n")[3]



    # date reported

    reported = re.findall("Richmond, VA(.*)", data)

    date = dparser.parse(reported[0],fuzzy=True).date()



    # only proceed from here if there was one or more sales

    if len(sales) > 0:

        

        df_temp = pd.DataFrame.from_records(sales, columns=column_labels)

        df_temp["market"] = market

        df_temp["date_reported"] = date



    # add data about markets with no sheep sales

    else:

        df_temp = pd.DataFrame(data={"category": ["none"], "head_sold": [0]})

        df_temp["market"] = market

        df_temp["date_reported"] = date

    

    # append data to our cleaned data

    cleaned_data = cleaned_data.append(df_temp, ignore_index=True)
# now let's take a peek at the cleaned data and make sure it looks good

cleaned_data
# save our data as a .csv file

cleaned_data.to_csv("/kaggle/working/cleaned_lamb_data.csv", index=False)