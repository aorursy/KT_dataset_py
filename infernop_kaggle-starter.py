# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # data visualization

import datetime

%matplotlib inline

#sns.set(style="ticks", color_codes=True)

plt.rcParams["figure.figsize"] = [20, 8]

plt.xticks(rotation=70)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/dsb_2019/"))



# Any results you write to the current directory are saved as output.
PATH = '../input/dsb_2019/'



product_master = pd.read_csv(PATH+'product_master.csv')

training_data = pd.read_csv(PATH+'training_data.csv')

customer_master = pd.read_csv(PATH+'customer_master.csv')

sample_submission = pd.read_csv(PATH+'sample_submission_file.csv')



# Displaying a dataset

display(product_master.head())

display(training_data.head())

display(customer_master.head())

display(sample_submission.head())
print("customer_master.empty : ", customer_master.empty) # Return a boolean variable. True if the dataset is empty, else False.

print("customer_master.ndim : ", customer_master.ndim) #Returns the number of dimensions.

print("customer_master.shape : ", customer_master.shape) #Returns the shape of the dataset in (rows, columns) form.

print("customer_master.axes : ", customer_master.axes) #Returns the list of the labels of the series.
display(product_master.describe())

display(training_data.describe())

display(customer_master.describe())
customer_master.head()
print(sum(customer_master['emailDomain'].isna())) #Boolean to check number of NaN records in the emailDomain column
customer_master['clean_emailDomain'] = customer_master['emailDomain'].fillna("BLANK") #Handling NaN values by substitution

print("Number of NaN records in the emailDomain column: ", sum(customer_master['clean_emailDomain'].isna())) # All the records with NaN values have been substituted with '_' literal.
display(customer_master.drop(columns='emailDomain').head()) # It returns a copy of the dataframe object without the column.  

display(customer_master.head()) # Makes no changes in the actual dataframe object. See? All columns are present.



#How to make the delete-column operation permanent?

display(customer_master.drop(columns='emailDomain', inplace = True)) ## Uncomment to execute this command
# Substituting values

replacement = 'UnknownGender'  ## Feel free to enter your own substitute value!



customer_master['gender'][customer_master['gender'] == '-1'] = replacement



display(customer_master.head())
print(customer_master['gender'].value_counts())

customer_master['gender'].value_counts().plot(kind = 'bar')
# This counts the number of Customer_IDs for a gender group  

print(customer_master[['Customer_Id', 'gender']].groupby(['gender'], sort=True).count())

customer_master[['Customer_Id', 'gender']].groupby(['gender'], sort=True).count().plot(kind='bar',legend=False)
## Let's choose the dimension of Loyalty Tiers and the metric of count()

print(customer_master[['Customer_Id', 'gender','anon_loyalty']].groupby(['anon_loyalty']).count())



customer_master[['Customer_Id','anon_loyalty']].groupby(['anon_loyalty']).count().plot(kind='bar',legend = False)
## Let's choose the dimensions of Gender and Loyalty Tiers and the metric of count()

print(customer_master[['Customer_Id', 'gender','anon_loyalty']].groupby(['anon_loyalty']).count())



customer_master[['Customer_Id','anon_loyalty','gender']].groupby(['gender','anon_loyalty']).count().plot(kind='bar',legend = False)
training_data.head(1)
print("Rows, Columns : ", training_data.shape)

print("Start Date : ", training_data.Bill_Date.min(),"\nEnd Date : ",training_data.Bill_Date.max())



print("Number of Unique Stores : ", training_data.Store_Id.nunique())

#print("List of Unique Stores : ", training_data.Store_Id.unique()) ## Uncomment to check out the unique store codes



print("Number of Unique Products:", training_data.Product_Id.nunique())



print("Number of Unique Customers:", training_data.Customer_Id.nunique())



print("Number of Unique Bills:", training_data.Bill_Header_Id.nunique())

# Shows the number of items bought per store

print(training_data[['Bill_Details_Id', 'Store_Id']].groupby(['Store_Id']).count())

training_data[['Bill_Details_Id', 'Store_Id']].groupby(['Store_Id']).count().plot(kind='bar')

training_data[['Store_Id','Sale_Qty']].groupby(['Store_Id']).sum().plot(kind='bar', rot=70)
# Total Revenue contributed by each Store 

print(training_data.groupby(['Store_Id']).Mrp.sum()) 

training_data.groupby(['Store_Id']).Mrp.sum().plot(kind='bar',rot=70)
# Minimum Revenue contributed by each Store 

print(training_data.groupby(['Store_Id']).Mrp.min()) 

training_data.groupby(['Store_Id']).Mrp.min().plot(kind='bar', rot=70)
# Maximum Revenue contributed by each Store 

print(training_data.groupby(['Store_Id']).Mrp.max()) 

training_data.groupby(['Store_Id']).Mrp.max().plot(kind='bar', rot=70)
# Maximum Revenue contributed by each Store 

print(training_data.groupby(['Store_Id']).Mrp.mean()) 

training_data.groupby(['Store_Id']).Mrp.mean().plot(kind='bar')
# Maximum Revenue contributed by each Store

list_of_operations = ['mean',]  ## Feel free to add any other aggregate operations

#print(training_data.groupby(['Store_Id']).Mrp.agg(list_of_operations)) 

training_data.groupby(['Store_Id']).Mrp.agg(list_of_operations).plot(kind='bar', rot=70)
display(customer_master.head(1))

display(training_data.head(1))
def give_me_length(email):

    return len(email)



customer_master['len_emailDomain'] = customer_master['clean_emailDomain'].apply(lambda x: give_me_length(x))



display(customer_master.head())
##1. Extracting elements from an array via slicing operations 

sample_string = 'Hello World'

print("First Element is sample_string[0] - ", sample_string[0] )

print("Second Element is sample_string[1] - ", sample_string[1] )

print("First 2 elements is sample_string[0:2] - ", sample_string[0:2])

print("Second 2 elements is sample_string[1:3] - ", sample_string[1:3])

print("Last element is sample_string[-1] - ", sample_string[-1])

print("Second Last element is sample_string[-2] - ", sample_string[-2])
def extract_first_letter(gender):

    return gender[0]



customer_master['new_gender'] = customer_master['gender'].apply(lambda x: extract_first_letter(x))

display(customer_master[['gender', 'new_gender']].head(5))
###2. Using the dictionary to map the values



gender_dict = {'UnknownGender':'U',

              'Male':'M',

              'Female':'F'}

print("Dictionary created manually: ", gender_dict)







print("--------------------------")



for g in customer_master['gender'].unique():

    print("Gender value is: ", g)

    print("Gender decode value is: ", g[0])

    

    #Updating the dictionary

    gender_dict[g] = g[0]

print("\nDictionary created the DRY way: ", gender_dict)



customer_master['new_gender'] = customer_master['gender'].map(gender_dict).fillna(customer_master['gender'])

display(customer_master[['gender', 'new_gender']].head())



# One Question : What if 'gender' column has a data value that is not included in my dictionary?

# Answer 1 - By default, those values are replaced with NaN

# Answer 2 - By using the fillna() function, we can use it to fill it with some suitable substitute. 

# Answer 3 - By using the fillna() function and inputting the original column,it retains the  original values of the 

#            entries that can't be mapped by the dictionary. 
default_value = 'NotMappable'

customer_master['new_gender'] = customer_master['gender'].map(gender_dict).fillna(default_value) #Using the default values

display(customer_master[['gender', 'new_gender']].head())





customer_master['new_gender'] = customer_master['gender'].map(gender_dict).fillna(customer_master['gender'])

display(customer_master[['gender', 'new_gender']].head())
display(training_data.head(1))

def give_me_day(date):

    parsed_date = datetime.datetime.strptime(date, "%Y-%m-%d")

    return parsed_date.day



print("Testing the function with give_me_day('2019-09-12') : ", give_me_day('2019-09-12'))





training_data['BillDate_Day'] = training_data['Bill_Date'].apply(lambda x: give_me_day(x))



display(training_data[['BillDate_Day','Bill_Date']].head(5))
def parse_my_date(date, attribute):  

    # This function takes 2 inputs - Date and the attribute to be extracted(day,month or year)

    parsed_date = datetime.datetime.strptime(date, "%Y-%m-%d")

    

    if attribute == 'day':

        return parsed_date.day

    elif attribute == 'month':

        return parsed_date.month

    elif attribute == 'year':

        return parsed_date.year

    else:

        print("Bad Attribute value!")

        exit(1) #Break out of the function!

        return 0

        

        
training_data['BillDate_day'] = training_data['Bill_Date'].apply( lambda x: parse_my_date(x, 'day'))

training_data['BillDate_month'] = training_data['Bill_Date'].apply( lambda x: parse_my_date(x, 'month'))

training_data['BillDate_year'] = training_data['Bill_Date'].apply( lambda x: parse_my_date(x, 'year'))



## Simple interface - No need to remember different functions, just give the attribute the user cares about. That's it. 



display(training_data[['Bill_Date','BillDate_year','BillDate_month','BillDate_day']].head())
## Let's look at the sale quantity basis year

training_data[['BillDate_year','Sale_Qty']].groupby(['BillDate_year']).sum().plot(kind='bar', rot=70)
## Let's look at the sale quantity basis month

training_data[['BillDate_month','Sale_Qty']].groupby(['BillDate_month']).sum().plot(kind='bar', rot=70)
## Let's look at the sale quantity basis month and year.

training_data[['BillDate_year','BillDate_month','Sale_Qty']].groupby(['BillDate_year','BillDate_month']).sum().plot(kind='bar', rot=70)