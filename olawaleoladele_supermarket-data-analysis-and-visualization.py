import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
my_filepath = "../input/groceries-dataset/Groceries_dataset.csv"
my_data = pd.read_csv(my_filepath)
my_data
uniques = my_data["itemDescription"].unique()

count_data=[]
for data in uniques:
    item = str(data)
    filt = my_data["itemDescription"] == item
    new=my_data.loc[filt]
    x=new["itemDescription"].count()
    count_data.append(x)
data = {'itemDescription':  uniques,
        'Count Of Items': count_data
        }
count_df = pd.DataFrame (data, columns = ['itemDescription','Count Of Items'])
count_df

truncated_count_df = count_df.nlargest(20,['Count Of Items'])
# Create a plot
plt.figure(figsize=(12,14))
sns.set_style("whitegrid")
chart = sns.barplot(y=truncated_count_df["itemDescription"], x=truncated_count_df['Count Of Items'],palette='Set2', data=truncated_count_df) # Your code here
plt.title("Most Commonly Bought Items")
#chart.set_yticklabels(chart.get_yticklabels(), rotation=90)
# sns.set_context("talk", font_scale=1.4)
my_data.Date = pd.to_datetime(my_data.Date) 

my_data['Year'] = my_data.Date.apply(lambda x : x.year)
my_data['Month'] = my_data.Date.apply(lambda x : x.month)
my_data['Days of Week'] = my_data.Date.apply(lambda x : x.dayofweek)
my_data
plt.figure(figsize=(12,10))
sns.set_style("whitegrid")
sns.countplot(my_data.Year,palette='Set2')
plt.ylabel("Volume of Sales")
plt.show()
filt = my_data["Year"] == 2015
my_data.loc[filt].count()
customers = my_data["Member_number"].unique()
years = my_data["Year"].unique()
num_of_customers = []


def no_of_customersfn(customers,year):
    uni = 0
    for member in customers:
        filt = (my_data["Member_number"] == member) & (my_data["Year"] == year)
        temp_df= my_data.loc[filt]
        temp_no = temp_df["Member_number"].nunique()
        if temp_no == 1:
            uni += 1
    return uni


for year in years:
    x=no_of_customersfn(customers, year)
    num_of_customers.append(x)
    
    

print(num_of_customers)
print(years)

#2014 is 3443
# 2015 is 3314
data = {'Years':  years,
        'No of Customers': num_of_customers
        }
nuniques_df = pd.DataFrame (data, columns = ['Years','No of Customers'])
nuniques_df
plt.figure(figsize=(14,10))
sns.set_style("whitegrid")
chart = sns.barplot(y=nuniques_df["No of Customers"], x=nuniques_df['Years'],palette='Set2', data=nuniques_df) # Your code here
plt.title("No of Unique Customers By Years")