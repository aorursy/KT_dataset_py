# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Import relevant packages



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/BlackFriday.csv")
df.head()
df.info()
# To check which columns contain null values first

df.isna().any()
# To check the unique values in Product_Category_2 and Product_Category_3 so that no null value would be missed out

print("Before")

print(df["Product_Category_1"].unique())

print(df["Product_Category_2"].unique())

print(df["Product_Category_3"].unique())



# To replace the only 1 null value which is nan values in Product_Category_2 and Product_Category_3 with 0



df["Product_Category_2"].fillna(0, inplace=True)

df["Product_Category_3"].fillna(0, inplace=True)



print("")



print("After")

print(df["Product_Category_1"].unique())

print(df["Product_Category_2"].unique())

print(df["Product_Category_3"].unique())
df = df.astype({"Product_Category_2": int, "Product_Category_3": int})

df.head()
#checking the results

print(sorted(df["Age"].unique()))

print(sorted(df["Occupation"].unique()))

print(sorted(df["City_Category"].unique()))



# creating a dict file  

#Age_range = {'0-17': 0,'18-25': 1, '26-35': 2, '36-45': 3, '46-50': 4, '51-55': 5, '55+': 6}



#Changing the age range to a numerical value

print(sorted(df["Marital_Status"].unique()))
df.head()
#Removing all User_ID duplicates to get an accurate count of gender and age groups

gender_unique_df = df.copy()

gender_unique_df = gender_unique_df.drop_duplicates(subset='User_ID', keep="first")

gender_unique_df.head()
gender_count_df = gender_unique_df.groupby(["Gender"]).size().reset_index(name="Number of buyers")

gender_count_df
#create a dataframe to see the breakdown by gender and age

age = gender_unique_df.groupby(["Age"]).size().reset_index(name="counts")

age
customers_city = gender_unique_df.groupby("City_Category").size().reset_index(name = "No. of Customers")

customers_city.head()
occupation_count = gender_unique_df.groupby("Occupation").size().reset_index(name = "Occupation count")

occupation_count.head()
marital_count = gender_unique_df.groupby("Marital_Status").size().reset_index(name = "Marital count")

marital_count.head()
print(sorted (df["Product_Category_1"].unique()))

print(sorted (df["Product_Category_2"].unique()))

print(sorted (df["Product_Category_3"].unique()))
check = df.copy()

category = df["Product_Category_1"] == 1

check = check[category]

check.head()
#Creating a combined category field

products = df.copy()

products["Full Category"] = products["Product_Category_1"].apply(str) + ", " + products["Product_Category_2"].apply(str) + ", " + products["Product_Category_3"].apply(str)

products.head()
# Start by creating the figure and add the subplot

fig1 = plt.figure(figsize=(10,6))

ax1 = fig1.add_subplot(111)



# Find out the total selling quantity of each product category

sales_total_cat1 = df[df["Product_Category_1"] != 0]["Product_Category_1"].count()

sales_total_cat2 = df[df["Product_Category_2"] != 0]["Product_Category_2"].count()

sales_total_cat3 = df[df["Product_Category_3"] != 0]["Product_Category_3"].count()



# Convert the total selling quantity of each product category into a DataFame

df_sales_cat = pd.DataFrame({"Product Category": ["Product_Category_1", "Product_Category_2", "Product_Category_3"],

                            "Selling Quantity": [sales_total_cat1, sales_total_cat2, sales_total_cat3]})



# Plot the bar graph here

df_sales_cat.plot(kind="bar", x="Product Category" ,y="Selling Quantity", ax=ax1, color="skyblue")



# Set the title, label of y-axis of the bar graph

ax1.set_title("Total Selling Quantity of each product category")

ax1.set_ylabel("Selling Quantity")



labels1 = [sales_total_cat1, sales_total_cat2, sales_total_cat3]



for rect, label in zip( ax1.patches, labels1):

    height = rect.get_height()

    x_value = rect.get_x() + rect.get_width() / 2

    plt.text(x_value, height + 5, label,

            ha='center', va='bottom')



plt.show()
# Find out the selling quantity of the sub-catagories within each product category

sales_cat1 = products.groupby("Product_Category_1").size().reset_index(name="Selling Quantity")

sales_cat2 = products.groupby("Product_Category_2").size().reset_index(name="Selling Quantity")

sales_cat3 = products.groupby("Product_Category_3").size().reset_index(name="Selling Quantity")

sales_combined = products.groupby("Full Category").size().reset_index(name="Selling Quantity")



# Drop the row where sub-category is 0

sales_cat2 = sales_cat2.drop([0])

sales_cat3 = sales_cat3.drop([0])
#only taking the top 10 combined category products as the resultant plot is too large

sales_combined.sort_values(by=['Selling Quantity'], inplace=True,  ascending=False)

top15_sales = sales_combined.head(15)
purchase_fullcat = products.groupby(["Full Category",  "Purchase"])["Product_ID"].count().reset_index(name = "cost count")

purchase_fullcat.head()
price = products.copy()

comb_cat_price = price.groupby(["Full Category"])["Purchase"].mean().reset_index(name="Cost")

cat1_price = products.groupby("Product_Category_1")["Purchase"].mean().reset_index(name="Cost")

cat2_price = products.groupby("Product_Category_2")["Purchase"].mean().reset_index(name="Cost")

cat3_price = products.groupby("Product_Category_3")["Purchase"].mean().reset_index(name="Cost")



# Drop the row where sub-category is 0

cat2_price = cat2_price.drop([0])

cat3_price = cat3_price.drop([0])



#only taking the top 10 combined category products as the resultant plot is too large

comb_cat_price.sort_values(by=['Cost'], inplace=True,  ascending=False)

top15_price = comb_cat_price.head(15)



#Retrieving the price of the top selling products

cond_top_15_Sold = comb_cat_price["Full Category"].isin(top15_sales["Full Category"])

top_15_sold_price = comb_cat_price[cond_top_15_Sold]
#Combining the selling quantity and cost together

cat1 = pd.merge(cat1_price, sales_cat1,  on='Product_Category_1')

cat1 = cat1.sort_values('Cost', ascending=False)



cat2 = pd.merge(cat2_price, sales_cat2,  on='Product_Category_2')

cat2 = cat2.sort_values('Cost', ascending=False)



cat3 = pd.merge(cat3_price, sales_cat3,  on='Product_Category_3')

cat3 = cat3.sort_values('Cost', ascending=False)



top15 = pd.merge(top_15_sold_price, top15_sales, on='Full Category')

top15 = top15.sort_values('Cost', ascending=False)
# Amt earned

comb_cat_earn = price.groupby(["Full Category"])["Purchase"].sum().reset_index(name="Earn")

cat1_earn = products.groupby("Product_Category_1")["Purchase"].sum().reset_index(name="Earn")

cat2_earn = products.groupby("Product_Category_2")["Purchase"].sum().reset_index(name="Earn")

cat3_earn = products.groupby("Product_Category_3")["Purchase"].sum().reset_index(name="Earn")



# Drop the row where sub-category is 0

cat2_earn = cat2_earn.drop([0])

cat3_earn = cat3_earn.drop([0])



#only taking the top 10 combined category products as the resultant plot is too large

comb_cat_earn.sort_values(by=['Earn'], inplace=True,  ascending=False)

top15_earn = comb_cat_earn.head(15)
# Create the figure and add the subplot

fig1 = plt.figure(figsize=(10,20))

ax1_0 = fig1.add_subplot(511)

ax1_1 = fig1.add_subplot(512)

"""ax2_0 = fig1.add_subplot(513)

ax2_1 = fig1.add_subplot(514)

ax3_0 = fig1.add_subplot(515)

ax3_1 = fig1.add_subplot(516)

ax4_0 = fig1.add_subplot(517)"""



width=0.20



# Plot the sorted bar graph (Cat 1) according to cost

cat1.plot( 

    kind="bar", x="Product_Category_1" ,y="Cost", ax=ax1_0, position=2, width=width, color='orange')



# Plot the sorted bar graph (category 1) according to qty sold

cat1.plot(

    kind="bar", x="Product_Category_1" ,y="Selling Quantity", ax=ax1_0, position=1, width=width, color='purple')



ax1_0.set_title("Cost of sub-categories in category 1 compared to quantity sold")

ax1_0.set_xlabel("Cost/Selling Qty")

ax1_0.set_ylabel("sub-categories in category 1")



#plot earnings

cat1_earn.sort_values(by = "Earn", ascending = False).plot(

    kind="bar", x="Product_Category_1" ,y="Earn", ax=ax1_1, color='tomato')



ax1_1.set_title("Earnings of sub-categories in category 1")

ax1_1.set_xlabel("Earnings")

ax1_1.set_ylabel("sub-categories in category 1")



fig1.tight_layout()



plt.show()
gender_df = df.copy()

gender_df = gender_df.groupby(["Gender"])["Purchase"].sum().reset_index(name = "Total Purchase ($)")

gender_df
# Create the figure and add the subplot

fig2 = plt.figure(figsize=(8,5))

ax2 = fig2.add_subplot(121)

ax3 = fig2.add_subplot(122)



# Create a pie chart

ax2.pie(

    gender_count_df['Number of buyers'],

    explode = None,

    labels=gender_count_df['Gender'],

    #colors = colors,

    autopct='%1.1f%%', 

    shadow=False,

    )



# Create the line chart

gender_df.plot(kind='bar', x='Gender', y='Total Purchase ($)', ax=ax3, legend=None)



# Set the title here

#

ax3.set_title("Total Purchase by Gender")



# Set the y axis label

#

ax3.set_ylabel("Total Purchase ($) Billion")



labels1 = ["1164624021", "3853044357"]



# To add the number label on top of the bars

for rect, label in zip( ax3.patches, labels1):

    height = rect.get_height()

    x_value = rect.get_x() + rect.get_width() / 2

    plt.text(x_value, height + 5, label,

            ha='center', va='bottom')



# View the plot

plt.tight_layout()

plt.show()
# Lets create a datafram that sums everything up by customer

total_by_cust = df.groupby(["User_ID", "Gender", "Age", "Occupation", "City_Category","Marital_Status", "Stay_In_Current_City_Years"])["Purchase"].sum().reset_index(name="Total Purchase")

total_by_cust.head()
# Create a box plot of the spending of females vs males

# For a more accurate comparison

fig4 = plt.figure(figsize=(6,14))

ax4 = fig4.add_subplot(111)



sns.boxplot(x = 'Gender', y = 'Total Purchase', data = total_by_cust, ax=ax4)

ax4.set_title("Boxplot of cutomer's individual purchase amount by gender")

plt.show()
#mean, median and mode of the 2 purchasing powers of the gender

print(total_by_cust.groupby("Gender").mean())

print()

print(total_by_cust.groupby("Gender").median())

print()

print("Females spend about " + str((911963.16-699054.03)/(911963.16 + 699054.03)*100) + "% less than males")

#We can see that Males spend more than females.
#calculate the number of females/males that spend above the IQR
#are females buying more of a certain category that is more expensive?

#fig5 = plt.figure(figsize=(6,14))
#create a dataframe to see the breakdown by gender and age

gender_age = gender_unique_df.groupby(["Age", "Gender"]).size().reset_index(name="counts")

gender_age_purchase = gender_unique_df.groupby(["Age", "Gender"])["Purchase"].sum().reset_index(name="Purchase Total")

gender_age_ave_purchase = gender_unique_df.groupby(["Age", "Gender"])["Purchase"].median().reset_index(name="Purchase Total")
gender_age_pivot = pd.pivot_table(

    gender_age,

    index="Age",

    columns="Gender",

    values="counts",

    aggfunc=sum

)



gender_age_pivot.columns = ["F", "M"]

gender_age_pivot = gender_age_pivot.reset_index()

gender_age_pivot

gender_age_purchase_pivot = pd.pivot_table(

    gender_age_purchase,

    index="Age",

    columns="Gender",

    values="Purchase Total",

    aggfunc=sum

)



gender_age_purchase_pivot.columns = ["F", "M"]

gender_age_purchase_pivot = gender_age_purchase_pivot.reset_index()

gender_age_purchase_pivot
gender_age_ave_purchase_pivot = pd.pivot_table(

    gender_age_ave_purchase,

    index="Age",

    columns="Gender",

    values="Purchase Total",

    aggfunc=sum

)



gender_age_ave_purchase_pivot.columns = ["F", "M"]

gender_age_ave_purchase_pivot = gender_age_ave_purchase_pivot.reset_index()

gender_age_ave_purchase_pivot
width=0.20



#Create the figure

fig6 = plt.figure(figsize=(15, 15))



#Add the subplot

ax6 = fig6.add_subplot(221)

ax7 = fig6.add_subplot(222)

ax8 = fig6.add_subplot(223)



#Plot the number of buyers by gender and age group

gender_age_pivot.plot(kind='bar', x='Age', y='F', 

                            ax=ax6, position=1, width=width, color='purple')

gender_age_pivot.plot(kind='bar', x='Age', y='M', 

                            ax=ax6, position=2, width=width, color='tomato')



# Add the title of the plot

ax6.set_title("No. of Buyers by age group and gender")

ax6.set_ylabel("Number of Buyers")



#Plot the purchase by gender and age group

gender_age_purchase_pivot.plot(kind='bar', x='Age', y='F', 

                            ax=ax7, position=1, width=width, color='purple')

gender_age_purchase_pivot.plot(kind='bar', x='Age', y='M', 

                            ax=ax7, position=2, width=width, color='tomato')



ax7.set_title("Purchase by age group and gender")

ax7.set_ylabel("Total Purchase (Billions)")



#Plot the ave purchase by gender and age group

gender_age_ave_purchase_pivot.plot(kind='bar', x='Age', y='F', 

                            ax=ax8, position=1, width=width, color='purple')

gender_age_ave_purchase_pivot.plot(kind='bar', x='Age', y='M', 

                            ax=ax8, position=2, width=width, color='tomato')



ax8.set_title("Median Purchase by age group and gender")

ax8.set_ylabel("Median Purchase (Billions)")



#Finally, show the plot

fig6.tight_layout()



plt.show()

#Look at amount spend by each age group (bar and box plots)

#fig9 = plt.figure(figsize=(18, 15))
#Look at the breakdown of the categories for each age group/gender

#fig10 = plt.figure(figsize=(18, 15))
occupation_count.sort_values(by=['Occupation count'], inplace=True,  ascending=False)

occupation_count.head()
# Find out the total purchase amount of each occupation

total_purchase_occupation = df.groupby("Occupation")["Purchase"].sum().reset_index(name = "Total Amount")

total_purchase_occupation.head()
occupation = pd.merge(total_purchase_occupation, occupation_count,  on='Occupation')

occupation.head()
fig11 = plt.figure(figsize=(12,6))

ax11 = fig11.add_subplot(111)



sns.boxplot(x = 'Occupation', y = 'Total Purchase', data = total_by_cust, ax=ax11)

ax11.set_title("Boxplot of purchase amount by occupation")

plt.show()
#In general, there seem to be quite a lot of outliers. Therefore, the median will be used

#Median spending by occupation

median_purchase_occupation = df.groupby("Occupation")["Purchase"].median().reset_index(name = "Median Amount")

median_purchase_occupation.head()
occupation.head()
# Create the figure and add the subplot

fig12 = plt.figure(figsize=(12,8))

ax12 = fig12.add_subplot(221)

ax13 = fig12.add_subplot(222)

ax14 = fig12.add_subplot(223)



colors = {0: 'gold', 1: 'navy', 2: 'orange', 3: 'g', 4: 'r', 5: 'purple', 6:'chocolate', 7:'m', 8:'slategrey', 

          9:'darkolivegreen', 10:'teal', 11:'deepskyblue', 12:'darkseagreen', 13:'lightcoral',

          14:'tan', 15:'bisque', 16:'mediumaquamarine', 17:'violet', 18:'thistle', 19:'pink', 20:'steelblue'}



# Plot the sorted bar graph here (Total Purchase Amount)

occupation.sort_values(by = "Total Amount", ascending = False).plot(

    kind="bar", x="Occupation" ,y="Total Amount", ax=ax12, legend = False, 

    color=[colors[i] for i in occupation['Occupation']])



ax12.set_title("Total purchase amount by occupation")

ax12.set_ylabel("Purchase amount")



# Plot the sorted bar graph here (Median Purchase Amount)

median_purchase_occupation.sort_values(by = "Median Amount", ascending = False).plot(

    kind="bar", x="Occupation" ,y="Median Amount", ax=ax13, legend = False,

    color=[colors[i] for i in occupation['Occupation']])

ax13.set_title("Median purchase amount by occupation")

ax13.set_ylabel("Purchase amount")



# Plot number of buyers

occupation.sort_values(by = "Occupation count", ascending = False).plot(

    kind='bar', x='Occupation', y='Occupation count', ax=ax14, 

    color=[colors[i] for i in occupation['Occupation']])



ax14.set_title("Buyers by occupation")

ax14.set_ylabel("Number of buyers")



fig12.tight_layout()



plt.show()



#I have no idea why the colours are not showing up by group
customers_city.head()
# Find out the total purchase amount of each city category

total_purchase_city = total_by_cust.groupby("City_Category")["Total Purchase"].sum().reset_index()

total_purchase_city.head()



# Find out the median purchase amount of each city category

median_purchase_city = total_by_cust.groupby("City_Category")["Total Purchase"].median().reset_index(name = "Median Amount")

median_purchase_city.head()
# Create the figure and add the subplot

fig15 = plt.figure(figsize=(16,6))

ax15 = fig15.add_subplot(121)

ax16 = fig15.add_subplot(122)



# Plot the bar graph here

total_purchase_city.plot(kind="bar", x="City_Category" ,y="Total Purchase", ax=ax15)

ax15.set_title("Total purchase amount by city category")

ax15.set_ylabel("Total amount")



median_purchase_city.plot(kind="bar", x="City_Category" ,y="Median Amount", ax=ax16)

ax16.set_title("Median purchase amount by city category")

ax16.set_ylabel("Median amount")



plt.show()
item_count_city = df.groupby("User_ID")["Product_ID"].count().reset_index(name = "Item Count")

item_count_city.head()
cust_city = total_by_cust.drop(['Gender', 'Age', "Occupation",  

                                "Stay_In_Current_City_Years", "Marital_Status", "Total Purchase"], axis=1)

cust_city.head()
def get_city(x): 

    user = total_by_cust["User_ID"] == x

    line = total_by_cust[user]

    city_letter = line["City_Category"].iloc[0]

    return city_letter
item_count_city["City"] = df["User_ID"].apply(get_city)

item_count_city.head()
#Are people in city a/b buying more items?



# Create the figure and add the subplot

fig16 = plt.figure(figsize=(16,6))

ax17 = fig16.add_subplot(121)

ax18 = fig16.add_subplot(122)



# Find out the mean purchase amount of each city category

mean_count_items = item_count_city.groupby("City")["Item Count"].mean().reset_index(name = "Mean Amount")

#print(mean_count_items)



median_count_items = item_count_city.groupby("City")["Item Count"].median().reset_index(name = "Median Amount")

#print(mean_count_items)



# Plot the mean bar graph

mean_count_items.plot(kind="bar", x="City" ,y="Mean Amount", ax=ax17)

ax17.set_title("Mean number of items purchased by city category")

ax17.set_ylabel("Mean number of items purchased")



# Plot the median bar graph

median_count_items.plot(kind="bar", x="City" ,y="Median Amount", ax=ax18)

ax18.set_title("Median number of items purchased by city category")

ax18.set_ylabel("Median number of items purchased")



plt.show()
marital_count
total_by_cust.head()
# Find out the total purchase amount of married or single customers with different gender

purchase_marital_gender = total_by_cust.groupby(

    ["Marital_Status", "Gender"])["Total Purchase"].sum().reset_index()



# Convert the purchase_marital_gender dataframe into a pivot table

purchase_pivot2 = purchase_marital_gender.pivot_table(index = "Marital_Status",

                                                      columns = "Gender",

                                                     values = "Total Purchase",

                                                     aggfunc=np.sum)



purchase_pivot2.head()
# Create the figure and add the subplot

fig19 = plt.figure(figsize=(8,8))

ax19 = fig19.add_subplot(111)



# Plot the stacked bar chart here

purchase_pivot2.plot(kind="bar", stacked = True, ax=ax19)

ax19.set_title("Total purchase amount of married and single customers by gender")

ax19.set_ylabel("Total amount")



plt.show()
# Find out the total purchase amount of married or single customers with age groups

purchase_marital_gender = total_by_cust.groupby(

    ["Marital_Status", "Age"])["Total Purchase"].sum().reset_index()



purchase_marital_gender_pivot = pd.pivot_table(

    purchase_marital_gender,

    index="Age",

    columns="Marital_Status",

    values="Total Purchase",

    aggfunc=sum

)



purchase_marital_gender_pivot.columns = ["0", "1"]

purchase_marital_gender_pivot = purchase_marital_gender_pivot.reset_index()



purchase_marital_gender_pivot["1"].fillna(0, inplace=True)

purchase_marital_gender_pivot = purchase_marital_gender_pivot.astype({"0": int, "1": int})





purchase_marital_gender_pivot
# Find out the total purchase amount of married or single customers with age groups

median_purchase_marital_gender = total_by_cust.groupby(

    ["Marital_Status", "Age"])["Total Purchase"].median().reset_index(name="Median Purchase")



median_purchase_marital_gender_pivot = pd.pivot_table(

    median_purchase_marital_gender,

    index="Age",

    columns="Marital_Status",

    values="Median Purchase",

    aggfunc=np.median,

    fill_value=0

)



median_purchase_marital_gender_pivot.columns = ["0", "1"]

median_purchase_marital_gender_pivot = median_purchase_marital_gender_pivot.reset_index()



median_purchase_marital_gender_pivot
median_purchase_marital_gender_pivot["total"] = median_purchase_marital_gender_pivot["0"] + median_purchase_marital_gender_pivot["1"]

    

median_perc = median_purchase_marital_gender_pivot.copy()



median_perc['0 %'] = round(median_perc['0'] / median_perc['total'] * 100,2)

median_perc['1 %'] = round(median_perc['1'] / median_perc['total'] * 100,2)

#print(median_perc)



perc = []

for index, row in median_perc.iterrows():

    perc.append(str(row["0 %"]) + "%")



for index, row in median_perc.iterrows():

    perc.append(str(row["1 %"]) + "%")



perc
width=0.20



#Create the figure

fig20 = plt.figure(figsize=(10, 9))



#Add the subplot

ax20 = fig20.add_subplot(211)

ax21 = fig20.add_subplot(212)



#Plot the values (Total Purchase)

purchase_marital_gender_pivot.plot(kind='bar', x='Age', y='0', 

                            ax=ax20, position=2, width=width, color='navy')

purchase_marital_gender_pivot.plot(kind='bar', x='Age', y='1', 

                            ax=ax20, position=1, width=width, color='limegreen')



#Add the title of the plot

ax20.set_title("Total Purchase by age group and maritial status")

ax20.set_ylabel("Total Purchase (Billion)")



#Plot the values (Median Purchase)

median_purchase_marital_gender_pivot.plot(kind='bar', x='Age', y='0', 

                            ax=ax21, position=2, width=width, color='navy')

median_purchase_marital_gender_pivot.plot(kind='bar', x='Age', y='1', 

                            ax=ax21, position=1, width=width, color='limegreen')



#Add the title of the plot

ax21.set_title("Median Purchase by age group and maritial status")

ax21.set_ylabel("Median Purchase $")



# To add the number label on top of the bars

for rect, label in zip( ax21.patches, perc):

    height = rect.get_height()

    x_value = rect.get_x() + rect.get_width() / 2

    ax21.text(x_value, height + 5, label,

            ha='left', va='bottom')



#Finally, show the plot

plt.tight_layout()

plt.show()