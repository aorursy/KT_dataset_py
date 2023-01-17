#Import and Clean the Data Set 



import pandas as pd



sales_data = pd.read_csv("../input/Sales Data-Table 1.csv")

df =  pd.DataFrame(sales_data)

df.to_csv (r'x.csv') #Don't forget to add '.csv' at the end of the path



sales_data['Gender'] = sales_data['Gender'].str.upper()

sales_data['Sales_USD'] = sales_data['Sales (USD)'].str.replace(",", "")

sales_data['Sales_USD'] = pd.to_numeric(sales_data['Sales_USD'])

#sales_data = sales_data.query('weeknumber == 1')



sales_data_groups = sales_data.groupby(['Brand Name', 'Family', 'Category', 'Gender', 'Season', 'Flag_if_sale']).sum()#.sort_values(by=['Qty_Sold'], ascending=False)

sales_data_df = sales_data_groups.reset_index()



sales_data_df['Average_Price'] = sales_data_df['Sales_USD'] / 4

sales_data_df['Average_Sales'] = sales_data_df['Qty_Sold'] / 4



stock_data = pd.read_csv("../input/Stock Data-Table 1.csv")

stock_data['Flag_if_sale'] = stock_data['Price Type'].apply(lambda x: "Yes" if x == "Mark down" else "No")



stock_data.rename(columns= lambda x: x.replace(" ", "_"), inplace=True)

sales_data_df.rename(columns= lambda x: x.replace(" ", "_"), inplace=True)





sales_and_stock_avg=stock_data.merge(sales_data_df, how="left", on=['Gender', 'Family', 'Category', 'Season', 'Brand_Name', 'Flag_if_sale'])

sales_and_stock_avg.fillna(0, inplace=True)



##data validation

sales_and_stock_avg[(sales_and_stock_avg['Price_Type'] == "Mark down") & (sales_and_stock_avg['Flag_if_sale'] != "Yes")]

#Function to filter the data set then calculate Cover and Sales Mix 



def sales_and_stock_summary(df, gender="MEN", season="AW17", brand=None, category=None, groupby_vals=None, filter_query=None, nlargest=None):

    df = df.query('Flag_if_sale == "Yes"')

    df = df.sort_values(by=['Qty_Sold'], ascending=False)

    if gender:

        df = df.query('Gender =="{}"'.format(gender))

    if season: 

        df = df.query('Season =="{}"'.format(season))

    if brand:

        df = df[df['Brand_Name'] == brand]

    if category:

        df = df[df['Category'] == category]

    if filter_query:

        df = df.query(filter_query)

    if groupby_vals:

        df = df.groupby(groupby_vals)

        df = df.sum().sort_values(by=['Current_Stock_Units'], ascending=False)

#     df['cover_text'] = "average cover..."

#     df['mix_text'] = "% stock mix vs % sales mix"

    df['Cover'] = df['Current_Stock_Units'] / df['Average_Sales']

    df['Cover'] = df['Cover'].astype(int, errors='ignore')

    df['%_Total_Stock'] = df['Current_Stock_Units'] / df['Current_Stock_Units'].sum() * 100

    df['%_Total_Units_Sold'] = df['Qty_Sold'] / df['Qty_Sold'].sum() * 100

    df['Net_Sales_Mix'] =  df['%_Total_Units_Sold'] - df['%_Total_Stock']

    if nlargest:

        df = df.nlargest(nlargest, columns="Current_Stock_Units")

    return df #ascending=False)#.reset_index(drop=False)

# best selling categories 

aw17_fam_summary = sales_and_stock_summary(sales_and_stock_avg,gender=None, groupby_vals=['Family'])[['Current_Stock_Units', 'Average_Sales', 'Cover', '%_Total_Stock', '%_Total_Units_Sold', 'Net_Sales_Mix']]                                                                                  

                                                                                  
aw17_fam_summary.head(10)
import matplotlib.pyplot as plt



import seaborn as sns

from seaborn import barplot



aw17_fam_gender_summary = sales_and_stock_summary(sales_and_stock_avg,gender=None, groupby_vals=['Family', "Gender"])[['Current_Stock_Units', 'Average_Sales', 'Cover', '%_Total_Stock', '%_Total_Units_Sold', 'Net_Sales_Mix']]                                                                                  







# Set axes style to white for first subplot

fig, ax = plt.subplots(figsize=(20,15))

plt.title('AW17 Markdown Stocks & Sales distribution by Family and Gender')

plt.subplot(211)

#plt.xscale('log')

sns.boxplot(x="Family", y="Cover", data=aw17_fam_gender_summary.reset_index().query('Average_Sales > 100 & Current_Stock_Units >100'));



# Initialize second subplot

plt.subplot(212)

sns.barplot(y="Current_Stock_Units", x="Family", data=aw17_fam_gender_summary.reset_index().query('Average_Sales > 0'));

acc_data = sales_and_stock_summary(sales_and_stock_avg,gender=None, groupby_vals=['Family', "Gender", "Brand_Name", "Category"])[['Current_Stock_Units', 'Average_Sales', 'Cover', '%_Total_Stock', '%_Total_Units_Sold', 'Net_Sales_Mix']]                             



acc_data.query('(Family == "Accessories" | Family == "Bags") & Current_Stock_Units > 500').sort_values(by=['Cover'], ascending=False).head(10)
mens_summary = sales_and_stock_summary(sales_and_stock_avg,gender='MEN', groupby_vals=['Family','Category'], filter_query='Family == "Clothing"')[['Current_Stock_Units', 'Average_Sales', 'Cover', '%_Total_Stock', '%_Total_Units_Sold', 'Net_Sales_Mix']]

mens_summary
# Import necessary libraries

import matplotlib.pyplot as plt

import seaborn as sns



# Load data

mens_summary_plot = sales_and_stock_summary(sales_and_stock_avg,gender='MEN', groupby_vals=['Family', 'Category', 'Brand_Name'], filter_query='Family == "Clothing"')[['Current_Stock_Units', 'Average_Sales', 'Cover', '%_Total_Stock', '%_Total_Units_Sold', 'Net_Sales_Mix']]                                                                                  

mens_summary_plot = mens_summary_plot.query('Average_Sales > 0')

mens_bar_plot = sales_and_stock_summary(sales_and_stock_avg,gender='MEN', groupby_vals=['Family', 'Category'], filter_query='Family == "Clothing"')[['Current_Stock_Units', 'Average_Sales', 'Cover', '%_Total_Stock', '%_Total_Units_Sold', 'Net_Sales_Mix']]



# Set plot size

fig, ax = plt.subplots(figsize=(20,12))



# Initialize second subplot

plt.subplot(311)

sns.barplot(y="Cover", x="Category", data=mens_bar_plot.reset_index().sort_values(by='Current_Stock_Units', ascending=False));



# Initialize second subplot

plt.subplot(312)

sns.barplot(y="Average_Sales", x="Category", data=mens_bar_plot.reset_index().sort_values(by='Current_Stock_Units', ascending=False));



order_mens = [i for (k,i) in list(mens_bar_plot.index)]



plt.subplot(313)

sns.swarmplot(x="Category", y="Current_Stock_Units", order=order_mens, data=(mens_summary_plot.reset_index().sort_values(by='Current_Stock_Units', ascending=False)));





# Show the plot                   

plt.show()

sales_and_stock_summary(sales_and_stock_avg,gender='MEN', groupby_vals=['Family','Brand_Name', 'Category'], filter_query='Family == "Clothing"')[['Current_Stock_Units', 'Average_Sales', 'Cover', '%_Total_Stock', '%_Total_Units_Sold', 'Net_Sales_Mix']].head(20)  

womens_clothes = sales_and_stock_summary(sales_and_stock_avg,gender='WOMEN', groupby_vals=['Family','Category'], filter_query='Family == "Clothing"')[['Current_Stock_Units', 'Average_Sales', 'Cover', '%_Total_Stock', '%_Total_Units_Sold', 'Net_Sales_Mix']]                                                     

womens_clothes
womens_summary_plot = sales_and_stock_summary(sales_and_stock_avg,gender='WOMEN', groupby_vals=['Family', 'Category', 'Brand_Name'], filter_query='Family == "Clothing"')[['Current_Stock_Units', 'Average_Sales', 'Cover', '%_Total_Stock', '%_Total_Units_Sold', 'Net_Sales_Mix']]                                                                                  

womens_summary_plot = womens_summary_plot.query('Average_Sales > 1')

womens_bar_plot = sales_and_stock_summary(sales_and_stock_avg,gender='WOMEN', groupby_vals=['Family', 'Category'], filter_query='Family == "Clothing"')[['Current_Stock_Units', 'Average_Sales', 'Cover', '%_Total_Stock', '%_Total_Units_Sold', 'Net_Sales_Mix']]



fig, ax = plt.subplots(figsize=(20,12))

plt.subplot(211)

#plt.yscale('log')

sns.boxplot(x="Category", y="Cover", data=womens_summary_plot.reset_index());



# Initialize second subplot

plt.subplot(212)

sns.swarmplot(y="Current_Stock_Units", x="Category", data=womens_summary_plot.reset_index());



# Show the plot                   

plt.show()

womens_summary_plot.head(20).sort_values(by='Cover', ascending=False).head(20)
sales_and_stock_summary(sales_and_stock_avg,gender=None, groupby_vals=['Family','Gender', 'Category'], filter_query='Family == "Shoes"')[['Current_Stock_Units', 'Average_Sales', 'Cover', '%_Total_Stock', '%_Total_Units_Sold', 'Net_Sales_Mix']].head(10)

best_cover = sales_and_stock_summary(sales_and_stock_avg,gender=None, groupby_vals=['Family','Gender', "Category", "Brand_Name"]).reset_index().sort_values(by="Cover", ascending=True).query('Current_Stock_Units > 1000 & Qty_Sold > 1')

trainers_all_genders = sales_and_stock_summary(sales_and_stock_avg,gender=None, groupby_vals=['Family', 'Brand_Name','Gender', 'Category'], filter_query='Family == "Shoes" & Category== "Trainers"')

trainers_all_genders = trainers_all_genders.query('Current_Stock_Units > 200 & Average_Sales > 1')

fig, ax = plt.subplots(figsize=(20,12))

plt.subplot(211)

sns.boxplot(y="Gender", x="Current_Stock_Units", data=trainers_all_genders.reset_index());



# Initialize second subplot

plt.subplot(212)

sns.boxplot(x="Cover", y="Gender", data=trainers_all_genders.reset_index());



# Show the plot                   

plt.show()
trainers_all_genders = sales_and_stock_summary(sales_and_stock_avg,gender=None, groupby_vals=['Family', 'Brand_Name','Category'], filter_query='Family == "Shoes" & Category== "Trainers"')

trainers_all_genders = trainers_all_genders.query('Current_Stock_Units > 200').head(20)



fig, ax = plt.subplots(figsize=(20,12))

plt.subplot(211)

plt.xlabel = "1"

plt.ylabel= "11"

sns.barplot(x="Brand_Name", y="Average_Sales", data=trainers_all_genders.reset_index());



# Initialize second subplot

plt.subplot(212)

plt.xlabel = "2"

plt.ylabel= "22"

sns.barplot(y="Current_Stock_Units", x="Brand_Name", data=trainers_all_genders.reset_index());



# Show the plot                   

plt.show()

sales_and_stock_summary(sales_and_stock_avg,gender=None, groupby_vals=['Family','Gender', "Brand_Name"], filter_query='Category == "Trainers" & Average_Sales > 1')[['Current_Stock_Units', 'Average_Sales', 'Cover', '%_Total_Stock', '%_Total_Units_Sold', 'Net_Sales_Mix']].sort_values(by='Cover', ascending=False).head(10)
sales_and_stock_summary(sales_and_stock_avg,gender=None, groupby_vals=['Family', "Brand_Name"], filter_query='Average_Sales <= 0')[['Current_Stock_Units', 'Average_Sales', 'Cover', '%_Total_Stock', '%_Total_Units_Sold', 'Net_Sales_Mix']].sort_values(by='Current_Stock_Units', ascending=False).head(20)