from IPython.display import Image
Image("D:../input/image.jpg", height = '200', width = '800')
# Import the required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
%matplotlib inline
# Display pandas DataFrame of floats using a format string for columns   

pd.options.display.float_format = '{:20,.2f}'.format
# Use a function to change the permalink and company permalink into lower-case

def lowercase(val):
    return val.lower()

# We will use a method to remove the special character from the data sets

def remove_spe_chr(val):
    removeliteral=""
    for i in re.compile(r'[0-9a-zA-Z-+/.]').findall(val):
        removeliteral += str(i)
    return str(removeliteral)
# Read the file by using the encoder

companies = pd.read_csv('D:../input/companies.txt', sep="\t", encoding = 'ISO-8859-1')
companies.head(5)
rounds2 = pd.read_csv('D:../input/rounds2.csv',encoding = 'ISO-8859-1')
rounds2.head(5)
# Format the fields of a dataframe into lowercase

companies["permalink"] = companies["permalink"].apply(lowercase)
rounds2["company_permalink"] = rounds2["company_permalink"].apply(lowercase)
# Format the fields of a dataframe for special characters

companies["permalink"] = companies["permalink"].apply(remove_spe_chr)
rounds2["company_permalink"] = rounds2["company_permalink"].apply(remove_spe_chr)
# No. of unique companies are present in rounds2

rounds2.company_permalink.nunique()
# No. of unique companies are present in companies

companies.permalink.nunique()
# Permalink

companies['permalink'].head(5)
# Any companies in the rounds2 file which are not present in companies

val1 = pd.DataFrame(rounds2.company_permalink.unique())
val2 = pd.DataFrame(companies.permalink.unique())
val2.equals(val1)
# Answer yes or no: Y/N

if val2.equals(val1):
    print('Y')
else:
    print('N')
# To merge the dataframes we will rename the company column name

companies.rename(columns = {'permalink':'company_permalink'}, inplace = True)
# Now we will merge both the dataframes together

master_frame = pd.merge(companies, rounds2, how = 'left', on = 'company_permalink')
master_frame.head(8)
# No. of observations which are present in master_fram

observations = len(master_frame)
observations
# Remove the NaN values from 

master_frame = master_frame[pd.notnull(master_frame['country_code'])]
master_frame = master_frame[pd.notnull(master_frame['category_list'])]
master_frame = master_frame[pd.notnull(master_frame['raised_amount_usd'])]
# For venture funding

venture_funding = master_frame[master_frame["funding_round_type"].isin(["venture"])]
print("Average funding amount of venture type : "+ str(venture_funding.raised_amount_usd.mean()))
# For seed funding

seed_funding = master_frame[master_frame["funding_round_type"].isin(["seed"])]
print("Average funding amount of seed type : "+ str(seed_funding.raised_amount_usd.mean()))
# For angle funding

angel_funding = master_frame[master_frame["funding_round_type"].isin(["angel"])]
print("Average funding amount of angle type : "+ str(seed_funding.raised_amount_usd.mean()))
# For Private Equity

private_equity_funding = master_frame[master_frame["funding_round_type"].isin(["private_equity"])]
print("Average Private Equity Funding : "+ str(private_equity_funding.raised_amount_usd.mean()))
# Group by the data as per the fund rounding type

funding_round_type = master_frame.groupby('funding_round_type').raised_amount_usd.mean().reset_index()
funding_round_type
# Filter the data so it only contains the chosen investment type

funding_round_type = funding_round_type.loc[(funding_round_type.raised_amount_usd >= 5000000) & (funding_round_type.raised_amount_usd <= 15000000)]
funding_round_type
# The top nine countries which have received the highest total funding (across ALL sectors for the chosen investment type)

venture_cc = venture_funding.groupby("country_code").raised_amount_usd.sum().reset_index()
venture_cc.sort_values(['raised_amount_usd'], axis = 0, ascending = False, inplace = True)
top9 = venture_cc.head(9)
top9
# The top three English-speaking countries in the data frame top9 are

english_country_list_codes = ['USA','AUS','CAN','IND','BMU','GBR','NZL','GIB','IRL']
top_3 = top9.loc[top9.country_code.isin(english_country_list_codes)].head(3)
top_3
# Extracting the primary sector of each category list from the category_list column

sectors_seperator = lambda x:x.split('|')[0].title()
master_frame['primary_sector'] = master_frame.category_list.apply(sectors_seperator)
# Read the mapping.csv file & replace all the na & 2.na values with 0

mapping = pd.read_csv('D:../input/mapping.csv')
mapping = mapping[pd.notnull(mapping['category_list'])]
mapping.category_list = mapping.category_list.replace({'0':'na', '2.na' :'2.0'}, regex=True)
mapping.head()
mapping = pd.melt(mapping, id_vars =['category_list'], value_vars =['Manufacturing','Automotive & Sports','Cleantech / Semiconductors','Entertainment','Health','News, Search and Messaging','Others','Social, Finance, Analytics, Advertising']) 
mapping.head(8)
mapping = mapping[~(mapping.value == 0)]
mapping = mapping.drop('value', axis = 1)
mapping.head(8)
mapping = mapping.rename(columns = {"variable":"main_sector"})
mapping.head()
# Merged data frame with each primary sector mapped to its main sector

master_frame_sector = pd.merge(master_frame, mapping, how = 'inner', on = 'category_list')
master_frame_sector.head(8)
# For D1

D1_SA2 = master_frame_sector.loc[(master_frame_sector.country_code == "USA") & (master_frame_sector.funding_round_type == "venture") & 
                             (master_frame_sector.raised_amount_usd >= 5000000) & (master_frame_sector.raised_amount_usd <= 15000000), :]
# The total amount invested for each main sector in a separate column

D1_total_sum = D1_SA2.groupby("main_sector").raised_amount_usd.sum().sort_values(ascending = False).to_frame(name='total_sum_invested')
# The total number (or count) of investments for each main sector in a separate column

D1_total_count = D1_SA2.groupby("main_sector").raised_amount_usd.count().sort_values(ascending = False).to_frame(name='count')
# Merging frames to created final D1

D1_SA2 = pd.merge(D1_SA2, D1_total_count, how='inner', on='main_sector')
D1 = pd.merge(D1_SA2, D1_total_sum, how='inner', on='main_sector')
D1.head(8)
# For D2

D2_SA2 = master_frame_sector.loc[(master_frame_sector.country_code=="GBR") & (master_frame_sector.funding_round_type=="venture") & 
                             (master_frame_sector.raised_amount_usd >=5000000) & (master_frame_sector.raised_amount_usd <= 15000000), :]
# The total amount invested for each main sector in a separate column

D2_total_sum = D2_SA2.groupby("main_sector").raised_amount_usd.sum().sort_values(ascending = False).to_frame(name='total_sum_invested')
# The total number (or count) of investments for each main sector in a separate column

D2_total_count = D2_SA2.groupby("main_sector").raised_amount_usd.count().sort_values(ascending = False).to_frame(name='count')
# Merging frames to created final D2

D2_SA2 = pd.merge(D2_SA2, D2_total_count, how='inner', on='main_sector')
D2 = pd.merge(D2_SA2, D2_total_sum, how='inner', on='main_sector')
D2.head(8)
# For D3

D3_SA2 = master_frame_sector.loc[(master_frame_sector.country_code == "IND") & (master_frame_sector.funding_round_type == "venture") & 
                                  (master_frame_sector.raised_amount_usd >= 5000000) & (master_frame_sector.raised_amount_usd <= 15000000), :]
# The total amount invested for each main sector in a separate column

D3_total_sum = D3_SA2.groupby("main_sector").raised_amount_usd.sum().sort_values(ascending = False).to_frame(name='total_sum_invested')
# The total number (or count) of investments for each main sector in a separate column

D3_total_count = D3_SA2.groupby("main_sector").raised_amount_usd.count().sort_values(ascending = False).to_frame(name='count')
# Merging frames to created final D3

D3_SA2 = pd.merge(D3_SA2, D3_total_count, how='inner', on='main_sector')
D3 = pd.merge(D3_SA2, D3_total_sum, how='inner', on='main_sector')
D3.head(8)
# 1. Total number of investments (count)

print("Total number of investments for Country1 : " + str(D1.raised_amount_usd.count()))
print("Total number of investments for Country2 : " + str(D2.raised_amount_usd.count()))
print("Total number of investments for Country3 : " + str(D3.raised_amount_usd.count()))
# 2. Total amount of investment (USD)

print("Total amount of investments for Country1 : " + str(D1.raised_amount_usd.sum()))
print("Total amount of investments for Country2 : " + str(D2.raised_amount_usd.sum()))
print("Total amount of investments for Country3 : " + str(D3.raised_amount_usd.sum()))
#  3.Top sector (based on count of investments) for all the 3 countries

D1_total_count_investments = D1.groupby("main_sector").total_sum_invested.count().sort_values(ascending = False).to_frame(name = 'total_count')
print("Top 3 sector for Country1 :")
print(D1_total_count_investments.head(3))
Country1 = D1_total_count_investments.sum()
print(Country1)
D2_total_count_investments = D2.groupby("main_sector").total_sum_invested.count().sort_values(ascending = False).to_frame(name = 'total_count')
print("Top 3 sector for Country2 :")
print(D2_total_count_investments.head(3))
Country2 = D2_total_count_investments.sum()
print(Country2)
D3_total_count_investments = D3.groupby("main_sector").total_sum_invested.count().sort_values(ascending = False).to_frame(name='total_count')
print("Top 3 sector for Country3 :")
print(D3_total_count_investments.head(3))
Country3 = D3_total_count_investments.sum()
print(Country3)
print("Top 3 sector for Country1 :")
D1_total_count_investments.head(3)
print("Top 3 sector for Country2 :")
D2_total_count_investments.head(3)
print("Top 3 sector for Country3 :")
D3_total_count_investments.head(3)
#  4. Second-best sector (based on count of investments)

print("Second best sector for Country1 :")
D1_total_count_investments.iloc[1:2]
print("Second best sector for Country2 :")
D2_total_count_investments.iloc[1:2]
print("Second best sector for Country3 :")
D3_total_count_investments.iloc[1:2]
# 5. Third-best sector (based on count of investments)

print("Third best sector for Country1 :")
D1_total_count_investments.iloc[2:3]
print("Third best sector for Country2 :")
D2_total_count_investments.iloc[2:3]
print("Third best sector for Country3 :")
D3_total_count_investments.iloc[2:3]
# 6. Number of investments in the top sector (refer to point 3)

Country1 = D1_total_count_investments.iloc[:1]
Country1
Country2 = D2_total_count_investments.iloc[:1]
Country2
Country3 = D3_total_count_investments.iloc[:1]
Country3
# 7. Number of investments in the second-best sector (refer to point 4)

Country1 = D1_total_count_investments.iloc[1:2]
Country1
Country2 = D2_total_count_investments.iloc[1:2]
Country2
Country3 = D2_total_count_investments.iloc[1:2]
Country3
# 8. Number of investments in the third-best sector (refer to point 5)

Country1 = D3_total_count_investments.iloc[2:3]
Country1
Country2 = D2_total_count_investments.iloc[2:3]
Country2
Country3 = D3_total_count_investments.iloc[2:3]
Country3
# 9. For the top sector count-wise (point 3), which company received the highest investment?

D1_val = D1[D1.main_sector == "Others"]
D1_val = D1_val.groupby(["company_permalink","name"]).raised_amount_usd.sum().reset_index()
D1_val.sort_values(["raised_amount_usd"], axis = 0, ascending = False, inplace = True)
d1_top_list = (D1_val.iloc[0:1, 1:3].values.tolist())[0]

D2_val = D2[D2.main_sector == "Others"]
D2_val = D2_val.groupby(["company_permalink","name"]).raised_amount_usd.sum().reset_index()
D2_val.sort_values(["raised_amount_usd"], axis = 0, ascending = False, inplace = True)
d2_top_list = (D2_val.iloc[0:1, 1:3].values.tolist())[0]

D3_val = D3[D3.main_sector == "Others"]
D3_val = D3_val.groupby(["company_permalink","name"]).raised_amount_usd.sum().reset_index()
D3_val.sort_values(["raised_amount_usd"], axis = 0, ascending = False, inplace = True)
d3_top_list = (D3_val.iloc[0:1, 1:3].values.tolist())[0]

print(" '{0}' company recived highest investments for Country1 worth total (sum) $".format(d1_top_list[0]),d1_top_list[1])
print(" '{0}' company recived highest investments for Country2 worth total (sum) $".format(d2_top_list[0]),d2_top_list[1])
print(" '{0}' company recived highest investments for Country3 worth total (sum) $".format(d3_top_list[0]),d3_top_list[1])
# 10. For the second-best sector count-wise (point 4), which company received the highest investment?

D1_val = D1[D1.main_sector == "Social, Finance, Analytics, Advertising"]
D1_val = D1_val.groupby(["company_permalink","name"]).raised_amount_usd.sum().reset_index()
D1_val.sort_values(["raised_amount_usd"], axis = 0, ascending = False, inplace = True)
d1_top_list = (D1_val.iloc[0:1, 1:3].values.tolist())[0]

D2_val = D2[D2.main_sector == "Social, Finance, Analytics, Advertising"]
D2_val = D2_val.groupby(["company_permalink","name"]).raised_amount_usd.sum().reset_index()
D2_val.sort_values(["raised_amount_usd"], axis = 0, ascending = False, inplace = True)
d2_top_list = (D2_val.iloc[0:1, 1:3].values.tolist())[0]

D3_val = D3[D3.main_sector == "Social, Finance, Analytics, Advertising"]
D3_val = D3_val.groupby(["company_permalink","name"]).raised_amount_usd.sum().reset_index()
D3_val.sort_values(["raised_amount_usd"], axis = 0, ascending = False, inplace = True)
d3_top_list = (D3_val.iloc[0:1, 1:3].values.tolist())[0]

print(" '{0}' company recived highest investments for Country1 worth total (sum) $".format(d1_top_list[0]),d1_top_list[1])
print(" '{0}' company recived highest investments for Country2 worth total (sum) $".format(d2_top_list[0]),d2_top_list[1])
print(" '{0}' company recived highest investments for Country3 worth total (sum) $".format(d3_top_list[0]),d3_top_list[1])
# A plot showing the fraction of total investments (globally) in angel, venture, seed, and private equity, and the average amount of investment in each funding type.

plotting_frame = master_frame_sector[master_frame_sector["funding_round_type"].isin(["angle","venture","seed","private_equity"])]
plotting_frame = plotting_frame.loc[(plotting_frame.raised_amount_usd >=5000000) & (plotting_frame.raised_amount_usd <= 15000000),:]
sns.violinplot(x = 'funding_round_type', y = 'raised_amount_usd', split = True, inner = "quart", data = plotting_frame)
sns.despine(left = True)
plt.title('Fraction of total investments')
plt.yscale('log')
plt.show()
# A plot showing the top 9 countries against the total amount of investments of funding type FT.

plotting = top9.set_index("country_code")
plotting.plot.bar(logy = True);
plotting
# A plot showing the number of investments in the top 3 sectors of the top 3 countries on one chart (for the chosen investment type FT).

D1_plot = D1
D1_plot = D1_plot.groupby("main_sector").raised_amount_usd.count().reset_index()
D1_plot.sort_values(["raised_amount_usd"], axis = 0, ascending = False, inplace = True)
D1_plot = D1_plot.head(3)

D2_plot = D2
D2_plot = D2_plot.groupby("main_sector").raised_amount_usd.count().reset_index()
D2_plot.sort_values(["raised_amount_usd"], axis = 0, ascending = False, inplace = True)
D2_plot = D2_plot.head(3)

D3_plot = D3
D3_plot = D3_plot.groupby("main_sector").raised_amount_usd.count().reset_index()
D3_plot.sort_values(["raised_amount_usd"], axis = 0, ascending = False, inplace = True)
D3_plot = D3_plot.head(3)
D12 = pd.merge(D1_plot, D2_plot, how = 'outer', on = 'main_sector')
D123 = pd.merge(D12, D3_plot, how = 'outer', on = 'main_sector')
D123 = D123.rename(columns = {"raised_amount_usd_x": "USD", "raised_amount_usd_y": "GBR" ,"raised_amount_usd": "IND"})
D123= D123.set_index("main_sector")
D123.fillna(0)
D123.T.plot.bar(logy = True)