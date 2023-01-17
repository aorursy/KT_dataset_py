import pandas as pd

import numpy as np

import re



pd.options.display.float_format = '{:20,.2f}'.format
# Below are the encoding that used to read files

encoding_cp = "cp1252"

encoding_iso = "ISO-8859-1"

encoding_utf = "utf-8"

encoding_latin_1="latin-1"

encoding_latin="latin"

encoding_utf_sig="utf-8-sig"

encoding_unicode="unicode-escape"

encoding_raw_unicode="raw-unicode-escape"



companies = pd.read_csv("../input/companies.txt", sep="\t", encoding = encoding_latin)

rounds2 = pd.read_csv("../input/rounds2.csv",encoding = encoding_latin)
# Can use lambda x:x.lower() but it needs to be repeated a lot so function is better 

def lowerCase(name):

    return name.lower()



# Method to remove special character 

def cleanNameField(name):

    cleanedLiteral=""

    for item in re.compile(r'[0-9a-zA-Z-+/.]').findall(name):

        cleanedLiteral +=str(item)

    return str(cleanedLiteral)
companies["permalink"] = companies["permalink"].apply(lowerCase)

rounds2["company_permalink"] = rounds2["company_permalink"].apply(lowerCase)



# Removing special character from unique fields to compare the dataset correctly

companies["permalink"] = companies["permalink"].apply(cleanNameField)

rounds2["company_permalink"] = rounds2["company_permalink"].apply(cleanNameField)
rounds2.head(5)
companies.head(5)
rounds2.company_permalink.nunique()
companies.permalink.nunique()
# Create index using company unique ID

companies_index = pd.Index(companies["permalink"])

rounds2_index = pd.Index(rounds2["company_permalink"])



rounds2_index.difference(companies_index)



# Difference between companis and rounds2 only when data is not cleaned and has special character



# No diffrence if data is cleaned of special character  



# Answer is N , no diffrence 
# Renaming company column name so to merge smoothly

companies.rename(columns={'permalink':'company_permalink'},inplace=True)



# Mering both to create a master_frame

master_frame = pd.merge(companies,rounds2,how='inner',on='company_permalink')



print(len(master_frame))

print((master_frame.shape))
master_frame.category_list = master_frame.category_list.astype(str)



master_frame["status"] = master_frame["status"].apply(lowerCase)

master_frame["funding_round_type"] = master_frame["funding_round_type"].apply(lowerCase)



# Removing NaN values rows as this field will play important role with analysis

master_frame = master_frame[pd.notnull(master_frame['country_code'])]

master_frame = master_frame[pd.notnull(master_frame['category_list'])]



# Removing NaN values rows as this field will play important role with analysis

master_frame = master_frame[pd.notnull(master_frame['raised_amount_usd'])]
# Capitlaizing column names for better visuals and diffrentiation 

master_frame.rename(columns=lambda x: x.title(), inplace=True)



master_frame.head()
print(master_frame.Funding_Round_Type.unique())


venture_funding_frame = master_frame[master_frame["Funding_Round_Type"].isin(["venture"])]

angel_funding_frame = master_frame[master_frame["Funding_Round_Type"].isin(["angel"])]

seed_funding_frame = master_frame[master_frame["Funding_Round_Type"].isin(["seed"])]

private_equity_funding_frame = master_frame[master_frame["Funding_Round_Type"].isin(["private_equity"])]



print("Average Venture Funding : "+ str(venture_funding_frame.Raised_Amount_Usd.mean()))

print("Average Angle Funding : "+ str(angel_funding_frame.Raised_Amount_Usd.mean()))

print("Average Seed Funding : "+ str(seed_funding_frame.Raised_Amount_Usd.mean()))

print("Average Private Equity Funding : "+ str(private_equity_funding_frame.Raised_Amount_Usd.mean()))

# Grouping data as per fund rounding type

funding_round_type = master_frame.groupby('Funding_Round_Type').Raised_Amount_Usd.mean().reset_index()



# Applying business restrictions and making sure to filter out values not included 

funding_round_type = funding_round_type.loc[ (funding_round_type.Raised_Amount_Usd >= 5000000) & 

                                             (funding_round_type.Raised_Amount_Usd <= 15000000)]



# Sorting vales as per amount

funding_round_type.sort_values(["Raised_Amount_Usd"], axis=0,ascending=False, inplace=True)

 

 # Get the top value    

funding_round_type
# To be removed 

invetment_type_df_temp = master_frame.groupby('Funding_Round_Type').Raised_Amount_Usd.mean().reset_index()

invetment_type_df_temp.sort_values(["Raised_Amount_Usd"], axis=0,ascending=False, inplace=True)

invetment_type_df_temp
# Calculating total sum country wise

venture_country_code = venture_funding_frame.groupby("Country_Code").Raised_Amount_Usd.sum().reset_index()

venture_country_code.sort_values(["Raised_Amount_Usd"], axis=0,ascending=False, inplace=True)



# Seleting top 9 countries 

top9 = venture_country_code.head(9)

top9

# Sample list created by refrencing above links

english_country_list_codes = ['USA','AUS','CAN','IND','BMU','GBR','NZL','GIB','IRL']
top3_funded = top9.loc[top9.Country_Code.isin(english_country_list_codes)].head(3)

top3_funded
master_frame['Primary_Sector'] = master_frame.Category_List.apply(lambda x:x.split('|')[0].title())
mapping_frame = pd.read_csv("../input/mapping.csv",encoding = "unicode-escape")

mapping_frame = mapping_frame[pd.notnull(mapping_frame['category_list'])]
def replace_zero_with_na(name) :

    index = name.find('0')    

    if index!= 0 :

        if name[index-1]!='.' :  # making sure that Expressoin 2.0 wont get replaced

            name = name.replace('0','na')

    elif index== 0 :             # making sure that first do get replaced    

            name = name.replace('0','na')

            

    return name.title()        
# Creating data frame with category list and main sector mapping only 

list_category = []

list_main = []

main_sectors_list = list(mapping_frame.columns)

main_sectors_list.pop(0)



# Iterating over dataframe

for row in mapping_frame.itertuples():

    for iCount in range(1,11):

        if row[iCount] == 1 :

            list_category.append(row[1]) 

            list_main.append(main_sectors_list[iCount-2])

         





 # Making sure that we do not have any incorrect sector       

for iCnt in range(len(list_category)): 

      list_category[iCnt] = replace_zero_with_na(list_category[iCnt]) 

        

mapping_frame_consolidated = pd.DataFrame({'Primary_Sector': list_category,'Main_Sector': list_main })

mapping_frame_consolidated.head(5)

 



sector_master_frame = pd.merge(master_frame,mapping_frame_consolidated,how='inner',on='Primary_Sector')

sector_master_frame.head(5)
# Creating alias with shorter name for readable code

smf = sector_master_frame
D1_temp = smf.loc[(smf.Country_Code=="USA") & 

                  (smf.Funding_Round_Type=="venture") & 

                  (smf.Raised_Amount_Usd >=5000000) & (smf.Raised_Amount_Usd <= 15000000), :]



#Total amount invested in each main sector in a separate column

D1_total_sum_investments = D1_temp.groupby("Main_Sector").Raised_Amount_Usd.sum().sort_values(ascending = False).to_frame(name='Total_Sum_Invested')



#Total number (or count) of investments for each main sector in a separate column

D1_total_count_investments = D1_temp.groupby("Main_Sector").Raised_Amount_Usd.count().sort_values(ascending = False).to_frame(name='Counter')



# Merging frames to created final D1

D1_temp= pd.merge(D1_temp,D1_total_count_investments,how='inner',on='Main_Sector')

D1 = pd.merge(D1_temp,D1_total_sum_investments,how='inner',on='Main_Sector')



D1.head(5)
D2_temp = smf.loc[(smf.Country_Code=="GBR") & 

                  (smf.Funding_Round_Type=="venture") & 

                  (smf.Raised_Amount_Usd >=5000000) & (smf.Raised_Amount_Usd <= 15000000), :]



D2_total_sum_investments = D2_temp.groupby("Main_Sector").Raised_Amount_Usd.sum().sort_values(ascending = False).to_frame(name='Total_Sum_Invested')

D2_total_count_investments = D2_temp.groupby("Main_Sector").Raised_Amount_Usd.count().sort_values(ascending = False).to_frame(name='Counter')



D2_temp= pd.merge(D2_temp,D2_total_count_investments,how='inner',on='Main_Sector')

D2 = pd.merge(D2_temp,D2_total_sum_investments,how='inner',on='Main_Sector')



D2.head(5)

D3_temp = smf.loc[(smf.Country_Code=="IND") & 

                  (smf.Funding_Round_Type=="venture") & 

                  (smf.Raised_Amount_Usd >=5000000) & (smf.Raised_Amount_Usd <= 15000000), :]



D3_total_sum_investments = D3_temp.groupby("Main_Sector").Raised_Amount_Usd.sum().sort_values(ascending = False).to_frame(name='Total_Sum_Invested')

D3_total_count_investments = D3_temp.groupby("Main_Sector").Raised_Amount_Usd.count().sort_values(ascending = False).to_frame(name='Counter')



D3_temp= pd.merge(D3_temp,D3_total_count_investments,how='inner',on='Main_Sector')

D3 = pd.merge(D3_temp,D3_total_sum_investments,how='inner',on='Main_Sector')



D3.head(5)
print("Total number of investments for D1 : " + str(D1.Raised_Amount_Usd.count()))

print("Total number of investments for D2 : " + str(D2.Raised_Amount_Usd.count()))

print("Total number of investments for D3 : " + str(D3.Raised_Amount_Usd.count()))
print("Total sum of investments for D1 : " + str(D1.Raised_Amount_Usd.sum()))

print("Total sum of investments for D2 : " + str(D2.Raised_Amount_Usd.sum()))

print("Total sum of investments for D3 : " + str(D3.Raised_Amount_Usd.sum()))

D1_total_count_investments = D1.groupby("Main_Sector").Total_Sum_Invested.count().sort_values(ascending = False).to_frame(name='Total_Count')

D2_total_count_investments = D2.groupby("Main_Sector").Total_Sum_Invested.count().sort_values(ascending = False).to_frame(name='Total_Count')

D3_total_count_investments = D3.groupby("Main_Sector").Total_Sum_Invested.count().sort_values(ascending = False).to_frame(name='Total_Count')



print("Top 3 sector for D1 :")

print(D1_total_count_investments.head(3))

print()

print("Top 3 sector for D2 :")

print(D2_total_count_investments.head(3))

print()

print("Top 3 sector for D3 :")

print(D3_total_count_investments.head(3))

print()

print("Top 3 sector for D1 :")

D1_total_count_investments.head(3)
print("Top 3 sector for D2 :")

D2_total_count_investments.head(3)

print("Top 3 sector for D3 :")

D3_total_count_investments.head(3)

# See above 
# See above
# See above
# See above
# See above
# Filter by Top sector and group by comany unique information and name

D1_temp = D1[D1.Main_Sector == "Others"]

D1_temp = D1_temp.groupby(["Company_Permalink","Name"]).Raised_Amount_Usd.sum().reset_index()

D1_temp.sort_values(["Raised_Amount_Usd"], axis=0,ascending=False, inplace=True)

d1_top_list = (D1_temp.iloc[0:1, 1:3].values.tolist())[0]



D2_temp = D2[D2.Main_Sector == "Others"]

D2_temp=D2_temp.groupby(["Company_Permalink","Name"]).Raised_Amount_Usd.sum().reset_index()

D2_temp.sort_values(["Raised_Amount_Usd"], axis=0,ascending=False, inplace=True)

d2_top_list = (D2_temp.iloc[0:1, 1:3].values.tolist())[0]



D3_temp = D3[D3.Main_Sector == "Others"]

D3_temp=D3_temp.groupby(["Company_Permalink","Name"]).Raised_Amount_Usd.sum().reset_index()

D3_temp.sort_values(["Raised_Amount_Usd"], axis=0,ascending=False, inplace=True)

d3_top_list = (D3_temp.iloc[0:1, 1:3].values.tolist())[0]



print("Analysis Results -")

print(" '{0}' company recived highest investments for D1 worth total (sum) $".format(d1_top_list[0]),d1_top_list[1])

print(" '{0}' company recived highest investments for D2 worth total (sum) $".format(d2_top_list[0]),d2_top_list[1])

print(" '{0}' company recived highest investments for D3 worth total (sum) $".format(d3_top_list[0]),d3_top_list[1])
# Filter by Top sector and group by comany unique information and name

D1_temp = D1[D1.Main_Sector == "Social, Finance, Analytics, Advertising"]

D1_temp = D1_temp.groupby(["Company_Permalink","Name"]).Raised_Amount_Usd.sum().reset_index()

D1_temp.sort_values(["Raised_Amount_Usd"], axis=0,ascending=False, inplace=True)

d1_top_list = (D1_temp.iloc[0:1, 1:3].values.tolist())[0]



D2_temp = D2[D2.Main_Sector == "Social, Finance, Analytics, Advertising"]

D2_temp=D2_temp.groupby(["Company_Permalink","Name"]).Raised_Amount_Usd.sum().reset_index()

D2_temp.sort_values(["Raised_Amount_Usd"], axis=0,ascending=False, inplace=True)

d2_top_list = (D2_temp.iloc[0:1, 1:3].values.tolist())[0]



D3_temp = D3[D3.Main_Sector == "Social, Finance, Analytics, Advertising"]

D3_temp=D3_temp.groupby(["Company_Permalink","Name"]).Raised_Amount_Usd.sum().reset_index()

D3_temp.sort_values(["Raised_Amount_Usd"], axis=0,ascending=False, inplace=True)

d3_top_list = (D3_temp.iloc[0:1, 1:3].values.tolist())[0]



print("Analysis Results -")

print(" '{0}' company recived highest investments for D1 worth total (sum) $".format(d1_top_list[0]),d1_top_list[1])

print(" '{0}' company recived highest investments for D2 worth total (sum) $".format(d2_top_list[0]),d2_top_list[1])

print(" '{0}' company recived highest investments for D3 worth total (sum) $".format(d3_top_list[0]),d3_top_list[1])

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
plotting_frame = sector_master_frame[sector_master_frame["Funding_Round_Type"].isin(

                       ["venture","seed","private_equity"])]



plotting_frame = plotting_frame.loc[(plotting_frame.Raised_Amount_Usd >=5000000) & 

                                    (plotting_frame.Raised_Amount_Usd <= 15000000),:]
sns.boxplot(x='Funding_Round_Type', y='Raised_Amount_Usd', data=plotting_frame)

plt.yscale('log')

plt.show()
plotting = top9.set_index("Country_Code")

plotting.plot.bar(logy=True);

plotting

D1_plot = D1



D1_plot = D1_plot.groupby("Main_Sector").Raised_Amount_Usd.count().reset_index()

D1_plot.sort_values(["Raised_Amount_Usd"], axis=0,ascending=False, inplace=True)

D1_plot = D1_plot.head(3)

D2_plot = D2



D2_plot = D2_plot.groupby("Main_Sector").Raised_Amount_Usd.count().reset_index()

D2_plot.sort_values(["Raised_Amount_Usd"], axis=0,ascending=False, inplace=True)

D2_plot = D2_plot.head(3)
D3_plot = D3



D3_plot = D3_plot.groupby("Main_Sector").Raised_Amount_Usd.count().reset_index()

D3_plot.sort_values(["Raised_Amount_Usd"], axis=0,ascending=False, inplace=True)

D3_plot = D3_plot.head(3)
D12 = pd.merge(D1_plot,D2_plot,how='outer',on='Main_Sector')

D123 = pd.merge(D12,D3_plot,how='outer',on='Main_Sector')
D123 = D123.rename(columns={"Raised_Amount_Usd_x": "USD", "Raised_Amount_Usd_y": "GBP" ,"Raised_Amount_Usd": "INR"})

D123= D123.set_index("Main_Sector")

D123.fillna(0)
D123.T.plot.bar(logy=True)