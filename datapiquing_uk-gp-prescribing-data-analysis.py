import numpy as np

import pandas as pd

import warnings

import json

import glob

import matplotlib.pyplot as plt
practices = pd.read_csv('../input/general-practice-prescribing-data/practices.csv', \

                        names=['practice', 'name', 'address1', 'address2', 'city', 'county', 'postcode'])

practices.head()
practices.info()
# Count the number of unique counties

len(practices['county'].unique())
# Display the first 10 unique counties

practices['county'].unique()[0:10]
# The list above is NOT all counties. Many are cities (e.g. Hartlepool) and towns (e.g. Yarm) which tends to suggest that

# the practice addresses are not uniformly categorised into 'city' and 'county' but spread inconsistently over 4 columns, i.e. address1, address2, city and county
# How many practices in the 'county' column are in Bradford?

practices[practices['county'] == 'BRADFORD']



# There appears to be 3
# How many practices in the 'city' column are in Bradford?

practices[practices['city'] == 'BRADFORD']



# There appears to be 110
# How many practices have a Bradford postcode 'BD'?

practices[practices['postcode'].str.startswith('BD')]



# There are 167 practices with a Bradford postcode

# The Bradford postcode, 'BD', extends beyond Bradford to other towns in Bradford Metropolitan 

# Council District (e.g. Skipton, Keighley, Ilkley)
# I live near Ilkley, West Yorkshire - so lets take a closer look at all GP practices in my LS29 postcode area

practices[practices['postcode'].str.startswith('LS29')]
# Load the first batch of prescribing data for May 2016

# Need to supress known Numpy error 'FutureWarning'. 

# See https://stackoverflow.com/questions/40659212/futurewarning-elementwise-comparison-failed-returning-scalar-but-in-the-futur





with warnings.catch_warnings():

    warnings.simplefilter(action='ignore', category=FutureWarning)

    bnf_2016_05 = pd.read_csv('../input/general-practice-prescribing-data/T201605PDPI+BNFT.csv')

    

bnf_2016_05.head()    
def extract_json(json_filename, node):

    '''Extract simple json'''



    # 

    with open(json_filename, encoding='utf-8') as data_file:

        data = json.loads(data_file.read())



    return data[node]
# Extract Preparation name 



file = '../input/general-practice-prescribing-data/column_remapping.json'



bnf_name = extract_json(file, 'bnf_name')



df_bnf_name = pd.DataFrame.from_dict(bnf_name, orient='index', columns=['bnf_name'])
print(df_bnf_name.head())
# Extract Preparation BNF code 



file = '../input/general-practice-prescribing-data/column_remapping.json'



bnf_code = extract_json(file, 'bnf_code')



df_bnf_code = pd.DataFrame.from_dict(bnf_code, orient='index', columns=['bnf_code'])
print( df_bnf_code.head())
# Concatenate all prescribing files for 2017 vertically [COMMENTED OUT]



# path = r'../input/general-practice-prescribing-data' # use your path

# all_files = glob.glob(path + "/T2017*.csv")



# li = []



# for filename in all_files:

#     df = pd.read_csv(filename, index_col=None, header=0)

#     li.append(df)



# df_2017 = pd.concat(li, axis=0, ignore_index=True)
# There are approximately 50 million rows of prescribed items in the 2017 datasets  [COMMENTED OUT]



# df_2017.info()
# [COMMENTED OUT]



# df_2017.head()
# Use the 50 million prescribed items in the 2017 data to get a representative DataFrame of all unique, official, bnf_code --> bnf_name mappings

# [COMMENTED OUT]



# df_bnf_mapping = df_2017[['bnf_code', 'bnf_name']].drop_duplicates()

# df_bnf_mapping.info()
# [COMMENTED OUT]



# df_bnf_mapping.head()
# Using the table above, get the BNF preparation name from it's bnf_name id number [COMMENTED OUT]



# df_bnf_name.iloc[2697]
# Get the associated BNF Code of the preparation it is mapped to above

# The bnf_code for 'Propranolol HCL_Cap 80mg M/R' is in Chapter 2, Section 4, Subsection 0 (bnf_code = 020400) which is 'Beta-adrenoreceptor blocking drugs'

# [COMMENTED OUT]



# df_bnf_code.iloc[2237]
# Save the mapping file  [COMMENTED OUT]

#df_bnf_mapping.to_csv('remapping.csv', index=False)
# Create a new column in the bnf_name and bnf_code DataFrames from their index converted to an integer from a string. [COMMENTED OUT]

# Use this column as the new index and drop the original column

# This allows both DataFrames to be joined with the df_bnf_mapping DataFrame as they have the same data type (int).



# df_bnf_name['id'] = df_bnf_name.index.astype(int)

# df_bnf_name.set_index('id', drop=True, inplace=True)

# df_bnf_code['id'] = df_bnf_code.index.astype(int)

# df_bnf_code.set_index('id', drop=True, inplace=True)



# df_bnf_name
# Inner join the df_bnf_mapping DataFrame to the df_bnf_name DataFrame [COMMENTED OUT]



# df_bnf_merge = pd.merge(df_bnf_mapping, df_bnf_name, left_on='bnf_name', right_on='id', suffixes=('_no', ''))

# df_bnf_merge.info()
# [COMMENTED OUT]



# df_bnf_merge.head()
# Inner join the merged mapping+name DataFrame to the df_bnf_code DataFrame [COMMENTED OUT]



# df_bnf = pd.merge(df_bnf_merge, df_bnf_code, left_on='bnf_code', right_on='id', suffixes=('_no', ''))

# df_bnf.info()
# [COMMENTED OUT]



# df_bnf.head()
# [NOT USED] Rename the column headers of the df_bnf mapping DataFrame

#df_bnf.rename(columns={'bnf_code_x':'code_id', 'bnf_name_x':'name_id', 'bnf_code_y':'bnf_code', 'bnf_name_y':'bnf_name' }, inplace=True)

#df_bnf.head()
# [COMMENTED OUT]

# Export the df_bnf DataFrame

# df_bnf.to_csv('full_mapping.csv')

# Read in the file created using the commented-out code above

file = '../input/full-bnf-mapping/full_mapping.csv'



df_bnf = pd.read_csv(file, usecols=[1,2,3,4])
df_bnf.head()
# Retrieve all preparations in BNF Chapter 1, Section 1, Subsection 1 (Antacids and Simethicone)



df_bnf[df_bnf['bnf_code'].str.startswith('010101')].head(10)
# Problems finding the file BNF file to read into a DataFrame

# Turns out the folder in which it resides is lower case and each word is separated by hyphens (and not as it appears

# in the right sidebar)



import os

print(os.listdir("../input"))


BNF = pd.read_csv("../input/bnf-chapter-mapping/BNF.csv", dtype={'ChapterNumber':str})

BNF.head()
# Create a column for the BNF Chapter Number

BNF['chapter_no'] = BNF.apply(lambda row: row.ChapterNumber[0:2], axis=1)
# Create a column for the BNF Chapter and Section Number

BNF['section_no'] = BNF.apply(lambda row: row.ChapterNumber[0:4], axis=1)
# Create a column for the BNF Chapter, Section and Subsection Number

BNF['subsection_no'] = BNF.apply(lambda row: row.ChapterNumber[0:6], axis=1)
# Create a column for the BNF Chapter, Section, Subsection and Subsubsection Number

BNF['subsubsection_no'] = BNF.apply(lambda row: row.ChapterNumber[0:7], axis=1)
BNF.head()
# View all UK practice prescribing for May 2016

bnf_2016_05.head()
burley_practice_id = practices[practices['practice'] == 'B83019'].index[0]



burley_practice_id
# My local GP practice is in Burley In Wharfedale (practice# B83019, index# 629)



# Get the index value of the Burley GP practice from the practices dataframe

burley_practice_id = practices[practices['practice'] == 'B83019'].index[0]



# Use the index value to get all prescribing data for Burley GP practice in May 2016

burley_practice = bnf_2016_05[bnf_2016_05['practice'] == burley_practice_id]

burley_practice
#burley_practice['total_cost'] = burley_practice['act_cost'] * burley_practice['quantity']

#burley_practice.head()
# Join the Burley practice prescribing info to the BNF

burley_practice_prescribing = pd.merge(burley_practice, df_bnf, left_on='bnf_code', right_on='bnf_code_no', suffixes=('_x', ''))

burley_practice_prescribing.drop(['bnf_code_x', 'bnf_name_x'], axis='columns', inplace=True)

burley_practice_prescribing.tail()
# Create a new column in the prescribing data for the 7 digit BNF code prefix

# This prefix indicates the drugs hierarchical position in the BNF, i.e. Chapter > Section > Subsection > Sub-subsection, e.g. 1.3.2.1 or 0103021

burley_practice_prescribing['bnf_code_prefix'] = burley_practice_prescribing.apply(lambda row: row.bnf_code[0:7], axis=1)
burley_practice_prescribing
# Extract all prescribing figures for Angiotensin Converting Enzyme (ACE) inhibitors in the Burley practice, May 2017

# ACE inhibitors are in Chapter 2, Section 5, Subsection 5 Sub-subsection 1 of the BNF. Hence their BNF code begins with '0205051'

burley_practice_prescribing[burley_practice_prescribing['bnf_code'].str.startswith('0205051')]
# Combine the Burley prescribing DataFrame with the BNF DataFrame so drugs can be categorised 

# into therapeutic/pharmaceutical groups (BNF Chapters, Sections etc.) and the total cost per group calculated 

burley_practice_sections = pd.merge(burley_practice_prescribing, BNF, left_on='bnf_code_prefix', right_on='subsubsection_no')
burley_practice_sections.head()
#burley_practice_sections.info()
# Create a DataFrame showing the total cost of all drugs prescribed under the root-level parent Chapter numbers

#pd.options.display.float_format = '£{:20,.2f}'.format

burley_chapter = burley_practice_sections.groupby('chapter_no')[['act_cost']].sum()

burley_chapter.head(20).style.format({"act_cost": "£{:20,.0f}"})
# Combine the Chapter results above with the BNF DataFrame (on chapter_no) then filter out their children to get total costs by Chapter

# Include a vertical bar chart to visualise the relative costs

#pd.options.display.float_format = '£{:20,.2f}'.format

burley_chapter_total = pd.merge(burley_chapter, BNF, left_on='chapter_no', right_on='chapter_no', how='left')

df_chapter = burley_chapter_total[burley_chapter_total['Chapter'].str.count('\.') == 0][['Chapter', 'ChapterTitle', 'act_cost', 'Order']]

df_chapter.style.bar(subset=['act_cost'], color='#d65f5f').format({"act_cost": "£{:20,.0f}"})
# Create a DataFrame showing the total cost of all drugs prescribed under each second-level section number

burley_section = burley_practice_sections.groupby('section_no')[['act_cost']].sum()



# Combine the Section results above with the BNF DataFrame (on section_no) then filter out their children to get total costs by Section

burley_section_total = pd.merge(burley_section, BNF, left_on='section_no', right_on='section_no', how='left')

df_section = burley_section_total[burley_section_total['Chapter'].str.count("\.") == 1][['Chapter', 'ChapterTitle', 'act_cost', 'Order']]

#df_section.style.format({"act_cost": "£{:20,.0f}"})

# Create a DataFrame showing the total cost of all drugs prescribed under each third-level subsection number



burley_subsection = burley_practice_sections.groupby('subsection_no')[['act_cost']].sum()



# Combine the Subsection results above with the BNF DataFrame (on subsection_no) then filter out their children to get total costs by Subsection

burley_subsection_total = pd.merge(burley_subsection, BNF, left_on='subsection_no', right_on='subsection_no', how='left')

df_subsection = burley_subsection_total[burley_subsection_total['Chapter'].str.count("\.") == 2][['Chapter', 'ChapterTitle', 'act_cost', 'Order']]

#df_subsection.style.format({"act_cost": "£{:20,.0f}"})
# Create a DataFrame showing the total cost of all drugs prescribed under each forth-level subsubsection (bad name sorry!) number



burley_subsubsection = burley_practice_sections.groupby('subsubsection_no')[['act_cost']].sum()



# Combine the Subsubsection results above with the BNF DataFrame (on subsection_no) then filter out their children to get total costs by Subsubsection

burley_subsubsection_total = pd.merge(burley_subsubsection, BNF, left_on='subsubsection_no', right_on='subsubsection_no', how='left')

df_subsubsection = burley_subsubsection_total[burley_subsubsection_total['Chapter'].str.count("\.") == 3][['Chapter', 'ChapterTitle', 'act_cost', 'Order']]

#df_subsubsection.style.format({"act_cost": "£{:20,.0f}"})
# Default value of display.max_rows is 10 i.e. at max 10 rows will be printed.

# Set it None to display all rows in the dataframe

pd.set_option('display.max_rows', None)
# Concatenate the 4 DataFrames above to get total costs for all Chapters,Section, Subsections and Sub-subsections

# This table shows the cost of all prescribed drugs by BNF classification

# Note: Parents of sub-sections show the total cost of all child subsection, e.g. 'Chapter 1: Gastro-intestinal system' at £2,990 is the sum of all

# sub-section totals beneath it, i.e. 1.2 + 1.3 + 1.3.1 etc.

df_burley_may16 = pd.concat([df_chapter, df_section, df_subsection, df_subsubsection], ignore_index=True).sort_values('Order', ascending=True)

df_burley_may16.reset_index(drop=True)

df_burley_may16.style.bar(subset=['act_cost'], color='#d65f5f').format({"act_cost": "£{:20,.0f}"})
# Import file containing a numerical breakdown of patients registered at each GP practice in March 2020

import pandas as pd

practice_patients = pd.read_csv("../input/patients-registered-at-a-gp-practice-march-2020/gp-reg-pat-prac-quin-age.csv", dtype={'ORG_CODE':str})

practice_patients.head()
# Import file that maps GP practice code to its ONS_CCG_CODE

import pandas as pd

practice_mapping = pd.read_csv("../input/gp-practice-mapping/gp-reg-pat-prac-map.csv")

practice_mapping.head()
# Get the Burley practice ONS code from its Practice Code

Burley_ONS_CODE = practice_mapping[practice_mapping['PRACTICE_CODE'] == 'B83019']['ONS_CCG_CODE'].iloc[0]

Burley_ONS_CODE
# Use the Burley ONS code to get patient numbers for the Burley practice

burley_patients = practice_patients[practice_patients['ONS_CODE'] == Burley_ONS_CODE]

burley_patients.head()
# Get the total number of patients in Grange Park Surgery, Burley In Wharfedale

burley_total_patients = burley_patients[burley_patients['AGE_GROUP_5'] == 'ALL']['NUMBER_OF_PATIENTS'].iloc[0]

burley_total_patients
# Get the number of female patients by age group

burley_females = burley_patients[(burley_patients['SEX'] == 'FEMALE') & (burley_patients['AGE_GROUP_5'] != 'ALL')][['AGE_GROUP_5', 'NUMBER_OF_PATIENTS']]

burley_females
# Get the number of male patients by age group

burley_males = burley_patients[(burley_patients['SEX'] == 'MALE') & (burley_patients['AGE_GROUP_5'] != 'ALL')][['AGE_GROUP_5', 'NUMBER_OF_PATIENTS']]
# Plot age range vs number of patients for each gender

width = 15

height = 5

plt.rcParams['figure.figsize'] = [width, height]

plt.plot(burley_females['AGE_GROUP_5'], burley_females['NUMBER_OF_PATIENTS'], label='Females')

plt.plot(burley_males['AGE_GROUP_5'], burley_males['NUMBER_OF_PATIENTS'], label='Males')

plt.xlabel('Age Range')

plt.ylabel('No of patients')

plt.title('Grange Park Surgery: Number of patients by age group')

plt.legend()

plt.show()
# Identify all NHS regions

practice_mapping['COMM_REGION_NAME'].unique()
practice_mapping[practice_mapping['PRACTICE_NAME'].str.startswith('GRANGE PARK')]
# Join all GP practices with the May 2016 prescribing data DataFrame

df_may2016 = pd.merge(practices, bnf_2016_05, left_index=True, right_on='practice', how='inner', suffixes=('_code', '_y'))

df_may2016.drop(['practice', 'practice_y', 'name', 'address1', 'address2', 'city', 'county', 'postcode'], axis='columns', inplace=True)

df_may2016.head()
# Join the May 2016 prescribing/practice DataFrame with the practice mapping DataFrame to allow aggregation by region

df_may2016_regions = pd.merge(df_may2016, practice_mapping, left_on='practice_code', right_on='PRACTICE_CODE', how='inner')

df_may2016_regions.drop(['PRACTICE_CODE', 'PRACTICE_NAME', 'PRACTICE_POSTCODE', 'PUBLICATION', 'STP_CODE', 'STP_NAME'], axis='columns', inplace=True)

df_may2016_regions.head()
# Join the May 2016 regional prescribing DataFrame with the BNF mapping DataFrame

df_may2016_regions_prescribing = pd.merge(df_may2016_regions, df_bnf, left_on='bnf_code', right_on='bnf_code_no', suffixes=('_x', ''))

df_may2016_regions_prescribing.drop(['bnf_code_x', 'bnf_name_x'], axis='columns', inplace=True)

df_may2016_regions_prescribing.tail()
# Create a new column in the prescribing data for the 7 digit BNF code prefix

# This prefix indicates the drugs hierarchical position in the BNF, i.e. Chapter > Section > Subsection > Sub-subsection, e.g. 1.3.2.1 or 0103021

df_may2016_regions_prescribing['bnf_code_prefix'] = df_may2016_regions_prescribing.apply(lambda row: row.bnf_code[0:7], axis=1)

df_may2016_regions_prescribing.info()
# Combine the all prescribing DataFrame with the BNF DataFrame so drugs can be categorised 

# into therapeutic/pharmaceutical groups (BNF Chapters, Sections etc.) and the total cost per group calculated 

df_all_practice_prescribing = pd.merge(df_may2016_regions_prescribing, BNF, left_on='bnf_code_prefix', right_on='subsubsection_no')
df_all_practice_prescribing.head()
# For each English region calculate the total cost of drugs in each BNF chapter in May 2016 

df_regional_prescribing = df_all_practice_prescribing.groupby(['COMM_REGION_NAME', 'chapter_no'])[['act_cost']].sum()



#df_regional_prescribing.style.format({"act_cost": "£{:20,.0f}"})
# Remove levels from the multi-index dataframe above

df_regional_prescribing.reset_index(inplace=True)  

#df_regional_prescribing.style.format({"act_cost": "£{:20,.0f}"})
# Calculate the total cost of all GI System drugs (BNF Chapter 1) in each English region in May 2016



df_regional_gastro = df_regional_prescribing[df_regional_prescribing['chapter_no'] == '01']

df_regional_gastro.style.format({"act_cost": "£{:20,.0f}"})
# Plot a bar chart of GI system drug costs (BNF Chapter 1) for each English region in May 2016

width = 15

height = 5

plt.rcParams['figure.figsize'] = [width, height]

plt.bar(df_regional_gastro['COMM_REGION_NAME'], df_regional_gastro['act_cost'])

plt.xticks(rotation=45)

plt.xlabel('COMM REGION')

plt.ylabel('Actual Cost (£)')

plt.title('Actual Cost of GI System drugs by region May 2016')

#plt.legend()

plt.show()
# For each English region calculate the total cost of drugs in each BNF subsection in May 2016 

df_regional_prescribing_subsection = df_all_practice_prescribing.groupby(['COMM_REGION_NAME', 'subsection_no'])[['act_cost']].sum()

#df_regional_prescribing_subsection.style.format({"act_cost": "£{:20,.0f}"})
# Remove levels from the multi-index dataframe above

df_regional_prescribing_subsection.reset_index(inplace=True)  

#df_regional_prescribing_subsection.style.format({"act_cost": "£{:20,.0f}"})
# Calculate the total cost of all Laxative drugs (BNF Chapter 1) in each English region in May 2016



df_regional_laxative = df_regional_prescribing_subsection[df_regional_prescribing_subsection['subsection_no'].str.contains('010601|010602|010603|010604|010605|010606|010607', regex=True)]

#df_regional_laxative.style.format({"act_cost": "£{:20,.0f}"})
# Plot a chart of Laxative costs by category(BNF Chapter 1.6) for each English region in May 2016

width = 15

height = 10

plt.rcParams['figure.figsize'] = [width, height]

plt.plot(df_regional_laxative[df_regional_laxative['subsection_no'] == '010601']['COMM_REGION_NAME'], \

        df_regional_laxative[df_regional_laxative['subsection_no'] == '010601']['act_cost'], label='Bulk-forming', marker='o')

plt.plot(df_regional_laxative[df_regional_laxative['subsection_no'] == '010602']['COMM_REGION_NAME'], \

        df_regional_laxative[df_regional_laxative['subsection_no'] == '010602']['act_cost'], label='Stimulant', marker='o')

plt.plot(df_regional_laxative[df_regional_laxative['subsection_no'] == '010603']['COMM_REGION_NAME'], \

        df_regional_laxative[df_regional_laxative['subsection_no'] == '010603']['act_cost'], label='Faecal Softeners', marker='o')

plt.plot(df_regional_laxative[df_regional_laxative['subsection_no'] == '010604']['COMM_REGION_NAME'], \

        df_regional_laxative[df_regional_laxative['subsection_no'] == '010604']['act_cost'], label='Osmotic', marker='o')

plt.plot(df_regional_laxative[df_regional_laxative['subsection_no'] == '010605']['COMM_REGION_NAME'], \

        df_regional_laxative[df_regional_laxative['subsection_no'] == '010605']['act_cost'], label='Bowel cleansing preps', marker='o')

plt.plot(df_regional_laxative[df_regional_laxative['subsection_no'] == '010606']['COMM_REGION_NAME'], \

        df_regional_laxative[df_regional_laxative['subsection_no'] == '010606']['act_cost'], label='Peripheral Opioid-Receptor Antagonists', marker='o')

plt.plot(df_regional_laxative[df_regional_laxative['subsection_no'] == '010607']['COMM_REGION_NAME'], \

        df_regional_laxative[df_regional_laxative['subsection_no'] == '010607']['act_cost'], label='5HT4-receptor agonists', marker='o')

plt.xticks(rotation=45)

plt.xlabel('COMM REGION')

plt.ylabel('Total Cost (£)')

plt.title('Total Cost of laxative drugs by type for each English region in May 2016')

plt.legend()

plt.show()
# Join practice patient population and practice region DataFrames



practice_patient_totals = practice_patients[(practice_patients['SEX'] == 'ALL') & (practice_patients['ORG_TYPE'] == 'GP') ]

df_region_patients_org = pd.merge(practice_patient_totals, practice_mapping, left_on='ORG_CODE', right_on='PRACTICE_CODE')
# Calculate the total number of patients per region and remove levels from index



df_patients_by_region = df_region_patients_org.groupby('COMM_REGION_NAME')[['NUMBER_OF_PATIENTS']].sum()

df_patients_by_region.reset_index(inplace=True)
df_patients_by_region
# Calculate the cost of each laxative type (per 1,000 patients) in each region



df_region_patients_lax = pd.merge(df_patients_by_region, df_regional_laxative, left_on='COMM_REGION_NAME', right_on='COMM_REGION_NAME')

df_region_patients_lax['cost_per_1000'] = (df_region_patients_lax['act_cost']/df_region_patients_lax['NUMBER_OF_PATIENTS']) * 1000

#df_region_patients_lax.style.format({"act_cost": "£{:20,.2f}", "cost_per_1000": "£{:20,.2f}"})
# Plot a chart of Laxative costs (per 1,000 patients) by category(BNF Chapter 1.6) for each English region in May 2016

width = 15

height = 10

plt.rcParams['figure.figsize'] = [width, height]

plt.plot(df_region_patients_lax[df_region_patients_lax['subsection_no'] == '010601']['COMM_REGION_NAME'], \

        df_region_patients_lax[df_region_patients_lax['subsection_no'] == '010601']['cost_per_1000'], label='Bulk-forming', marker='o')

plt.plot(df_region_patients_lax[df_region_patients_lax['subsection_no'] == '010602']['COMM_REGION_NAME'], \

        df_region_patients_lax[df_region_patients_lax['subsection_no'] == '010602']['cost_per_1000'], label='Stimulant', marker='o')

plt.plot(df_region_patients_lax[df_region_patients_lax['subsection_no'] == '010603']['COMM_REGION_NAME'], \

        df_region_patients_lax[df_region_patients_lax['subsection_no'] == '010603']['cost_per_1000'], label='Faecal Softeners', marker='o')

plt.plot(df_region_patients_lax[df_region_patients_lax['subsection_no'] == '010604']['COMM_REGION_NAME'], \

        df_region_patients_lax[df_region_patients_lax['subsection_no'] == '010604']['cost_per_1000'], label='Osmotic', marker='o')

plt.plot(df_region_patients_lax[df_region_patients_lax['subsection_no'] == '010605']['COMM_REGION_NAME'], \

        df_region_patients_lax[df_region_patients_lax['subsection_no'] == '010605']['cost_per_1000'], label='Bowel cleansing preps', marker='o')

plt.plot(df_region_patients_lax[df_region_patients_lax['subsection_no'] == '010606']['COMM_REGION_NAME'], \

        df_region_patients_lax[df_region_patients_lax['subsection_no'] == '010606']['cost_per_1000'], label='Peripheral Opioid-Receptor Antagonists', marker='o')

plt.plot(df_region_patients_lax[df_region_patients_lax['subsection_no'] == '010607']['COMM_REGION_NAME'], \

        df_region_patients_lax[df_region_patients_lax['subsection_no'] == '010607']['cost_per_1000'], label='5HT4-receptor agonists', marker='o')

plt.xticks(rotation=45)

plt.xlabel('COMM REGION')

plt.ylabel('Total Cost per 1000 patients (£)')

plt.title('Total Cost of laxative drugs (per 1000 patients) by type for each English region in May 2016')

plt.legend()

plt.show()
# In the above graph, the North West appears to prescribe approximately 20% more 

# Osmotic and Stimulant Laxatives than other regions. 