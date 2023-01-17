#import packages
import numpy as np
import pandas as pd
import os
import re

# listing out all variable for use later
# good practice to set all constants as upper case variables so they can be easily modified
GLASSDOOR_FILE_PATH='/kaggle/input/data-jobs-listings-glassdoor/glassdoor.csv'
GLASSDOOR_COLUMN_MAPPINGS = {
    'header.jobTitle': 'Jobtitle',
    'job.description': 'Jobdescription',
    'map.country': 'Country',
    'gaTrackerData.location': 'City',
    'header.employerName':'Employer',
    'overview.industry': 'Industry',
    'overview.size': 'CompanySize',
    'rating.starRating': 'Companystarrating'
}
COUNTRY_CODE_FILE_PATH = '/kaggle/input/data-jobs-listings-glassdoor/country_names_2_digit_codes.csv'
TRANSFORMATIONS = {'City': 'title', 'Country':'title', 'Jobdescription':'lower', 'Jobtitle':'lower', 'Industry':'lower'}
CONSOLIDATIONS = {}

CONSOLIDATIONS['Industry'] = {
    ' technology ':
        ['IT Services',
         'Internet',
         'Computer Hardware & Software',
         'Enterprise Software & Network Solutions',
         'cable, technology & telephone providers'
        ],
    ' financial services ':
        [ 'Accounting',
         'Banks & Building Societies',
         'Investment Banking & Asset Management',
         'Financial Analytics & Research',
         'Financial Transaction Processing',
         'Lending',
         'stock exchanges',
         'Insurance Operators'
        ],
    ' health ':
        ['Healthcare Product Manufacturing',
         'Healthcare Services & Hospitals',
         'Health, Beauty & Fitness',
         'Biotech & Pharmaceuticals'
        ],
    ' education ':
        ['Colleges & Universities',
         'Education Training Services'
        ],
    ' professional services':
        ['Consulting',
         'Advertising & Marketing',
         'Legal',
         'Staffing & Outsourcing'
        ],
    ' government agencies':
        ['Government Agencies'],
    ' consumer/retail':
        ['Other Retail Shops', 'Publishing', 'TV Broadcasting & Cable Networks'],
    ' other':
        ['Aerospace & Defence',
         'Sports & Recreation',
         'Membership Organisations',
         'Gambling',
         'Airlines',
         'Wood Product Manufacturing',
         'Charitable Foundations',
         'null'
        ],
    ' eur':
        ['Utilities', 'Building & Construction', 'Mining', 'Building & Personnel Services'],
    ' consumer/retail':
        ['Grocery Shops & Supermarkets'],
    ' unspecified industry':
        ['null']
}

## consolidated the cloud cleanup into one sep with the industry cleanup
CONSOLIDATIONS['Jobdescription'] = {
    ' alibaba cloud ': ['alibaba'],
    ' amazon web services (aws) ': [
        'amazon web services',
        'aws'
    ],
    ' google cloud platform (gcp) ': [
        'google cloud platform',
        'gcp',
        'google cloud'
    ],
    ' ibm cloud ': [
        'ibm'
    ],
    ' microsoft azure ': [
        'azure'
    ],
    ' oracle cloud ': [
        'oracle'
    ],
    ' red hat cloud ': [
        'red hat'
    ],
    ' sap cloud ': [
        'sap'
    ],
    ' salesforce cloud ': [
        'salesforcre'
    ],
    ' vmware cloud ': ['vmware']
    
} 


CLOUDS = ['AWS', 'GCP', 'Azure']

JOB_CATEGORIES = [
    ('analyst|analytics', 'data analyst'),
    ('scientist|science', 'data scientist'),
    ('engineer|developer', 'data engineer'),
    ('architect', 'data architect'),
    ('database administrator', 'database administrator')
]

LANGUAGES = ['python', 'r', 'sql', 'c', 'java', 'javascript', 'matlab', 'scala', 'swift', 'julia']
# Loading the main file for glassdoor listings
glassdoor_data = pd.read_csv(GLASSDOOR_FILE_PATH)
glassdoor_data.head()
# This approach utilizes a dictionary to limit the columns and appropriately map the column names
# by externalizing this to a constant, it is easy to add, delete, or rename columns
glassdoor = glassdoor_data[GLASSDOOR_COLUMN_MAPPINGS.keys()].copy().rename(columns=GLASSDOOR_COLUMN_MAPPINGS)
glassdoor.head()
# This table has a list of country names vs 2 digit codes
country_codes = pd.read_csv(COUNTRY_CODE_FILE_PATH)
country_codes.head()

# We merge both by 2 digit code, and then fill the NaNs with the full country name
glassdoor = pd.merge(glassdoor, country_codes, left_on='Country', right_on='Code', how='left')

# Then replace the 2 digits codes with full name
glassdoor.Country = glassdoor.Name.fillna(glassdoor.Country)
glassdoor = glassdoor.drop(['Name', 'Code'], axis=1)
glassdoor = pd.merge(glassdoor, country_codes, left_on='Country', right_on='Name', how='inner') #changed so that we can remove the dropna below
glassdoor.head()

# # Finally, this block removes any other values that do not match standard naming
# glassdoor.dropna(subset=['Name'], inplace=True) ## Assuming there are no missing values in country codes, this can just use an inner join
glassdoor = glassdoor.drop(['Name', 'Code'], axis=1)
listings_after = glassdoor.shape[0]
print('After removing countries names that don\'t match standard naming there' \
      f'are {listings_after} job listings.')
# Clean up casing on the columns that are going to have user input/transformation applied to them later on in the notebook

## Converting transformations to metadata and applying string transformations based on the value in the dictionary
for k, v in TRANSFORMATIONS.items():
    glassdoor[k] = getattr(glassdoor[k].str, v)()
glassdoor.head()
## since similar logic is used in multiple places, I created a function
def get_valid_response(cleansing_method, valid_values, invalid_response):
    while True:
        user_input = input()
        user_value = getattr(user_input, cleansing_method)()
        if user_value not in valid_values:
            print(invalid_response)
        else:
            break
    return user_value
    
# This block asks the user to narrow down the job location by either City or Country

# enhancement: refactor to check that the answer that the user gives for city/country is a valid value in the dataset
# enhancement: refactor to ask if they would like to analyse by a country or city at all




statement_question = print("Would you like to analyse by Country or City?")
# using the function defined above instead
country_city = get_valid_response('title', ['Country','City'], 'Sorry, the value you entered does not match City or Country, please try again')
glassdoor.dropna(subset=[country_city], inplace=True)
if country_city == 'Country': 
    print("You have selected to analyse by Country. What Country would you like to analyse?")
elif country_city == 'City':
    print("You have selected to analyse by City. What City would you like to analyse?")
else:
    print("Program Error")

# We can reuse the function here so that the user doesn't have to move forward if the city or country doesn't have any records
country_city_value = country_city_value = get_valid_response('title', glassdoor[country_city].unique(), 'There are no records matching that request. Please try again')


#This block takes the column the user selected - country or city - as well as the value to narrow down the location
glassdoor = glassdoor[glassdoor[country_city] == (country_city_value)]
glassdoor.head()

#Unhash for double checking/testing
#list = glassdoor['City'].unique().tolist()
#list[-200:]

listings_after = glassdoor.shape[0]
print('After removing countries names that don\'t match standard naming there' \
      f'are {listings_after} job listings.')
# List of job titles that are data specific - note that the below filters for any that CONTAIN the following key terms in any combination 
# Seeing as we are mostly interested in 'data', titles that contain the word 'data' should cover most of the roles we are interested in. A few other terms have been added in for good measure.
job_titles = ['data', 'analytics', 'machine learning']


# Creating masks for each job title to identify where they appear
## Since you already know that you lowercased the jobtitle, you could lowercase the job titles, too instead of ignorecase, but this is explicit and perfectly fine
job_masks = [glassdoor.Jobtitle.str.contains(Jobtitle, flags=re.IGNORECASE, regex=True) for Jobtitle in job_titles]
# Combining all masks where any value is True, return True
combined_mask = np.vstack(job_masks).any(axis=0)
combined_mask

# Applying the mask to the dataset
glassdoor = glassdoor[combined_mask].reset_index(drop=True)
listings_after = glassdoor.shape[0]
print(f'After refining job titles there were {listings_after} job listings.')
glassdoor.head()

print('The below is a sample of the remaining job titles. From the list is there any remaining key words that you would like removed from the analysis? e.g. you might find it useful to remove certain levels such as graduate')

job_title_list = glassdoor['Jobtitle'].unique().tolist()
job_title_list[-200:]
#This question asks the user if they would like to remove any terms from job titles
#e.g. you can choose to remove key terms related to level such as graduate or intern

# job_question = print('Would you like to remove any key terms from the Job Title? Please enter Y or N')
# #validate that the user is entering only Y or N
# ## replaced to use common function

## use recursion for the entire interaction
def cleanse_terms(glassdoor, continued=False):
    if continued:
        print('would you like to remove more terms?')
    else:
        print('Would you like to remove any key terms from the Job Title? Please enter Y or N')
    user_answer = get_valid_response('upper', ['Y','N'], 'Sorry, you did not enter a valid input, please try again')
    if user_answer == 'Y':
        print ("Enter the term you would like to remove. Please only enter a single term")
        user_input = input().lower()
        term_mask = glassdoor['Jobtitle'].str.contains(user_input)
        glassdoor = glassdoor[~term_mask]
        listings_after_rem = glassdoor.shape[0]
        print(f'After refining job titles there were {listings_after_rem} job listings.')
        return cleanse_terms(glassdoor, True)
    else:
        print ("You selected to continue with the data set as is")
        return glassdoor

cleanse_terms(glassdoor)
   
#unhash statement to test/double check the list
#list = newdp['Jobtitle'].unique().tolist()
#print(list[-200:])
# clean up and consolidation of industries
# Initially we made everything lower case in this column, so no need to worry about casing

### This metdata may be more readable if it's reversed and a bit easier to update - see CONSOLIDATIONS above

# using regex to replace the group of terms instead of one at a time to save on performance
for field, consolidation in CONSOLIDATIONS.items():
    for repl, finds in consolidation.items():
        regex_pattern = r"(?=("+'|'.join([f.lower() for f in finds])+ r"))"
        function = lambda x: re.sub(regex_pattern, repl.lower(), x) if x==x else x
        glassdoor[field] = list(map(function, glassdoor[field]))


glassdoor['Industry'] = glassdoor['Industry'].fillna('other')   
glassdoor.head()
#this block creates a column that pulls out the reference to the 3 main cloud platforms. 
#The reason we flatten it and pull it into 3 different columns, rather than 1, is that is makes it easier for filtering and analysis in front end visualisation tools
#It also allows for easy analysis and comparison of cloud platforms as a Job description may mention more than one cloud platform

## Converted to metadata for maintainability and to reduce redundent code


for cloud in CLOUDS:
    glassdoor['CloudPlatform({})'.format(cloud)] = glassdoor['Jobdescription'].str.extract('({})'.format(cloud.lower()), expand=True)


glassdoor.head()
#this block of code uses categorises the jobs into 6 categories: data engineer, data analyst, data scientist, business analyst, database administrator, data architect
#this creates a new column called job category

## moving this to metadata and using a recursive function to apply the nested ternary logic



def get_category(jobtitles, job_categories, index=0):
    if index == (len(job_categories)-1):
        terms, category = job_categories[index]
        return np.where(jobtitles.str.contains(terms), category, 'other')
    else:
        terms, category = job_categories[index]
        return np.where(jobtitles.str.contains(terms), category, get_category(jobtitles, job_categories, index+1))
        
        
glassdoor['job_category'] = get_category(glassdoor['Jobtitle'], JOB_CATEGORIES)
    

glassdoor.head()
#this block creates a column that pulls out the reference to the the top 10 coding languages for data. This was pulled from https://www.analyticsinsight.net/top-10-data-science-programming-languages-for-2020/ 
#The reason we flatten it and pull it into different columns, rather than 1, is that is makes it easier for filtering and analysis in front end visualisation tools
#It also allows for easy analysis and comparison of coding languages as a Job description often mentions more than one coding language

## Moving to metadata and iterating over it to reduce redundency in code


for language in LANGUAGES:
    glassdoor['language_{}'.format(language)] = glassdoor['Jobdescription'].str.extract('(\\b{}\\b)'.format(language), expand=True)


glassdoor.head()
# Since the counting and threshold logic is command, this is a function for applying it generically

def add_count_and_required(count_field, field_pattern, field_items, required_field, minimum):
    count = glassdoor[[field_pattern.format(x) for x in field_items]].copy()
    glassdoor[count_field] = count.apply(lambda x: x.count(), axis=1)
    glassdoor[required_field] = list(map(lambda x: 1 if x >= minimum else 0, glassdoor[count_field]))
#Select the columns required for this analysis

## moving to use a generic implementation
add_count_and_required('count_languages', 'language_{}', LANGUAGES, 'language_req', 1)

glassdoor.head()
## moving to use a generic implementation
add_count_and_required('count_platforms', 'CloudPlatform({})', CLOUDS, 'platform_req', 1)


glassdoor.head()


# This creates another data frame with the counts of values in cloud platform and code language platforms

CODE_COLUMN_MAPPINGS = {'language_python': 'Python', 'language_r': 'R', 'language_sql': 'SQL', 'language_c': 'C', 'language_java': 'Java', 'language_javascript': 'Javascript', 'language_matlab': 'MatLab', 'language_scala': 'Scala', 'language_swift': 'Swift', 'language_julia': 'Julia', 'CloudPlatform(AWS)': 'AWS', 'CloudPlatform(Azure)': 'Azure', 'CloudPlatform(GCP)': 'GCP' }

glassdoor_code = glassdoor[CODE_COLUMN_MAPPINGS.keys()].copy().rename(columns=CODE_COLUMN_MAPPINGS)

code_platform_counts = glassdoor_code.count()

#code_platform_counts.head()

code_file_name = 'code_platform_counts'

code_platform_counts.to_csv('{}.csv'.format(code_file_name),index=True)

#Now to export the finalised data set
print("input a name for your final file ")
input7 = input()
file_name = input7.lower()

glassdoor.to_csv('{}.csv'.format(file_name),index=False)
print("to access your file go to the right side panel and select 'data'>'Output'. Click the refresh button, the select the 3 dots to download the file")