import pandas as pd 

import matplotlib.pyplot as plt

import nltk

from nltk.corpus import stopwords # Filter out stopwords, such as 'the', 'or', 'and'

import re # Regular expressions

from bs4 import BeautifulSoup # For HTML parsing

import math



import urllib.request as urllib2

from IPython.display import clear_output

from time import sleep # To prevent overwhelming the server between connections

from collections import Counter # Keep track of our term counts





%matplotlib inline
# This is a function that will create a URL to indeed.ca. The inputs to this function are job title, city, and province/state.



def create_URL(job, city, province):



    job = job.replace(' ','+') # Where the job title has two parts, in the URL there will be a + between the two parts.



    

    city = city.strip().replace(' ', '-') # Make sure the city specified works properly if it has more than one word (such as St. Johns) or it has leading or ending spaces

    site_list = ['http://www.indeed.ca/jobs?q=', job, '&l=', city,

                        '%2C+', province] # Joining all of our strings together to create the URL   

    if(city == ''):

        site_list = ['https://www.indeed.ca/jobs?q=', job , '&l=',province] # Creating the URL when city is not specified (is blank)



    final_URL = ''.join(site_list) # Merge the html address together into one string



    return final_URL
print('Please click on the link below to visit the job posting results from indeed.ca. We are going to extract these job posting information:')

print(create_URL('Data Scientist','Toronto','ON'))
def find_number_of_jobs(url):

    

    html = urllib2.urlopen(url).read() # Open up the URL

    soup = BeautifulSoup(html, 'lxml') # Get the html source from the URL



   

    # Now we will find out how many jobs there were



    num_jobs_string = soup.find(id = 'searchCountPages').string # searchCountPages is an html tag in the URL source that holds the number of jobs, 

                                                              # so we find it, then we get the string part out (check the screenshot below)



    job_numbers = re.findall('\d+', num_jobs_string) #Here we want to get the digits out of the string that we found in previous command. 



    if(len(job_numbers)>2): # If there are more than 999 jobs, then there will be a "," as a thousand separator, 

                                #so when we get the digits only, there will be two parts, numbers before "," and after it.

        total_num_jobs = int(job_numbers[1]) * 1000 + int(job_numbers[2]) 

    else:  

        total_num_jobs = int(job_numbers[1])



    num_pages = math.ceil(total_num_jobs/20) # In each page, there is usually more than 20 job postings, so we divide the number of jobs by 20, to get a rough 

                                             # estimation of the number of pages we need to scrape

                                          

    #job_descriptions = [] # Store all our descriptions in this list



    return total_num_jobs, num_pages
URL = create_URL('Data Scientist','Toronto','ON') # Getting all the jobs in Toronto



print('There are ',find_number_of_jobs(URL)[0], 'jobs available in',find_number_of_jobs(URL)[1], 'pages')
URL = create_URL('Data Scientist','','ON') # Getting all the jobs in all Ontario



print('There are ',find_number_of_jobs(URL)[0], 'jobs available in',find_number_of_jobs(URL)[1], 'pages')
def extract_jobs(url):



    job_links = []

    companies = []

    job_titles = []

    job_links = []

    job_descriptions = []

    locations = []



    base_url = 'http://www.indeed.ca'



    for i in range(1,num_pages+1): # Loop through all of our search result pages

        clear_output() # We clear the output because we are going to print he URLs each time, and there will be alot of them.

        print ('Getting page', i)

        start_num = str(i*20-20) # Assign the multiplier of 20 to view the pages we want

        current_page = ''.join([url, '&start=', start_num]) # This is the URL that indeed uses for each page of the results. 

                                                            # For example, the second page of the results would have a URL like "https://www.indeed.ca/jobs?q=Data+Scientist&l=Toronto%2C+ON&start=20"

                                                            # Note the "&start=20" part in the URL.

        print(current_page)



        html_page = urllib2.urlopen(current_page).read() # Get the current page



        page_obj = BeautifulSoup(html_page, 'lxml')

        

        my_divs = page_obj.find_all("div") # We get all the div tags from the page

        counter=0

        for div in my_divs:

            if (div.find("div", class_="title")): # Finding a div that has a class='title'. That's where we can find that specific job posting's URL (because we want to capture the URL as well)

                job_link = str(div.a.get('href')) # Getting the URL part from the a tag inside the div tag we found.

                job_link = 'http://www.indeed.ca'+job_link # Because the URL that is inside the a tag is missing the base URL part, we add it manually.

                print(job_link)

                

                try: # need to open with try

                    job_page = urllib2.urlopen(job_link).read() # Now that we found the URL to the specific job posting, we open it

                    job_object = BeautifulSoup(job_page, 'lxml')

                    if(job_object.find("span", class_="jobsearch-JobMetadataHeader-iconLabel")): # This is the tag that holds the location of the job

                        job_location = job_object.find("span", class_="jobsearch-JobMetadataHeader-iconLabel").text



                    else:

                        job_location='' # We have this part in case if there is no location mentioned in the job add.



                    counter=counter+1    

                    if (job_object.find("div", class_="icl-u-lg-mr--sm icl-u-xs-mr--xs")): # This is the tag that has the company name

                        job_company_name = job_object.find("div", class_="icl-u-lg-mr--sm icl-u-xs-mr--xs").text

                    else:

                        job_company_name = 'Unknown' # If the company name is not accessible for any reason



                    if(job_object.find("div", class_="jobsearch-JobInfoHeader-title-container")): # The tag that has the job title

                        job_title = job_object.find("div", class_="jobsearch-JobInfoHeader-title-container").text

                    else:

                        job_title = 'Unknown'



                    if(job_object.find("div", id="jobDescriptionText")): # This is where we find the job description

                        job_description = job_object.find("div", id="jobDescriptionText").text

                    else:

                        job_description = 'Unknown'





                    # Adding the finding into lists so that we can create a data frame from these lists.

                    

                    companies.append(job_company_name)

                    job_titles.append(job_title)

                    job_links.append(job_link)

                    job_descriptions.append(job_description)

                    locations.append(job_location)

                    

                    

                # This is where we tell the function to continue if there is a 404 page not found error.    

                except urllib2.HTTPError as e:

                    if e.getcode() == 404: # check the return code

                        continue # If there is an error and the link is not accessible, continue to the next job posting URL.

                    raise # if other than 404, raise the error        

                



    # Now that we have extracted the information that we need, we put them into a data frame.

    dictionary = {'Job Title': job_titles, 'Company Name': companies,

                     'Location': locations, #'Key_to_Link':jks

                     'job URL': job_links,

                     'Job Description': job_descriptions,

                     }                 



    df = pd.DataFrame(dictionary)  

    

    

    filename='Jobs.csv'

    df.to_csv(filename)

    print ('*********************** Data was saved in',filename, 'file. *********************')

    df = df.drop_duplicates()

    df.head(3)

    return df

        
jobs = ['Data Scientist', 'Carpenter', 'Registered Nurse', 'Custome Service Representative']





#job = ''

city = ''

state = 'ON'

dfs=[]



i=0

for job in jobs:

    i = i+1

    url = create_URL(job, city, state)

    

    print (url)

    total_num_jobs,num_pages  = find_number_of_jobs(url)

    print('The job title', job, 'has', total_num_jobs,'jobs in', num_pages,'pages')

    if (num_pages>50):

        num_pages = 50    

    df = extract_jobs(url)

    filename = 'Jobs_' + city.replace(' ','_') + '.csv'

    df.to_csv(filename)

    dfs.append(df) # This will give us a series of data frames that will need to be concatenated.

# Checking to see if we have multiple data frames

print(dfs[0].head(3))

print(dfs[1].head(3))
# Concatenating all data frames and saving the results into a csv file

df_all_jobs= pd.concat(dfs)

df_all_jobs.to_csv('All_Jobs.csv')
URL = create_URL('Data Scientist','','') # Getting all Data Scientist jobs within Canada

print(URL)

print('There are ',find_number_of_jobs(URL)[0], 'jobs available in',find_number_of_jobs(URL)[1], 'pages')

total_num_jobs,num_pages  = find_number_of_jobs(URL)

df = extract_jobs(URL)
df.keys()
#Getting rid of the extra column

df = df[['Job Title', 'Company Name', 'Location', 'job URL',

       'Job Description']]

df[6:9]
# There are still a lot of duplicate job postings:

df[df['Location']=='Welland, ON']
# Removing the duplicate jobs



print('There are',len(df), 'jobs in the data set.\n')

df = df.drop_duplicates(subset = ['Job Title','Company Name','Location','Job Description'], keep="first")



print('There are',len(df), 'jobs in the data set after removing the duplicates.\n')

df = df.reset_index(drop=True)
# Taking care of missing values

df = df.fillna('Unknown')
# Getting number of job postings by city:

jobs_by_city= df.groupby('Location')['Job Title'].count().sort_values(ascending=False)[:10]

print("Number of jobs by city:\n\n ",jobs_by_city)
# Generating a bar chart for number of job postings by city:



# Getting number of job postings by city:

jobs_by_city= df.groupby('Location')['Job Title'].count().sort_values(ascending=False)[:10]







plt.figure(figsize=(10,8))

xvals = jobs_by_city.index

yvals = jobs_by_city.tolist()

plt.bar(xvals, yvals, color='#006fb9')

plt.xticks(rotation=90)

#plt.margins(0.2)



plt.subplots_adjust(bottom=0.3, left=0.2) # To make sure the x axis labels are visible

plt.title('Number of Data Scientist Jobs by City (Top 10 Cities)')

plt.xlabel('Top 10 Cities by the Number of Jobs')

plt.ylabel('Number of Job Postings')



#Having the y axis formatted as thousand separated in case there are more than a thousand jobs

ax = plt.gca()

ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))



ax.spines['top'].set_visible(False) # Removing the top border

ax.spines['right'].set_visible(False) # Removing the righ border



for i, v in enumerate(yvals): # Showing the data values on top of the bars

    ax.text(i-0.17, v+5, str(v)) # i and v show the position of the data labels


