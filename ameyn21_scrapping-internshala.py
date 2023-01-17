# importing utility libraries

import requests
import bs4
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
URL = "https://internshala.com/internships/data%20science-internship"
#conducting a request of the stated URL above:
page = requests.get(URL)
#specifying a desired format of “page” using the html parser - this allows python to read the various components of the page, rather than treating it as one long string.
soup = BeautifulSoup(page.text, "html.parser")
#printing soup in a more structured tree format that makes for easier reading
print(soup.prettify())
def extract_job_title_from_result(job_div, job_post):    
    for a in job_div.find(name="div", attrs={"class":"company"}).find(name="div", attrs={"class":"profile"}).find(name="a"):
        job_post.append(a)
def extract_company_from_result(job_div, job_post): 
    for a in job_div.find(name="div", attrs={"class":"individual_internship_header"}).find(name="div", attrs={"class":"company"}).find(name="div", attrs={"class":"company_name"}).find(name="a"):
        job_post.append(str(a).strip())
def extract_location_from_result(job_div, job_post): 
    for a in job_div.find(name="div", attrs={"class":"individual_internship_details"}).find(name="div", attrs={"id":"location_names"}).find(name="a", attrs={"class":"location_link"}):
        job_post.append(str(a).strip())
def extract_salary_from_result(job_div, job_post): 
    for stipend in job_div.find(name="div", attrs={"class":"individual_internship_details"}).find(name="div", attrs={"class":"internship_other_details_container"}).find(name="span", attrs={"class":"stipend"}):        
        if isinstance(stipend, bs4.element.NavigableString):
            job_post.append(str(stipend).strip())
def extract_duration_from_result(job_div, job_post): 
    for stipend in job_div.find(name="div", attrs={"class":"individual_internship_details"}).find(name="div", attrs={"class":"internship_other_details_container"}).find(name="span", attrs={"class":"stipend"}):        
        if isinstance(stipend, bs4.element.NavigableString):
            job_post.append(str(stipend).strip())
def Get_URL_Of_page(base_url, skill):
    return base_url + "/internships/" + skill + "-internship"    
def Get_total_pages(url):
    # get total number of pages
    page = requests.get(url)
    soup = BeautifulSoup(page.text, "html.parser", from_encoding="utf-8")
    total_pages = soup.find(name="span", attrs={"id":"total_pages"})
    return int(total_pages.text.strip())
def Get_Internship_Description_Page_Url(job_div, base_url):
    for a in job_div.find(name="div", attrs={"class":"button_container"}).find_all(name="a", attrs={"class":"view_detail_button"}, href=True):
        return base_url + a['href']
def extract_description_from_result(job_div, job_post):
    page = requests.get(Get_Internship_Description_Page_Url(job_div, base_url))
    soup_desc = BeautifulSoup(page.text, "html.parser", from_encoding="utf-8")
    
    for div in soup_desc.find_all(name="div", attrs={"class":"section_heading heading_5_5"}):
        if any(word in div.text.strip() for word in ["job/internship", "internship", "work from home", "part time", "job"]):
            job_post.append(div.next_sibling.next_sibling.text.strip())
#scraping code:
columns = ["job_title", "company_name", "location", "salary", "description"]
sample_df = pd.DataFrame(columns = columns)
base_url = "https://internshala.com"
skill = "data science"


def Scrap_Internshala(base_url, skill):
    url = Get_URL_Of_page(base_url, skill)
    total_pages = Get_total_pages(url)
    for page_number in range(total_pages):
        page = requests.get(url + "/page-" + str(page_number + 1))
        time.sleep(1)  #ensuring at least 1 second between page grabs
        soup = BeautifulSoup(page.text, "html.parser")

        for div in soup.find_all(name="div", attrs={"class":"individual_internship"}): 
            #specifying row num for index of job posting in dataframe
            num = (len(sample_df) + 1)
            #creating an empty list to hold the data for each posting
            job_post = []
            #grabbing job title
            extract_job_title_from_result(div, job_post)
            #grabbing company name
            extract_company_from_result(div, job_post)
            #grabbing location name
            extract_location_from_result(div, job_post)
            #grabbing salary
            extract_salary_from_result(div, job_post)
            #grabbing internship description
            extract_description_from_result(div, job_post)
            #appending list of job post info to dataframe at index num
            sample_df.loc[num] = job_post
            
    return sample_df

df_internships = Scrap_Internshala(base_url, skill)
df_internships.head()
#saving df_internships as a local csv file — define your own local path to save contents 
df_internships.to_csv("job_scrapping_internshala.csv", encoding="utf-8")