import bs4

from urllib.request import urlopen as ureq

from bs4 import BeautifulSoup as soup

import pandas as pd

import csv
def total_cases(page_soup):

    rep_date = page_soup.find('span', class_="text-red").text

    list_of_positive_cases = page_soup.find('div', class_="card-body bg-white").find('ul').find_all('li')

    summary = {'Date':rep_date,'total': list_of_positive_cases[0].text,

                'deaths': list_of_positive_cases[1].text,

                'jurisdictions': list_of_positive_cases[2].text

            }

    return summary
my_url = 'https://www.cdc.gov/coronavirus/2019-ncov/cases-updates/cases-in-us.html'

uClient = ureq(my_url)

page_html = uClient.read()

uClient.close()

page_soup = soup(page_html,"html.parser")
summary = total_cases(page_soup)

summary
positive_cases_details_table_html = page_soup.find('table',class_='table table-bordered nein-scroll')

positive_cases_details = pd.read_html(str(positive_cases_details_table_html))[0]

positive_cases_details