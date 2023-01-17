!pip install textract
import textract
!pip install wget

import wget

def fetch_pdf(link):

    wget.download(link)

    return link.split('/')[-1]
!pip install bs4

!pip install urllib3
import bs4 as bs

import urllib.request

def get_links():

    links=[]

    source = urllib.request.urlopen('https://www.google.com/covid19/mobility/')

    print(f'Fetching links from {source}')

    soup = bs.BeautifulSoup(source,'lxml')

    print(soup.title)

    for anchors in soup.find_all('a',class_="download-link"):

        links.append(anchors.get('href'))

    print(f'Successfully retrieved links')

    return links
import os

def delete_pdf(pdf):

    os.remove(pdf)

    print(f'removed:{pdf}')
def extract_text(pdf):

    text = textract.process(pdf)

    print(f'Extracting text from {pdf}')

    paras=text.decode().split('\n\n')

    country=paras[1][0:paras[1].find('March')].strip()

    retail_val=paras[7]

    grocery_val=paras[10]

    park_val=paras[13]

    transit_val=paras[43]

    work_val=paras[46]

    resi_val=paras[49]

    res=[country,retail_val,grocery_val,park_val,transit_val,work_val,resi_val]

    return res
import csv
def write_results_to_csv(res,csv_file):

    with open(csv_file,'a',newline='') as file:

        writer=csv.writer(file)

        writer.writerow(res)

    print(f'wrote {res} to {csv_file}')
csv_file='mobility_google.csv'

!rm {csv_file}

res=['Country','Retail & recreation','Grocery & pharmacy', 'Parks', 'Transit stations', 'Workplaces', 'Residential']

write_results_to_csv(res,csv_file)
links=get_links()

for link in links:

    pdf=fetch_pdf(link)

    print(f'Link fetched. File:{pdf}')

    res=extract_text(pdf)

    write_results_to_csv(res,csv_file)

    delete_pdf(pdf)
!cat mobility_google.csv