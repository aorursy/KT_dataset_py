!pip install PyPDF2
!pip install tabula-py
import requests
import urllib.request
import tabula
import PyPDF2
import pandas as pd
df_countries = pd.read_csv('.../countries.csv') # Change path to Countries file here
df_countries.at[150,'Abbrev'] = 'NA'
df_countries.set_index('Abbrev',inplace=True)
df_countries.Country = [word.title().replace(' ','_') for word in df_countries.Country]
df_countries.drop_duplicates(inplace=True)
df_states = pd.read_csv('.../US_States.csv').set_index('State') # Change path to States file here
df_states.index = [word.replace(' ','_') for word in df_states.index]
df_states
for country in df_countries.index:
  download_url = 'https://www.gstatic.com/covid19/mobility/2020-03-29_'+country+'_Mobility_Report_en.pdf' # update this link to make sure data is availiable
  path = '.../country/'+country+'.pdf' # Change path to save your file here
  try:
    urllib.request.urlretrieve(download_url,path)
  except IOError:
    pass  
for state in df_states.index:
  download_url = 'https://www.gstatic.com/covid19/mobility/2020-04-05_US_'+state+'_Mobility_Report_en.pdf' # update this link to make sure data is availiable
  path = '.../state/'+state+'.pdf' # Change path to save your file here
  try:
    urllib.request.urlretrieve(download_url,path)
  except IOError:
    pass

def extract_pdf_data(directory):
  import os

  # define empty lists that will hold the six sets of values
  index=[]

  rr_p = []
  gp_p = []
  pa_p = []
  ts_p = []
  wp_p = []
  rs_p = []

  for filename in os.listdir(directory):
    path = directory+filename
    pdfFileObj = open(path, 'rb')

    pageObj = PyPDF2.PdfFileReader(pdfFileObj).getPage(0)

    df = tabula.read_pdf(path, area = pageObj.mediaBox, pages=0)[0]
    
    index.append(filename.split('.')[0])

    try:   
      rr_p.append(int(df.iloc[15,0].split('%')[0])) # Retail & recreation percent
    except:    
      rr_p.append(0)
      
    try:
      gp_p.append(int(df.iloc[23,0].split('%')[0])) # Grocery & pharmacy percent
    except:
      gp_p.append(0)

    try:
      pa_p.append(int(df.iloc[31,0].split('%')[0])) # Parks percent
    except:
      pa_p.append(0)

    df = tabula.read_pdf(path, area = pageObj.mediaBox, pages = 2)[0]
    
    try:
      ts_p.append(int(df.iloc[3,0].split('%')[0])) # Transit stations percent
    except:
      ts_p.append(0)

    try:
      wp_p.append(int(df.iloc[11,0].split('%')[0])) # Workplaces percent
    except:
      wp_p.append(0)

    try:
      rs_p.append(int(df.iloc[19,0].split('%')[0])) # Residential percent
    except:
      rs_p.append(0)

  columns = ['Retail_Recreation','Grocery_Pharmacy','Parks','Transit_Stations','Workplaces','Residential']
  df = pd.DataFrame(data = [rr_p,gp_p,pa_p,ts_p,wp_p,rs_p]).T
  df.columns = columns
  df.index = index
  
  return df
path ='...' # Change path to the saved pdf files

df_us_state_percent = extract_pdf_data(path+'state/')
df_us_country_percent = extract_pdf_data(path+'country/')
df_us_state_percent['Abbrev'] = df_states.Abbrev
df_us_state_percent['State_Name'] = df_us_state_percent.index
df_us_state_percent.set_index('Abbrev',inplace=True)
df_us_state_percent

df_country_percent.index.name = 'Abbrev'
df_country_percent['Country_Name'] = df_countries.Country
df_country_percent
df_country_percent.to_csv(path+'COVID19_Google_Mobility_Report_Country.csv')
df_us_state_percent.to_csv(path+'COVID19_Google_Mobility_Report_US_State.csv')