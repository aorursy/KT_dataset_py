import pandas as pd
from bs4 import BeautifulSoup
from time import sleep
from selenium import webdriver
import matplotlib.pyplot  as plt
import seaborn as sns
driver = webdriver.Chrome(executable_path='../chromedriver/chromedriver.exe')
driver.get('https://www.worldometers.info/coronavirus/#countries')
html = driver.page_source
soup2 = BeautifulSoup(html,"html")
table = soup2.find('table',class_="table table-bordered table-hover main_table_countries dataTable no-footer")
# create list for column name 
heading = []
for i in table.find_all('th'):
    heading.append(i.text)
heading
# create list for each row
tr = []
#append each row in the list
for i in table.find_all('tr'):
    tr.append(i)
#list for each column
Country = []
TotalCases = []
NewCases = []
TotalDeaths = []
NewDeaths = []
TotalRecovered = []
ActiveCases = []
Serious = []
Total_cases = []
Deaths1m  = []
TotalTests = []
Tests1m = []
Continen = []
#itirate for each coulmn data to append it row by row
for i in tr[1:]:
    try:
        Country.append(i.find_all('td')[0].text)
        TotalCases.append(i.find_all('td')[1].text)
        NewCases.append(i.find_all('td')[2].text)
        TotalDeaths.append(i.find_all('td')[3].text)
        NewDeaths.append(i.find_all('td')[4].text)
        TotalRecovered.append(i.find_all('td')[5].text)
        ActiveCases.append(i.find_all('td')[6].text)
        Serious.append(i.find_all('td')[7].text)
        Total_cases.append(i.find_all('td')[8].text)
        Deaths1m.append(i.find_all('td')[9].text)
        TotalTests.append(i.find_all('td')[10].text)
        Tests1m.append(i.find_all('td')[11].text)
        Continen.append(i.find_all('td')[12].text)
    except:
          print('error')
          break
# create dataframe
df = pd.DataFrame({
    'country':Country,
    'total_cases':TotalCases,
    'new_case':NewCases,
    'total_deaths':TotalDeaths,
    'today_death':NewDeaths,
    'total_recovery':TotalRecovered,
    'total_active_cases':ActiveCases,
    'critical_cases':Serious,
    'Total1m':Total_cases,
    'Deaths1m':Deaths1m,
    'TotalTests':TotalTests,
    'Tests1m':Tests1m,
    'continen':Continen
    
})
df.info()
df.isnull().sum() 
#there is white space values which mean 0 
new_case_new = []
today_death_new = []
for i in df.new_case:
    if i == '':
        new_case_new.append(0)
    else:
        new_case_new.append(i)
for i in df.today_death:
    if i == '':
        today_death_new.append(0)
    else:
        today_death_new.append(i)
df.new_case = new_case_new
df.today_death = today_death_new
# I drop these columns beacuse i dont understand them :)
df.drop(['Total1m','Deaths1m','TotalTests','Tests1m'],axis=1,inplace=True)
# save csv file
df.to_csv('../data/cases_by_conutries.csv')