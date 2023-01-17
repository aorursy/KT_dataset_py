from bs4 import BeautifulSoup
import requests
import pandas as pd
all_years=pd.DataFrame(columns=['link','title','speaker','event','year'])
years=range(1996,2006)
for year in years:
    speeches_one_year = pd.DataFrame()
    page = requests.get(f'https://www.federalreserve.gov/newsevents/speech/{year}speech.htm')
    soup = BeautifulSoup(page.text, 'html.parser')
    title = soup.select(".title")
    speakers = soup.select(".speaker")
    locations = soup.select(".location")
    for i in range(len(title)):
        speeches_one_year.at[i,'link'] = 'https://www.federalreserve.gov'+title[i].find_all('a', href=True)[0]['href']
        speeches_one_year.at[i,'title'] = title[i].text.split('\n')[1]
        speeches_one_year.at[i,'speaker'] = speakers[i].text.split('\n')[1].strip()
        speeches_one_year.at[i,'event'] = locations[i].text.split('\n')[1].strip()
        speeches_one_year.at[i,'year'] = year
    all_years=all_years.append(speeches_one_year,ignore_index=True)
years=range(2006,2021)
for year in years:
    if year > 2010:
        page = requests.get(f'https://www.federalreserve.gov/newsevents/speech/{year}-speeches.htm')
    else:
        page = requests.get(f'https://www.federalreserve.gov/newsevents/speech/{year}speech.htm')
    soup = BeautifulSoup(page.text, 'html.parser')
    events = soup.select(".eventlist__event")
    speeches_one_year = pd.DataFrame()
    for i,speech in enumerate(events):
        speeches_one_year.at[i,'link'] = 'https://www.federalreserve.gov'+events[i].find_all('a', href=True)[0]['href']
        speeches_one_year.at[i,'title'] = events[i].text.split('\n')[2]
        if events[i].text.split('\n')[3]=='Watch Live' or events[i].text.split('\n')[3]=='Video':
            speeches_one_year.at[i,'speaker'] = events[i].text.split('\n')[4]
            speeches_one_year.at[i,'event'] = events[i].text.split('\n')[5]
        else:
            speeches_one_year.at[i,'speaker'] = events[i].text.split('\n')[3]
            speeches_one_year.at[i,'event'] = events[i].text.split('\n')[4]
        speeches_one_year.at[i,'year'] = year
    all_years=all_years.append(speeches_one_year,ignore_index=True)
old_site_version_length = sum(all_years['year']<1999)
for i in range(old_site_version_length):
    print(i)
    page = requests.get(all_years.loc[i,'link'])
    soup = BeautifulSoup(page.text, 'html.parser')
    text_list = [i for i in soup.find('p').getText().split('\n') if i] 
    text_list=text_list[:-8]
    text_list = ' '.join(text_list)
    text_list = text_list.replace('--', ' ')
    text_list = text_list.replace('\r', '')
    text_list = text_list.replace('\t', '')
    all_years.loc[i,'text'] = text_list
for i in range(len(all_years)):
    if ((all_years.loc[i,'year']>1998) & (all_years.loc[i,'year']<2006)):
        print(i)
        page = requests.get(all_years['link'].iloc[i])
        soup = BeautifulSoup(page.text, 'html.parser')
        events = soup.select("table")
        if len(str(events[0].text))>600:
            text_list = [i for i in events[0].text if i] 
        else:
            text_list = [i for i in events[1].text if i]
        text_list = ''.join(text_list)
        text_list = text_list.replace('--', '')
        text_list = text_list.replace('\r', '')
        text_list = text_list.replace('\t', '')
        if ((i>=383) & (i<=536)):
            text_list = text_list.replace('     ', ' ')
            text_list = text_list.replace('    ', ' ')
        all_years.loc[i,'text'] = text_list
black_listed=[744,748]
for i in range(1,len(all_years)):
    if ((all_years.loc[i,'year']>2005) and (i not in black_listed)):
        print(i)
        page = requests.get(all_years.loc[i,'link'])
        soup = BeautifulSoup(page.text, 'html.parser')
        events = soup.select(".col-md-8")
        text_list = events[1].text
        text_list = text_list.replace('\xa0', ' ')
        text_list = text_list.replace('\n', ' ')
        all_years.loc[i,'text'] = text_list
all_years['date'] = all_years['link'].str.extract('(\d\d\d\d\d\d\d\d)')
all_years = all_years[~all_years['text'].isna()]
all_years['text_len'] = all_years['text'].str.split().apply(len)
all_years['location'] = all_years.event.str.split(', ').apply(lambda x: x[-1])
all_years.loc[all_years['speaker']=='Chairman  Ben S. Bernanke','speaker'] = 'Chairman Ben S. Bernanke'
all_years.loc[all_years['speaker']=='Governor Ben S. Bernanke and Vincent R. Reinhart, Director, Division of Monetary Affairs','speaker'] = 'Governor Ben S. Bernanke'
all_years.loc[all_years['speaker']=='Governor Donald L. Kohn and Brian P. Sack, Senior Economist','speaker'] = 'Governor Donald L. Kohn'
all_years.loc[all_years['speaker']=='Governor Susan Schmidt Bies','speaker'] = 'Governor Susan S. Bies'
all_years.loc[all_years['speaker']=='Vice Chair for Supervision and Chair of the Financial Stability Board Randal K. Quarles','speaker'] = 'Vice Chair for Supervision Randal K. Quarles'
all_years.loc[all_years['speaker']=='Vice Chairman for Supervision and Chair of the Financial Stability Board Randal K. Quarles','speaker'] = 'Vice Chair for Supervision Randal K. Quarles'
all_years.loc[all_years['speaker']=='Vice Chairman for Supervision Randal K. Quarles','speaker'] = 'Vice Chair for Supervision Randal K. Quarles'
all_years.loc[all_years['speaker']=='Vice Chairman Roger W. Ferguson, Jr','speaker'] = 'Vice Chairman Roger W. Ferguson'
all_years.loc[all_years['speaker']=='Vice Chairman Roger W. Ferguson, Jr.','speaker'] = 'Vice Chairman Roger W. Ferguson'
all_years.loc[all_years['speaker']=='Chair Jerome H. Powell','speaker'] = 'Chairman Jerome H. Powell'
all_years.loc[all_years['speaker']=='Vice Chair Richard H. Clarida','speaker'] = 'Vice Chairman Richard H. Clarida'
all_years = all_years[all_years['speaker']!='Brian F. Madigan, Director, Division of Monetary Affairs']
all_years = all_years[all_years.text_len!=0]
all_years.to_csv('fed_speeches_1996_2020.csv',index=False)