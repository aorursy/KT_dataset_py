import requests
from bs4 import BeautifulSoup
import pandas as pd

headers = ['Notice Number', "Recipient's Name", "Issue Date","Main Activity", 'Page URL', 'Notice URL', 'Notice Type',
 'Description',
 'Compliance Date',
 'Revised Compliance Date',
 'Result',
 'Address',
 'Region',
 'Local Authority',
 'Industry',
 'Main Activity',
 'Type of Location',
 'HSE Group',
 'HSE Directorate',
 'HSE Area ',
 'HSE Division']
result = pd.DataFrame(columns = headers)
#######################
from_page = 1
to_page = 1
#######################
row_n = 0

for page_n in range(from_page, to_page+1):
    
    print("scraping page:", page_n)
    
    page_url = f'https://resources.hse.gov.uk/notices/notices/notice_list.asp?PN={page_n}&ST=N&rdoNType=&NT=&SN=F&EO=LIKE&SF=RN&SV=&SO=DNIS'
    r = requests.get(page_url)
    page_soup = BeautifulSoup(r.text, 'html.parser')
    notices = []
    
    for i, td in enumerate(page_soup.find_all('td')):
        if i%6 == 0:
            notices.append(td.string.strip())
            result.loc[row_n+i//6] = ''
        
        if i%6 == 1:
            result.loc[row_n+i//6]["Recipient's Name"] = td.string
        elif i%6 == 3:
            result.loc[row_n+i//6]["Issue Date"] = td.string
        elif i%6 == 5:
            result.loc[row_n+i//6]["Main Activity"] = td.string
        

    for n in notices:
        temp = []
        result.loc[row_n]['Page URL'] = page_url
        
        notice_url = "https://resources.hse.gov.uk/notices/notices/notice_details.asp?SF=CN&SV=" + n
        
        result.loc[row_n]['Notice Number'] = n
        result.loc[row_n]['Notice URL'] = notice_url
        
        
        r = requests.get(notice_url)
        notice_soup = BeautifulSoup(r.text, 'html.parser')

        t1, t2 = notice_soup.find_all('table')

        tds = t1.find_all('td')[:-1] + t2.find_all('td')[:12] + t2.find_all('td')[13:]
        
        for i, td in enumerate(tds):
            if i%2 == 0:
                h = td.string
            else:
                if h == 'Address':
                    s = "\n".join([s.string for s in td.contents if s.string != None])
                else:
                    s = td.contents[0] if len(td.contents) > 0 else None
                
                temp.append(s)

                result.loc[row_n][h] = s
                
        row_n += 1

result.to_csv('result.csv', sep=',')
result
