import requests
from bs4 import BeautifulSoup
for x in range(1, 6):
    response = requests.get("http://www1.cuny.edu/mu/forum/?sf_paged=%i" % x)
    soup = BeautifulSoup(response.content, 'html.parser')
    column = soup.find_all("li", class_="vc_col-sm-12")
    
    for c in column:
        text = c.find_all('h2')
        if text:
            for t in text:
                final = t.get_text
                if final:
                    print(t.string)
                    print(t.find('a')['href'])
                    
                date = c.find(class_="date")
                if date:
                    print(date.string)