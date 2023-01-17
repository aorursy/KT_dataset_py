# importing required libraries
from urllib.request import urlopen as uReq
from bs4 import BeautifulSoup
import pandas as pd
# Creating list in which program will add scraped data
NT_list= []

# looping pages in aruodas
for i in range(150):
    # link for page, formated to each page
    aruodas = 'https://www.aruodas.lt/butai/vilniuje/puslapis/{}/'.format(i)
    #opening page
    page = uReq(aruodas)
    # reading html code
    page_html = page.read()
    # formating with soup
    page_soup = BeautifulSoup(page_html, 'html.parser')
    #closing page
    page.close()
    # all container which contain every information about appartment
    containers = page_soup.findAll('tr', {'class': 'list-row'})
    

    # looping each element in all container                                   
    for container in containers:
        try:
            # taking separate information blocks. text, taking only text and strip removes spaces
            title = container.a.img['title']           
            whole_price = container.findAll('span',{'class':'list-item-price'})[0].text
            number_of_rooms = container.findAll('td',{'class':'list-RoomNum'})[0].text.strip() 
            area = container.findAll('td',{'class':'list-AreaOverall'})[0].text.strip()
            floor_number = container.findAll('td',{'class':'list-Floors'})[0].text.strip()
            price_area = container.findAll('span',{'class':'price-pm'})[0].text.strip()
            
            # adding information to created list
            NT_list.append((title,  whole_price, number_of_rooms, area, floor_number, price_area))
            
            # if there is no information, we skip that advertise
        except AttributeError:
            pass
    
# Transforming NT list to dataframe
NT = pd.DataFrame(NT_list)
# And taking csv file
NT.to_csv('NT_raw_csv.csv', index = False)
