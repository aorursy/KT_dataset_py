# Import useful libraries and classes.

from urllib.request import urlopen as uReq

from bs4 import BeautifulSoup as soup
#html page upload and read in web_page variable.

my_url=  "https://www.flipkart.com/search?p%5B%5D=facets.brand%255B%255D%3DSamsung&sid=tyy%2F4io&sort=recency_desc&wid=1.productCard.PMU_V2_1"

web_page= uReq(my_url)

page_html= web_page.read()
#html parser. It is to beautify the HTML code.

page_soup= soup(page_html)

page_soup


containers= page_soup.findAll("div", {"class": "_1-2Iqu row"})

print(len(containers))


model_name_phone= containers[5].findAll("div", {"class": "_3wU53n"})



print(model_name_phone[0].text.strip())

# how many elements does 'tVe95H' class contained in it?

Specs_phone= containers[7].findAll("li", {"class": "tVe95H"})



print(Specs_phone[2].text.strip())