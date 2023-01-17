!pip install git+https://github.com/alirezamika/autoscraper.git
from autoscraper import AutoScraper



url = 'https://stackoverflow.com/questions/2081586/web-scraping-with-python'



# We can add one or multiple candidates here.

# You can also put urls here to retrieve urls.

wanted_list = ["How to call an external command?"]



scraper = AutoScraper()

result = scraper.build(url, wanted_list)

print(result)
scraper.get_result_similar('https://stackoverflow.com/questions/606191/convert-bytes-to-a-string')
from autoscraper import AutoScraper



url = 'https://finance.yahoo.com/quote/AAPL/'



wanted_list = ["124.81"]



scraper = AutoScraper()



# Here we can also pass html content via the html parameter instead of the url (html=html_content)

result = scraper.build(url, wanted_list)

print(result)
scraper.get_result_exact('https://finance.yahoo.com/quote/MSFT/')

url = 'https://github.com/alirezamika/autoscraper'



wanted_list = ['A Smart, Automatic, Fast and Lightweight Web Scraper for Python', '662', 'https://github.com/alirezamika/autoscraper/issues']



scraper.build(url, wanted_list)
# Give it a file path

scraper.save('yahoo-finance')
scraper.load('yahoo-finance')