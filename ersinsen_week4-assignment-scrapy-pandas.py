# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
## scrapy ana py dosyasi

import scrapy
from ..items import ImbdItem
import requests
#from bs4 import BeautifulSoup

# film ismi, tarihi(detayli yayinlanma tarihi ),
# hangi ulkede cekildigi,yonetmen,oyuncular,
# rates puani
# scrapy crawl imbd_top330 -o imbd_top.json


class Imbd_top330Spider(scrapy.Spider):
    name = 'imbd_top330'
    # kullanacagimiz url tanimliyoruz
    start_urls = [
        "https://www.imdb.com/list/ls004610270/?st_dt=&mode=detail&page=1&sort=list_order,asc&ref_=ttls_vm_dtlhttps://www.imdb.com/list/ls004610270/?st_dt=&mode=detail&page=1&sort=list_order,asc&ref_=ttls_vm_dtl"]
    # page_count = 0

    def parse(self, response):
        # detay sayfasina gitmek icin linklere ulasiyoruz
        hrefs = response.css(
            "div.lister-item.mode-detail h3 a ::attr(href)").getall()
        for href in hrefs:  # filmlerin kayitli oldugu div in adi
            # tum url leri birlestiriyoruz response.urljoin -scrapy metod
            url = response.urljoin(href)
            # html = requests.get(url).content
            # self.soup = BeautifulSoup(html, "html.parser")
            yield scrapy.Request(url, callback=self.parse_page)

        next_page = response.css(
            "a.flat-button.lister-page-next.next-page ::attr(href)")
        if next_page:
            url = response.urljoin(next_page[0].extract())
            yield scrapy.Request(url, self.parse)

    def parse_page(self, response):
        item = ImbdItem()

        movie_name = response.css("div.title_wrapper h1::text").get()
        movie_date_country = response.css(
            'div.subtext :last-child::text').get()
        movie_date = movie_date_country.split(' (')
        movie_country = movie_date[1].split(')')
        movie_director = response.xpath(
            '//*[@class="plot_summary "]/div[2]/a/text()').get()
        movie_rate = response.xpath(
            '//*[@id="title-overview-widget"]/div[1]/div[2]/div/div[1]/div[1]/div[1]/strong/span/text()').get()
        movie_stars = response.css(
            'div.plot_summary  div:nth-child(4) a:not(:last-child) ::text').getall()

        item['movie_name'] = movie_name
        item['movie_country'] = movie_country[0]
        item['movie_date'] = movie_date[0]
        item['movie_director'] = movie_director
        item['movie_rate'] = movie_rate
        item['movie_stars'] = movie_stars

        yield item
# elde edilen csv dosyasi ile pandas ile asagida devam edilecek
imbd_data=pd.read_csv('/kaggle/input/imbd-top330movies-csv/imbd_top.csv')
imbd_data.info()
imbd_data.describe()
sort_byRate=imbd_data.sort_values(by='movie_rate',ascending=False)
sort_byRate
