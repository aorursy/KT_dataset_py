from pprint import pprint
import requests
from lxml import html
from pymongo import MongoClient
news_mail = 'https://news.mail.ru/'
lenta = 'https://lenta.ru/'
yandex_news = 'https://yandex.ru/news'


header= {"User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.61 Safari/537.36"}


response = requests.get(yandex_news, headers = header)
dom = html.fromstring(response.text)

title = dom.xpath("//h2[contains(@class,'title')]/text()")
title
