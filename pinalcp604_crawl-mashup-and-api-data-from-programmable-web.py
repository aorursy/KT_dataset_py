# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#c:\users\kbotange\Appdata\local\continuum\anaconda3\python.exe c:\users\kbotange\desktop\apicrawl.py

#c:\\users\\kbotange\\desktop\\api_data.csv

import requests

from lxml import etree

import re

import csv

import multiprocessing

import time

import os

from urllib.parse import quote



'''

https://www.programmableweb.com/category/all/mashups?order=created&sort=desc

这个是mushups的

'''

headers = {

    ''

}



username = quote('pishah')

password = quote('jalaram123')

proxy = 'cache.aut.ac.nz'

proxy_port = '3128'



proxy_http = 'http://' + username + ':' + password + '@' + proxy + ':' + proxy_port

proxy_https = 'https://' + username + ':' + password + '@' + proxy + ':' + proxy_port



proxies = {'http' : proxy_http, 'https': proxy_https}







def get_programm_mashoup_urls(urls):

	try:           

		mashup_data = requests.get(url=urls)

		mashup_data.encoding = 'utf-8'

		et = etree.HTML(mashup_data.text)

		mashoup_name_hrefs = et.xpath('//*[@id="block-system-main"]/article/div[7]/div[1]/table/tbody/tr/td[1]/a/@href')

		Categorys = et.xpath('//*[@id="block-system-main"]/article/div[7]/div[1]/table/tbody/tr/td[3]/a/text()')

		Submitteds = et.xpath('//*[@id="block-system-main"]/article/div[7]/div[1]/table/tbody/tr/td[4]/text()')

		for masoup_name_href,Category,Submitted in zip(mashoup_name_hrefs,Categorys,Submitteds):

			data = {

				'masoup_name_href':masoup_name_href.strip(),

				'Category':Category.strip(),

				'Submitted':Submitted.strip()

			}

			try:

				programm_mashoup_info(data)

			except Exception as e:

				print(e)

				pass

	except Exception as e:

		print(e)

		pass

def programm_mashoup_info(data):

	try:

		url = 'https://www.programmableweb.com' + data['masoup_name_href']

		mashoup_info_data = requests.get(url=url)

		info_et = etree.HTML(mashoup_info_data.text)

		mashoup_name = ''.join(info_et.xpath('/html/body/div[5]/div[1]/section/div[2]/section/article/header/div[2]/div/h1/text()'))

		tags = info_et.xpath('/html/body/div[5]/div[1]/section/div[2]/section/article/header/div[2]/div[2]/div/a/text()')

		desc = re.sub('\\n|\\r','',''.join(info_et.xpath('//*[@id="tabs-header-content"]/div/div[1]/div/div/div/text()')))

		apis = info_et.xpath('//*[@id="tabs-content"]/div[1]/div[label[text()="Related APIs"]]/span/a/text()')

		link = info_et.xpath('//*[@id="tabs-content"]/div[1]/div[label[text()="URL"]]/span/a/text()')

		company = info_et.xpath('//*[@id="tabs-content"]/div[1]/div[label[text()="Company"]]/span/text()')

		type = info_et.xpath('//*[@id="tabs-content"]/div[1]/div[label[text()="Mashup/App Type"]]/span/text()')

		info_data = {

			'mashoup_name':mashoup_name.strip(),

			'Category':data['Category'].strip(),

			'tags':tags,

			'Submitted':data['Submitted'].strip(),

			'desc':desc.strip(),

			'apis':apis,

			'link':link,	

			'company':company,	

			'type':type,

		}

		print(info_data)

		# try:

		sav_data(info_data)

		# except Exception as e:

		#     print(e)

		#     pass

	except:

		pass

def sav_data(info_data):

    f_w = open('/kaggle/working/mashup_data.csv','a+',encoding='utf-8',newline='')

    writer = csv.writer(f_w)

    writer.writerow((info_data['mashoup_name'],info_data['Category'],info_data['tags'],info_data['Submitted'],info_data['desc'],info_data['apis'],info_data['link'],info_data['company'],info_data['type']))

    f_w.flush()

    f_w.close()

if __name__ == '__main__':

    f = open('/kaggle/working/mashup_data.csv', 'w+', encoding='utf-8', newline='')

    urls = ['https://www.programmableweb.com/category/all/mashups?order=created&sort=desc&page={page}'.format(page=page) for page in range(1,260)]

    pool = multiprocessing.Pool(processes=4)

    pool.map(get_programm_mashoup_urls,urls)

    pool.close()

    pool.join()
import requests

from lxml import etree

import re

import csv

import multiprocessing

import time

import random

'''

https://www.programmableweb.com/category/all/mashups?order=created&sort=desc

这个是mushups的

'''

headers = {

    ''

}



proxies = {

	'http': 'http://cache.aut.ac.nz:3128',

	'https': 'http://cache.aut.ac.nz:3128',

}



def get_programm_api_urls(urls):

	try:           

		api_data = requests.get(url=urls)

		api_data.encoding = 'utf-8'

		et = etree.HTML(api_data.text)

		Names = et.xpath('//*[@id="block-system-main"]/article/div[7]/div[2]/table/tbody/tr/td[1]/a/@href')

		Categorys = et.xpath('//*[@id="block-system-main"]/article/div[7]/div[2]/table/tbody/tr/td[3]/a/text()')

		Submitteds = et.xpath('//*[@id="block-system-main"]/article/div[7]/div[2]/table/tbody/tr/td[4]/text()')

		time.sleep(random.uniform(0,5))

		for Name,Category,Submitted in zip(Names,Categorys,Submitteds):

			data = {

				'Name':Name.strip(),

				'Category':Category.strip(),

				'Submitted':Submitted.strip()

			}

			try:

				programm_api_info(data)

			except Exception as e:

				print(e)

				pass

	except Exception as e:

		print(e)

		pass

def programm_api_info(data):

	try:

		url = 'https://www.programmableweb.com' + data['Name']

		api_info_data = requests.get(url=url)

		info_et = etree.HTML(api_info_data.text)

		api_name = ''.join(info_et.xpath('/html/body/div[5]/div[1]/section/div[2]/section/article/header/div[2]/div/h1/text()'))

		tags = info_et.xpath('/html/body/div[5]/div[1]/section/div[2]/section/article/header/div[2]/div[2]/div/a/text()')

		desc = re.sub('\\n|\\r','',''.join(info_et.xpath('//*[@id="tabs-header-content"]/div/div[1]//text()')))

		try:

			mashup = info_et.xpath('//*[@id="block-views-api-mashups-new-list-top"]/div[1]/span/text()')					

			follower = info_et.xpath('//*[@id="block-views-api-followers-row-top"]/div[1]/span/text()')

			provider = info_et.xpath('//*[@id="tabs-content"]/div[2]/div[label[text()="API Provider"]]/span/a/text()')

			endpoint = info_et.xpath('//*[@id="tabs-content"]/div[2]/div[label[text()="API Endpoint"]]/span/a/text()')

			portal = info_et.xpath('//*[@id="tabs-content"]/div[2]/div[label[text()="API Portal / Home Page"]]/span/a/text()')

							

		except IndexError as e:

			print(e)

			mashup = None

			follower = None

			provider = None

			endpoint = None

			portal = None

		try:

			info_data = {

				'api_name':api_name.strip(),

				'Category':data['Category'].strip(),

				'tags':tags,

				'Submitted':data['Submitted'].strip(),

				'desc':desc.strip(),

				'mashup':mashup,

				'follower':follower,

				'provider':provider,

				'endpoint':endpoint,

				'portal':portal,				

			}

		except Exception as e:

				print(e)

				pass			

		print(info_data)

		sav_data(info_data)		

	except:

		pass

def sav_data(info_data):

    f_w = open('/kaggle/working/api_data.csv','a+',encoding='utf-8',newline='')

    writer = csv.writer(f_w)

    writer.writerow((info_data['api_name'],info_data['Category'],info_data['tags'],info_data['Submitted'],info_data['desc'],info_data['mashup'],info_data['follower'],info_data['provider'],info_data['endpoint'],info_data['portal']))

    f_w.flush()

    f_w.close()

if __name__ == '__main__':

    f = open('/kaggle/working/api_data.csv', 'w+', encoding='utf-8', newline='')

    urls = ['https://www.programmableweb.com/category/all/apis?page={page}'.format(page=page) for page in range(0,720)]

    pool = multiprocessing.Pool(processes=4)

    pool.map(get_programm_api_urls,urls)

    pool.close()

    pool.join()