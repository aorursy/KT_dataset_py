import requests
from bs4 import BeautifulSoup as bs
import re
import json
import random
import time
import pandas
import os

gap = 0
def get_detail(extra_url='/wiki/AECOM'):
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.81 Safari/537.36'
    }
    url = 'https://en.wikipedia.org' + extra_url
    print("\t\tanalyse " + url)
    response = requests.get(url, headers=headers).content
    soup = bs(response, 'html.parser')

    for sup in soup.find_all("sup"):
        sup.clear()

    info_box = soup.find("table", class_="infobox vcard")
    detail_dict = {tr.find("th").get_text(): tr.find("td").get_text() for tr in info_box.find_all("tr") if
                   tr.find("th")}

    ps = [p for p in info_box.find_all_next("p") if p in soup.find("h2").find_all_previous("p")]
    detail = "".join([p.get_text() for p in ps])

    return detail, detail_dict
def get_info(extra_url="/wiki/Companies_listed_on_the_New_York_Stock_Exchange_(0%E2%80%939)", stock_name='Stock name'):
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.81 Safari/537.36'
    }
    url = 'https://en.wikipedia.org' + extra_url
    print("\tanalyse " + url)
    response = requests.get(url, headers=headers).content
    soup = bs(response, 'html.parser')
    table = soup.find("div", id='mw-content-text').find("table", style='background:transparent;').find_all("tr")
    results = []
    keys = [column.get_text().strip() for column in table[0].find_all("th")]
    for row in table[1:]:
        sleep_time = random.random() * gap
        result = {}
        for key, column in zip(keys, row.find_all("td")):
            result[key] = column.get_text().strip()
            result['detail'] = None
            result['detail_dict'] = None
            if key == stock_name and column.find("a"):
                try:
                    detail, detail_dict = get_detail(column.find("a")['href'])
                    result['detail'] = detail
                    result['detail_dict'] = detail_dict
                except:
                    print("No details " + url)
                    continue
        results.append(result)
        time.sleep(sleep_time)
        print("\tsleep {} seconds".format(sleep_time))
    return results

def get_all():
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.81 Safari/537.36'
    }
    url = 'https://en.wikipedia.org/wiki/New_York_Stock_Exchange'
    response = requests.get(url).content
    soup = bs(response, 'html.parser')
    url_list = [a['href'] for a in
                soup.find("table", style="margin:0 auto", class_="toccolours").find("p").find_all("a")]
    results = []
    for extra_url in url_list:
        if os.path.exists("{}.json".format(extra_url.split("(")[1].split(")")[0])):
            continue
        result = get_info(extra_url)
        results.append(result)
        sleep_time = random.random() * gap
        print("sleep {} seconds".format(sleep_time))
        time.sleep(sleep_time)
    with open("temp.json", "w") as f:
        json.dump(results, f)
get_all()