import requests

from bs4 import BeautifulSoup
sao_paulo_loc = "https://forecast.weather.gov/MapClick.php?lat=31.604&lon=-106.2511#XoOR725v9PU"
page = requests.get(sao_paulo_loc)
soup = BeautifulSoup(page.content, 'html.parser')
seven_day = soup.find(id = "seven-day-forecast")
forecast_items = seven_day.find_all(class_ = "tombstone-container")
def fahrenheit2celsius(tf):

    return (tf - 32) * 5/9
import re

temps_tags = seven_day.select(".temp")

temps = []

for tp in temps_tags:

    tf_str_suja = tp.get_text()

    tf_str_limpa = re.findall("\d+", tf_str_suja)

    tf = float(tf_str_limpa[0])

    tc = fahrenheit2celsius(tf)

    temps.append(tc)

temps