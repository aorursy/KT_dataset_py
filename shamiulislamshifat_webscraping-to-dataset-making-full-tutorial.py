import requests
page = requests.get("http://dataquestio.github.io/web-scraping-pages/simple.html")
page
page.status_code
page.content
from bs4 import BeautifulSoup
soup = BeautifulSoup(page.content, 'html.parser')
print(soup.prettify())
soup = BeautifulSoup(page.content, 'html.parser')
soup.find_all('p')
soup.find_all('p')[0].get_text()
page = requests.get("http://dataquestio.github.io/web-scraping-pages/ids_and_classes.html")
soup = BeautifulSoup(page.content, 'html.parser')
soup
soup.find_all('p', class_='outer-text')
soup.find_all(class_="outer-text")
page = requests.get("http://forecast.weather.gov/MapClick.php?lat=37.7772&lon=-122.4168")
soup = BeautifulSoup(page.content, 'html.parser')
seven_day = soup.find(id="seven-day-forecast")
forecast_items = seven_day.find_all(class_="tombstone-container")
tonight = forecast_items[0]
print(tonight.prettify())
period = tonight.find(class_="period-name").get_text()
short_desc = tonight.find(class_="short-desc").get_text()
#temp = tonight.find(class_="temp").get_text()
print(period)
print(short_desc)
#print(temp)
img = tonight.find("img")
desc = img['title']
print(desc)
#lets extract all info
period_tags = seven_day.select(".tombstone-container .period-name")
periods = [pt.get_text() for pt in period_tags]
periods
short_descs = [sd.get_text() for sd in seven_day.select(".tombstone-container .short-desc")]
temps = [t.get_text() for t in seven_day.select(".tombstone-container .temp")]
descs = [d["title"] for d in seven_day.select(".tombstone-container img")]
print(short_descs)
print(temps)
print(descs)
import pandas as pd
weather = pd.DataFrame({
    "period": periods,
    "short_desc": short_descs,
    "desc":descs
})
weather
temp_nums = weather["temp"].str.extract("(?P<temp_num>d+)", expand=False)
weather["temp_num"] = temp_nums.astype('int')
temp_nums
weather["temp_num"].mean()
is_night = weather["temp"].str.contains("Low")
weather["is_night"] = is_night
is_night
weather[is_night]
