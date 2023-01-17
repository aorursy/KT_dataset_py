# Note: for show only. Meant to be run natively on a machine, not in Kaggle.

# Web Scraper 1.0 for Upwork.com - shhh

import time
from selenium import webdriver
import pickle
import bs4
import pandas as pd
import pyautogui

# Set Gecko Driver, declare data objects
driver = webdriver.Firefox()
nameSeries = []
priceSeries = []
worktitleSeries = []
earningsSeries = []
profileSeries = []

# Scraping 50 pages worth
x = 1
while x < 50:
    y = str(x)
    baseurl = "https://www.upwork.com/ab/profiles/search/?q=data analyst&user_pref=2&page=" + y
    driver.get(baseurl)

    # saving and loading cookies into driver
    pickle.dump(driver.get_cookies(), open("cookies.pkl", "wb"))
    cookies = pickle.load(open("cookies.pkl", "rb"))
    for cookie in cookies:
            driver.add_cookie(cookie)
    time.sleep(0.5)

    # extract HTML
    soup = bs4.BeautifulSoup(driver.page_source)

    # Captcha check! Plus "magic" - returns to top of loop
    captchacheck = soup.find("script")
    if captchacheck.get("src") == "https://www.google.com/recaptcha/api.js":
        time.sleep(5)
        pyautogui.click(x=700, y=110)
        time.sleep(2)
        pyautogui.press("enter")
        x = x-1

    # keep going
    else:
        x = x+1

        # pull individual lists of tags containing desired information - using class name
        name = soup.find_all("a", class_="freelancer-tile-name")
        price = soup.find_all("strong", class_="pull-left")
        worktitle = soup.find_all("h4", class_="m-0 freelancer-tile-title ellipsis d-none d-lg-block d-xl-block")
        earnings = soup.find_all("div", class_="col-xs-3 p-0-right")

    # This part could have been done via some kind of method. May mess with this later to simplify
        # 1. Skip when information isn't found
        # 2. Append desired information into "Series" list for each tag in list
        for index in range(len(name)):
            if name == None:
                nameSeries.append("")
            else:
                nameSeries.append(name[index].get("title"))
                profileSeries.append(name[index].get("href"))
        for index in range(len(price)):
            if price == None:
                priceSeries.append("")
            else:
                priceSeries.append(price[index].string.strip())

        # this one deals with when there's HTML in the title
        for index in range(len(worktitle)):
            if worktitle[index] == None:
                worktitleSeries.append("")
            else:
                try:
                    worktitleSeries.append(worktitle[index].string.strip())
                except:
                    worktitleSeries.append("")

        # this one gets information from a child node of the original tag
        for index in range(len(earnings)):
            if (index + 2) % 2 == 1:
                earningstag = earnings[index].div
                if earningstag == None:
                    earningsSeries.append("")
                else:
                    earningsSeries.append(earningstag.get("data-combined-earnings"))

# declare each "Series" list as a Pandas series
nameSeries = pd.Series(nameSeries[0::2])
priceSeries = pd.Series(priceSeries)
worktitleSeries = pd.Series(worktitleSeries)
earningsSeries = pd.Series(earningsSeries)
profileSeries = pd.Series(profileSeries[0::2])

# Combine all Pandas "series" into a list of series
totalSeries = [nameSeries, priceSeries, worktitleSeries, earningsSeries, profileSeries]

# Concatenate list of series into Pandas Dataframe
df = pd.concat(totalSeries, axis=1)

# Print to console, export results to CSV
print(df)
df.to_csv("results.csv")
