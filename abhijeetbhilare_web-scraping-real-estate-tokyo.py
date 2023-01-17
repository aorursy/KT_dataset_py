import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import re

import time

from datetime import datetime

import matplotlib.dates as mdates

import seaborn as sns

import matplotlib.ticker as ticker

from urllib.request import urlopen

from bs4 import BeautifulSoup

import requests

from datetime import datetime

import warnings

warnings.filterwarnings('ignore')

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
def get_data(url):  

    page = requests.get(url)

    soup = BeautifulSoup(page.content, 'html.parser')

    real_estate_data = soup.find_all("dl", class_='dl-horizontal-border')

    is_property_found = "Yes"

    data_dict = {}

    data_dict["URL"] = url

    if real_estate_data:

        for d in real_estate_data:

            dt = d.find_all('dt')

            dd = d.find_all('dd')

            

            for i,j in zip(dt, dd):

                i = i.contents[0].strip()

                j = j.contents[0].strip()

                if i == "Unit Number":

                    if j == "-":

                        data_dict["Unit Number"] = np.nan

                    else:

                        data_dict["Unit Number"] = j

                elif i == "Price":

                    data_dict["Price(¥)"] = float(j.replace(",","").replace("¥","").strip())

                elif i == "Building Name":

                    data_dict["Building Name"] = j

                elif i == "Available From":

                    if "Please Inquire" in j:

                        data_dict["Available From"] = np.nan

                    else:

                        data_dict["Available From"] = datetime.strptime(j, '%b %d, %Y')

                elif i == "Type":

                    data_dict["Type"] = j.replace(" ", "")

                elif i == "Size":

                    data_dict["Size(m²)"] = float(j.replace("m²", "").replace(",", "").strip())

                elif i == "Gross Yield":

                    data_dict["Gross Yield(%)"] = float(j.replace("%", "").strip())

                elif i == "Land Rights":

                    data_dict["Land Rights"] = j

                elif i == "Maintenance Fee":

                    data_dict["Maintenance Fee(¥/mnt)"] = float(j.replace("¥", "").replace(" / mth", "").strip().replace(",",""))

                elif i == "Location":

                    data_dict["Location"] = j.replace(",", "")

                elif i == "Occupancy":

                    data_dict["Occupancy"] = j

                elif i == "Floor":

                    data_dict["Floor"] = j.replace(" ", "")

                elif i == "Nearest Station":

                    data_dict["Nearest Station"] = j.split("(")[0].strip()

                    if len(j.split("(")) > 1:

                        if "walk" in j:

                            data_dict["Way to Nearest Station"] = "Walk"

                            data_dict["Distance From Station(min)"] = j.split("(")[1].split("min")[0].strip()

                        elif "bus" in j:

                            data_dict["Way to Nearest Station"] = "Bus"

                            data_dict["Distance From Station(min)"] = j.split("(")[1].split("min")[0].strip()

                elif i == "Layout":

                    data_dict["Layout"] = j

                elif i == "Year Built":

                    data_dict["Year Built"] = j

                elif i == "Direction Facing":

                    data_dict["Direction Facing"] = j.replace(",", "")

                elif i == "Transaction Type":

                    data_dict["Transaction Type"] = j

                elif i == "Balcony Size":

                    data_dict["Balcony Size(m²)"] = float(j.replace("m²", "").replace(",", "").strip())

                elif i == "Building Description":

                    data_dict["Building Description"] = j.replace(",", "")

                elif i == "Other Expenses":

                    j = j.replace(",", "").replace(" ", "").replace("，", "")

                    lst = re.findall(r'\d+', j)

                    if len(lst) > 0:

                        lst = [int(i) for i in lst] 

                        data_dict["Other Expenses"] = sum(lst)

                elif i == "Parking":

                    data_dict["Parking Available"] = j.split()[0].replace(",", "")

                    if len(j.split()) > 1:

                        if j.split()[0].replace(",", "") == "Available":

                            data_dict["Parking Fee(¥/mnt)"] = float(j.split()[1].replace(",", "").replace("¥", "").strip())

                elif i == "Date Updated":

                    if "Please Inquire" in j:

                        data_dict["Date Updated"] = np.nan

                    else:

                        data_dict["Date Updated"] = datetime.strptime(j, '%b %d, %Y')

                elif i == "Next Update Schedule":

                    if "Please Inquire" in j:

                        data_dict["Next Update Schedule"] = np.nan

                    else:

                        data_dict["Next Update Schedule"] = datetime.strptime(j, '%b %d, %Y')

                    

    else:

        is_property_found = "No"

    data_dict["Is_Prop_Avl"] = is_property_found

    return data_dict

        
df = pd.read_csv("/kaggle/input/japanese-property-urls/tokyo_property_urls.csv")

print(df.shape)

df.drop_duplicates(inplace=True)

df.shape
urls = df.URL.tolist()

len(urls)
real_estate_df = pd.DataFrame(columns=["URL", "Is_Prop_Avl", "Unit Number", "Price(¥)", "Building Name", "Floor", "Available From", "Type", "Size(m²)", "Gross Yield(%)",

                                      "Land Rights", "Maintenance Fee(¥/mnt)", "Location", "Occupancy", "Nearest Station", "Way to Nearest Station", "Distance From Station(min)",

                                      "Layout", "Year Built", "Direction Facing", "Transaction Type", "Balcony Size(m²)", "Building Description", "Other Expenses",

                                      "Parking Available", "Parking Fee(¥/mnt)", "Date Updated", "Next Update Schedule"])

for url in urls:

    res = get_data(url)

    real_estate_df = real_estate_df.append(res, ignore_index=True)



real_estate_df.to_csv("real_estate.csv")

real_estate_df.head(10)
df = real_estate_df[real_estate_df.Is_Prop_Avl == "Yes"]

df.shape
df.head()
df.describe()
df["Price(¥)"].hist()
df["Size(m²)"].hist(bins=10)