import numpy as np

import pandas as pd

from bs4 import BeautifulSoup

import requests

import matplotlib.pyplot as plt

import matplotlib.image as mpimg



from matplotlib.pyplot import figure



img=mpimg.imread('/kaggle/input/skyrim/skyrimtable.bmp')

figure(num=None, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')



imgplot = plt.imshow(img)
page = requests.get("https://elderscrolls.fandom.com/wiki/Races_(Skyrim)").text

soup = BeautifulSoup(page, 'html.parser')
img=mpimg.imread('/kaggle/input/skyrim/skyrimrightclick.bmp')

figure(num=None, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')



imgplot = plt.imshow(img)
img=mpimg.imread('/kaggle/input/skyrim/skyrimhtmltable.bmp')

figure(num=None, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')



imgplot = plt.imshow(img)
skills_table = soup.find_all('table')[0]
skills_table.find_all('tr')[0:3]
races_row = skills_table.find_all('tr')[0]

races_row
races_row.text
races = races_row.text.split('\n')

races
col = races[1][0:4]

col
races = races[2:-1]

races
dict2 = {}

dict2[col] = races

dict2
for i in range(1,19):

    skill = skills_table.find_all('tr')[i].text.split('\n')

    col = skill[1]

    skill = skill[2:-1]

    dict2[col] = skill
skyrimstats = pd.DataFrame(dict2)

skyrimstats
skyrimstats.to_csv("skyrimstats.csv")