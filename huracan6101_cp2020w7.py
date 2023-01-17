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
# use "python read url file data" search"
# get "https://stackoverflow.com/questions/1393324/given-a-url-to-a-text-file-what-is-the-simplest-way-to-read-the-contents-of-the"
# since we use python 3 therefore try to use the following script to get cp1a list
import urllib.request  # the lib that handles the url stuff
target_url = "https://nfulist.herokuapp.com/?semester=1091&courseno=0762"
cp1a = []
for line in urllib.request.urlopen(target_url):
    cp1a.append(line.decode('utf-8'))
    #print(line.decode('utf-8'), end = "") #utf-8 or iso8859-1 or whatever the page encoding scheme is
print(cp1a)
# need to chop \n for each line

# use "python chop new line" search 
# get "https://stackoverflow.com/questions/275018/how-can-i-remove-a-trailing-newline"
# found .rstrip() maybe work
import urllib.request  # the lib that handles the url stuff
target_url = "https://nfulist.herokuapp.com/?semester=1091&courseno=0762"
cp1a = []
for line in urllib.request.urlopen(target_url):
    cp1a.append(line.decode('utf-8').rstrip())
    #print(line.decode('utf-8'), end = "") #utf-8 or iso8859-1 or whatever the page encoding scheme is
print(cp1a)
# yes, we got the needed list.
# now we can get the cp1b by using the same method
# https://nfulist.herokuapp.com/?semester=1091&courseno=0776
import urllib.request  # the lib that handles the url stuff
target_url = "https://nfulist.herokuapp.com/?semester=1091&courseno=0776"
cp1a = []
for line in urllib.request.urlopen(target_url):
    # we can use int() to convert string into integer
    cp1a.append(int(line.decode('utf-8').rstrip()))
    #print(line.decode('utf-8'), end = "") #utf-8 or iso8859-1 or whatever the page encoding scheme is
print(cp1a)
''' now can we read [[40823148, 40923203, 40923208, 40923209, 40923210, 40923223, 40923225, 40923230, 40923238, 40923239, 40923244, 40923249], [40523148, 40923201, 40923202, 40923218, 40923219, 40923228, 40923231, 40923232, 40923240, 40923247, 40923248, 40923250], [40823152, 40923205, 40923212, 40923214, 40923217, 40923226, 40923236, 40923241, 40923242, 40923246, 40923251], [40723217, 40728238, 40923206, 40923216, 40923220, 40923227, 40923233, 40923237, 40923243, 40923252, 40923253], [40523138, 40923204, 40923207, 40923211, 40923213, 40923221, 40923224, 40923229, 40923234, 40923235, 40923245]] 
into one dimensional list
'''
cp1bGroup = [[40823148, 40923203, 40923208, 40923209, 40923210, 40923223, 40923225, 40923230, 40923238, 40923239, 40923244, 40923249], [40523148, 40923201, 40923202, 40923218, 40923219, 40923228, 40923231, 40923232, 40923240, 40923247, 40923248, 40923250], [40823152, 40923205, 40923212, 40923214, 40923217, 40923226, 40923236, 40923241, 40923242, 40923246, 40923251], [40723217, 40728238, 40923206, 40923216, 40923220, 40923227, 40923233, 40923237, 40923243, 40923252, 40923253], [40523138, 40923204, 40923207, 40923211, 40923213, 40923221, 40923224, 40923229, 40923234, 40923235, 40923245]] 
# len() can be used to get the length of a list
#print(len(cp1bGroup))
# so we can use the for loop to read group member out
groupNum = len(cp1bGroup)
cp1b = []
for i in range(groupNum):
    # use len() to get student number for each group
    studNum = len(cp1bGroup[i])
    #print(cp1bGroup[i])
    for j in range(studNum):
        cp1b.append(cp1bGroup[i][j])
print(cp1b)
# yes, we transfer two dimensional list into one diimension
# for the next step we may need to compare two lists to find the discrepancy
# since we don't have cp1a grouping list, we need to get it from http://mde.tw/cp2020/content/W3.html
import requests
from bs4 import BeautifulSoup

url = "http://mde.tw/cp2020/content/W3.html"
response = requests.get(url) # request object from url
html_doc = response.text # get html from request object
# can we use split to seperate file content
#print(html_doc.split("=============================="))
# use "1b" to cut the page
cut1a = html_doc.split("1b")
# get first part of cut1a
# use "==============================" to cut the 1a html content
cutPage = cut1a[0].split("==============================")
#print(len(cutPage))
#soup = BeautifulSoup(response.text, "lxml") # use lxml to parse html
# we only test the first element of the html page
cp1a = []
for group in range(len(cutPage)):
    soup = BeautifulSoup(cutPage[group], "lxml")

    '''
    print(soup.title) # get title
    print("---")
    print(soup.title.name) # get title name
    print("---")
    print(soup.title.string) # get title string
    print("---")
    print(soup.title.parent.name) # get title parent name
    print("---")
    print(soup.a) # get first anchor tag
    print("---")
    '''

    # get all tags
    #print(soup.find_all('a'))
    allA = soup.find_all('a')
    # search with "bs4 get all tag with certain text" and get "https://stackoverflow.com/questions/866000/using-beautifulsoup-to-find-a-html-tag-that-contains-certain-text"
    index = 0
    cp1aElement = []
    for i in allA:
        #print(i)
        if "github" in str(i):
            #print(i)
            #print(allA[index].contents[0])
            stud = allA[index].contents[0]
            if stud not in cp1aElement:
                cp1aElement.append(stud)
        index = index + 1
    #print("1a group " + str(group + 1) + ":")
    #print(cp1aElement)
    cp1a.append(cp1aElement)
print(cp1a)
for i in range(len(cp1a)):
    print("group " + str(i+1) + ":", cp1a[i])
# 1b student_no and github_account can be retrieved from "http://mde.tw/cp2020/downloads/hw2/cpb_github_account.txt"
import urllib.request  # the lib that handles the url stuff
target_url = "http://mde.tw/cp2020/downloads/hw2/cpb_github_account.txt"
cp1b = []
for line in urllib.request.urlopen(target_url):
    cp1bTemp = line.decode('utf-8').rstrip()
    #print(line.decode('utf-8'), end = "") #utf-8 or iso8859-1 or whatever the page encoding scheme is
    cp1b.append(cp1bTemp.split('\t'))
#print(cp1b)
# drop the first element of cp1b and convert into dictionary
cp1bAccount = dict(cp1b[1:])
# check into the cp1bAccount dict for "40923208"
print(cp1bAccount["40923208"])