for i in range(5):

    print("i = "+ str(i))
import urllib.request  # the lib that handles the url stuff

target_url = "https://nfulist.herokuapp.com/?semester=1091&courseno=0762"

cp1a = []

for line in urllib.request.urlopen(target_url):

    cp1a.append(line.decode('utf-8').rstrip())

    #print(line.decode('utf-8'), end = "") #utf-8 or iso8859-1 or whatever the page encoding scheme is

print(cp1a)

'''

<a href="https://github.com/40623219/cp2020">40623219</a>



'''
a = [1, 2, 3, 4]

a_len = len(a)

for i in range(a_len):

    print(a[i])

print(a[1:])
import urllib.request

target_url = "https://nfulist.herokuapp.com/?semester=1091&courseno=0762"

for line in urllib.request.urlopen(target_url):

    print(line.decode('utf-8'))
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

cp1aGroup = []

for group in range(len(cutPage)):

    soup = BeautifulSoup(cutPage[group], "lxml")

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

    cp1aGroup.append(cp1aElement)

print(cp1aGroup)

'''

for i in range(len(cp1aGroup)):

    print("group " + str(i+1) + ":", cp1aGroup[i])

'''
import urllib.request  # the lib that handles the url stuff

target_url = "http://mde.tw/cp2020/downloads/hw2/cpa_github_account.txt"

cp1a = []

for line in urllib.request.urlopen(target_url):

    cp1aTemp = line.decode('utf-8').rstrip()

    #print(line.decode('utf-8'), end = "") #utf-8 or iso8859-1 or whatever the page encoding scheme is

    cp1a.append(cp1aTemp.split('\t'))

#print(cp1b)

# drop the first element of cp1b and convert into dictionary

cp1aAccount = dict(cp1a[1:])

# check into the cp1bAccount dict for "40923208"

print(cp1aAccount["40923137"])
import urllib.request

# check if the student is still registered in the course

# 1a student list from registar server

target_url = "https://nfulist.herokuapp.com?semester=1091&courseno=0762"

# 1a registered student list

cp1aReg = []

for line in urllib.request.urlopen(target_url):

    cp1aReg.append(line.decode('utf-8').rstrip())

# cp1aReg element is string

#print(cp1aReg)

# cp1aGroup element is also string

#print(cp1aGroup)

# generate https://github.com/ + account + cad2020 and https:// + account + .github.io/cad2020

# 假如根據先前分組資料,不進行目前是否選課人員檢查, 則無法剔除已經退選的學員

# read in cp1aGroup

# 利用 dropStud 收集已經退選名單

dropStud = []

for gpNum in range(len(cp1aGroup)):

    # cp1aGroup[gpNum] is the member list of group number (gpNum + 1) 

    if gpNum != 0:

        print("<br />"*2)

        print("==============================")

        print("<br />"*2)

    print("group " + str(gpNum + 1) + ":" + "<br />"*2)

    for i in range(len(cp1aGroup[gpNum])):

        # check if cp1aGroup[gpNum][i] still in cp1aReg

        if cp1aGroup[gpNum][i] in cp1aReg:

        #try:

            memberNum = cp1aGroup[gpNum][i]

            # from number to check account

            memberAccount = cp1aAccount[str(memberNum)]

            #print(memberAccount)

            courseTitle = "cp2020"

            print("Repository: <a href='https://github.com/" + str(memberAccount) + "/" + courseTitle + "'>" + str(memberNum) + "</a> | ", end="")

            print("Site: <a href='https://" + str(memberAccount) + ".github.io/" + courseTitle + "'>" + str(memberNum) + "</a><br />")

        else:

        #except:

            #print(cp1aGroup[gpNum][i] + "已經不在修課名單中")

            dropStud.append(cp1aGroup[gpNum][i])

# 列出已經退選名單

for i in range(len(dropStud)):

    print(dropStud[i] + "已經退選")