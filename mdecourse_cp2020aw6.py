# 導入 urllib.request: https://docs.python.org/3/library/urllib.request.html

# 透過 urllib.request 模組中的 urlopen() 開啟網路 url 連結資料

import urllib.request  # the lib that handles the url stuff

# 定義一個輸入學期與課號,就能夠輸出各課程修課人員數列的函式



def getRegList(semester, courseno):

    # 因為課號可能以 0 開頭, 因此採字串型別輸入, 為了一致, 輸入變數一律採字串輸入

    target_url = "https://nfulist.herokuapp.com/?semester=" + semester + "&courseno=" + courseno

    regList = []

    for line in urllib.request.urlopen(target_url):

        # 由於 urlopen() 取下的網際資料為 binary 格式, 可以透過 decode() 解碼為 ASCII 資料

        regList.append(line.decode('utf-8').rstrip())

    # 此一函式利用 return 將資料傳回

    return regList

'''

各班在 1091 課號

cp

1a 1091/0762

1b 1091/0776

cad

2a 1091/0788

2b 1091/0801

'''

print("一甲選課學號名單:",getRegList("1091", "0762"))

print("一乙選課學號名單:",getRegList("1091", "0776"))

print("二甲選課學號名單:",getRegList("1091", "0788"))

print("二乙選課學號名單:",getRegList("1091", "0801"))

# 以下透過 url 取得網路資料, 並利用 bs4 解讀網頁內容.

# 導入 requests 模組: https://requests.readthedocs.io/en/master/

import requests

from bs4 import BeautifulSoup

import urllib.request



url = "http://mde.tw/cp2020/content/W3.html"

# 利用 requests 模組中的 get() 取下資料

response = requests.get(url) # request object from url

html_doc = response.text # get html from request object

# 上面兩行取下的資料與 urllib.request.urlopen().decode() 所取得的資料有何不同?或者相同

# 設定 urlData 為一個空字串, 後續將逐行從網際檔案逐行讀出的資料串接成 html 檔案

urlData = ""

for line in urllib.request.urlopen(url):

    # 由於 urlopen() 取下的網際資料為 binary 格式, 可以透過 decode() 解碼為 ASCII 資料

    # 之前為了將每一列的資料放入數列, 因此將每一行最後面的 \n 跳行符號去除

    #urlData += line.decode('utf-8').rstrip()

    urlData += line.decode('utf-8')

#print(html_doc)

#print(urlData)

'''

if html_doc == urlData:

    print("兩個檔案完全相同")

else:

    print("兩個檔案不相同")

'''

# 之後可以透過 html_doc 或 urlData 取得特定 url 的網際資料

# W3 網際頁面可以利用 1b 字串進行分割

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

    # allA 為 cutPage 逐組帶有分組學員學號的數列

    allA = soup.find_all('a')

    # search with "bs4 get all tag with certain text" and get "https://stackoverflow.com/questions/866000/using-beautifulsoup-to-find-a-html-tag-that-contains-certain-text"

    index = 0

    cp1aElement = []

    for i in allA:

        #print(i)

        # 只取下有 "github" 字串的資料

        if "github" in str(i):

            #print(i)

            #print(allA[index].contents[0])

            stud = allA[index].contents[0]

            # 由於頁面中的 anchor 有倉儲連結與網頁連結, 以下不重複選

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

# 利用 ethercalc 取得使用者 github 帳號, 理論上可以建立網際表單, 讓使用者以從表單輸入帳號資料

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
# 特別注意: 單行註解的 # 不可使用中文全形字元

# 直接利用前面已經定義的 getRegList() 函式取得 cp1a 選課學員名單

# 以下處理 cp1a W3 頁面超文件資料

cp1aReg = getRegList("1091", "0762")

# cp1aGroup 在前面已經定義完成, 代表第三週所完成的分組資料

#print("註冊名單: ", cp1aReg)

#print("分組名單: ", cp1aGroup)

# 以下還需要取得 cp1aAccount

# classTitle = 'cp2020'

# 以下設法根據註冊名單與分組名單建立倉儲與網頁超文件.

# 倉儲: https://github.com/ + account + / + classTitle 

# 網頁: https:// + account + .github.io/ + classTitle

# 必須進行學員是否選課檢查, 否則無法剔除已經退選的學員

# 利用 dropStud 收集已經退選名單

dropStud = []

for gpNum in range(len(cp1aGroup)):

    # cp1aGroup[gpNum] is the member list of group number (gpNum + 1) 

    # 從第一組與第二組中間才加入組別分隔符號

    if gpNum != 0:

        print("<br />"*2)

        print("==============================")

        print("<br />"*2)

    print("group " + str(gpNum + 1) + ":" + "<br />"*2)

    # 接下來逐組列出組員, 先判定是否在選課名單中, 再利用學號與 github 帳號拼出所需要的 html

    for i in range(len(cp1aGroup[gpNum])):

        # check if cp1aGroup[gpNum][i] still in cp1aReg

        if cp1aGroup[gpNum][i] in cp1aReg:

        # 因為若學員不在選課名單中, 則以選課名單產生的 dict 資料中無法取得對應資料

        # 因此除了判定學號是否在選課名單中之外, 也可以用 try: except: 進行處理

        #try:

            # 從分組數列中按照順序取出各學員學號

            memberNum = cp1aGroup[gpNum][i]

            # 根據學號至 github 帳號 dict 中查出對應的 github 帳號

            memberAccount = cp1aAccount[str(memberNum)]

            #print(memberAccount)

            # 1a 的課程代號

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