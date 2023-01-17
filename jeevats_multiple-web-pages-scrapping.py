#importing required packages
import requests
import urllib.request
from bs4 import BeautifulSoup
from html.parser import HTMLParser
import pandas as pd
#web scrapping
e=[]
t=[]
for i in range(1,1000):
    try:
        r=requests.get("https://ilearntamil.com/tamil-to-english-dictionary/?letter&cpage="+str(i))
        soup = BeautifulSoup(r.text, "lxml")
        review=review=soup.select("td")
        print(len(review))
        if len(review)==0:
            print(len(e))
            break
        j=2
        while(j<len(review)):
            t.append(review[j])
            e.append(review[j+1])
            j=j+2
    except:
        print(len(e))
        print("Ended page for read")
        break
#Tamil Dictionary Scrapping
import pandas as pd
df=pd.DataFrame()
df["English"]=e
df["Tamil"]=t
df.head()
