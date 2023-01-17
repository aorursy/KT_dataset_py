###### INPUT A SEARCH TERM For CASES IN SP UNKNOWNS: https://digital.pathology.johnshopkins.edu/repos/451 #########

#txt = input("Search Term: ")

txt="adenoma"



import json

import re

import pandas as pd

import random

import urllib3

import requests

from bs4 import BeautifulSoup

from googleapiclient.discovery import build



filename = 'thisfile.csv'



commands = {}

with open(filename) as fh:

    for line in fh:

        #print(line)

        if bool(re.search(":", line)):

            command,desc= line.rsplit(':',1);

            command=command.replace(" ","")

            desc=desc.replace(",\n","")

            commands[command] = desc



def flatten(d,sep="_"):

    import collections



    obj = collections.OrderedDict()



    def recurse(t,parent_key=""):

        

        if isinstance(t,list):

            for i in range(len(t)):

                recurse(t[i],parent_key + sep + str(i) if parent_key else str(i))

        elif isinstance(t,dict):

            for k,v in t.items():

                recurse(v,parent_key + sep + k if parent_key else k)

        else:

            obj[parent_key] = t



    recurse(d)



    return obj







ran=bool(random.getrandbits(1))



if ran:

    my_api_key = "AIzaSyDSWjeCe5YDjpMyUo1ASvPs_YCiezla55U"

    my_cse_id = "012186595233664989134:ueeop7vueol"

else:

    my_api_key="AIzaSyB9nMMH7YuvDwhHVi4jSsPg6Q5RW7tSnTY"

    my_cse_id="012186595233664989134:duay3pheitw"





def google_search(search_term, api_key, cse_id, **kwargs):

    service = build("customsearch", "v1", developerKey=api_key)

    res = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()

    try:

        return res['items']

    except:

        print("No results found")

        



results = google_search(

    txt, my_api_key, my_cse_id,start=1)



results2 = google_search(

    txt, my_api_key, my_cse_id,start=11)





try:

    rr=flatten(results+results2)

    r2=[i for i in rr.values()]

    r3=[re.search("Week.+Case [0-9]+",i) for i in r2]

    r4=[j for j in r3 if j is not None]

    r5=[j[0].replace(" ","") for j in r4]



    url="http://apps.pathology.jhu.edu/sp/all-cases/"

    #page = urllib3.urllib3(page)

    response = requests.get( url)

    soup = BeautifulSoup(response.content)

    pd.set_option('display.max_colwidth', -1)



    ss=soup.find_all('a')

    ss3=[j["href"] for j in ss]

    ss2=[j.text.replace(" ","") for j in ss]

    ss4=pd.DataFrame(list(zip(ss2,ss3)))

    ss4.columns=["week","url"]

    csv=pd.DataFrame.from_dict(list(commands.items()))

    csv.columns=["week","diagnosis"]

    ss5=pd.merge(csv,ss4,on="week")

    ss6=ss5.drop_duplicates()

    seares=pd.DataFrame(r5,columns=["week"]).drop_duplicates()

    sear=pd.merge(seares,ss6,on="week")

    sear



#df = pd.DataFrame(['http://google.com', 'http://duckduckgo.com'])



    def make_clickable(val):

        return '<a href="{}">{}</a>'.format(val,val)



    pp=pd.DataFrame(sear).style.format(make_clickable,subset="url")



except:

        print("")

        print("")

        print("")

        print("")

        

pp
