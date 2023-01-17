# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
"""

Target data set:

[

    {

        name : String,

        id : String,

        country : String,

        followers : Number,

        biography: String,

        aliases : [String],

        events: [

            {

                id : String,

                participants : [String],

                date : [String],

                region: [String],

                country: [String]

            

            }

        ],

        tracks : {

            total_tracks : Number,

            track_ids: [String]

        }

    },

    ...

]

"""
#imports

from bs4 import BeautifulSoup as bs

import os

from requests import get

import re

import json

import locale



locale.setlocale(locale.LC_ALL, 'en_US.UTF-8') 
#constants

html_data_dir = "./data/html/"

json_data_dir = "./data/json/"

base_url = "https://www.residentadvisor.net/"

base_dj_url = base_url + "/dj/"
#helpers

def read_file_or_write_request(file_name, url):

    if os.path.exists(file_name):

        fd = open(file_name, "r")

        contents = fd.read()

        fd.close()

        return contents

    else:

        resp = get(url)

        if (resp.ok):

            fd = open(file_name, "w+")

            fd.write(resp.text)

            fd.close()

        else :

            raise ("Could not get " + url)

        return resp.text



def read_data_or_parse_html(file_name, data_fn):

    if os.path.exists(file_name):

        fd = open(file_name, "r", encoding="utf8")

        data = fd.read()

        fd.close()

        return json.loads(data)

    else:

        data = data_fn()

        fd = open(file_name, "w+", encoding="utf8")

        json.dump(data, fd, ensure_ascii=False)

        return data
"""

Get list of top 1000 DJs

"""

#Target fields : name, id

def get_top_1000_data():

    dj_links = get_dj_links()

    return list(map(

            lambda a: ({ "name" : a.get_text(), "id" : a["href"][4:] }),

            dj_links

        ))



def get_top_1000_html():

    top_1000_list_url = base_url + "dj.aspx"

    top_1000_list_file = html_data_dir + "top_1000.html"

    return read_file_or_write_request(

            top_1000_list_file,

            top_1000_list_url

        )



# Looking for ...

# - hyperlinks with an href that begins with "/dj/"

# - has text

# - has no children

# - isn't erroneous "Following" link

def unique_dj_names(item):

    return (

        item.name == "a"

        and item.get('href')

        and re.compile("^\/dj\/").search(item["href"])

        and len(item.text) > 0

        and len(item.contents) == 1

        and item.text != 'Following'

    )



def get_dj_links():

    dj_list_html = get_top_1000_html()

    dj_list_tree = bs(dj_list_html, "html.parser")

    dj_links = dj_list_tree.find_all(unique_dj_names)

    return dj_links



def get_top_1000():

    dj_list_loc = json_data_dir + "top_1000.json"

    return read_data_or_parse_html(

        dj_list_loc,

        get_top_1000_data

    )
""""

Get DJ profile data

"""

#Target fields : aliases, country, followers

def get_dj_profile_data(html):

    profile = {}

    tree = bs(html, "html.parser")



    #aliases

    alias_elems = tree.find_all("div", text="Aliases /")

    profile["aliases"] = [] if len(alias_elems) == 0 else alias_elems[0].parent.text.split(", ")

    if (len(profile["aliases"])):

        profile["aliases"][0] = profile["aliases"][0].replace("Aliases /", "")

    

    #country

    country_elem = tree.find("span", itemprop="country")

    profile["country"] = country_elem.text if country_elem else ""



    #followers

    followers_elem = tree.find("h1", id="MembersFavouriteCount")

    profile["followers"] = locale.atoi(followers_elem.text)

    return profile



def get_dj_profile_html(dj):

    dj_html_loc =  html_data_dir + dj["id"] + ".html"

    dj_url = base_dj_url + dj["id"]

    dj_html = read_file_or_write_request(

        dj_html_loc,

        dj_url

    )

    return dj_html



def get_dj_profile(dj):

    dj_profile_loc = json_data_dir + dj["id"] + ".json"

    data_fn = lambda: get_dj_profile_data(get_dj_profile_html(dj))

    return read_data_or_parse_html(

        dj_profile_loc,

        data_fn

    )
"""

Get DJ biography

"""



#Target fields : biography

def get_dj_bio_data(html):

    tree = bs(html, "html.parser")

    bio_article = tree.find("article")

    bio_divs = bio_article.find_all("div")

    bio_cont = bio_divs[0].contents[0]

    if (len(bio_cont.contents)):

        bio_header = bio_cont.contents[0].text

        bio_body = bio_cont.contents[2].text

        return bio_header + bio_body

    else:

        return ""



def get_dj_bio_html(dj):

    dj_bio_html_loc = html_data_dir + dj["id"] + ".bio.html"

    dj_bio_url = base_dj_url + dj["id"] + "/biography"

    dj_bio_html = read_file_or_write_request(

        dj_bio_html_loc,

        dj_bio_url

    )

    return dj_bio_html



def get_dj_biography(dj):

    dj_bio_loc = json_data_dir + dj["id"] + ".bio.json"

    data_fn = lambda: get_dj_bio_data(get_dj_bio_html(dj))

    return read_data_or_parse_html(

        dj_bio_loc,

        data_fn

    )
"""

Get DJ events

"""

#Target fields :

#events : [id, time, region, country]

def get_dj_events_data(html):

    events = []

    tree = bs(html, "html.parser")

    year_links = tree.find_all("a", href=re.compile("dates\?yr=[1-9][0-9]{3}$"))

    for year_link in year_links:

        resp = get(base_url + year_link["href"])

        if (resp.ok):

            year_tree = bs(resp.text, "html.parser")

            event_elems = year_tree.find_all("article", class_="event")

            for event_elem in event_elems:

                event = {}

                event["id"] = event_elem.find("a", href=re.compile("^/events/"))["href"].replace("/events/", "")

                event["time"] = event_elem.find("time").text

                region_elem =  event_elem.find("span", itemprop="region")

                event["region"] = region_elem.text if region_elem else ""

                country_elem = event_elem.find("span", itemprop="country-name")

                event["country"] = country_elem.text if country_elem else ""

                events.append(event)

    return (events)



def get_dj_events_html(dj):

    dj_events_html_loc = html_data_dir + dj["id"] + ".events.html"

    dj_events_url = base_dj_url + dj["id"] + "/dates"

    dj_events_html = read_file_or_write_request(

        dj_events_html_loc,

        dj_events_url

    )

    return dj_events_html



def get_dj_events(dj):

    dj_events_loc = json_data_dir + dj["id"] + ".events.json"

    data_fn = lambda: get_dj_events_data(get_dj_events_html(dj))

    return read_data_or_parse_html(

        dj_events_loc,

        data_fn

    )

"""

Get DJ tracks

"""

def get_dj_tracks_data(html):

    tracks = {}

    tracks["total"] = None

    tracks["ids"] = []

    tree = bs(html, "html.parser")

    track_list = tree.find("ul", id="tracks")

    if not track_list:

        return tracks

    track_elems = track_list.find_all("li")

    tracks["total"] = len(track_elems)

    for track_elem in track_elems:

        track_link = track_elem.find("a", href=re.compile("^\/tracks\/"))

        if (track_link):

            track_id = track_link["href"].replace("/tracks/", "")

            tracks["ids"].append(track_id)

    return (tracks)



def get_dj_tracks_html(dj):

    dj_tracks_html_loc = html_data_dir + dj["id"] + ".tracks.html"

    dj_tracks_url = base_dj_url + dj["id"] + "/tracks"

    dj_tracks_html = read_file_or_write_request(

        dj_tracks_html_loc,

        dj_tracks_url

    )

    return dj_tracks_html



def get_dj_tracks(dj):

    dj_tracks_loc = json_data_dir + dj["id"] + ".tracks.json"

    data_fn = lambda: get_dj_tracks_data(get_dj_tracks_html(dj))

    return read_data_or_parse_html(

        dj_tracks_loc,

        data_fn

    )
def get_all_dj_data():

    dj_list = get_top_1000()

    dj_data = []

    for dj in dj_list:

        dj_full_loc = json_data_dir + dj["id"] + ".full.json"

        print(dj["name"])

        if os.path.exists(dj_full_loc):

            fd = open(dj_full_loc, "r")

            dj_data_text = fd.read()

            dj_obj = json.loads(dj_data_text)

            dj_data.append(dj_obj)

        else:

            profile = get_dj_profile(dj)

            bio = get_dj_biography(dj)

            events = get_dj_events(dj)

            tracks = get_dj_tracks(dj)

            dj = {**profile, **dj}

            dj["bio"] = bio

            dj["events"] = events

            dj["tracks"] = tracks

            fd = open(dj_full_loc, "w")

            json.dump(dj, fd, ensure_ascii=False)

            dj_data.append(dj)

    return dj_data