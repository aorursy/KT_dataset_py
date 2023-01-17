# install dependencies

#! pip3 install beautifulsoup4, plotly, pandas



import plotly.express as px

import plotly.graph_objects as go

import time



from datetime import datetime, timedelta



import pandas as pd

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)
import requests, json

from bs4 import BeautifulSoup



def fetchBestSnapshot(url, timestamp):

    waybackmachine_url = 'http://archive.org/wayback/available?url=%s&timestamp=%s' % (url, timestamp)

    print(waybackmachine_url)



    response = requests.get(waybackmachine_url)

    if response.status_code != 200:

        raise Exception("could not get availability from waybackmachine", response.text)



    data = json.loads(response.text)

    

    # extract timestamp and snapshot_url from WBM's json response

    closest = data['archived_snapshots']['closest']



    if closest['available'] != True or closest['status'] != '200':

        raise Exception("closest snapshot not available", closest)



    timestamp_str = closest['timestamp']

    timestamp = datetime.strptime(timestamp_str, '%Y%m%d%H%M%S')

    snapshot_url = closest['url']

    print("found snapshot_url=%s at timestamp=%s" % (snapshot_url, timestamp))



    # fetch the historical snapshot of the target url

    response = requests.get(snapshot_url)

    if response.status_code != 200:

        raise Exception("could not fetch snapshot", response.text)



    return response.content



def fetchExactSnapshot(url, timestamp):

    archive_snapshot_url = 'http://web.archive.org/web/%s/%s' % (timestamp, url)

    print(archive_snapshot_url)



    response = requests.get(archive_snapshot_url)

    if response.status_code != 200:

        raise Exception("could not fetch snapshot", response.text)



    return response.content



# handles two different designs

def extractDataFrameFeb27(page_snapshot):

    soup = BeautifulSoup(page_snapshot, 'html.parser')



    table = soup.find('table', attrs={'id':'case_count_table'})

    table_rows = table.find_all('tr')



    res = []

    for tr in table_rows:

        td = tr.find_all('td', limit=4)

        row = [tr.text.strip() for tr in td if tr.text.strip()]

        if row:

            res.append(row)

    

    if len(res) > 0 and res[0][0] == 'Positive Cases':

        res = [

            ['New York State (Outside of NYC)', res[0][1]],

            ['New York City', res[0][2]],

        ]

        return pd.DataFrame(res, columns=["Location", "Count"])

    

    res = []

    for tr in table_rows:

        td = tr.find_all('td', limit=2)

        row = [tr.text.strip() for tr in td if tr.text.strip()]

        if row:

            res.append(row)



    return pd.DataFrame(res, columns=["Location", "Count"])



# handle yet another redesign

def extractDataFrameMar17(page_snapshot):

    soup = BeautifulSoup(page_snapshot, 'html.parser')



    table = soup.select_one('body div.wysiwyg--field-webny-wysiwyg-body table')

    

    #table = soup.find('table', attrs={'class':'nothead'})

    table_rows = table.find_all('tr')



    res = []

    for tr in table_rows:

        td = tr.find_all('td', limit=2)

        row = [tr.text.strip() for tr in td if tr.text.strip()]

        if row:

            res.append(row)



    return pd.DataFrame(res, columns=["Location", "Count"])



def getArchiveIndex(url):

    archive_url = 'http://web.archive.org/cdx/search/cdx?url=%s&output=json' % (url)

    response = requests.get(archive_url)

    if response.status_code != 200:

        raise Exception("could not get availability from waybackmachine", response.text)



    archive_snapshots = json.loads(response.text)



    archive_df = pd.DataFrame(archive_snapshots[1:], columns=archive_snapshots[0])

    archive_df = archive_df[(archive_df['mimetype'] == 'text/html') & (archive_df['statuscode'] == '200')]

    archive_df = archive_df.drop_duplicates(subset='digest')

    archive_df['datetime'] = pd.to_datetime(archive_df['timestamp'])

    return archive_df



# translate waybackmachine's timestamp into a datetime object

def wbm_to_datetime(wbm_timestamp):

    return datetime.strptime(str(wbm_timestamp), "%Y%m%d%H%M%S")
# check if we have previously loaded data

try:

    previous = pd.read_csv('../input/covid-19-ny-data-scrapper/ny_confirmed_cases.csv')

    previous_timestamp = pd.to_datetime(previous['Timestamp'].max())

except:

    # build from scratch

    print("prior data not found, starting from zero")

    previous = pd.DataFrame()

    previous_timestamp = datetime(2020, 2, 27, 9) # first wide-format data



# the ny health website changed a few times, each time needing a slightly diff scrapping approach

targets = [

    # February 27th and March 8th redesigns

    [ datetime(2020,2,27,9), datetime(2020,3,17,19), 'www.health.ny.gov/diseases/communicable/coronavirus/', extractDataFrameFeb27 ],

    # March 17 redesign and new location

    [ datetime(2020,3,17,19), datetime(2021,1,1,1), 'coronavirus.health.ny.gov/county-county-breakdown-positive-cases', extractDataFrameMar17 ]

]



now = datetime.now()



# iterate through all the versions of the coronavirus cases website and incrementally update

# when new versions become available via the WayBackMachine (thank you archive.org)

# even if this script is only run once a week, it will backfill any gaps

for start_time, end_time, url, extractor in targets:

    archive_df = getArchiveIndex(url)

    # resume from where we last left off

    archive_df = archive_df[archive_df['datetime'] > max(start_time, previous_timestamp)]



    print("examining archive snapshots: %s" % url)

    print("  there are %d additional snapshots to fetch and process" % len(archive_df))



    for index, archive_snapshot in archive_df.iterrows():

        url = archive_snapshot['original']

        timestamp = archive_snapshot['timestamp']

        snapshot_html = fetchExactSnapshot(url, timestamp)

        snapshot_df = extractor(snapshot_html)

        snapshot_df['Timestamp'] = wbm_to_datetime(timestamp) # use snapshot's timestamp as the data timestamp

        previous = previous.append(snapshot_df, ignore_index=True)

        time.sleep(1) # be nice to archive.org

        if wbm_to_datetime(timestamp) >= end_time:

            break



current = previous.drop_duplicates(subset=['Location', 'Timestamp'])
# normalize and cleanup naming changes during March

current['Location'] = current['Location'].str.strip(':')

current['Location'] = current['Location'].str.replace(' County', '')



# normalize early March to late March naming conventions for New York State total count

current['Location'] = current['Location'].str.replace('Total Positive Cases .*', 'Total Number of Positive Cases')



# drop a metric no longer reported (as of March 19th)

current = current[~(current['Location'] == 'New York State (Outside of NYC)')]



current = current.sort_values(by=['Location','Timestamp', 'Count'])
# persist data with UTC timestamps based on when data was seen on the web

current.to_csv('ny_confirmed_cases.csv', index=False)
# manual smoke test: convert timestamps into NY time to validate metrics

df = current.copy()

df['Timestamp'] = pd.to_datetime(df['Timestamp']).dt.tz_localize('UTC').dt.tz_convert('US/Eastern')

df = df.sort_values(by=['Location','Timestamp', 'Count'])



# keep only the last value for each day

df['Date'] = df['Timestamp'].dt.strftime("%Y%m%d")

df = df.drop_duplicates(subset=['Location', 'Date'], keep='last')



# Print NYC confirmed case counts per day

df[df['Location'] == 'New York City']