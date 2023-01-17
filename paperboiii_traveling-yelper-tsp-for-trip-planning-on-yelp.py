import os

import pandas as pd

#PLOTLY

import plotly

import plotly.plotly as py

import plotly.offline as offline

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

import cufflinks as cf

from plotly.graph_objs import Scatter, Figure, Layout

from concorde.tsp import TSPSolver

import time

import pylab as pl

import numpy as np

from matplotlib import collections  as mc

import colorlover as cl

from IPython.display import HTML

# imports from yelp starter

from __future__ import print_function

import argparse

import json

import pprint

import requests

import sys

import urllib

import os

try:

    # For Python 3.0 and later

    from urllib.error import HTTPError

    from urllib.parse import quote

    from urllib.parse import urlencode

except ImportError:

    # Fall back to Python 2's urllib2 and urllib

    from urllib2 import HTTPError

    from urllib import quote

    from urllib import urlencode

cf.set_config_file(offline=True)
with open('../input/nacho-stuff/nacho_credentials/nacho_credentials/yelp_cred.json') as fh:

    creds = json.loads(fh.read())

client = creds['client']

key = creds['key']



with open('../input/nacho-stuff/nacho_credentials/nacho_credentials/mapbox.json') as fh:

    mapcreds = json.loads(fh.read())



mapkey = mapcreds['key']





# Yelp Fusion no longer uses OAuth as of December 7, 2017.

# You no longer need to provide Client ID to fetch Data

# It now uses private keys to authenticate requests (API Key)

# You can find it on

# https://www.yelp.com/developers/v3/manage_app

API_KEY = key





# API constants, you shouldn't have to change these.

API_HOST = 'https://api.yelp.com'

SEARCH_PATH = '/v3/businesses/search'

BUSINESS_PATH = '/v3/businesses/'  # Business ID will come after slash.





# Defaults for our simple example.

DEFAULT_TERM = 'dinner'

DEFAULT_LOCATION = 'San Francisco, CA'

SEARCH_LIMIT = 3





def request(host, path, api_key, url_params=None):

    """Given your API_KEY, send a GET request to the API.

    Args:

        host (str): The domain host of the API.

        path (str): The path of the API after the domain.

        API_KEY (str): Your API Key.

        url_params (dict): An optional set of query parameters in the request.

    Returns:

        dict: The JSON response from the request.

    Raises:

        HTTPError: An error occurs from the HTTP request.

    """

    url_params = url_params or {}

    url = '{0}{1}'.format(host, quote(path.encode('utf8')))

    headers = {

        'Authorization': 'Bearer %s' % api_key,

    }



    print(u'Querying {0} ...'.format(url))



    response = requests.request('GET', url, headers=headers, params=url_params)



    return response.json()





def search(api_key, term, location):

    """Query the Search API by a search term and location.

    Args:

        term (str): The search term passed to the API.

        location (str): The search location passed to the API.

    Returns:

        dict: The JSON response from the request.

    """



    url_params = {

        'term': term.replace(' ', '+'),

        'location': location.replace(' ', '+'),

        'limit': SEARCH_LIMIT

    }

    return request(API_HOST, SEARCH_PATH, api_key, url_params=url_params)





def get_business(api_key, business_id):

    """Query the Business API by a business ID.

    Args:

        business_id (str): The ID of the business to query.

    Returns:

        dict: The JSON response from the request.

    """

    business_path = BUSINESS_PATH + business_id



    return request(API_HOST, business_path, api_key)





def query_api(term, location):

    """Queries the API by the input values from the user.

    Args:

        term (str): The search term to query.

        location (str): The location of the business to query.

    """

    response = search(API_KEY, term, location)



    businesses = response.get('businesses')



    if not businesses:

        print(u'No businesses for {0} in {1} found.'.format(term, location))

        return



    business_id = businesses[0]['id']



    print(u'{0} businesses found, querying business info '

          'for the top result "{1}" ...'.format(

              len(businesses), business_id))

    response = get_business(API_KEY, business_id)



    print(u'Result for business "{0}" found:'.format(business_id))

    pprint.pprint(response, indent=2)





def destination_ask():

    all_bus = []

    all_city = []

    try:

        while True:

            bus = raw_input(

                'Please enter the NAME of the business. Leave blank if done. \n')

            #city=raw_input('Please enter the name of the CITY the business is in. Leave blank if done. \n')

            if bus != "":

                all_bus.append(bus)

                # all_city.append(city)

                all_city.append('San Francisco')

                print('\n')

                continue



            elif bus == "":

                return zip(all_bus, all_city)

                break



            else:

                print('Goodbye')

                break



    except:

        print('Please fix errors and run script again.')





def plot_route(to_yelp):



    yelp_returns = []

    for call in to_yelp:

        yelp_returns.append(search(API_KEY, call[0], call[1]))



    business_names = []

    latitudes = []

    longitudes = []

    addresses = []

    ratings = []

    review_counts = []



    for dest in yelp_returns:

        business_names.append(dest['businesses'][0]['name'])

        latitudes.append(dest['businesses'][0]['coordinates']['latitude'])

        longitudes.append(dest['businesses'][0]['coordinates']['longitude'])

        addresses.append(dest['businesses'][0]

                         ['location']['display_address'][0])

        ratings.append(dest['businesses'][0]['rating'])

        review_counts.append(dest['businesses'][0]['review_count'])



    df = pd.DataFrame({

        'business_name': business_names,

        'latitude': latitudes,

        'longitude': longitudes,

        'addresses': addresses,

        'ratings': ratings,

        'review_counts': review_counts

    })



    # Instantiate solver

    solver = TSPSolver.from_data(

        df.latitude,

        df.longitude,

        norm="EUC_2D"

    )



    t = time.time()

    # solve() doesn't seem to respect time_bound for certain values?

    tour_data = solver.solve(time_bound=6000.0, verbose=True, random_seed=42)

    print('Solved in %s Seconds' % (time.time() - t))

    print('Found the most optimal path: %s' % (tour_data.found_tour))



    df['path'] = tour_data.tour

    df = df.sort_values(by=['path']).reset_index(drop=True)

    df['path'] = df['path'].astype(str)

    df['next_latitude'] = df['latitude'].shift(periods=-1)

    df['next_longitude'] = df['longitude'].shift(periods=-1)

    df['next_business_name'] = df['business_name'].shift(periods=-1)

    df['next_addresses'] = df['addresses'].shift(periods=-1)

    df['path_name'] = df['addresses']+' to '+df['next_addresses']

    df['path_b_name'] = df['business_name']+' to '+df['next_business_name']



    cpal = cl.scales['11']['div']['Spectral']

    #df['colors'] = cl.interp(cpal, len(df))

    df['colors'] = ['hsl(346.0, 100.0%, 32.0%)',

                    'hsl(10.3333333333, 81.6666666667%, 56.3333333333%)',

                    'hsl(34.0, 97.3333333333%, 71.0%)',

                    'hsl(60.0, 100.0%, 87.0%)',

                    'hsl(82.3333333333, 64.3333333333%, 66.6666666667%)',

                    'hsl(127.0, 50.0%, 48.6666666667%)',

                    'hsl(151.0, 100.0%, 20.0%)']



    def create_line_marker():

        line_marker = []

        # for line

        for place in np.arange(len(df['longitude'])-1):

            line_marker.append(go.Scattermapbox(

                #    lat=[list(pair) for pair in zip(df['longitude'].tolist()[:-1],df['next_longitude'].tolist()[:-1])],

                #    lon=[list(pair) for pair in zip(df['longitude'].tolist()[:-1],df['next_longitude'].tolist()[:-1])],

                lon=[list(pair) for pair in zip(df['longitude'].tolist()[

                    :-1], df['next_longitude'].tolist()[:-1])][place],

                lat=[list(pair) for pair in zip(df['latitude'].tolist()[

                    :-1], df['next_latitude'].tolist()[:-1])][place],

                mode='lines',

                name=df['path_b_name'].tolist()[:-1][place],

                opacity=0.5,

                line=dict(

                    width=2,

                    color=df['colors'].tolist()[:-1][place],

                ),

            ))

        # for markers

        line_marker.append(go.Scattermapbox(

            lat=df['latitude'],

            lon=df['longitude'],

            customdata=df['business_name'],

            mode='markers',

            name='Businesses',

            marker=dict(

                size=15,

                color=df['colors'],

                opacity=.9,

            ),

            text=df['path']+' '+df['business_name']+' ('+df['addresses']+')',

            hoverinfo='text'

        ))

        return line_marker

    line_marker = create_line_marker()



    data = line_marker

    layout = go.Layout(autosize=False,

                       mapbox=dict(accesstoken=mapkey,

                                   bearing=10,

                                   # bearing=180,

                                   pitch=60,

                                   zoom=10.5,

                                   center=dict(

                                       lat=37.7749,

                                       lon=-122.4194),

                                   style="mapbox://styles/paperboi/cjrzixa4b2irp1hmwj8zewpjo"                                   

                                   ),

                       width=900,

                       height=500, title="Best Path For Your Destinations")

    fig = dict(data=data, layout=layout)

    iplot(fig)
plot_route(

    zip(['bobs donut and pastry shop','dynamo donut','twisted donuts','the jelly donut','mr holmes bakehouse','donut farm','milkbomb ice cream'],['San Francisco']*7)

)