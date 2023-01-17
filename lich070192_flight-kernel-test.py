import datetime

import pandas as pd

from urllib.parse import urljoin

from time import time

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import requests



class Kiwicom(object):

    """

    Parent class for initialisation

    """

    _TIME_ZONES = 'gmt'

    API_HOST = {

        'search': 'https://api.skypicker.com/',

        'booking': 'https://booking-api.skypicker.com/api/v0.1/',

        'location': 'https://locations.skypicker.com/',

        'zooz_sandbox': 'https://sandbox.zooz.co/mobile/ZooZPaymentAPI/',

        'zooz': 'Need to add'

    }



    def __init__(self, time_zone='gmt', sandbox=True):

        if time_zone.lower() not in self._TIME_ZONES:

            raise ValueError(

                'Unknown time zone: {}, '

                'supported time zones are {}'.format(time_zone, self._TIME_ZONES)

            )

        self.time_zone = time_zone.lower()

        self.sandbox = sandbox



    def make_request(self, service_url, method='get', data=None,

                     json_data=None, request_args=None, headers=None, **params):



        request = getattr(requests, method.lower())

        try:

            response = request(service_url, params=params, data=data, json=json_data, headers=headers, **request_args)

        except TypeError as err:

            response = request(service_url, params=params, data=data, json=json_data, headers=headers)



        try:

            response.raise_for_status()

            return response

        except Exception as e:

            return self._error_handling(response, e)



    def _error_handling(self, response, error):

        if isinstance(error, requests.HTTPError):

            if response.status_code == 400:

                error = requests.HTTPError(

                    '%s: Request parameters were rejected by the server'

                    % error, response=response

                )

            elif response.status_code == 429:

                error = requests.HTTPError(

                    '%s: Too many requests in the last minute'

                    % error, response=response

                )

            raise error

        else:

            return response



    @staticmethod

    def _validate_date(dt):

        try:

            datetime.datetime.strptime(str(dt), '%Y-%m-%d')

            return True

        except ValueError:

            return False



    def _reformat_date(self, params):

        """

        Reformatting datetime.datetime and YYYY-mm-dd to dd/mm/YYYY

        :param params: takes dict with parameters

        :return: dict with reformatted date

        """

        for k in ('dateFrom', 'dateTo'):

            if k in params:

                if isinstance(params[k], datetime.date):

                    params[k] = datetime.date.strftime(params[k], "%d/%m/%Y")

                elif isinstance(params[k], datetime.datetime):

                    params[k] = datetime.date.strftime(params[k], "%d/%m/%Y")

                elif self._validate_date(params[k]):

                    params[k] = datetime.datetime.strptime(params[k], "%Y-%m-%d").strftime("%d/%m/%Y")

        return params





class Search(Kiwicom):

    """

    Search Class

    """

    def search_flights(self, headers=None, request_args=None, **params):

        """  

        :param headers: headers

        :param request_args: extra args to requests.get

        :param params: parameters for request

        :return: response

        """

        # params.update(self._make_request_params(params, req_params))

        self._reformat_date(params)



        service_url = urljoin(self.API_HOST['search'], 'flights')

        return self.make_request(service_url,

                                 headers=headers,

                                 request_args=request_args,

                                 **params)



    def search_flights_multi(self, json_data=None, data=None, headers=None, request_args=None, **params):

        """

        Sending post request

        :param json_data: takes post data dict

        :param data: takes json formatted data

        :param headers: headres

        :param request_args: extra args to requests.get

        :param params: parameters for request

        :return: response

        """

        if json_data:

            for item in json_data['requests']:

                json_data.update(self._reformat_date(item))

        if data:

            for item in data['requests']:

                data.update(self._reformat_date(item))



        service_url = urljoin(self.API_HOST['search'], 'flights_multi')

        return self.make_request(service_url,

                                 method='post',

                                 json_data=json_data,

                                 data=data,

                                 headers=headers,

                                 request_args=request_args,

                                 **params)
s = Search()

observed_date = pd.to_datetime(datetime.date.today())

flight_date_list = pd.date_range(start=observed_date, end='12/31/2020')

flight_from = 'city:LON'

flight_to_list = ['city:MIL', 'TRN', 'VIE']

observed_date
#output_df = pd.DataFrame(columns = ['observed_date', 'flight_date', 'flight_from', 'flight_to', 'flight_number', 'price'])

output_df = pd.read_csv('../input/flight-data/data.csv')

output_df
for flight_date in flight_date_list:

    for flight_to in flight_to_list:

        if (len(output_df[(output_df['flight_date'] == flight_date.strftime("%d/%m/%Y")) 

                          & (output_df['observed_date'] == observed_date.strftime("%d/%m/%Y")) 

                          & (output_df['flight_from'] == flight_from) 

                          & (output_df['flight_to'] == flight_to)]) == 0):

            res = s.search_flights(flyFrom=flight_from, 

                                   to=flight_to, 

                                   dateFrom=flight_date.strftime("%d/%m/%Y"), 

                                   dateTo=flight_date.strftime("%d/%m/%Y"), 

                                   partner='picky',

                                   one_for_city=1,

                                   curr='GBP',

                                   max_stopovers=0).json()

            print('calling ', flight_to, ' on ', flight_date)

            for r in res['data']:

                for route in (r['route']):

                    output_df = output_df.append({'observed_date': observed_date.strftime("%d/%m/%Y"), 

                        'flight_date': flight_date.strftime("%d/%m/%Y"),

                        'flight_from': flight_from, 

                        'flight_to': flight_to, 

                        'flight_number': (route['airline'] + str(route['flight_no'])), 

                        'price': r['price']},

                        ignore_index=True)



output_df.to_csv('data.csv', index=False)
output_df.flight_date = pd.to_datetime(output_df.flight_date, dayfirst = True)

output_df.observed_date = pd.to_datetime(output_df.observed_date, dayfirst = True)

sns.lmplot('flight_date', 

           'price', 

           data = output_df[(output_df.observed_date == observed_date) &

                            (output_df.flight_date.dt.month == 12) &

                            (output_df.price < 100)], 

           hue = 'flight_to', 

           fit_reg=False, 

           height=6, 

           aspect=14/6)
sns.lmplot('flight_date', 

           'price', 

           data = output_df[(output_df.flight_to == 'TRN') &

                            (output_df.flight_date.dt.month == 12) &

                            (output_df.price < 100)], 

           hue = 'observed_date', 

           fit_reg=False, 

           height=6, 

           aspect=14/6)
plt.rcParams['figure.figsize'] = [14, 6]

#plt.style.use('ggplot')

plot_df = output_df[(output_df.flight_to == 'TRN') &

                    (output_df.flight_date.dt.month == 12) &

                    (output_df.price < 100)]

plt.scatter(plot_df['flight_date'], 

            plot_df['price'], 

            cmap = plot_df['observed_date'].astype(str), 

            alpha=0.5)
output_df[(output_df.flight_to == 'TRN') &

         (output_df.flight_date <= '2020-04-30') &

         (output_df.flight_date >= observed_date)]