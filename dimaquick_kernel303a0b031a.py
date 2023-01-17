# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import csv

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
with open('/kaggle/input/covid2-submissions/days30_normal.csv', 'rt') as infile:

    with open('submission.csv', 'wt') as outfile:

        writer = csv.writer(outfile)

        for line in csv.reader(infile):

            writer.writerow(line)
from collections import defaultdict

import collections

import csv

from datetime import datetime, timedelta

import glob

import logging

from math import *

import numpy as np

import optparse

import os

import time

import sys

import copy

from typing import Any, Dict, List, Optional, Tuple, Union

import xgboost as xgb

from catboost import Pool, CatBoostRegressor

from xml.etree import ElementTree

from scipy.optimize import curve_fit



RUN = False

PROD = True

PROD_DAYS = 30

EVAL_DAYS = 7
POINTS = {}



PIVOTS = []

PIVOTS.append([43.0, 12.0]) #italy

PIVOTS.append([32.0, 53.0]) #iran

PIVOTS.append([30.9756, 112.2707]) #hubei

PIVOTS.append([36.1162, -119.6816]) #california

PIVOTS.append([47.4009, -121.4905]) #seattle, WA

PIVOTS.append([42.1657, -74.9481]) #NY



Point = collections.namedtuple('Point', ['latitude', 'longitude'])

PointName = collections.namedtuple('PointName', ['country', 'state'])

PointData = collections.namedtuple('PointData', ['cases', 'fatalities', 'recovered'])

CountryData = collections.namedtuple('CountryData', ['population', 'area', 'density', 'coastline', 'migration', 'infant_mortality', 'gdp', 'literacy', 'phones', 'arable', 'crops', 'other', 'climate', 'birthrate', 'deathrate', 'agriculture', 'industry', 'service'])

CountryData.__new__.__defaults__ = (-1,) * len(CountryData._fields)



FEATURE_NAME_PREFIX = 'F_'

DATE_FORMAT = '%Y-%m-%d'



USELESS_FEATURES = set([line.rstrip('\n') for line in open('/kaggle/input/covidshared/useless_features', 'rt')])



Coordinates = {}



with open('/kaggle/input/covidshared/points.csv') as f:

    for line in csv.DictReader(f):

        key = (line['Province/State'], line['Country/Region'])

        latitude = float(line['Lat']) if line['Lat'] != '' else 0.0

        longitude = float(line['Long']) if line['Long'] != '' else 0.0

        Coordinates[key] = (latitude, longitude)

        

def div(a, b):

    return -1 if abs(b) < 1e-5 else a / b



def feature_name(s: str) -> str:

    return FEATURE_NAME_PREFIX + s



def is_feature_name(s: str) -> bool:

    return s.startswith(FEATURE_NAME_PREFIX)



def date_add_days(date: str, num_days: int) -> str:

    return (datetime.strptime(date, DATE_FORMAT) + timedelta(days=num_days)).strftime(DATE_FORMAT)



def date_days_diff(date_start: str, date_end: str):

    delta = datetime.strptime(date_end, DATE_FORMAT) - datetime.strptime(date_start, DATE_FORMAT)

    return delta.days



def in_range(x: Any, full_range: Tuple[Any, Any]) -> bool:

    return full_range[0] <= x <= full_range[1]



def in_left_range(x: Any, left_range: Tuple[Any, Any]) -> bool:

    return left_range[0] <= x < left_range[1]



def normalize_string(s):

    return s.strip().lower()



def haversine(lat1, lon1, lat2, lon2):

    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1 

    dlat = lat2 - lat1 

    tmp = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2

    return asin(sqrt(tmp))



class CountryMetaData(object):

    TOTAL_KEY = 'TOTAL'

    US = 'us'

    UNK_REGION = 'UNK'

    CRUISE_SHIP = 'cruise ship'



    def __init__(self) -> None:

        # Map from country name to 

        self.country_data = defaultdict(CountryData)

        self.us_data = defaultdict(CountryData)

        self.region_id = {}

        self.country_to_region_id = {}

        self.lockdown = {}

        self.tests = {}

        self.not_found = 0



        self.read_world_stats('/kaggle/input/covidshared/world_stats.csv')

        self.read_us_stats('/kaggle/input/covidshared/us_stats.csv')

        self.read_lockdown_data('/kaggle/input/covidshared/lockdown.csv')

        self.read_tests('/kaggle/input/covidshared/tests.html')



    def map_name(self, name: str) -> str:

        if name == 'the gambia':

            return normalize_string('Gambia, The')

        if name == 'trinidad and tobago':

            return normalize_string('Trinidad & Tobago')

        if name == 'the bahamas':

            return normalize_string('Bahamas, The')

        if name == 'taiwan*':

            return normalize_string('Taiwan')

        if name == 'republic of the congo' or name == 'congo (kinshasa)' or name == 'congo (brazzaville)':

            return normalize_string('Congo, Repub. of the')

        if name == 'czechia':

            return normalize_string('Czech Republic')

        if name == 'central african republic':

            return normalize_string('Central African Rep.')

        if name == 'bosnia and herzegovina':

            return normalize_string('Bosnia & Herzegovina')

        if name == 'antigua and barbuda':

            return normalize_string('Antigua & Barbuda')

        if name == 'north macedonia':

            return normalize_string('Macedonia')

        return name



    def read_world_stats(self, file_name) -> None:

        with open(file_name) as f:

            reader = csv.DictReader(f)

            for row in reader:

                country = normalize_string(row['Country'])

                population = float(row['Population'])

                region = normalize_string(row['Region'])

                area = float(row['Area (sq. mi.)'] or -1)

                density = population / area

                coastline = float(row['Coastline (coast/area ratio)'].replace(',','.') or -1)

                migration = float(row['Net migration'].replace(',','.') or -1)

                infant_mortality = float(row['Infant mortality (per 1000 births)'].replace(',','.') or -1)

                gdp = float(row['GDP ($ per capita)'] or -1)

                literacy = float(row['Literacy (%)'].replace(',','.') or -1)

                phones = float(row['Phones (per 1000)'].replace(',','.') or -1)

                arable = float(row['Arable (%)'].replace(',','.') or -1)

                crops = float(row['Crops (%)'].replace(',','.') or -1)

                other = float(row['Other (%)'].replace(',','.') or -1)

                climate = float(row['Climate'].replace(',','.') or -1)

                birthrate = float(row['Birthrate'].replace(',','.') or -1)

                deathrate = float(row['Deathrate'].replace(',','.') or -1)

                agriculture = float(row['Agriculture'].replace(',','.') or -1)

                industry = float(row['Industry'].replace(',','.') or -1)

                service = float(row['Service'].replace(',','.') or -1)



                self.country_data[country] = CountryData(

                  population=population,

                  area=area, 

                  density=density,

                  coastline=coastline,

                  migration=migration,

                  infant_mortality=infant_mortality,

                  gdp=gdp,

                  literacy=literacy,

                  phones=phones,

                  arable=arable,

                  crops=crops,

                  other=other,

                  climate=climate,

                  birthrate=birthrate,

                  deathrate=deathrate,

                  agriculture=agriculture,

                  industry=industry,

                  service=service)



                total = self.country_data[self.TOTAL_KEY]

                self.country_data[self.TOTAL_KEY] = CountryData(population=population + total.population)



                if region not in self.region_id:

                    self.region_id[region] = len(self.region_id) + 1

                self.country_to_region_id[country] = self.region_id[region]



    def read_us_stats(self, file_name) -> None:

        us_data = defaultdict(CountryData)

        with open(file_name) as f:

            reader = csv.DictReader(f)

            for row in reader:

                state = normalize_string(row['Location'])

                population = float(row['Total'])



                us_data[state] = CountryData(population=population)



        us_stats = self.country_data[self.US]

        for state, stats in us_data.items():

            area = us_stats.area / len(us_data)

            self.us_data[state] = CountryData(

                population=stats.population,

                area=area,

                density = stats.population / area,

                migration=us_stats.migration,

                infant_mortality=us_stats.infant_mortality,

                gdp=us_stats.gdp,

                literacy=us_stats.literacy,

                phones=us_stats.phones,

                arable=us_stats.arable,

                crops=us_stats.crops,

                other=us_stats.other,

                climate=us_stats.climate,

                birthrate=us_stats.birthrate,

                deathrate=us_stats.deathrate,

                agriculture=us_stats.agriculture,

                industry=us_stats.industry,

                service=us_stats.service)



    def read_lockdown_data(self, file_name) -> None:

        us_data = defaultdict(CountryData)

        with open(file_name) as f:

            reader = csv.DictReader(f)

            for row in reader:

                country = normalize_string(row['country'])

                state = normalize_string(row['state'])

                lockdown = row['lockdown_date']

                self.lockdown[PointName(country=country, state=state)] = lockdown



    def read_tests(self, file_name):

        with open(file_name) as f:

            html = f.read().strip().replace('<br>', '')



            trs = ElementTree.XML(html)[0]

            headers = [td.text for td in trs[0]]



            for i, tr in enumerate(trs):

                if i == 0 or tr[1].text is None:

                    continue



                values = [td.text for td in tr]

                row = dict(zip(headers, values))



                country = normalize_string(row['Country or territory']).split('-')[0].strip()

                date = datetime.strptime(row['Date'], '%d %b %Y').strftime(DATE_FORMAT)

                tests = int(row['Total tests'].replace(',', ''))



                if country == 'united states':

                    country = 'us'



                if country in self.tests:

                    prev = self.tests[country]

                    self.tests[country] = (max(date, prev.date), tests + prev.tests)

                else:

                    self.tests[country] = (date, tests)



    def _get_data_impl(self, country, state):

        if country == self.CRUISE_SHIP:

            if state == 'diamond princess':

                return CountryData(population=3711)



        if country == self.US:

            if state in self.country_data:

                return self.country_data[state]



        if state in self.us_data:

            return self.us_data[state]



        if country in self.country_data:

            return self.country_data[country]



        return None



    def get_data(self, country, state):

        country = self.map_name(normalize_string(country))

        state = self.map_name(normalize_string(state))



        data = self._get_data_impl(country, state)

        if data:

            return data



        self.not_found += 1



        if country == self.US:

            return CountryData(population=self.us_data[self.US].population / len(self.us_data))



        return CountryData(population=self.country_data[self.TOTAL_KEY].population / len(self.country_data))



    def get_region(self, country):

        country = self.map_name(normalize_string(country))

        if country not in self.country_to_region_id:

            self.region_id[country] = len(self.region_id) + 1

            self.country_to_region_id[country] = self.region_id[country]



        return self.country_to_region_id[country]



    def _get_days_since_lockdown(self, lockdown: str, today: str) -> int:

        days_since_lockdown = date_days_diff(lockdown, today)

        return days_since_lockdown if days_since_lockdown >= 0 else -1



    def get_days_since_lockdown(self, country: str, state: str, today: str) -> int:

        country = normalize_string(country)

        state = normalize_string(state)

        country_state_key = PointName(country=country, state=state)

        country_key = PointName(country=country, state='')

        if country_state_key in self.lockdown:

            return self._get_days_since_lockdown(self.lockdown[country_state_key], today)

        if country_key in self.lockdown:

            return self._get_days_since_lockdown(self.lockdown[country_key], today)

        return -1



    def get_tests_data(self, country: str) -> Optional[Tuple[str, int]]:

        country = normalize_string(country)

        return self.tests[country] if country  in self.tests else None

    

class Example(object):

    HEADER = ['key', 'state', 'country', 'latitude', 'longitude', 'date', 'cases', 'fatalities']



    def __init__(

      self,

      key: int,

      state: str,

      country: str,

      latitude: float,

      longitude: float,

      date: str,

      cases: float,

      fatalities: float) -> None:

        self.key = key

        self.state = state

        self.country = country

        self.latitude = latitude

        self.longitude = longitude

        self.date = date

        self.cases = cases

        self.fatalities = fatalities

        self.recovered = None

        self.features = {}



    @classmethod

    def from_dict(cls, row: Dict[str, str]) -> 'Example':

        if 'Id' in row or 'ForecastId' in row:

            key = (row['Province_State'], row['Country_Region'])

            cur_lat, cur_long = Coordinates[key] # if key in Coordinates else (0.0, 0.0)

            return Example(

                key=int(row['Id']) if 'Id' in row else int(row['ForecastId']),

                state=row['Province_State'],

                country=row['Country_Region'],

                latitude=cur_lat,

                longitude=cur_long,

                date=row['Date'],

                cases=float(row['ConfirmedCases']) if 'ConfirmedCases' in row else None,

                fatalities=float(row['Fatalities']) if 'Fatalities' in row else None)

        elif 'key' in row:

            e = Example(

                key=int(row['key']),

                state=row['state'],

                country=row['country'],

                latitude=float(row['latitude']),

                longitude=float(row['longitude']),

                date=row['date'],

                cases=float(row['cases']) if row['cases'] else None,

                fatalities=float(row['fatalities']) if row['fatalities'] else None)



            for key in row:

                if is_feature_name(key):

                    e.features[key] = float(row[key])

            return e

        assert False, row



    def to_row(self) -> List[Union[int, float, str]]:

        row = []

        for h in self.HEADER:

            row.append(getattr(self, h))



        for f in self.features:

            row.append(self.features[f])



        return row



    @property

    def point(self) -> Point:

        return Point(latitude=self.latitude, longitude=self.longitude)



    @property

    def parsed_date(self):

        return datetime.strptime(self.date, DATE_FORMAT)



    @property

    def point_name(self) -> PointName:

        return PointName(country=self.country, state=self.state)



    @property

    def point_data(self) -> PointData:

        return PointData(cases=self.cases, fatalities=self.fatalities, recovered=self.recovered)



    def set_feature(self, name: str, value: float) -> None:

        name = feature_name(name)

        if name in USELESS_FEATURES:

            return

        assert name not in self.features

        self.features[name] = value



    def get_feature(self, name: str) -> float:

        return self.features[feature_name(name)]





class PointDataSeries(object):

    def __init__(self, dataset: List[Example]) -> None:

        self.point_data_series = collections.defaultdict(dict)

        self.locations = {}

        for e in dataset:

            assert e.date not in self.point_data_series[e.point_name], (e.point_name, e.date)

            if e.point_name not in self.locations:

                self.locations[e.point_name] = len(self.locations)

            self.point_data_series[e.point_name][e.date] = e.point_data



    @property

    def series_len(self) -> int:

        res = None

        for series in self.point_data_series.values():

            if res:

                assert res == len(series)

            else:

                res = len(series)

        return res



    @property

    def date_range(self) -> Tuple[str, str]:

        res = None

        for point_name in self.point_data_series:

            dates = self.point_data_series[point_name].keys()

            if res:

                assert res == (min(dates), max(dates))

            else:

                res = (min(dates), max(dates))

        return res



    def get_full_series(self, point_name: PointName, start_date: str) -> List[PointData]:

        return self.get_series(point_name, start_date, self.series_len)



    def get_series(self, point_name: PointName, start_date: str, num: int) -> List[PointData]:

        res = []

        for i in range(num):

            date = date_add_days(start_date, -i)

            if date in self.point_data_series[point_name]:

                res.append(self.point_data_series[point_name][date])

            else:

                res.append(None)

        return res





class GeoZones(object):

    # A B

    # C D



    LATITUDE_RANGE = (-90.0, 90.0)

    LONGITUDE_RANGE = (-180.0, 180.0)



    def __init__(self, points: Dict[PointName, Point], max_zone_points: int) -> None:

        self.max_zone_points = max_zone_points

        self.geo_zones = {}

        self.leaf_geohashes = []



        self.build(points)



        self.point_geohash = {}

        for geohash in self.leaf_geohashes:

            for p_name, p in self.geo_zones[geohash].items():

                self.point_geohash[p_name] = geohash

        assert set(points.keys()) == set(self.point_geohash.keys())



    def build(self, 

      points: Dict[PointName, Point],

      geohash: str = '',

      latitude_range: Tuple[float, float] = LATITUDE_RANGE,

      longitude_range: Tuple[float, float] = LONGITUDE_RANGE) -> None:

        for p in points.values():

            assert in_left_range(p.latitude, latitude_range), (p, latitude_range)

            assert in_left_range(p.longitude, longitude_range), (p, longitude_range)



        self.geo_zones[geohash] = points

        if len(points) <= self.max_zone_points:

            self.leaf_geohashes.append(geohash)

            return



        quadrants = self._get_quadrants(latitude_range, longitude_range)

        for c, params in quadrants.items():

            self.build(

                points={

                  p_name: p

                  for p_name, p in points.items()

                  if in_left_range(p.latitude, params['latitude_range']) 

                  and in_left_range(p.longitude, params['longitude_range'])

                },

                geohash=geohash + c,

                latitude_range=params['latitude_range'],

                longitude_range=params['longitude_range'])



    def _get_center_point(

      self,

      latitude_range: Tuple[float, float],

      longitude_range: Tuple[float, float]) -> Point:

        return Point(

          latitude=(latitude_range[0] + latitude_range[1]) / 2.0,

          longitude=(longitude_range[0] + longitude_range[1]) / 2.0)



    def _get_quadrants(

      self,

      latitude_range: Tuple[float, float],

      longitude_range: Tuple[float, float]) -> Dict[str, Dict[str, Tuple[float, float]]]:

        center = self._get_center_point(latitude_range, longitude_range)

        return {

          'A': {

            'latitude_range': (latitude_range[0], center.latitude),

            'longitude_range': (longitude_range[0], center.longitude),

          },

          'B': {

            'latitude_range': (center.latitude, latitude_range[1]),

            'longitude_range': (longitude_range[0], center.longitude),

          },

          'C': {

            'latitude_range': (latitude_range[0], center.latitude),

            'longitude_range': (center.longitude, longitude_range[1]),

          },

          'D': {

            'latitude_range': (center.latitude, latitude_range[1]),

            'longitude_range': (center.longitude, longitude_range[1]),

          },

        }



    def _get_same_zone_points_impl(

      self,

      p: Point,

      geohash: str = '',

      latitude_range: Tuple[float, float] = LATITUDE_RANGE,

      longitude_range: Tuple[float, float] = LONGITUDE_RANGE) -> Dict[PointName, Point]:

        assert geohash in self.geo_zones



        quadrants = self._get_quadrants(latitude_range, longitude_range)

        for c, params in quadrants.items():

            if in_left_range(p.latitude, params['latitude_range']) and in_left_range(p.longitude, params['longitude_range']):

                if geohash + c in self.geo_zones:

                    return self._get_same_zone_points_impl(

                        p=p,

                        geohash=geohash + c,

                        latitude_range=params['latitude_range'],

                        longitude_range=params['longitude_range'])

                else:

                    return self.geo_zones[geohash]



        assert False





def read_dataset(path: str) -> List[Example]:

    dataset = []

    with open(path) as f:

        reader = csv.DictReader(f, quotechar='"', delimiter=',')

        for row in reader:

            e = Example.from_dict(row)

            dataset.append(e)

            if e.point_name in POINTS:

                assert POINTS[e.point_name] == e.point

            else:

                POINTS[e.point_name] = e.point



    return dataset



def write_features(path: str, dataset: List[Example]) -> None:

    feature_names = []

    for e in dataset:

        if feature_names:

            assert feature_names == list(e.features.keys())

        else:

            feature_names = list(e.features.keys())



    with open(path, 'w') as f:

        writer = csv.writer(f, lineterminator='\n', delimiter=',')

        writer.writerow(Example.HEADER + feature_names)

        for e in dataset:

            writer.writerow(e.to_row())



def first_n_infected(element, pds, n, predict_window):

    idx = predict_window

    pds = pds.point_data_series



    while True:

        cur_date = date_add_days(element.date, -idx)

        if element.point_name not in pds: return -1

        e = pds[element.point_name]

        if cur_date not in e: return -1

        if e[cur_date].cases < n: return idx

        idx += 1

    



def prev(element, pds, n, predict_window):

    pds = pds.point_data_series

    result = []

    date = element.date

    for i in range(n):

        cur_date = date_add_days(date, -(i + predict_window))

        cases, fatalities, recovered = 0, 0, 0

        if element.point_name in pds and cur_date in pds[element.point_name]:

            e = pds[element.point_name][cur_date]

            if e.cases:

                cases = e.cases

            if e.fatalities:

                fatalities = e.fatalities

            if e.recovered:

                recovered = e.recovered



        result.append(PointData(cases=cases, fatalities=fatalities, recovered=recovered))

    return result



def fit_and_predict(cases, predict_window):

    cases = cases[::-1]

    x, y, logy = [], [], []

    for i in range(len(cases)):

        x.append(i+1)

        y.append(cases[i])

        logy.append(log(1 + cases[i]))

    x, y, logy = map(np.asarray, [x, y, logy])

    z = np.poly1d(np.polyfit(x, y, 3))

    logz = np.poly1d(np.polyfit(x, logy, 3))

    return z(x[-1] + predict_window), logz(x[-1] + predict_window)



def fit_delta(cases, predict_window):

    cases = [log(1 + x) for x in cases[::-1]]

    x, y = [], []

    for i in range(len(cases) - 1):

        x.append(i+1)

        y.append(cases[i + 1] - cases[i])

    x, y = map(np.asarray, [x, y])

    z3 = np.poly1d(np.polyfit(x, y, 3))

    z1 = np.poly1d(np.polyfit(x, y, 1))

    return sum([z3(x[-1] + i + 1) for i in range(predict_window)]), sum([z1(x[-1] + i + 1) for i in range(predict_window)])



def delta_embedding(example, pds, predict_window, size, use_cases):

    pds = pds.point_data_series

    dates = [date_add_days(example.date, -predict_window)]

    for i in range(size):

        dates.append(date_add_days(dates[-1], -1))

    dates = dates[::-1]

    emb = []

    for d in dates:

        if example.point_name in pds and d in pds[example.point_name]:

            e = pds[example.point_name][d]

            emb.append(log(1 + (e.cases if use_cases else e.fatalities)))

        else:

            emb.append(0)

    return [emb[i + 1] - emb[i] for i in range(len(emb) - 1)]



def cluster_distance(emb, cluster_center):

    return sqrt(sum([(emb[i] - cluster_center[i]) ** 2 for i in range(len(emb))]))



def build_emb_bayes(dataset, pds, predict_window):

    emb_bayes = defaultdict(list)

    for e in dataset:

        emb_cases = delta_embedding(e, pds, predict_window, 5, True)

        prev_value = prev(e, pds, 1, predict_window)[0]

        delta_diff = (log(1.0 + e.cases) - log(1.0 + prev_value.cases), log(1.0 + e.fatalities) - log(1.0 + prev_value.fatalities))

        emb_bayes[e.date].append((emb_cases, e.point_data, delta_diff, e.country))

    return emb_bayes



def build_features(dataset: List[Example], country_data: CountryMetaData, pds: PointDataSeries, predict_window: int, emb_bayes) -> List[Example]:

    for e in dataset:

        e_emb_cases = delta_embedding(e, pds, predict_window, 5, True)



        best_cases = None

        best_delta = None

        best_dist_cases = None

    

        for i in range(10):

            date = date_add_days(e.date, -predict_window - i)

            assert date < e.date

            if date in emb_bayes:

                for e2_emb_cases, point_data, delta_diff, country in emb_bayes[date]:

                    dist_cases = cluster_distance(e_emb_cases, e2_emb_cases)

                    if best_dist_cases is None or best_dist_cases > dist_cases:

                        best_dist_cases = dist_cases

                        best_cases = point_data

                        best_delta = delta_diff

           

        prev_values = prev(e, pds, 1, predict_window)

        if best_cases is None:

            e.set_feature('emb_bayes_cases_cases', -1)

            e.set_feature('emb_bayes_cases_fatal', -1)

            e.set_feature('emb_bayes_cases_cases_delta', -1)

            e.set_feature('emb_bayes_cases_fatal_delta', -1)

            e.set_feature('emb_bayes_cases_cases_delta_diff', 0)

            e.set_feature('emb_bayes_cases_fatal_delta_diff', 0)

        else:

            e.set_feature('emb_bayes_cases_cases', best_cases.cases)

            e.set_feature('emb_bayes_cases_fatal', best_cases.fatalities)

            e.set_feature('emb_bayes_cases_cases_delta', log(1 + best_cases.cases) - log(1 + prev_values[0].cases))

            e.set_feature('emb_bayes_cases_fatal_delta', log(1 + best_cases.fatalities) - log(1 + prev_values[0].fatalities))

            e.set_feature('emb_bayes_cases_cases_delta_diff', best_delta[0])

            e.set_feature('emb_bayes_cases_fatal_delta_diff', best_delta[1])

     

    for e in dataset:

        location_index = pds.locations[e.point_name]

        for i in range(294):

            e.set_feature('L' + str(i), 1 if i == location_index else 0)

        for i in range(32):

            e.set_feature('predict_window_' + str(i), 1 if (i + 1) == predict_window else 0)

        e.set_feature('state_len', len(e.state))

        e.set_feature('country_len', len(e.country))

        e.set_feature('latitude', e.latitude / 90.0)

        e.set_feature('longitude', e.longitude / 180.0)



        data = country_data.get_data(e.country, e.state)

        e.set_feature('country_population', data.population)

        # e.set_feature('country_population_log', log(data.population) / 25.0)

        # e.set_feature('country_area', data.area)

        # e.set_feature('country_density', data.density)

        # e.set_feature('country_coastline', data.coastline)

        # e.set_feature('country_migration', data.migration)

        # e.set_feature('country_infant_mortality', data.infant_mortality)

        # e.set_feature('country_gdp', data.gdp)

        # e.set_feature('country_literacy', data.literacy)

        # e.set_feature('country_phones', data.phones)

        # e.set_feature('country_arable', data.arable)

        # e.set_feature('country_crops', data.crops)

        # e.set_feature('country_other', data.other)

        # e.set_feature('country_climate', data.climate)

        # e.set_feature('country_birthrate', data.birthrate)

        # e.set_feature('country_deathrate', data.deathrate)

        # e.set_feature('country_agriculture', data.agriculture)

        # e.set_feature('country_industry', data.industry)

        # e.set_feature('country_service', data.service)



        #cluster_info = point2cluster[e.point_name]

        #e.set_feature('clusterid', cluster_info[0])

        #for i in range(len(cluster_info[1])):

        #  e.set_feature('cluster_distance_' + str(i), cluster_info[1][i])



        #pivots

        #for i in range(len(PIVOTS)):

        #  dist = haversine(PIVOTS[i][0], PIVOTS[i][1], e.latitude, e.longitude)

        #  e.set_feature('pivots_dist_' + str(i), dist)



        tests_data = country_data.get_tests_data(e.country)



        weekday = e.parsed_date.weekday()

        assert 0 <= weekday < 7

        for i in range(7):

            e.set_feature('day_' + str(i), int(i == weekday))



        #for n in [1, 10, 100, 1000]:

        #  e.set_feature('first_' + str(n), first_n_infected(e, pds, n, predict_window))



        prev_values = prev(e, pds, 50, predict_window)

        for i in [10, 20, 50]:

            fit = fit_and_predict([x.cases for x in prev_values[:i]], predict_window)

            e.set_feature('fit_' + str(i), fit[0])

            e.set_feature('fitlog_' + str(i), fit[1])

            fit = fit_and_predict([x.fatalities for x in prev_values[:i]], predict_window)

            e.set_feature('fit_fatal_' + str(i), fit[0])

            e.set_feature('fitlog_fatal_' + str(i), fit[1])

            fit = fit_delta([x.cases for x in prev_values[:i]], predict_window)

            e.set_feature('fit_delta3_cases_' + str(i), fit[0])

            e.set_feature('fit_delta1_cases_' + str(i), fit[1])

            fit = fit_delta([x.fatalities for x in prev_values[:i]], predict_window)

            e.set_feature('fit_delta3_fatal_' + str(i), fit[0])

            e.set_feature('fit_delta1_fatal_' + str(i), fit[1])

        

        #res = fit_exp([x.cases for x in prev_values], predict_window, e.country, e.date)

        #fit_gauss([x.cases for x in prev_values], [x.fatalities for x in prev_values], predict_window, e.country, e.date)

        #e.set_feature('fitexp', res[0])

        #e.set_feature('fitexp_b', res[1])

        #e.set_feature('fitexp_e', res[2])



        e.set_feature('infected_share', div(prev_values[0].cases, data.population))

        e.set_feature('mortality_share', div(prev_values[0].fatalities, prev_values[0].cases))

        # e.set_feature('mortality_to_recovered_share', div(prev_values[0].fatalities, prev_values[0].recovered))



        e.set_feature('recovered', log(1 + prev_values[0].recovered))

        e.set_feature('recovered_share', div(prev_values[0].recovered, prev_values[0].cases))

        e.set_feature('headroom', prev_values[0].cases - prev_values[0].fatalities - prev_values[0].recovered)

        # e.set_feature('headroom_share', div(prev_values[0].cases - prev_values[0].fatalities - prev_values[0].recovered, prev_values[0].cases))



        for i in range(20):

            if i > 0:

                e.set_feature('prev_cases_speed_' + str(i+1), -log(1 + prev_values[i].cases) + log(1 + prev_values[0].cases))

                e.set_feature('prev_cases_speed_window_' + str(i+1), (log(1 + prev_values[0].cases) - log(1 + prev_values[i].cases)) / i * predict_window)

                e.set_feature('prev_fatal_speed_' + str(i+1), -log(1 + prev_values[i].fatalities) + log(1 + prev_values[0].fatalities))

                e.set_feature('prev_fatal_speed_window_' + str(i+1), (log(1 + prev_values[0].fatalities) - log(1 + prev_values[i].fatalities)) / i * predict_window)

            if i < 10:

                e.set_feature('prev_cases_' + str(i+1), log(1 + prev_values[i].cases))

                e.set_feature('prev_cases_accum_' + str(i+1), prev_values[i].cases)

                e.set_feature('prev_fatal_' + str(i+1), log(1 + prev_values[i].fatalities))

                e.set_feature('prev_fatal_accum_' + str(i+1), prev_values[i].fatalities)



        e.set_feature('region', country_data.get_region(e.country))

        e.set_feature('days_since_lockdown', country_data.get_days_since_lockdown(e.country, e.state, date_add_days(e.date, -predict_window)))



    return dataset



def build_geohash_features(dataset: List[Example], gz: GeoZones) -> List[Example]:

    F_PREFIXES = [feature_name('prev_cases_'), feature_name('prev_fatal_'), feature_name('infected_share')]



    feature_lists = defaultdict(lambda: defaultdict(list))

    for e in dataset:

        gh = gz.point_geohash[e.point_name]

        for fp in F_PREFIXES:

            for f_name, f_value in e.features.items():

                if f_name.startswith(fp):

                    feature_lists[gh][f_name].append(f_value)



    feature_stats = defaultdict(dict)

    for gh, feature_values in feature_lists.items():

        for f_name, f_values in feature_values.items():

            feature_stats[gh][f_name] = {

                'min': min(f_values),

                'max': max(f_values),

                'sum': 1.0 * sum(f_values) / len(f_values),

                'cnt': len(f_values),

              }



    for e in dataset:

        gh = gz.point_geohash[e.point_name]

        e.set_feature('geohash', hash(gh))

        for f_name, f_stats in feature_stats[gh].items():

            e.set_feature('geozone_min_' + f_name, f_stats['min'])

            e.set_feature('geozone_max_' + f_name, f_stats['max'])

            e.set_feature('geozone_sum_' + f_name, f_stats['sum'])

            e.set_feature('geozone_cnt_' + f_name, f_stats['cnt'])

    return dataset



def main_FeaturesStage(opts):

    start = time.time()

    raw_train_set = read_dataset(opts.train_csv)

    raw_test_set = read_dataset(opts.test_csv)

    country_data = CountryMetaData()

    #reports = read_reports()

  

    pds = PointDataSeries(raw_train_set)

    data_date_range = pds.date_range

  

    gz12 = GeoZones(points=POINTS, max_zone_points=12)

  

    test_date_range = (date_add_days(opts.start_date, 1 - opts.num_days), opts.start_date)

    train_date_range = (data_date_range[0], date_add_days(test_date_range[0], -1))

    no_data_test_date_range = (date_add_days(data_date_range[1], 1), test_date_range[1])



    train_set = [e for e in raw_train_set if in_range(e.date, train_date_range)]

    test_set = [e for e in raw_train_set if in_range(e.date, test_date_range) and date_add_days(e.date, -opts.predict_window) == date_add_days(opts.start_date, -opts.num_days)]

    if no_data_test_date_range[0] <= no_data_test_date_range[1]:

        test_set += [e for e in raw_test_set if in_range(e.date, no_data_test_date_range) and date_add_days(e.date, -opts.predict_window) == date_add_days(opts.start_date, -opts.num_days)]



    if opts.last_n_days != -1:

        train_set = train_set[-opts.last_n_days:]

        test_set = test_set[-opts.last_n_days:]

    # point2cluster = kmeans(pds, opts.predict_window, [x for x in raw_train_set if x.date == date_add_days(opts.start_date, -opts.num_days)])



    emb_bayes = build_emb_bayes(raw_train_set, pds, opts.predict_window)



    #import_reports_data(train_set, reports)



    train_features = build_features(train_set, country_data, pds, opts.predict_window, emb_bayes)

    train_features = build_geohash_features(train_features, gz12)

    write_features(opts.train_features_csv, train_features)

    train_features = []

  

    test_features = build_features(test_set, country_data, pds, opts.predict_window, emb_bayes)

    test_features = build_geohash_features(test_features, gz12)

    write_features(opts.test_features_csv, test_features)

    end = time.time()

    print('Took {} seconds'.format(end - start))

    if opts.verbose:

        print('Info:')

        print('  Data series len: {}'.format(pds.series_len))

        print('  Data date range: {}'.format(data_date_range))

        print('  Num geo-points: {}'.format(len(POINTS)))

        print('  Num geo-zones (up to 10 points): {}'.format(len(gz12.leaf_geohashes)))

        print('  Num features: {}'.format(len(train_set[0].features)))

        print('  Train date range: {}'.format(train_date_range))

        print('  Test date range: {}'.format(test_date_range))

        print('  Num train examples: {}'.format(len(train_features)))

        print('  Num test examples: {}'.format(len(test_features)))

        if no_data_test_date_range[0] <= no_data_test_date_range[1]:

            print('  Test date range (no data): {}'.format(no_data_test_date_range))
# Learning

import math

def read_learn_datasets(path, days):

    examples = []

    for i in range(days):

        examples.extend(read_dataset(path + '_' + str(i+1) + '.csv'))

    return examples



def get_pool_data(mode, data, prefix):

    def get(e):

        return e if e else 0.0

    x, delta_y, abs_y, fit, prev = [], [], [], [], []

    for e in data[prefix + 'examples']:

        if mode == 'cases':

            cur_prev, cur_fit, cur_y = e.features['F_prev_cases_1'], e.features['F_fit_20'], get(e.cases)

        else:

            cur_prev, cur_fit, cur_y = e.features['F_prev_fatal_1'], e.features['F_fit_fatal_20'], get(e.fatalities)

        cur_y = math.log(1 + cur_y)

        abs_y.append(cur_y)

        delta_y.append(cur_y - cur_prev)

        x.append(list(e.features.values()))

        fit.append(cur_fit)

        prev.append(cur_prev)

        data['names'] = list(e.features.keys())

    data[prefix + 'x'] = np.asarray(x)

    data[prefix + 'delta_y'] = np.asarray(delta_y)

    data[prefix + 'abs_y'] = np.asarray(abs_y)

    data[prefix + 'fit'] = fit

    data[prefix + 'prev'] = prev



def read_everything(opts):

    data = {}  

    data['train_examples'] = read_learn_datasets(opts.train_features_csv, opts.days)

    data['test_examples'] = read_learn_datasets(opts.test_features_csv, opts.days)

    get_pool_data(opts.mode, data, 'train_')

    get_pool_data(opts.mode, data, 'test_')

    return data



def get_pool(data, prefix, is_delta, is_catboost):

    y_key = prefix + 'delta_y' if is_delta else prefix + 'abs_y'

    if is_catboost:

        return Pool(data[prefix + 'x'], data[y_key])

    return xgb.DMatrix(data[prefix + 'x'], label=data[y_key])



def write_predictions(path, data, mode, prefix, predicted, is_delta, name):

    with open('_'.join([path, name, mode]) + '.csv', 'w') as f:

        writer = csv.writer(f, lineterminator='\n', delimiter=',')

        writer.writerow(['key', 'state', 'country', 'date', 'y', 'prediction', 'fit20', 'prev'])

        examples, y, fit, prev = data[prefix + 'examples'], data[prefix + 'abs_y'], data[prefix + 'fit'], data[prefix + 'prev']

        for i in range(len(examples)):

            e = examples[i]

            pred = predicted[i]

            if is_delta:

                pred += prev[i]

            writer.writerow([e.key, e.state, e.country, e.date, y[i], max(pred, 0), fit[i], prev[i]])



def model_name(is_delta, is_linear, is_catboost):

    prefix = 'delta' if is_delta else 'abs'

    if is_linear:   return prefix + '_linear'

    if is_catboost: return prefix + '_catboost'

    return prefix + '_trees'



def do_learn(data, opts, is_delta, is_linear, is_catboost, iterations):

    train = get_pool(data, 'train_', is_delta, is_catboost)

    test  = get_pool(data, 'test_', is_delta, is_catboost)

  

    if is_catboost:

        #model = CatBoostRegressor(iterations=opts.iterations, depth=opts.depth, learning_rate=opts.eta, loss_function='RMSE', logging_level='Verbose')

        #model.fit(train, eval_set=test)

        return

    else:

        param = {'nthread' : 16, 'objective' : 'reg:squarederror', 'seed' : opts.seed, 'eta' : opts.eta}

        if is_linear:

            param['booster'] = 'gblinear'

            param['alpha'] = 0.00001

        else:

            param['max_depth'] = opts.depth

            param['subsample'] = 0.8

        evallist = [(test, 'eval'), (train, 'train')] if opts.verbose else []

        booster = xgb.train(param, train, 2 if opts.super_quick else iterations, evallist)

        train_predicted, test_predicted = booster.predict(train), booster.predict(test)

        #if opts.dump:

            #dump_model('data/model.txt', data['names'], booster)

  

    write_predictions(opts.train_predicted_csv, data, opts.mode, 'train_', train_predicted, is_delta, model_name(is_delta, is_linear, is_catboost))

    write_predictions(opts.test_predicted_csv,  data, opts.mode, 'test_',  test_predicted,  is_delta, model_name(is_delta, is_linear, is_catboost))

    #if not is_linear and not is_catboost: print_importance(opts.importance, booster, data['names'])



def main_LearningStage(opts):

    data = read_everything(opts)



    iterations = {}

    iterations[('delta', 'trees', 'cases')] = 170

    iterations[('delta', 'trees', 'fatal')] = 260

    iterations[('abs', 'trees', 'cases')] = 210

    iterations[('abs', 'trees', 'fatal')] = 260

    iterations[('delta', 'linear', 'cases')] = 4

    iterations[('delta', 'linear', 'fatal')] = 10

    iterations[('abs', 'linear', 'cases')] = 40

    iterations[('abs', 'linear', 'fatal')] = 10

    

    for is_linear in [False, True]:

        for is_delta in [True, False]:

            key = ('delta' if is_delta else 'abs', 'linear' if is_linear else 'trees', opts.mode)

            print('__'.join(['Key'] + list(key)))

            do_learn(data, opts, is_delta, is_linear, False, iterations[key])
def get_final_score(dt, at, dl, al, weights):

    prev = float(dt['prev'])

    dt, at, dl, al = map(lambda x: float(x['prediction']), [dt, at, dl, al])

    pred = 0

    pred += weights['wdt'] * dt

    pred += weights['wat'] * at

    pred += weights['wdl'] * dl

    pred += weights['wal'] * al

    return max(pred, prev)



def coeff():

    return [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.5, 0.75, 0.85, 0.9, 1.0]



def ordered_loc_data(loc_data):

    cur = [[k, loc_data[k]] for k in loc_data.keys()]

    cur.sort(key=lambda x: x[0])

    return [x[1] for x in cur]



def get_for_country(weights, values):

    error, n = 0.0, 0

    for v in values:

        sc, sf = v['scores_cases'], v['scores_fatal']

        cases = sc[0] * weights['wdt'] + sc[1] * weights['wat'] + sc[2] * weights['wdl'] + sc[3] * weights['wal']

        fatal = sf[0] * weights['wdt'] + sf[1] * weights['wat'] + sf[2] * weights['wdl'] + sf[3] * weights['wal']

        n += 2

        error += (v['y_cases'] - cases) ** 2

        fatal += (v['y_fatal'] - fatal) ** 2

    return error



def prepare_index(examples_delta_trees, examples_abs_trees, examples_delta_linear, examples_abs_linear, opts, weights, country_weights):

    index = {}

    for e_dt, e_at, e_dl, e_al in zip(examples_delta_trees, examples_abs_trees, examples_delta_linear, examples_abs_linear):

        key = '__'.join(map(str, [e_dt['country'], e_dt['state']]))

        if country_weights is None:

            pred = get_final_score(e_dt, e_at, e_dl, e_al, weights)

        else:

            pred = get_final_score(e_dt, e_at, e_dl, e_al, country_weights[key])

        date = e_dt['date']

        if key not in index: index[key] = {}

        loc_index = index[key]

        if date not in loc_index: loc_index[date] = {}

        dct = loc_index[date]

        dct['pred_' + e_dt['mode']] = pred

        dct['y_'    + e_dt['mode']] = float(e_dt['y'])

        dct['obj_'  + e_dt['mode']] = e_dt

        dct['prev_' + e_dt['mode']] = float(e_dt['prev'])

        dct['scores_' + e_dt['mode']] = list(map(lambda x: float(x['prediction']), [e_dt, e_at, e_dl, e_al]))



    for key in index.keys():

        index[key] = ordered_loc_data(index[key])

    return index



def apply_rules(loc_data, min_mortality, max_mortality):

    prev_cases = []

    prev_fatalities = []

    

    for v in loc_data:

        cases, fatalities = v['pred_cases'], v['pred_fatal']

      

        if len(prev_cases) > 0 and cases < prev_cases[-1]:

            cases = prev_cases[-1]

      

        if len(prev_fatalities) > 0 and fatalities < prev_fatalities[-1]:

            fatalities = prev_fatalities[-1]

      

        if cases - fatalities > min_mortality:

            fatalities = cases - min_mortality

        if cases - fatalities < max_mortality and cases > 5:

            fatalities = cases - max_mortality

        v['final_cases'] = max(cases, v['prev_cases'])

        v['final_fatal'] = max(fatalities, v['prev_fatal'])

        prev_cases.append(v['final_cases'])

        prev_fatalities.append(v['final_fatal'])



def copy_and_set_prediction(e, pred, lst):

    tmp = copy.deepcopy(e)

    tmp['prediction'] = pred

    lst.append(tmp)



def calc_error(examples_delta_trees, examples_abs_trees, examples_delta_linear, examples_abs_linear, opts, weights, country_weights):

    index = prepare_index(examples_delta_trees, examples_abs_trees, examples_delta_linear, examples_abs_linear, opts, weights, country_weights)

    for loc_data in index.values():

        apply_rules(loc_data, math.log(opts.min_mortality), math.log(opts.max_mortality))

    

    error = 0

    cases_list, fatalities_list = [], []

    n = 0

    for loc_data in index.values():

        for v in loc_data:

            if not opts.only_fatal:

                n += 1

                error += (v['final_cases'] - v['y_cases']) ** 2

            if not opts.only_cases:

                n += 1

                error += (v['final_fatal'] - v['y_fatal']) ** 2

            copy_and_set_prediction(v['obj_cases'], v['final_cases'], cases_list)

            copy_and_set_prediction(v['obj_fatal'], v['final_fatal'], fatalities_list)

  

    return (error / n) ** 0.5, cases_list, fatalities_list



def read_examples(pattern, mode):

    result = []

    with open('{}_{}.csv'.format(pattern, mode), 'r') as f:

        reader = csv.DictReader(f, quotechar='"', delimiter=',')

        for row in reader:

            row['mode'] = mode

            result.append(row)

    return result



def read_examples_for_model(opts, model_name):

    test_examples = []

    test_examples.extend(read_examples(opts.predicted_pattern + model_name, "cases"))

    test_examples.extend(read_examples(opts.predicted_pattern + model_name, "fatal"))

    return test_examples



def write_final_predictions(path, lst):

    with open(path, 'wt') as f:

        w = csv.writer(f)

        w.writerow(lst[0].keys())

        for e in lst:

            w.writerow([e[k] for k in e.keys()])



def print_error(error, opts, w, win):

    s = 'Test error = {}, Win = {}. min_mortality={}, max_mortality={}, formula = {} * DT + {} * AT + {} * DL + {} * AL'

    return s.format(error, win, opts.min_mortality, opts.max_mortality, w['wdt'], w['wat'], w['wdl'], w['wal'])



def main_FinalizingStage(opts):

    examples_delta_trees  = read_examples_for_model(opts, '_delta_trees')

    examples_abs_trees    = read_examples_for_model(opts, '_abs_trees')

    examples_delta_linear = read_examples_for_model(opts, '_delta_linear')

    examples_abs_linear   = read_examples_for_model(opts, '_abs_linear')



    country_weights = None #read_country_weights(opts.country_weights)

    weights = {'wdt' : opts.wdt, 'wat' : opts.wat, 'wdl' : opts.wdl, 'wal' : opts.wal}

    prod_error, cases_list, fatalities_list = calc_error(examples_delta_trees, examples_abs_trees, examples_delta_linear, examples_abs_linear, opts, weights, country_weights)

    print(prod_error)

    write_final_predictions(opts.predicted_pattern + '_cases.csv', cases_list)

    write_final_predictions(opts.predicted_pattern + '_fatalities.csv', fatalities_list)

  

    #if opts.optimize:

    #errors = []

    #for wdt in coeff():

    #  for wat in coeff():

    #    if wdt + wat > 1: break

    #    for wdl in coeff():

    #      wal = 1.0 - wdt - wat - wdl

    #      if wal < 0: break

    #      weights = {'wdt' : wdt, 'wat' : wat, 'wdl' : wdl, 'wal' : wal}

    #      test_error, _, __ = calc_error(examples_delta_trees, examples_abs_trees, examples_delta_linear, examples_abs_linear, opts, weights, country_weights)

    #      errors.append([test_error, print_error(test_error, opts, weights, prod_error - test_error)])

    #errors.sort(key=lambda x: x[0])

    #for e in errors:

    #  print(e[1])
# Run week2.py

def main_Week2Adjustment(opts):

    data = {}

    with open(opts.train, 'r') as f:

        reader = csv.DictReader(f, quotechar='"', delimiter=',')

        for row in reader:

            key = '__'.join([row['Province_State'], row['Country_Region']])

            if key not in data: data[key] = [-1, -1, -1, -1]

            if row['Date'] == '2020-03-27':

                data[key][0] = math.log(1 + float(row['ConfirmedCases']))

                data[key][1] = math.log(1 + float(row['Fatalities']))

            if row['Date'] == '2020-03-31':

                data[key][2] = math.log(1 + float(row['ConfirmedCases']))

                data[key][3] = math.log(1 + float(row['Fatalities']))

        hack = {}

        with open(opts.cases_predicted_csv, 'r') as f:

            writer = csv.writer(open(opts.modified_cases, 'wt'))

            reader = csv.DictReader(f, quotechar='"', delimiter=',')

            last_country = 'None'

            writer.writerow(['key','state','country','date','prediction', 'fit20'])

            for row in reader:

                key = '__'.join([row['state'], row['country']])

                val = data[key]

                if last_country != key:

                    last_country = key

                    last_value = data[key][2]

                    delta = (val[2] - val[0]) / 4

                    delta = max(0, delta)

                else:

                    delta *= 0.925

                pred = float(row['prediction'])

                at_most = last_value + delta

                pred = min(pred, at_most)

                last_value = pred

                hack['__'.join([row['state'], row['country'], row['date']])] = pred

                writer.writerow([row['key'], row['state'], row['country'], row['date'], pred, 0])



        with open(opts.fatal_predicted_csv, 'r') as f:

            writer = csv.writer(open(opts.modified_fatal, 'wt'))

            reader = csv.DictReader(f, quotechar='"', delimiter=',')

            last_country = 'None'

            writer.writerow(['key','state','country','date','prediction', 'fit20'])

            for row in reader:

                key = '__'.join([row['state'], row['country']])

                val = data[key]

                if last_country != key:

                    last_country = key

                    last_value = data[key][3]

                    delta = (val[2] - val[0]) / 4 # not a bug

                    delta = max(0, delta)

                else:

                    delta *= 0.93

                pred = float(row['prediction'])

                at_most = last_value + delta

                pred = min(pred, at_most)

                cases_value = hack['__'.join([row['state'], row['country'], row['date']])]

                if pred < cases_value - opts.max_mortality_log:

                    pred = cases_value - opts.max_mortality_log

                last_value = pred

                writer.writerow([row['key'], row['state'], row['country'], row['date'], pred, 0])

def read_one(path):

    res = []

    with open(path, 'r') as f:

        for row in csv.DictReader(f, quotechar='"', delimiter=','):

            res.append(row)

    return res



def main_Submission(opts):

    submission = read_one(opts.submission)

    cases = read_one(opts.predicted_cases)

    fatal = read_one(opts.predicted_fatalities)

  

    cases_index = {}

    for case in cases:

        cases_index[case['state'] + '__' + case['country'] + '__' + case['date']] = math.exp(float(case['prediction'])) - 1

    fatal_index = {}

    for case in fatal:

        fatal_index[case['state'] + '__' + case['country'] + '__' + case['date']] = math.exp(float(case['prediction'])) - 1

    n, found = 0, 0

    with open(opts.output, 'w') as f:

        writer = csv.writer(f, lineterminator='\n', delimiter=',')

        writer.writerow(['ForecastId', 'ConfirmedCases', 'Fatalities'])

        for rec in submission:

            key = rec['Province_State'] + '__' + rec['Country_Region'] + '__' + rec['Date']

            cases_answer, fatal_answer = 0, 0

            n += 1

            if key in cases_index:

                found += 1

                cases_answer, fatal_answer = cases_index[key], fatal_index[key]

            writer.writerow([rec['ForecastId'], cases_answer, fatal_answer])

    print('N={}'.format(n))

    print('Found={}'.format(found))

    print('Share={}'.format(float(found) / n))
# Run all.py

class FeaturesOpts:

    def __init__(self, num_days, predict_window, verbose, start_date, last_n_days = -1):

        self.train_csv = '/kaggle/input/covid19-global-forecasting-week-2/train.csv'

        self.test_csv = '/kaggle/input/covid19-global-forecasting-week-2/test.csv'

        self.train_features_csv = 'train_features_{}.csv'.format(predict_window)

        self.test_features_csv = 'test_features_{}.csv'.format(predict_window)

        self.num_days = num_days

        self.predict_window = predict_window

        self.verbose = verbose

        self.start_date = start_date

        self.last_n_days = last_n_days

        

start = time.time()

if RUN:

    if PROD:

        for i in range(PROD_DAYS):

            print('Day ' + str(i + 1))

            main_FeaturesStage(FeaturesOpts(PROD_DAYS, i + 1, False, '2020-04-02'))

    else:

        for i in range(EVAL_DAYS):

            print('Day ' + str(i + 1))

            main_FeaturesStage(FeaturesOpts(EVAL_DAYS, i + 1, False, '2020-03-31'))

end = time.time()

print('Overall took {} minutes'.format((end - start) / 60))
# Run all.py

class LearningOpts:

    def __init__(self, num_days, mode, super_quick = False):

        self.train_features_csv = 'train_features'

        self.test_features_csv = 'test_features'

        self.train_predicted_csv = 'train_predicted'

        self.test_predicted_csv = 'test_predicted'

        self.mode = mode

        self.days = num_days

        self.verbose = True

        self.eta = 0.03

        self.seed = 0

        self.depth = 6

        self.super_quick = super_quick

        

start = time.time()

if RUN:

    if PROD:

        main_LearningStage(LearningOpts(PROD_DAYS, 'cases'))

        main_LearningStage(LearningOpts(PROD_DAYS, 'fatal'))

    else:

        main_LearningStage(LearningOpts(EVAL_DAYS, 'cases', False))

        main_LearningStage(LearningOpts(EVAL_DAYS, 'fatal'))

end = time.time()

print('Overall took {} minutes'.format((end - start) / 60))
# Run stat.py

class FinalizingOpts:

    def __init__(self):

        self.predicted_pattern = 'test_predicted'

        self.verbose = True

        self.only_cases = False

        self.only_fatal = False

        self.wdt = 0.5

        self.wat = 0.3

        self.wdl = 0.1

        self.wal = 0.1

        self.min_mortality = 250.0

        self.max_mortality = 1.0

        self.optimize = False

        

if RUN:

    main_FinalizingStage(FinalizingOpts())
# Run week2.py

class Week2Opts:

    def __init__(self):

        self.train = '/kaggle/input/covid19-global-forecasting-week-2/train.csv'

        self.cases_predicted_csv = 'test_predicted_cases.csv'

        self.fatal_predicted_csv = 'test_predicted_fatalities.csv'

        self.modified_cases = 'modified_cases.csv'

        self.modified_fatal = 'modified_fatal.csv'

        self.max_mortality_log = 4.7

        

if RUN and PROD:

    main_Week2Adjustment(Week2Opts())

# Run make_submission.py

class SubmissionOpts:

    def __init__(self, use_modified = True):

        self.submission = '/kaggle/input/covid19-global-forecasting-week-2/test.csv'

        self.predicted_cases = 'modified_cases.csv' if use_modified else 'test_predicted_cases.csv'

        self.predicted_fatalities = 'modified_fatal.csv' if use_modified else 'test_predicted_fatalities.csv'

        self.output = 'submission.csv'

        self.verbose = True

        

if RUN and PROD:

    main_Submission(SubmissionOpts(True))
