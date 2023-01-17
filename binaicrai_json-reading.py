import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

warnings.filterwarnings("ignore")

import requests

import time
headers = {

'Referer': 'https://www.rottentomatoes.com/m/the_lion_king_2019/reviews?type=user',

'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.108 Safari/537.36',

'X-Requested-With': 'XMLHttpRequest',}
url = 'https://www.rottentomatoes.com/napi/movie/9057c2cf-7cab-317f-876f-e50b245ca76e/reviews/user'
s = requests.Session()
time.sleep(20)

payload = {

'direction': 'next',

'endCursor': 'eyJyZWFsbV91c2VySWQiOiJGYW5kYW5nb183NThmZWY1OS0zZDdlLTQ1NmItOGQ0Zi0xMGMzNzExOTI2MzgiLCJlbXNJZCI6IjkwNTdjMmNmLTdjYWItMzE3Zi04NzZmLWU1MGIyNDVjYTc2ZSIsImVtc0lkX2hhc1Jldmlld0lzVmlzaWJsZSI6IjkwNTdjMmNmLTdjYWItMzE3Zi04NzZmLWU1MGIyNDVjYTc2ZV9UIiwiY3JlYXRlRGF0ZSI6IjIwMTktMDgtMDdUMjM6MzI6MjEuMzc2WiJ9',

  'startCursor': 'eyJyZWFsbV91c2VySWQiOiJGYW5kYW5nb19iMDE1Nzg4Ny1hN2RkLTRlZjgtOTA3ZC01NzAwMmY0ZDE3MDUiLCJlbXNJZCI6IjkwNTdjMmNmLTdjYWItMzE3Zi04NzZmLWU1MGIyNDVjYTc2ZSIsImVtc0lkX2hhc1Jldmlld0lzVmlzaWJsZSI6IjkwNTdjMmNmLTdjYWItMzE3Zi04NzZmLWU1MGIyNDVjYTc2ZV9UIiwiY3JlYXRlRGF0ZSI6IjIwMTktMDgtMDhUMDA6MTM6NTAuMjA3WiJ9',}
r = s.get(url, headers=headers, params=payload) 

data = r.json()
data
reviews = [{'rating': 'STAR_3_5',

   'review': 'I loved it, there were afew things from the first one that I would have loved to see in this one but over all it was great.',

   'displayName': 'Vida H',

   'displayImageUrl': 'https://graph.facebook.com/v3.3/10157518547596120/picture',

   'isVerified': False,

   'isSuperReviewer': False,

   'hasSpoilers': False,

   'hasProfanity': False,

   'createDate': '2019-08-07T23:28:30.179Z',

   'updateDate': '2019-08-07T23:28:30.179Z',

   'user': {'userId': '978190610',

    'realm': 'RT',

    'displayName': 'Vida H',

    'accountLink': '/user/id/978190610'},

   'score': 3.5,

   'timeFromCreation': 'Aug 07, 2019'},

  {'rating': 'STAR_4_5',

   'review': 'It was cute but not as good as the original.  Timon and pumba werent as funny and zazu didnt sing the morning report.',

   'displayName': 'Jo',

   'displayImageUrl': None,

   'isVerified': True,

   'isSuperReviewer': False,

   'hasSpoilers': False,

   'hasProfanity': False,

   'createDate': '2019-08-07T23:14:34.048Z',

   'updateDate': '2019-08-07T23:14:34.048Z',

   'user': {'userId': '1bcfba84-080b-4b21-95d5-9961b120a5b8',

    'realm': 'Fandango',

    'displayName': 'Jo',

    'accountLink': None},

   'score': 4.5,

   'timeFromCreation': 'Aug 07, 2019'},

  {'rating': 'STAR_4',

   'review': 'The Lion King was visually stunning. The differences were small but worked. It was true to the original. As a Disney purist, I prefer the original, but this was a very entertaining movie.',

   'displayName': 'Visually stunning and true to the original',

   'displayImageUrl': None,

   'isVerified': True,

   'isSuperReviewer': False,

   'hasSpoilers': False,

   'hasProfanity': False,

   'createDate': '2019-08-07T23:11:43.548Z',

   'updateDate': '2019-08-07T23:11:43.548Z',

   'user': {'userId': '04652d8e-494b-4797-9d94-5af354cf23af',

    'realm': 'Fandango',

    'displayName': 'Visually stunning and true to the original',

    'accountLink': None},

   'score': 4,

   'timeFromCreation': 'Aug 07, 2019'},

  {'rating': 'STAR_5',

   'review': 'This is my favorite play and also saw the last movie. Asked my 14 year old granddaughter to go with me to a matinee. She loved it and admitted it. The animals look so real that little kids in the audience would scream for or against different scenes in the movie. Highly recommend.',

   'displayName': 'Margie S',

   'displayImageUrl': None,

   'isVerified': False,

   'isSuperReviewer': False,

   'hasSpoilers': False,

   'hasProfanity': False,

   'createDate': '2019-08-07T23:08:18.693Z',

   'updateDate': '2019-08-07T23:08:18.693Z',

   'user': {'userId': '529f6a5f-124b-4092-91b4-fc94edd26029',

    'realm': 'Fandango',

    'displayName': 'Margie S',

    'accountLink': None},

   'score': 5,

   'timeFromCreation': 'Aug 07, 2019'},

  {'rating': 'STAR_5',

   'review': 'amazing story now paired with the most amazing cgi.',

   'displayName': 'Jr',

   'displayImageUrl': None,

   'isVerified': False,

   'isSuperReviewer': False,

   'hasSpoilers': False,

   'hasProfanity': False,

   'createDate': '2019-08-07T22:43:08.931Z',

   'updateDate': '2019-08-07T22:43:08.931Z',

   'user': {'userId': 'EE4F9798-20D4-4E95-90D8-9EB1386F1152',

    'realm': 'Fandango',

    'displayName': 'Jr',

    'accountLink': None},

   'score': 5,

   'timeFromCreation': 'Aug 07, 2019'},

  {'rating': 'STAR_4',

   'review': "Beyonce is great and all but i don't believe she was the right person to play the role of nala.",

   'displayName': 'Giovanny',

   'displayImageUrl': None,

   'isVerified': True,

   'isSuperReviewer': False,

   'hasSpoilers': False,

   'hasProfanity': False,

   'createDate': '2019-08-07T22:40:05.542Z',

   'updateDate': '2019-08-07T22:40:05.542Z',

   'user': {'userId': 'a5545015-fdca-4c43-8630-e540555eadf2',

    'realm': 'Fandango',

    'displayName': 'Giovanny',

    'accountLink': None},

   'score': 4,

   'timeFromCreation': 'Aug 07, 2019'},

  {'rating': 'STAR_4',

   'review': 'I needed that 94 feel to it',

   'displayName': 'John G',

   'displayImageUrl': None,

   'isVerified': True,

   'isSuperReviewer': False,

   'hasSpoilers': False,

   'hasProfanity': False,

   'createDate': '2019-08-07T22:18:34.825Z',

   'updateDate': '2019-08-07T22:18:34.825Z',

   'user': {'userId': 'E919E659-B5E8-46DD-AF66-8BB1EFC3F789',

    'realm': 'Fandango',

    'displayName': 'John G',

    'accountLink': None},

   'score': 4,

   'timeFromCreation': 'Aug 07, 2019'},

  {'rating': 'STAR_5',

   'review': 'Awesome!! Can’t wait to watch again!',

   'displayName': 'Tammie Kinder',

   'displayImageUrl': None,

   'isVerified': True,

   'isSuperReviewer': False,

   'hasSpoilers': False,

   'hasProfanity': False,

   'createDate': '2019-08-07T22:18:13.872Z',

   'updateDate': '2019-08-07T22:18:13.872Z',

   'user': {'userId': 'FCC1D6D4-7557-4040-8BE8-D4A66E362DEA',

    'realm': 'Fandango',

    'displayName': 'Tammie Kinder',

    'accountLink': None},

   'score': 5,

   'timeFromCreation': 'Aug 07, 2019'},

  {'rating': 'STAR_5',

   'review': 'I love the movie. It was so much like the original & I loved that. I hate when they remake a movie and they change everything. Example-Winnie the Pooh. The little mermaid. ',

   'displayName': 'Nikki O',

   'displayImageUrl': 'https://graph.facebook.com/v3.3/513422926/picture',

   'isVerified': False,

   'isSuperReviewer': False,

   'hasSpoilers': False,

   'hasProfanity': False,

   'createDate': '2019-08-07T22:15:50.612Z',

   'updateDate': '2019-08-07T22:15:50.612Z',

   'user': {'userId': '968638814',

    'realm': 'RT',

    'displayName': 'Nikki O',

    'accountLink': '/user/id/968638814'},

   'score': 5,

   'timeFromCreation': 'Aug 07, 2019'},

  {'rating': 'STAR_5',

   'review': 'Loved it it was funny',

   'displayName': 'Patricia',

   'displayImageUrl': None,

   'isVerified': True,

   'isSuperReviewer': False,

   'hasSpoilers': False,

   'hasProfanity': False,

   'createDate': '2019-08-07T22:09:33.248Z',

   'updateDate': '2019-08-07T22:09:33.248Z',

   'user': {'userId': '573E6024-FF39-439B-BCA2-C0933C3BB72E',

    'realm': 'Fandango',

    'displayName': 'Patricia',

    'accountLink': None},

   'score': 5,

   'timeFromCreation': 'Aug 07, 2019'}]
from pandas.io.json import json_normalize

rev_df = pd.DataFrame.from_dict(json_normalize(reviews), orient='columns')
rev_df