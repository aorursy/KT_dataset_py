import pandas as pd

import numpy as np

import requests

import json

import requests

from datetime import datetime

from bs4 import BeautifulSoup
fifa = pd.read_csv('../input/FIFA19 - Ultimate Team players.csv', low_memory=False)
fifa.head()
fifa['player_ID'].duplicated().sum()