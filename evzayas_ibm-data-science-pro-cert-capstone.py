# usual imports & installs as necessary

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup # for HTML parsing
# get postal codes from wikipage & 
res = requests.get("https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M")
soup = BeautifulSoup(res,'lxml')
table = soup.find_all('table')[0]
df = pd.read_html(str(table))
df
