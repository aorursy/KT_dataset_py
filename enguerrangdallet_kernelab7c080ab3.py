"chart full-width"
import requests

import lxml.html as lh

import pandas as pd
df = pd.read_html("https://www.imdb.com/chart/top/?ref_=nv_mv_250")[0]
df[0]