import pandas as pd

read_html=pd.read_html("https://www.imdb.com/chart/top/")   # pandas powerful code line for reading tables in html pages

data=read_html[0]                                           # read_html returns list item. First item of that list returns pandas dataframe

data