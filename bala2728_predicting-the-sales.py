import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import gc

df_train = pd.read_csv("../input/train.csv")
df_train.head()
test  = pd.read_csv("../input/favorita-grocery-sales-forecasting/test.csv")

testg  = pd.read_csv("../input/favorita-grocery-sales-forecasting/test.csv")

store = pd.read_csv("../input/favorita-grocery-sales-forecasting/stores.csv")

holiday = pd.read_csv("../input/favorita-grocery-sales-forecasting/holidays_events.csv")

item = pd.read_csv("../input/favorita-grocery-sales-forecasting/items.csv")

oil = pd.read_csv("../input/favorita-grocery-sales-forecasting/oil.csv")

trans = pd.read_csv("../input/favorita-grocery-sales-forecasting/transactions.csv")