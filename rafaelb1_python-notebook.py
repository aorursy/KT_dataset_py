import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
dados = pd.read_csv("/kaggle/input/elixir-radar/Elixir Radar - Elixir Radar curated content.csv")



dados.head()
dados.info()
dados["Email publication date"] = pd.to_datetime(dados["Email publication date"], format = "%Y-%m-%d")



dados.info()
dados["Content type"].value_counts(normalize = True).mul(100)