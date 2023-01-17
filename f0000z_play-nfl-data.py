from bokeh.plotting import figure, output_file, show, output_notebook
import pandas as pd
import numpy as np

output_notebook()
nfl_data = pd.read_csv("../input/NFL Play by Play 2009-2017 (v4).csv")

nfl_data.sample(5)
