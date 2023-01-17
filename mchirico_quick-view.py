# First, we'll import pandas, a data processing and CSV file I/O library
import pandas as pd
import numpy as np
import datetime




# We'll also import seaborn, a Python graphing library
import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)

d = pd.read_csv("../input/2016-FCC-New-Coders-Survey-Data.csv")
d.head()