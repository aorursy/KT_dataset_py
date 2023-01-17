import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
names = pd.read_csv("../input/dummyfile/csv dummy fille - Sheet1.csv")

names.head()

names.set_index("LastN").head()
