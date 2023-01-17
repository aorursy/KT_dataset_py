import numpy as np

import pandas as pd



comps = pd.read_csv("../input/Competitions.csv")

comps[comps["DataSizeBytes"]>(10**10)][["CompetitionName", "DataSizeBytes"]].plot.barh(x="CompetitionName", figsize=(8,5))