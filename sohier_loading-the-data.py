import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
mysteries = pd.read_csv("../input/Rosary_Mysteries_CSV.csv")
mysteries.head(3)
prayers = pd.read_csv("../input/Rosary_Prayers_CSV.csv")
prayers = pd.read_csv("../input/Rosary_Prayers_CSV.csv", encoding="ISO-8859-1")
prayers.head(3)