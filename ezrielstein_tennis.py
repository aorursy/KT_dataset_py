import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import Counter

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/tennis-20112019/wta_picks.csv")
resultats = Counter(df.Result.map(lambda x : len(x.split())))
resultats
"Il y a  %f %% de jeux en 3 sets et %f %% en 2."%(100 * resultats[3] / len(df), 100 * resultats[2] / len(df))
