!pip install pandas --upgrade

import importlib

importlib.invalidate_caches()
import pandas as pd

!pip show pandas

print(pd.__version__) # NOT WORKING