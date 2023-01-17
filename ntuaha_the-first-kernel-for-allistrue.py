import pandas as pd
import numpy as np
!  ls ../input/esuntoygame2018
output = pd.read_csv("../input/allistruecsv/allistrue.csv")
output.to_csv('./output.csv',index=False)