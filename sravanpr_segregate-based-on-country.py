import numpy as np

import pandas as pd



import os



print(os.listdir("../input/icc-test-cricket-runs/"))



a = pd.read_excel('../input/icc-test-cricket-runs/ICC Test Bat 3001.xlsx')



s = a.loc[:,['Player']]



country = ((s.Player.str.split('(').str[1]).str.split(')').str[0])



country = country.rename('country')



df1 = pd.concat([s,country], axis=1)



df2 = df1.groupby('country')


