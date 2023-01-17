import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
from IPython.display import IFrame

df=pd.read_csv("../input/insurance.csv")
df.shape
df.isna().sum()

IFrame("https://public.tableau.com/views/IsBMIdependongender/IsBMIDependonGender?:embed=y&:showVizHome=no", width=1200, height=500)

IFrame("https://public.tableau.com/views/IsAgeHaveImpactonBMI/IsAgeHaveImpactonBMI?:embed=y&:showVizHome=no", width=1200, height=500)
#https://public.tableau.com/views/IsAgeHaveImpactonBMI/IsAgeHaveImpactonBMI?:embed=y&:display_count=yes&publish=yes
IFrame("https://public.tableau.com/views/Ageismajorfactorofcharges/Ageismajorfactorofcharges?:embed=y&:showVizHome=no", width=1200, height=500)
#https://public.tableau.com/views/Ageismajorfactorofcharges/Ageismajorfactorofcharges?:embed=y&:display_count=yes&publish=yes
IFrame("https://public.tableau.com/views/Smokeisdangerous/Smokeisdangerous?:embed=y&:showVizHome=no", width=1200, height=700)
#https://public.tableau.com/views/Smokeisdangerous/Smokeisdangerous?:embed=y&:display_count=yes&publish=yes
IFrame("https://public.tableau.com/views/BMIabove24notgood/BMIabove24notgood?:embed=y&:showVizHome=no", width=1200, height=700)
#https://public.tableau.com/views/BMIabove24notgood/BMIabove24notgood?:embed=y&:display_count=yes&publish=yes
