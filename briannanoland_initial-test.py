import numpy as np
import pandas as pd
from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))
prim_dat = pd.read_csv('../input/CLIWOC15.csv')
# Remove duplicate rows
prim_dat.describe()
#prim_dat.shape
prim_dat = prim_dat.loc[prim_dat.Duplicate == 0]
life_true = prim_dat.loc[prim_dat.LifeOnBoard == 1]
#list(life_true.columns)
temp = life_true[['LogbookLanguage',
                  'EnteredBy',
                  'VoyageFrom',
                  'VoyageTo',
                  'ShipName',
                  'ShipType',
                  'Company',
                  'OtherShipInformation',
                  'Nationality',
                  'Name1',
                  'Rank1',
                  'Name2', 
                  'Rank2',
                  'Name3',
                  'Rank3',
                  'DistToLandmarkUnits',
                  'DistTravelledUnits',
                  'LongitudeUnits',
                  'VoyageIni',
                  'UnitsOfMeasurement',
                  'Calendar',
                  'Year',
                  'Month',
                  'Day',
                  'DayOfTheWeek',
                  'PartDay',
                  'Distance',
                  'LatDeg',
                  'LongDeg',
                  'EncName',
                  'EncNat',
                  'EncRem',
                  'Anchored',
                  'AnchorPlace']]
temp.describe()