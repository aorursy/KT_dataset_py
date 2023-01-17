# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Handle table-like data and matrices
import numpy as np
import pandas as pd

# Visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

# Input data files are available in the "../input/" directory.
import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.
#Dataset: 2016 Diversity Survey for Silicon Valley
#Read dataset locally,to data variable
data = pd.read_csv("../input/cleaneddata/CleanedData.csv")
#Visual sample of datase, data is fairly straightforward
data.head()
#Make Catergorical Data Objects Binary using Pandas - See https://www.kaggle.com/dansbecker/using-categorical-data-with-one-hot-encoding
#data.dtypes.sample(4)
encoded_data = pd.get_dummies(data)
#new heads
encoded_data.head()
#Number calculations
total=encoded_data['count'].sum()
totalMales=encoded_data.groupby('gender_male')['count'].sum()[1]
totalFemales=encoded_data.groupby('gender_female')['count'].sum()[1]
totalExecs=encoded_data.groupby('job_category_Executive/Senior officials & Mgrs')['count'].sum()[1]
maleExecs=encoded_data.groupby(['job_category_Executive/Senior officials & Mgrs','gender_male'])['count'].sum()[1][1]
totalWhite=encoded_data.groupby('race_White')['count'].sum()[1]
totalNonWhite=total-totalWhite
whiteExecs=encoded_data.groupby(['job_category_Executive/Senior officials & Mgrs','race_White'])['count'].sum()[1][1]
whiteMaleExecs=encoded_data.groupby(['job_category_Executive/Senior officials & Mgrs','race_White','gender_male'])['count'].sum()[1][1][1]
print("---OVERALL STATS---")
print("There are a total of", total, "people suerveyed.\nOf these", totalMales,
      "or", (totalMales/total)*100,"% are male.\nThere are", totalFemales,"or", 
      (totalFemales/total)*100,"% are female")
print("Of the workforce", totalWhite, "or", (totalWhite/total)*100,"% are white\n")

print("---EXECUTIVE POSITIONS---",
      "\nThere are",totalExecs,"Executive or Senior Official positions.\nOf these", 
      maleExecs, "or", (maleExecs/totalExecs)*100, "% are males.")

print("Of all executives", whiteExecs, "or",(whiteExecs/totalExecs)*100,"are white"
      "\nand", whiteMaleExecs,"or",(whiteMaleExecs/totalExecs)*100,"% are both white and male")

