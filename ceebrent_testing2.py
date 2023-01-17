
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import collections

train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
train1 = train.dropna(how="any",subset=["Cabin"])
surviveArray = np.array(train1[['Cabin','Survived']])
surviveArray = np.sort(surviveArray, axis=0)
surviveArray = pd.DataFrame(surviveArray)
surviveArray
cabinCountsNumbers = cabinNumbersDF['a'].value_counts()
cabinCountsLetters = cabinLettersDF['a'].value_counts()
                              
cabinCountsNumbersSur = cabinNumbersSurDF['a'].value_counts()
cabinCountsLettersSur = cabinLettersSurDF['a'].value_counts()
cabinCountsNumbers
ax = cabinCountsNumbers.plot.bar()
cabinCountsNumbersSur.plot.bar(ax=ax,color="red")
cabinCountsLetters.plot.bar()
traincabin2 = train1[['Cabin','Name']]
