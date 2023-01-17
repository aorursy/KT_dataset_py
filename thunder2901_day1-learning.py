import pandas as pd

deliveries = pd.read_csv("../input/ipldata/deliveries.csv")

matches = pd.read_csv("../input/ipldata/matches.csv")
matches.shape
deliveries.shape
matches.head(3)
matches['city'][0:3]
import pandas as pd

StudentData = pd.read_csv("../input/studentdata/StudentData.csv")
StudentData.head(10)
StudentData.sort_values('roll',0)
%matplotlib inline

import matplotlib.pyplot as plt



plt.hist(StudentData['marks(out of 100)'],color='g')

plt.xlabel('marks out of 100')

plt.ylabel("Number of Students")
matches.isnull().any()
y = matches['winner']

X = matches.drop(['winner'],axis=1)
import seaborn as sns

matches.boxplot()