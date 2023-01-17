import pandas as pd

deliveries = pd.read_csv("../input/ipldata/deliveries.csv")

matches = pd.read_csv("../input/ipldata/matches.csv")
deliveries.isnull().any()



matches.boxplot(by=['winner'])
matches['city'][0:3]
import pandas as pd

std = pd.read_csv("../input/studentsdata/students.csv")

std.sort_values('roll',0)

import matplotlib.pyplot as plt

plt.hist(std['marks(out of 100)'],color='r')

plt.xlabel('Marks(Out of 100)')

plt.ylabel('Number of Students')