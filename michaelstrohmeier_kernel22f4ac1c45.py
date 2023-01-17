import numpy as np

import pandas as pd
df = pd.read_csv('../input/data.csv')
import matplotlib.pyplot as plt



temp = df['diagnosis'].value_counts()



green = (174/255, 255/255, 81/255)

red = (255/255, 45/255, 70/255)



plt.bar(1, temp[1], align = 'center', alpha = 0.80, color = red)

plt.bar(0, temp[0], align = 'center', alpha = 1.0, color = green)



plt.xticks([0, 1], ['B', 'M'])

plt.ylabel('Count')

plt.title('Patient Count')

 

plt.show()