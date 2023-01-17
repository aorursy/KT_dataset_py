import pandas as pd
path = r'../input/chopstick-effectiveness.csv'

dataFrame = pd.read_csv(path)

dataFrame
dataFrame['Food.Pinching.Efficiency'].mean()
meansByLength = dataFrame.groupby('Chopstick.Length')['Food.Pinching.Efficiency'].mean().reset_index()

meansByLength
# Causes plots to display within the notebook rather than in a new window

%pylab inline



import matplotlib.pyplot as plt



plt.scatter(x=meansByLength['Chopstick.Length'], 

            y=meansByLength['Food.Pinching.Efficiency'])

            # title="")

plt.xlabel("Length in mm")

plt.ylabel("Efficiency in PPPC")

plt.title("Average Food Pinching Efficiency by Chopstick Length")

plt.show()