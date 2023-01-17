import pandas as pd

import seaborn as sns



DATA_PATH = '../input/iris/Iris.csv'



df = pd.read_csv(DATA_PATH)



df.head(1)
g = sns.scatterplot(x = 'SepalWidthCm', y = 'SepalLengthCm', hue = 'Species', data = df)
g = sns.scatterplot(x = 'SepalWidthCm', y = 'SepalLengthCm', hue = 'Species', data = df)

g.legend(loc = 'center left', bbox_to_anchor = (1, 0.5), ncol = 1)