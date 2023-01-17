import numpy as np

import pandas as pd



df = pd.read_csv('/kaggle/input/nfl-2019-predictions/predictions_2019.csv')

df.head()

predictions = np.dot(np.diff(df.values[:, 1:-2], axis=1), np.arange(-98, 100, 1).reshape(-1,1)).reshape(-1).round()

stat = df.groupby(predictions)['Yards'].agg({"avg":np.mean, "num":len})

stat.head()
import matplotlib.pyplot as plt



plt.style.use(['default'])

fig = plt.figure(figsize = (7, 7))



plt.plot([-5, 20], [-5, 20], color='black', linestyle='--', linewidth=0.5)

plt.scatter(y=stat['avg'].values, x=stat.index, s=stat['num'].values*0.1)



plt.xlim((None, 20))

plt.ylim((None, 20))



plt.xlabel('Predicted yardage')

plt.ylabel('Observed yardage')



plt.show()