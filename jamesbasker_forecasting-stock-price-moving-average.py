import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



df = pd.read_csv("../input/GOOG_032016_032017.csv")

print(df)
import matplotlib.pyplot as plt

ax = df['Close'].plot(title="Moving average", fontsize=12)

moving_average = df['Close'].rolling(20).mean()

moving_average.plot(label='Rolling mean', ax=ax)

plt.show()
import matplotlib.pyplot as plt

ax = df['Close'].plot(title="Moving average", fontsize=12)

moving_average = df['Close'].rolling(3).mean()

moving_average.plot(label='Rolling mean', ax=ax)

plt.show()