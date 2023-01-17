# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = {}

for year in ["2015", "2016"]:

    filename = "../input/{}.csv".format(year)

    data[year] = pd.read_csv(filename)

data["2015"].head()
columns_to_drop = {"Standard Error", "Lower Confidence Interval", "Upper Confidence Interval"}

for year, df in data.items():

    columns_to_keep = [col for col in df.columns.tolist() if col not in columns_to_drop]

    data[year] = df[columns_to_keep]
data["2016"].corr()["Happiness Rank"].sort_values(ascending=True)
data["2016"].corr()["Happiness Score"].sort_values(ascending=False)
import matplotlib.pyplot as plt

cols = [

    ("Economy (GDP per Capita)", "blue"),

    ("Health (Life Expectancy)", "orange"),

    ("Family", "red")

]

for column, color in cols:

    label = column.split(" ")[0]

    plt.scatter(data["2016"]["Happiness Score"], data["2016"][column], label=label, c=color)

plt.legend()

plt.xlabel("Happiness Score")

plt.show()
data["2016"].shape
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split



model = LinearRegression()

train, test = train_test_split(data["2016"], test_size = 0.25)

features = [

 'Economy (GDP per Capita)',

 'Family',

 'Health (Life Expectancy)',

 'Freedom',

 'Trust (Government Corruption)',

 'Generosity',

 'Dystopia Residual'

]

model.fit(train[features], train["Happiness Score"])

predictions = model.predict(test[features])
# Let's compute our Root Mean Squared Error

from sklearn.metrics import mean_squared_error

rmse = np.sqrt(mean_squared_error(test["Happiness Score"], predictions))

rmse