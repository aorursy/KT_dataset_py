import pandas as pd



file_path = ""

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        file_path = os.path.join(dirname, filename)



        

data = pd.read_csv(file_path)
training_data = []

values = [ ]



for year in range(1970, 2010, 10):

    for index, row in data.iterrows():

        key = str(year) + " Population"

        training_data.append([row['CD Number'], year])

        values.append(row[key])
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor()

model.fit(training_data, values)



print ("Predict New York City's Population Of Community Districts By Years")

df = pd.DataFrame({'City': [], 'CD Number': [], 'CD Name': [], '1970': [], '1980': [],'1990':[], '2000':[], '2010':[], '2050': []})



for index, row in data.iterrows():

    no = row['CD Number']

    future = model.predict([[int(no), 2050]])

    df.loc[index] = [row['Borough']] + [row['CD Number']] + [row['CD Name']] + [row['1970 Population']] + [row['1980 Population']] + [row['1990 Population']] + [row['2000 Population']] + [row['2010 Population']] + [future[0]]

             

df

df = df.drop(columns=['CD Number'])

df.groupby('City').sum()