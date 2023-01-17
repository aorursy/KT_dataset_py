# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



import matplotlib.pyplot as plt

import seaborn as sns



from subprocess import check_output

from datetime import time

print(check_output(["ls", "../input"]).decode("utf8"))
df = pd.read_csv("../input/crowdedness-at-the-campus-gym/data.csv")

df_graphs = pd.read_csv("../input/crowdedness-at-the-campus-gym/data.csv")

df.head()
df.describe()
del df['is_holiday']
#show the correlation between features

correlation = df.corr()

#create and display a graph figure

plt.figure(figsize=(10,10))

#We use the heatmap from sns libray and we choose the options(color, shape)

sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='viridis')

#Display the title of the graph

plt.title('Correlation between different fearures')
# Drop columns

# First we delete the date column because it doesn't fit our random forest regression study

df = df.drop("date", axis=1)

# Then we delete the timestamp as it is resulted from the heatmap

del df['timestamp']

# one hot encode categorical columns

columns = ["day_of_week", "month", "hour"]

# get_dumies is a function from pandas that converts categorical variable into dummy/indicator variables

df = pd.get_dummies(df, columns=columns)

df.head(10)
# Extract the training and test data

data = df.values

X = data[:, 1:]  # all rows, no label

y = data[:, 0]  # all rows, label only

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# Scale the data to be between -1 and 1

scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
# Establish model of random forest algorithm

model = RandomForestRegressor(n_jobs=-1)
# Try different numbers of n_estimators - this will take a minute or so

estimators = np.arange(10, 200, 10)

# Create an array of scores

scores = []



for n in estimators:

    model.set_params(n_estimators=n) # Choose the parameters

    model.fit(X_train, y_train) # Use the training datasets to build the forest of trees

    scores.append(model.score(X_test, y_test)) # Fill the array with predictions

plt.title("Effect of n_estimators")

plt.xlabel("n_estimator")

plt.ylabel("score")

plt.plot(estimators, scores)
scores
round(max(scores), 4)
g = df_graphs[['hour','number_people','day_of_week']] #dataframe with hour, number of people and the day of the week columns



F = g.groupby(['hour','day_of_week'], as_index = False).number_people.mean().pivot('day_of_week','hour','number_people').fillna(0) #resharp the dataframe with the mean of people number



grid_kws = {"height_ratios": (.9, .05), "hspace": .3}



dow= 'Monday Tuesday Wednesday Thursday Friday Saturday Sunday'.split() #splitting the string





ax = sns.heatmap(F, cmap='RdBu_r',cbar_kws={"orientation": "horizontal"}) #cmap = Red/Blue, colorbar horizontal

ax.set_yticklabels(dow, rotation = 0) # axis labels

ax.set_ylabel('')

ax.set_xlabel('Hour')



cbar = ax.collections[0].colorbar

cbar.set_label('Average Number of People')
def get_date(series):

    return series.str.slice(8,11) #get characters between 8th and 11th place
df_graphs['day_of_month'] = df_graphs[['date']].apply(get_date) #dataframe with only the day of the month from the date

month_date_count_df = pd.pivot_table(df_graphs, columns=['day_of_month'],index=['month'], values='number_people', aggfunc=np.mean) #reshape the dataframe

month_date_count_df.fillna(0, inplace=True)

fig, ax = plt.subplots(figsize=(18,7)) 

heatmap = sns.heatmap(month_date_count_df, annot=True, ax=ax, cmap="OrRd") #heatmap with the reshaped dataframe, with colors from Orange to Red

heatmap.set_ylabel('Month')

heatmap.set_xlabel('Day of week')