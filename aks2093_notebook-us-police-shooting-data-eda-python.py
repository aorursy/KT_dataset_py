# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.express as px
from kmodes.kmodes import KModes
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
import matplotlib.pyplot as plt


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
        data = pd.read_csv(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data.columns
data.dtypes
print("Total number of rows are: {}".format(data.__len__()))
print("Total unique Ids are: {}".format(data["id"].unique().__len__()))
print("Total unique names are: {}".format(data["name"].unique().__len__()))
print("Total unique dates are: {}".format(data["date"].unique().__len__()))

data.describe()
data.head()
#1. "race" and "signs_of_mental_illness"

crosstab1 = pd.crosstab(data["race"], data["signs_of_mental_illness"])
print(crosstab1.columns)
print(crosstab1.index)
print("\n")
print(crosstab1)

crosstab1.plot.bar(stacked=False)
plt.legend(title='signs_of_mental_illness')
plt.show()
#2. "signs_of_mental_illness" and "threat_level"

pd.crosstab(data["signs_of_mental_illness"], data["threat_level"])
#3. "arms_category" and "race"

pd.crosstab(data["arms_category"], data["race"])
#4. "body_camera" and "race"

pd.crosstab(data["body_camera"], data["race"])
#5. "race" and "flee"

pd.crosstab(data["race"], data["flee"])
data["manner_of_death"].unique()
data['year'] = pd.DatetimeIndex(data['date']).year
data['month'] = pd.DatetimeIndex(data['date']).month
print(data["year"].unique())
print(data["month"].unique())
print(data.columns)
print(data.columns.__len__())
print(data.groupby(["month"])["id"].count())
month_wise_values = data.groupby(["month"])["id"].count().values.tolist()
month_df = pd.DataFrame({"month":data.month.unique().tolist(), "count": month_wise_values})

print(month_wise_values)
fig = px.pie(month_df, values="count",  names='month')
fig.show()
print(data.groupby(["year"])["id"].count())
year_wise_values = data.groupby(["year"])["id"].count().values.tolist()
year_df = pd.DataFrame({"year":data.year.unique().tolist(), "count": year_wise_values})

print(year_wise_values)
fig = px.pie(year_df, values="count",  names='year')
fig.show()
pd.crosstab(data["year"], data["month"], margins=True)
process_data = data.copy()
columns = ['manner_of_death', 'armed', 'age', 'gender',
       'race', 'city', 'state', 'signs_of_mental_illness', 'threat_level',
       'flee', 'body_camera', 'arms_category', 'year', 'month']

#Here we are taking age<=73, the reason is explained in the next steps

process_data = process_data[data["age"]<=73]

process_data['age_bin'] = pd.cut(process_data['age'], [0, 20, 40, 50, 60, 73], labels=['0-20', '20-40','40-50','50-60','60-73'])
variable_pairs = []
for i in range(len(columns)):
    for j in range(i, len(columns)):
        variable_pairs.append((columns[i], columns[j]))

dependent_variables = []
for pair in variable_pairs:
    column1 = pair[0]
    column2 = pair[1]

    obs = pd.crosstab(process_data[column1], process_data[column2])
    if np.all(obs>5):
        stat, p, dof, expected = chi2_contingency(obs)
        if p<=0.005:
            dependent_variables.append((column1, column2))
            print("Processing columns: {} , {}".format(column1, column2))
            print("p-value: {}".format(p))
            print(obs)
            print("\n")
            print("\n")
print(dependent_variables)
fig = px.box(data, y="age")
fig.show()
cluster_data = data[data["age"]<=73]
cluster_data.columns
fig = px.histogram(cluster_data, x="age")
fig.show()
cluster_data['age_bin'] = pd.cut(cluster_data['age'], [0, 20, 40, 50, 60, 73], labels=['0-20', '20-40','40-50','50-60','60-73'])

fig = px.histogram(cluster_data, x="age_bin")
fig.show()
cluster_data = cluster_data.drop(columns=["id",	"name",	"date", "age","city"], axis=1)
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
cluster_data = cluster_data.apply(le.fit_transform)
cluster_data.head()
number_of_clusers = 10
cost = []
for num_clusters in list(range(1,number_of_clusers)):
    kmode = KModes(n_clusters=num_clusters, init = "Cao", n_init = 1, verbose=1)
    kmode.fit_predict(cluster_data)
    cost.append(kmode.cost_)

x_axis = np.array([i for i in range(1,number_of_clusers,1)])

fig = px.line(x=x_axis, y=cost)
fig.show()
#So lets take number of clusters 4

num_clusters = 4
kmode = KModes(n_clusters=num_clusters, init = "Cao", n_init = 1, verbose=1)
arr = kmode.fit_predict(cluster_data)
cluster_data["cluster"] = arr
cluster_data.columns
columns = ['manner_of_death', 'armed', 'gender', 'race', 'state',
       'signs_of_mental_illness', 'threat_level', 'flee', 'body_camera',
       'arms_category', 'year', 'month', 'age_bin']
       
target = "cluster"

dependent_variables = []
for column in columns:

    obs = pd.crosstab(cluster_data[column], cluster_data[target])
    if np.all(obs>5):
        stat, p, dof, expected = chi2_contingency(obs)
        if p<=0.005:
            dependent_variables.append(column)
            print("Processing columns: {} , {}".format(column, target))
            print("p-value: {}".format(p))
            print(obs)
            print("\n")
            print("\n")

print(dependent_variables)