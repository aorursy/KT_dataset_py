import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt # data visualization
from sklearn.cluster import KMeans # clustering algorithm
from sklearn import metrics # evaluate the model
from scipy.spatial.distance import cdist # compute distance between 2 vecots (centroids and datapoints)
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv('../input/train.csv')
data1 = pd.read_csv('../input/test.csv')
train = pd.DataFrame(data)
test = pd.DataFrame(data1)
train.head(3)
train["Date"] = train.Dates.str.split('-')
train["Date"] = train.Date.str[0]
train.drop(columns={'Dates'},inplace = True)
# Useful Insights
pd.Series([0,train.Category.count(),train.Category.value_counts().index[0],train.Date.value_counts().index[0],train.DayOfWeek.value_counts().index[0]],index=['NaN Values','Number Of Crimes','Most Occured Crime','Black Year','Black Day'])
# Full Summary
train.describe(include='all').fillna(0)
# We won't need the following columns : Descript, X, Y beacause our goal is not about visualization
train.drop(columns={"Descript",'X','Y'}, inplace = True)
# When does crimes in state X often occur ?
_, ax = plt.subplots()
ax.bar(train.PdDistrict.str[0:2].sample(100), train.Date.sample(100))

ax.set_title("Year of Crime Occurences per States")
ax.set_xlabel("State")
ax.set_ylabel("Year")
# Drop the label columns
train.drop(columns={"Category",'Resolution'},inplace = True)
# Applying elbow method for finding the optimal number of k_clusters

for column in train.columns:
    if train[column].dtype == type(object):
        le = LabelEncoder()
        train[column] = le.fit_transform(train[column])

features = ["DayOfWeek", "PdDistrict"]
X_train = train[features]
model = KMeans()
model.fit(X_train)
for column in test.columns:
    if test[column].dtype == type(object):
        le = LabelEncoder()
        test[column] = le.fit_transform(test[column])
pred = model.predict(test.loc[:,["DayOfWeek",'PdDistrict']])
prediction = pd.DataFrame(data=pred,columns=['Predicition'])
prediction.to_csv('test.csv',index=False)