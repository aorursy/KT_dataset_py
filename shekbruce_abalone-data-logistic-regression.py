# Importing helpful package to load and hadle our data

import pandas as pd

import numpy as np



# import package for visualization

import matplotlib.pyplot as plt

import seaborn as sns



# Inline Priting of Visualizations

sns.set()

%matplotlib inline



# flexible and easy-to-use missing data visualizations

import missingno as msno



#  Import SK-Learn Library

from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import mean_absolute_error , mean_squared_error, r2_score , classification_report

from sklearn.preprocessing import StandardScaler

from scipy.stats import skew



# Ignore warnings :

import warnings

warnings.filterwarnings('ignore')
# Import data

#  Get data

abalone_data = pd.read_csv('../input/abalone.csv')

# Get glimpse of data

abalone_data.head()
# As Rings: +1.5 gives the age in years , So we will replace rings with age

abalone_data['Age'] = abalone_data['Rings'] + 1.5

abalone_data.head(5)
#  Now we will drop Rings

abalone_data.drop('Rings', axis=1, inplace=True)

abalone_data.head()
# Get information about our dat frame

abalone_data.info()
# confirm null

abalone_data.isnull().sum()
# Visualize missing data 

msno.dendrogram(abalone_data)

plt.show()
# Hence no null value, now we can do data visualization now look at description

abalone_data.describe()
#  Here we have height == 0 , but height(With meat in shell) can't be zero as its one parameter of dimension and 

# Shucked weight(Weight of meat) has some value

# But Sex say 'Infant', so it can be possible that infant has height equals to zero



# Check rows with height = 0 

abalone_data.loc[abalone_data['Height'] == 0]
#  count number of rows and percentage having height = 0

print("No. of rows with height == 0 is {}".format((abalone_data['Height'] == 0).sum()))

print("Percentage of rows with height == 0 is {0:.4f}%".format(((abalone_data['Height'] == 0).sum() * 100 )/len(abalone_data)))
# As we have only 2 rows with 0 height, so we will drop these two rows

abalone_data = abalone_data[abalone_data['Height'] != 0]
# Confirm 0 height row

print("No. of rows with height == 0 is {}".format((abalone_data['Height'] == 0).sum()))
# Correlation between all features

# Pandas dataframe.corr() is used to find the pairwise correlation of all columns in the dataframe.

# Any na values are automatically excluded. For any non-numeric data type columns in the dataframe it is ignored.



fig,ax = plt.subplots()

fig.set_size_inches(15,7)

sns.heatmap(abalone_data.corr(),annot=True,ax=ax)

plt.show()
# Length

fig,ax = plt.subplots()

fig.set_size_inches(15,5)

sns.distplot(abalone_data['Length'],hist_kws={'edgecolor':'black'})

plt.show()
# Relation of Length vs rings

fig,ax = plt.subplots()

fig.set_size_inches(15,5)

sns.scatterplot(x="Length", y="Age",data = abalone_data,ax=ax)

plt.show()
# Diameter

fig,ax = plt.subplots()

fig.set_size_inches(15,5)

sns.distplot(abalone_data['Diameter'],hist_kws={'edgecolor':'black'})

plt.show()
# Relation of Diameter vs rings

fig,ax = plt.subplots()

fig.set_size_inches(15,5)

sns.scatterplot(x="Diameter", y="Age",data = abalone_data,ax=ax)

plt.show()
# Height

fig,ax = plt.subplots()

fig.set_size_inches(15,5)

sns.distplot(abalone_data['Height'],hist_kws={'edgecolor':'black'})

plt.show()
# Relation of Height vs rings

fig,ax = plt.subplots()

fig.set_size_inches(15,5)

sns.scatterplot(x="Height", y="Age",data = abalone_data,ax=ax)

plt.show()
# Whole weight

fig,ax = plt.subplots()

fig.set_size_inches(15,5)

sns.distplot(abalone_data['Whole weight'],hist_kws={'edgecolor':'black'})

plt.show()
# Relation of Whole weight vs rings

fig,ax = plt.subplots()

fig.set_size_inches(15,5)

sns.scatterplot(x="Whole weight", y="Age",data = abalone_data,ax=ax)

# Shucked weightplt.show()
# Shucked weight

fig,ax = plt.subplots()

fig.set_size_inches(15,5)

sns.distplot(abalone_data['Shucked weight'],hist_kws={'edgecolor':'black'})

plt.show()
# Relation of Shucked weight vs rings

fig,ax = plt.subplots()

fig.set_size_inches(15,5)

sns.scatterplot(x="Shucked weight", y="Age",data = abalone_data,ax=ax)

plt.show()
# Viscera weight

fig,ax = plt.subplots()

fig.set_size_inches(15,5)

sns.distplot(abalone_data['Viscera weight'],hist_kws={'edgecolor':'black'})

plt.show()
# Relation of Shucked weight vs rings

fig,ax = plt.subplots()

fig.set_size_inches(15,5)

sns.scatterplot(x="Viscera weight", y="Age",data = abalone_data,ax=ax)

plt.show()
# Shell weight

fig,ax = plt.subplots()

fig.set_size_inches(15,5)

sns.distplot(abalone_data['Shell weight'],hist_kws={'edgecolor':'black'})

plt.show()
# Relation of Shell weight vs rings

fig,ax = plt.subplots()

fig.set_size_inches(15,5)

sns.scatterplot(x="Shell weight", y="Age",data = abalone_data,ax=ax)

plt.show()
# Sex

labels = abalone_data.Sex.unique().tolist()

sizes = abalone_data.Sex.value_counts().tolist()

explode = (0.1, 0, 0)

plt.pie(sizes, explode=explode, labels=labels,autopct='%1.1f%%', shadow=True, startangle=90)

plt.axis('equal')

plt.title("Percentage of Sex Categories")



plt.plot()

fig=plt.gcf()

fig.set_size_inches(6,6)

plt.show()
fig, ax = plt.subplots()

fig.set_size_inches(10, 5)

# sns.stripplot(x='Sex',y='Age',data=abalone_data,jitter=True,ax=ax)

sns.swarmplot(x = 'Sex', y = 'Age', data = abalone_data, hue = 'Sex')

sns.boxenplot(x='Sex',y='Age',data=abalone_data,ax=ax)

plt.show()

# jitter to bring out the distribution of values

# jitter option, a small amount of random noise is added to the vertical coordinate

# Jitter is a random value (or for our purposes pseudo-random) that is assigned to the dots to separate

# them so that they aren't plotted directly on top of each other
one_hot_encoders_abalone_df =  pd.get_dummies(abalone_data)

cols = one_hot_encoders_abalone_df.columns

abalone_clean_data = pd.DataFrame(one_hot_encoders_abalone_df,columns= cols)

abalone_clean_data.head(1)
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

scaled_data =  pd.DataFrame(

    sc_X.fit_transform(abalone_clean_data[['Length','Diameter','Height','Whole weight','Shucked weight','Viscera weight']]),

                           columns=['Length','Diameter','Height','Whole weight','Shucked weight','Viscera weight'],

                           index=abalone_clean_data.index)
scaled_data.head()
fig,ax = plt.subplots()

fig.set_size_inches(15,7)

sns.heatmap(scaled_data.corr(),annot=True,ax=ax)

plt.show()
# When deep=True (default), a new object will be created with a copy of the calling object’s data and indices.

# Modifications to the data or indices of the copy will not be reflected in the original object

abalone_clean_data_standard = abalone_clean_data.copy(deep=True)

abalone_clean_data_standard[['Length','Diameter','Height','Whole weight','Shucked weight','Viscera weight']] = scaled_data[['Length','Diameter','Height','Whole weight','Shucked weight','Viscera weight']]
abalone_clean_data_standard.head()
x = abalone_clean_data_standard.drop(["Age"],axis=1)

y = abalone_clean_data_standard.Age

# y is float value and we will categorize ouput in two categories 0 and 1

y = np.where(y > 10,1,0)
train_x, test_x, train_y, test_y = train_test_split(x, y,random_state = 23,test_size=0.3)
train_y
logreg = LogisticRegression()

logreg.fit(train_x,train_y)

y_pred = logreg.predict(test_x) 
print("accuracy: "+ str(accuracy_score(test_y,y_pred)*100) + "%")

print("Mean absolute error: {}".format(mean_absolute_error(test_y,y_pred)))

print("Mean squared error: {}".format(mean_squared_error(test_y,y_pred)))

print("R2 score: {}".format(r2_score(test_y, y_pred)))

print("intercept: {}".format(logreg.intercept_))
print(classification_report(test_y, y_pred))