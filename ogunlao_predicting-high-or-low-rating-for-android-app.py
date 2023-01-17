# Let's import the necessary tools
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #for data visualization
import seaborn as sns
# Input data files are available in the "../input/" directory.
import os
print(os.listdir("../input"))
# Get in the data
# I am using only the googleplaystore.csv file in this kernel
path = '../input'
play_store_data = pd.read_csv(path + "/googleplaystore.csv")
play_store_data.head(10)
play_store_data.shape
play_store_data.info()
# Starting with the easiest.
# Convert Reviews to numeric
play_store_data['Reviews'] = pd.to_numeric(play_store_data.Reviews, errors = 'coerce')
play_store_data.info()
#Let's look closely at the apps in the data 
play_store_data.App.value_counts().head(20)
#Taking 3 sample Apps for exploration
play_store_data[play_store_data['App'].isin(['ROBLOX', 'Candy Crush Saga','Granny'])].sort_values(by='App')
# Sort App in Ascending order of reviews
play_store_data_sorted = play_store_data.sort_values(by = ['App', 'Reviews'], ascending = True)

#drops other duplicate entries keeping the App with the highest reviews
play_store_data_sorted.drop_duplicates('App',keep='last',inplace=True)
#Let's verify that duplicates has been removed
play_store_data_sorted.App.value_counts().head(10)
play_store_data_sorted.shape
# Let's check out the App categories
play_store_data_sorted.Category.value_counts()
# Drop the category named 1.9, unknown category
play_store_data_sorted[play_store_data_sorted['Category'] == '1.9']
play_store_data_sorted = play_store_data_sorted.drop([10472])
#Let's check for null values and start dealing with them.
play_store_data_sorted.isnull().sum()
play_store_data_sorted.dropna(axis = 0, inplace = True, subset = ['Rating'])
play_store_data_sorted.isnull().sum()
play_store_data_sorted.Size.value_counts()
#Convert non numeric values in App size to NAN
play_store_data_sorted['Size'][play_store_data_sorted['Size'] == 'Varies with device'] = np.nan

#Replace M with 1 million and k with 1 thousand
play_store_data_sorted['Size'] = play_store_data_sorted.Size.str.replace('M', 'e6')
play_store_data_sorted['Size'] = play_store_data_sorted.Size.str.replace('k', 'e3')

#convert column to numeric, dropping non numeric values
play_store_data_sorted['Size'] = pd.to_numeric(play_store_data_sorted['Size'], errors = 'coerce')
play_store_data_sorted.info()
play_store_data_sorted['Installs'].value_counts()
# To eliminate the '+' and ',' signs and convert to numeric
play_store_data_sorted['Installs'] = play_store_data_sorted.Installs.str.replace('+', '')
play_store_data_sorted['Installs'] = play_store_data_sorted.Installs.str.replace(',', '')

# Convert to numeric type
play_store_data_sorted['Installs'] = pd.to_numeric(play_store_data_sorted['Installs'], errors = 'coerce')
play_store_data_sorted['Installs'].value_counts()
#Get the bin levels
bin_array = play_store_data_sorted.Installs.sort_values().unique()
#convert to array
bins = [x for x in bin_array]

# Added 5 billion for the higher range of app installs
bins.append(5000000000)
#Create bins for Installs
play_store_data_sorted['Installs_binned'] = pd.cut(play_store_data_sorted['Installs'], bins)

# Digitize the bins for encoding
Installs_digitized = np.digitize(play_store_data_sorted['Installs'], bins = bins )

#Add to the data frame as a column
play_store_data_sorted = play_store_data_sorted.assign(Installs_d = pd.Series(Installs_digitized).values)
play_store_data_sorted.info()
play_store_data_sorted.describe()
#as most machine learning models do not work well with NA, I have to drop rows having them.
attributes = ['Category', 'Reviews', 'Size' , 'Installs_d','Rating']
psa = play_store_data_sorted[attributes].dropna().copy()
psa.shape
#convert ratings to high and low categories.
Rating_cat = dict()
for i in range(0,len(psa['Rating'])):
    if psa['Rating'].iloc[i] >= 3.5:
        Rating_cat[i] = 'High'
    else: Rating_cat[i] = 'Low'
        
#Add the categorical column to the data 
psa = psa.assign(Rating_cat = pd.Series(Rating_cat).values)
psa['Rating_cat'].value_counts()
#drop the Ratings column
psa = psa.drop(['Rating'], axis = 1)

#To encode the Ratings labels for learning
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
psa['Rating_cat'] = le.fit_transform(psa['Rating_cat'])
#To view the encoded labels
list(le.classes_)
#Applying One-Hot Encoding to the Categorical Column 'Category' and 'Installs_d'
psa_encode = pd.get_dummies(psa, columns= ['Category','Installs_d'])
print(psa_encode.columns)
X = psa_encode.drop(['Rating_cat'], axis = 1)
y = psa_encode['Rating_cat']
#Checking for correlation using heatmap
plt.figure(figsize=(20,15)) 

sns.heatmap(X.corr())
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
print('Training Set Score: {} \nTest Set Score: {}'.format(knn.score(X_train, y_train),knn.score(X_test, y_test) ))
# Looking for optimum value of n_neighbours for the dataset.
for i in range(1,7):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train);
    print('For n = {}, Test = {}, Train = {}'.format(i,knn.score(X_train, y_train),knn.score(X_test, y_test) ))
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=10, max_depth = 10, random_state=2)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
forest.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(forest.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(forest.score(X_test, y_test)))
