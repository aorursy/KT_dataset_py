# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns
house_data = pd.read_csv("../input/melbourne-housing-market/Melbourne_housing_FULL.csv")
house_data.head()
house_data.isnull().sum()
house_data.index
house_data.columns
sns.heatmap(house_data.isnull(),yticklabels=False,cbar=False,cmap="YlGnBu")
house_data['Price'].fillna(house_data['Price'].mean(),inplace=True)
house_data.info()
house_data['Bedroom2'].fillna(house_data['Bedroom2'].mean(),inplace=True)
house_data['Bathroom'].fillna(house_data['Bathroom'].mean(),inplace=True)
house_data.drop('Car',axis=1,inplace=True)
house_data.columns
house_data['Landsize'].fillna(house_data['Landsize'].mean(),inplace=True)
house_data.drop('BuildingArea',axis=1,inplace=True)
house_data['YearBuilt'].fillna(house_data['YearBuilt'].mean(),inplace=True)
house_data['Lattitude'].fillna(house_data['Lattitude'].mean(),inplace=True)

house_data['Longtitude'].fillna(house_data['Longtitude'].mean(),inplace=True)
house_data['Regionname'].value_counts()
house_data['Regionname'].fillna(house_data['Regionname'].mode()[0],inplace=True)
house_data.isnull().sum()
house_data.dropna(inplace=True)
house_data.shape
house_data.isnull().sum()
house_data.head()
house_data['Type'].value_counts()
house_data.loc[house_data['Price'].idxmax()]
house_data.loc[house_data['Price']==house_data['Price'].min()].head()
plt.figure(figsize=[10,10])

sns.barplot(x="Rooms",y="Price",data=house_data,hue="Type")
from wordcloud import WordCloud
wordcloud = WordCloud(width = 1000, height = 800,  background_color ='white',  max_words=200,max_font_size=200 ,).generate("".join(house_data['Suburb'])) 

  

# plot the WordCloud image                        

plt.figure(figsize = (10, 10), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

  

plt.show() 
sns.set_style("whitegrid")

sns.FacetGrid(house_data,height=6).map(plt.scatter,'Rooms','Price').add_legend()

plt.show()
sns.set_style("whitegrid")

sns.FacetGrid(house_data,height=6).map(plt.scatter,'Bedroom2','Price').add_legend()

plt.show()
sns.set_style("whitegrid")

sns.FacetGrid(house_data,height=6).map(plt.scatter,'Bathroom','Price').add_legend()

plt.show()

sns.set_style("whitegrid")

sns.FacetGrid(house_data,height=6).map(plt.scatter,'Landsize','Price').add_legend()

plt.show()
plt.figure(figsize=[10,10])

sns.barplot(x="Rooms",y="Price",data=house_data)
sns.countplot(house_data['Type'])
data =house_data.corr()
plt.figure(figsize=[8,6])

sns.heatmap(data,annot=True)
plt.figure(figsize=[8,8])

house_data['Suburb'].value_counts().head(5).plot.bar()
house_data.columns
X= house_data[[ 'Rooms',  'Distance', 'Postcode', 'Bedroom2', 'Bathroom', 'Landsize',

       'YearBuilt', 'Lattitude', 'Longtitude', 

       'Propertycount']]

y= house_data['Price']
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.4)
lm =LinearRegression()
lm.fit(X_train,y_train)
print(lm.intercept_)
lm.coef_
X_train.columns
cdf = pd.DataFrame(lm.coef_,X.columns,columns=['coef'])
cdf
prediction = lm.predict(X_test)
prediction
y_test.head()
plt.scatter(y_test,prediction)
sns.distplot(y_test-prediction)
final_output = pd.DataFrame({'Actual':y_test,'prdict':prediction})
print(final_output.head())