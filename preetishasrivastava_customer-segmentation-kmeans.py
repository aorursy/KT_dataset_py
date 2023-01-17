# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from sklearn.cluster import KMeans
df= pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
df.head()  # To view the top few rows of the dataset.
df.info()  # To view thw information about the dataset
# Now we can drop the 'CustomerID' column as it is unique to every customer and cannot be used for any clustering.

df= df.drop('CustomerID', axis=1)

df.head(3)
sns.countplot(df['Gender'])
sns.distplot(df['Age']) # this will show the distribution of the age column.
sns.boxplot(df['Gender'], df['Spending Score (1-100)']) # This will show us if any outliers are present or not.

# Also, boxplot shows us the max, min and the 3 quartile ranges present in our dataset.
# In the above geaph, we find that there are no outliers present in the 'Spending Score(1-100)' column.



# Now let's check the same for 'Annual Income(k$)'.

sns.boxplot(df['Gender'], df['Annual Income (k$)'])
# We find that there is only 1 outlier point. So that can be ignored.
df['Gender_Male']= pd.get_dummies(df['Gender'], drop_first=True)

df.head()
df= df.drop('Gender', axis=1)

df.head()
x= df[['Age','Annual Income (k$)', 'Gender_Male']]

y= df['Spending Score (1-100)']

data=pd.DataFrame({'x1':df['Annual Income (k$)'],'x2':df['Gender_Male'],'y': y})
wcss=[]  # Empty list to store the value while plotting it on the graph.



# Since we do not know the exact value of 'K', we will select a range of values, let's say from 1-10.

# So the 'wcss' list will contain the sum of squares for 10 different values of K. Based on the graph, we will know the exact value of K

for i in range(1, 11):

    kmeans = KMeans(n_clusters = i, init = 'k-means++', 

                    max_iter = 300, n_init = 10, random_state = 42)

    kmeans.fit(data)

    wcss.append(kmeans.inertia_)

    

# Plotting the results onto a line graph, 

# allowing us to observe 'The elbow'

plt.figure(figsize=(10,5))

plt.plot(range(1, 11), wcss)

plt.title('The elbow method', fontweight="bold")

plt.xlabel('Number of clusters(K)')

plt.ylabel('within Clusters Sum of Squares(WCSS)') # Within cluster sum of squares
# Applying kmeans to the dataset / Creating the kmeans classifier

km= KMeans(n_clusters=5)

km.fit(data)
yp=km.predict(data) #Prediction of value based on the data.



plt.scatter(data['x1'],data['y'],c=yp)

plt.title("Clustering customers based on Annual Income and Spending score", fontsize=15,fontweight="bold")

plt.xlabel("Annual Income")

plt.ylabel("Spending Score")