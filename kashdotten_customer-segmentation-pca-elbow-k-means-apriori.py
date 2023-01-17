import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

#Standard data science libraries.

import pandas as pd

import numpy as np



from sklearn.cluster import KMeans #for kmeans algorithm



#For dimensionality reduction.

from sklearn.decomposition import PCA #pca from decomposition module.

from sklearn.preprocessing import StandardScaler

from sklearn import decomposition #decomposition module



#Plotting params.

%matplotlib inline

import matplotlib.pyplot as plt

from pylab import rcParams

import seaborn as sb

rcParams['figure.figsize'] = 12, 4

sb.set_style('whitegrid')



np.random.seed(42) # set the seed to make examples repeatable
#Since the files are zipped, they need to be imported with the following approach. 



prior = "order_products__prior.csv"

order_train = "order_products__train.csv"

orders = "orders.csv"

products = "products.csv"

aisles = "aisles.csv"

departments = "departments.csv"
import zipfile # Unzips the files

from subprocess import check_output    



#Prior Dataset

with zipfile.ZipFile("/kaggle/input/instacart-market-basket-analysis/"+prior+".zip","r") as z:

    z.extractall(".")

prior = pd.read_csv("order_products__prior.csv")



#Order_Train Dataset.

with zipfile.ZipFile("/kaggle/input/instacart-market-basket-analysis/"+order_train+".zip","r") as z:

    z.extractall(".")

order_train = pd.read_csv("order_products__train.csv")



#Orders Dataset.

with zipfile.ZipFile("/kaggle/input/instacart-market-basket-analysis/"+orders+".zip","r") as z:

    z.extractall(".")

orders = pd.read_csv("orders.csv")



#Products

with zipfile.ZipFile("/kaggle/input/instacart-market-basket-analysis/"+products+".zip","r") as z:

    z.extractall(".")

products = pd.read_csv("products.csv")



#Aisles

with zipfile.ZipFile("/kaggle/input/instacart-market-basket-analysis/"+aisles+".zip","r") as z:

    z.extractall(".")

aisles = pd.read_csv("aisles.csv")



#Departments

with zipfile.ZipFile("/kaggle/input/instacart-market-basket-analysis/"+departments+".zip","r") as z:

    z.extractall(".")

departments = pd.read_csv("departments.csv")
# Inspect all the dataframes, join them and make a combined df to form clusters. 
#Put them in a list to print shape.

combined_df_list = [products,orders, departments, aisles, prior, order_train]
#Check the size of the datasets.

for i in combined_df_list:

    print (i.shape)

#There are two df's which are very large in size, subset to use it on local machine with limited compute power.

del combined_df_list
#Products Dataframe

products.head(2)
#Departments Dataframe

departments.head(2)
#Aisles Dataframe - Products are kept in aisles.

aisles.head(2)
#Orders Dataframe

orders.head(2)
#Orders Train Dataframe

order_train.head(2)
#Products in Orders (Prior) - These files specify which products were purchased in each order. Contains Previous Orders.

prior.head(2) #notice the reordered feature.
#Since the dataframe is too big for in memory computation, reducing prior to only 500k rows. 

prior = prior [:500000]
#Merge 1 - Prior and Orders DF (Joining Orders to prior df)

#Combining the Prior and Orders dataframe - shows which user ordered what products and in which order.

df1 = pd.merge(prior, orders, on= 'order_id')

df1.head(2)
#Merge 2

#Combining the department and aisle df's to product df. 

prod_aisles = pd.merge(products, aisles, on = 'aisle_id')

df2 = pd.merge(prod_aisles, departments, on = 'department_id')

df2.head(2)
#Combining df1 anf df2

combined_df = pd.merge(df1, df2, on = 'product_id').reset_index(drop=True)

combined_df.head(2)
#Check Nulls

sb.heatmap(combined_df.isnull(), cbar=True)
#These are null values in the feature 'days_since_prior_order'

combined_df[combined_df['days_since_prior_order'].isnull()].head(2)



#To be dealt with later, as this does not influence the current scope of work.
#Most ordering customer. Favourite Customer?

pd.DataFrame(combined_df.groupby('user_id')['product_id'].count()).sort_values('product_id', ascending=False).head(2)



#User_id = 142131
#Most ordered items.

pd.DataFrame(combined_df['product_name'].value_counts()).head(5)
#Most sold items as per aisle.

pd.DataFrame(combined_df['aisle'].value_counts()).head(5)
combined_df.shape
#Using aisles and user_id. This shows the users that purchased items from which aisle.

user_by_aisle_df = pd.crosstab(combined_df['user_id'], combined_df['aisle'])

user_by_aisle_df.head(2)
#The final dataframe has about 134 features.

user_by_aisle_df.shape
#Standardization is not needed in this case.

user_by_aisle_df.describe() #this confirms that the values dont need to be standardized since they're all 'quantity'.
#Taking array of 'user_by_aisle_df'. To use for elbow method.

X = user_by_aisle_df.values
user_by_aisle_df.head()
#Implementing the Elbow method to identify the ideal value of 'k'. 



ks = range(1,10) #hit and trial, let's try it 10 times.

inertias = []

for k in ks:

    model = KMeans(n_clusters=k)    # Create a KMeans instance with k clusters: model

    model.fit(X)                    # Fit model to samples

    inertias.append(model.inertia_) # Append the inertia to the list of inertias

    

plt.plot(ks, inertias, '-o', color='black') #Plotting. The plot will give the 'elbow'.

plt.xlabel('number of clusters, k')

plt.ylabel('inertia')

plt.xticks(ks)

plt.show()
#Seeing the above plot, the ideal value for cluster (k) should be between 5 and 6 - since the features beyond these values,

# do not explain much of the variability in the dataset. 



#Decomposing the features into 6 using PCA (seeing the above plot, n_components = 6)

pca = decomposition.PCA(n_components=6)

pca_user_order = pca.fit_transform(X)



#You can do hit and trial here to change the number of components and see how much variation in the data 

#is explained by the chose n_components.
#Checking the % variation explained by the 6 pca components.

pca.explained_variance_ratio_.sum()

#More than half (50%) of the variability in the data can be explained by just 6 components.
# Plot the explained variances to verify the variation.

features = range(pca.n_components_)

plt.bar(features, pca.explained_variance_ratio_, color='black')

plt.xlabel('PCA features')

plt.ylabel('variance %')



#A majority of the variance can be explained by just five to six components. Anything beyond that does not capture much of the variation in the dataset.
#Chosen components.

PCA_components = pd.DataFrame(pca_user_order)

PCA_components.head(5)
#Build the model (kmeans using 5 clusters)

kmeans = KMeans(n_clusters=5)

X_clustered = kmeans.fit_predict(pca_user_order) #fit_predict on chosen components only.
#Visualize it.



label_color_mapping = {0:'r', 1: 'g', 2: 'b',3:'c' , 4:'m'}

label_color = [label_color_mapping[l] for l in X_clustered]



#Scatterplot showing the cluster to which each user_id belongs.

plt.figure(figsize = (15,8))

plt.scatter(pca_user_order[:,0],pca_user_order[:,2], c= label_color, alpha=0.3) 

plt.xlabel = 'X-Values'

plt.ylabel = 'Y-Values'

plt.show()
#This contains all the clusters which are to be mapped to each user_id in the user_by_aisle_df.

X_clustered.shape
#Mapping clusters to users.

user_by_aisle_df['cluster']=X_clustered
#Checking cluster concentration. 

user_by_aisle_df['cluster'].value_counts().sort_values(ascending = False)
#Check out cluster mapping.

user_by_aisle_df.head()
# Apriori Algorithm - Association Rules



#Some Theory - just a little.



#Fomatting Data

#Applying Apriori to get support in order to see what items go well together.

#Applying Association Rules to get the Confidence and Lift Scores

#How to come up with up-selling and cross-selling stratgies. The END.
#Importing Libraries

from mlxtend.frequent_patterns import apriori

from mlxtend.frequent_patterns import association_rules
#Checking with only a few samples. Concept is replicable.

np.random.seed(942) # set the seed to make examples repeatable

df2 = combined_df.sample(n=1000)[['user_id','product_name']]

basket = pd.crosstab(df2['user_id'],df2['product_name']).astype('bool').astype('int')

del df2
#Checking and removing index.

basket=basket.reset_index(drop=True)

basket.index
#Lets see if the format is correct.

basket.head(2)
#Calling apriori algorithm on dummified data - basket.

frequent_itemsets=apriori(basket, min_support=0.00002, use_colnames=True).sort_values('support', ascending=False) 



#These are all the POPULAR (Top 20) items purchased from the store.

frequent_itemsets.head(20)
#Lets check the length of the item sets using a tini lambda function.

frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

frequent_itemsets.head()
#Putting a new filter to get all items with length 3 or more. (this means items purchased together)

frequent_itemsets[frequent_itemsets['length'] >= 3]
#FIRST PART - CONFIDENCE



#For association rules, metric can be either confidence or lift. Second argument is minimum threshold (0.5).

#Trying confidence first.

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

rules.head()



#The minimum confidence level starts at 0.5 (confidence column). How likely is it for item C to be purchased if A was purchased?

#Hence if 'Kidz All Natural Baked Chicken Nuggets' was purchased, it is extremely likely that 'Quart Sized Easy Open Freezer Bags' will be purchased in the same transaction. 

#Confidence tells us if item C is purchased, how likely will item A be purchased too.
#SECOND PART - LIFT



#Changing metric to lift. Minimum threshold is 1.

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

rules.head()



#Lift tells how likely are items bought together as opposed to being bought individually.

#Row 0: If Kidz All Natural Baked Chicken Nuggets is purchased, then Quart Sized Easy Open Freezer Bags will be purchased too.

#Row 1: If Quart Sized Easy Open Freezer Bags item is purchased, then Kidz All Natural Baked Chicken Nuggets will be purchased. As there is SLIGHTLY more confidence in row1 (compared to row0).

#THIRD PART - CONFIDENCE AND LIFT



#Select life>5 and confidence >.5

rules[(rules['lift'] >= 5) & (rules['confidence']>= 0.5)] 



#Now these items will be mostly be bought together. So you can make Cross-sell/upsell strategies based on that.
#Next steps - Some tuning to improve performance. 