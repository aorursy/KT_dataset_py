#import important libraries

import pandas as pd

from matplotlib import pyplot as plt

%matplotlib inline

import seaborn as sns

import numpy as np

#read dataset 

df = pd.read_csv('../input/Pokemon.csv', index_col=0)
#check attributes

df.head()
#check types of pokemons

df.groupby('Type 1').size()
df.groupby('Type 2').size()
plt.figure(figsize=(12,5))

chart0=sns.countplot(x="Type 1", data=df)
plt.figure(figsize=(12,5))

sns.countplot(x="Type 2", data=df)
#Distribution of Total

sns.set(color_codes=True)

sns.distplot(df['Total'])
#We notice there are NaN in our data. Let's take a look

# checking the percentage of missing values in each variable

df.isnull().sum()/len(df)*100
#We also needs to check the variance of the attributes to see if it makes sense to keep all of them.

df.var()/len(df)*100
#change NaN to 0

df['Type 2'] = df['Type 2'].fillna(0) 
#create a new list to change non-NaN values to 1

Type_2 = []

for i in df['Type 2']:

    if i == 0:

        Type_2.append(0)

    else:

        Type_2.append(1)

        

#replace old column 'Type 2' with new binary column        

df['Type 2'] = Type_2

df.head()


#Histogram of attribute 'Attack' and 'Defense'

sns.distplot(df['Attack'])

sns.distplot(df['Defense'])
#A right-skewed normal distribution graph for both attributes.

#Similarly let's look at the distribution of other attributes:
sns.distplot(df['HP'])
sns.distplot(df['Speed'])
sns.distplot(df['Sp. Atk'])

sns.distplot(df['Sp. Def'])
#"wide-form" plots of the dataframe

sns.catplot(data=df, kind="box");
df.groupby('Type 1', sort=True).mean()
#Table gives the mean of each type but how much variance each type represents? 

plt.figure(figsize=(10,5))

chart1=sns.catplot(x="Type 1", y="Total", kind="bar", data=df)

chart1.set_xticklabels(rotation=60)
#Now let's breakdown and see what makes up the 'Total'

#A brief overlook of the correlations between each attribute

df.corr()
#Let's take a look at the 2D plots of 'Sp.Def' and 'Sp.Atk':

sns.relplot(x="Sp. Atk", y="Sp. Def",data=df);

#Overall we can see that the higher Sp. Atk a Pokemon has, the higher Sp. Def it has.

#It might make more sense to see if different type would give any more clues.



sns.relplot(x="Sp. Atk", y="Sp. Def",hue="Type 1",data=df);

sns.relplot(x="Sp. Atk", y="Sp. Def",hue="Type 2",data=df);
#Out of curiosity... Is the strength of pokemon higher when there is a secondary type present?

df.groupby('Type 2', sort=True).mean().sort_values('Total',ascending=False).Total
(456.6-412)/412*100
#Which pokemon types are more likely to get a secondary type?

chart2=sns.catplot(x="Type 1", kind="count", hue="Type 2", data=df);

chart2.set_xticklabels(rotation=60)
#Can we explain everything with our best friend - linear regression?

import statsmodels.api as sm
#First let's separate the predictors and the target - in this case -- Total.

df1 = df.drop(columns=['Total'])
df1
#Lower dimensionality approach using PCA

#import standard scaler package

from sklearn.preprocessing import StandardScaler
features = ['Total','Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']



#separating out the features

x = df.loc[:, features].values

y = df.loc[:,['Type 1']].values

#standardizing the features

x = StandardScaler().fit_transform(x)



from sklearn.decomposition import PCA

pca = PCA(n_components=2)

principalComponents = pca.fit_transform(x)

principalDf = pd.DataFrame(data = principalComponents

             , columns = ['principal component 1', 'principal component 2'])
principalDf
target=df.iloc[:, 1]

target.index = range(800)

target



finalDf = pd.concat([principalDf, target], axis=1)

finalDf
sns.relplot(x="principal component 1", y="principal component 2",hue="Type 1",data=finalDf);
#The plot did not seem to separate out types too well. Let's see if accuracy of this model:

pca.explained_variance_ratio_

#Split dataset

from sklearn.model_selection import train_test_split

dat = df.loc[:, features].values

dat_target = target

x_train, x_test, y_train, y_test = train_test_split(dat, dat_target, test_size=0.2, random_state=0)



#Fit on training set only

scaler.fit(x_train)



#Standardize using scaler

x_train = scaler.transform(x_train)

x_test = scaler.transform(x_test)



#Make an instance of the model. This means that the minimum number of principal components chosen have 95% of the variance retained.

pca=PCA(.95)
#Fit PCA on trainig set 

pca.fit(x_train)
#Now transform the training and the test sets... aka mapping

x_train = pca.transform(x_train)

x_test = pca.transform(x_test)
#Apply logistic regression to the transformed data

from sklearn.linear_model import LogisticRegression

logisticRegr = LogisticRegression(solver = 'lbfgs') #faster!



#Train the model on the data

logisticRegr.fit(x_train, y_train)
#Predict for one observation

logisticRegr.predict(x_test[0].reshape(1,-1))
logisticRegr.score(x_test, y_test)