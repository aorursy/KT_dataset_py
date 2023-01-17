#Import libraries.

from sklearn.datasets import load_boston

import pandas as pd

import matplotlib.pyplot as plt
#Loading the dataset

data = load_boston()

df = pd.DataFrame(data.data,columns =data.feature_names)

df['MEDV'] = data.target
df.head()
data.feature_names
X =df.drop('MEDV', axis =1)

y = df["MEDV"]
#Separate data into train and test

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.3,random_state = 0)

X_train.shape , X_test.shape
# Now we will use Pearson Correlation to remove the features which are correlated. Below is the example:

import seaborn as sns

plt.figure(figsize=(12,10))

cor = X_train.corr()

sns.heatmap(cor,annot=True,cmap= plt.cm.CMRmap_r)

plt.show()
# from the above figure we can conclude that, TAX and  RAD is highly correlated.i.e 0.91(91%) 

#So,we can use any one if it. It all depends on how much threshold you are keeping for correlation.

#If it is 70%. then, we can see AGE and NOX also.

# I know you dont have enough time to watch the figure and write down the correlated feature.

#Lets Write some code.



def correlation(dataset,threshold):

    col_corr = set() # set all the names of correlated columns

    corr_matrix = dataset.corr() # hey just execute  dataset.corr() to understand this line.

    for i in range(len(corr_matrix.columns)):

        for j in range(i):

            if(corr_matrix.iloc[i,j])> threshold:

                colname= corr_matrix.columns[i]  #getting column names

                col_corr.add(colname)

    return col_corr
corr_features = correlation(X_train,0.7)

len(set(corr_features))

corr_features
df= pd.read_csv("../input/santander-customer-satisfaction-traincsv/Santander Customer Satisfaction - TRAIN.csv", nrows =10000)
df.shape
df.head()
# OMG Here we have 371 features.Now in this case we can use the pearson correlation to remove the features

X = df.drop(labels = ["TARGET"],axis=1)

y= df['TARGET']

# separating dataset into training and test set

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state = 0)
X_train.shape
# Now we will use Pearson Correlation to remove the features which are correlated. Below is the example:

import seaborn as sns

plt.figure(figsize=(12,10))

corrmat = X_train.corr()

fig,ax = plt.subplots()

fig.set_size_inches(11,11)

sns.heatmap(corrmat)

plt.show()
cor_features = correlation(X_train,0.8)

len(set(cor_features))
#Ok so we can delete 191 columns.

cor_features

#Below are the names.
#Deleting those features

X_train.drop(cor_features,axis=1)
370-191