# Importing necessary packages



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings('ignore')

sns.set()
# loading the dataset

df = pd.read_csv('../input/habermans-survival-data-set/haberman.csv', header=None, names=['Age', 'Op_Year', 'axil_nodes', 'Surv_status'])

print(df.info())
df["Surv_status"]=df["Surv_status"].map({1:'positive',2:'negative'})

df.head()
print(df.describe())

print("Number of rows: " + str(df.shape[0]))

print("Number of columns: " + str(df.shape[1]))

print("Columns: " + ", ".join(df.columns))

# How many data points for each class are present?

print(df.iloc[:,-1].value_counts())

# Number of classes

print(df['Surv_status'].nunique())
#Dividing the dataset into 2 datasets of positive and negative result.

positive=df.loc[df['Surv_status']=='positive']

negative=df.loc[df['Surv_status']=='negative']
print (df.describe())
print ("Age")

print ("  Mean:")

print ("  positive result- "+str(np.mean(positive["Age"])))

print ("  negative result- "+str(np.mean(negative["Age"])))

print ()

print ("  Standard Devation:")

print ("  positive result- "+str(np.std(positive["Age"])))

print ("  negative result- "+str(np.std(negative["Age"])))

print ()



print ()

print ("Year of Operation")

print ("  Mean:")

print ("  positive result- "+str(np.mean(positive["Op_Year"])))

print ("  negative result- "+str(np.mean(negative["Op_Year"])))

print ()

print ("  Standard Devation:")

print ("  positive result- "+str(np.std(positive["Op_Year"])))

print ("  negative result- "+str(np.std(negative["Op_Year"])))

print ()

print ()

print ("No of Auxillary Nodes")

print ("  Mean:")

print ("  positive result- "+str(np.mean(positive["axil_nodes"])))

print ("  negative result- "+str(np.mean(negative["axil_nodes"])))

print ()

print ("  Standard Devation:")

print ("  positive result- "+str(np.std(positive["axil_nodes"])))

print ("  negative result- "+str(np.std(negative["axil_nodes"])))
#90th percentile

print ('90th Percentile')

print ()

print ("Age")

print ("  positive result- "+str(np.percentile(positive["Age"],90)))

print ("  negative result- "+str(np.percentile(negative["Age"],90)))



print ("Year of Operation")

print ("  positive result- "+str(np.percentile(positive["Op_Year"],90)))

print ("  negative result- "+str(np.percentile(negative["Op_Year"],90)))



print ("No of Auxillary Nodes")

print ("  positive result- "+str(np.percentile(positive["axil_nodes"],90)))

print ("  negative result- "+str(np.percentile(negative["axil_nodes"],90)))

print ("  general result- "+str(np.percentile(df["axil_nodes"],90)))
#PDF

sns.set_style("whitegrid")

for index, feature in enumerate(list(df.columns)[:-1]):

    sns.FacetGrid(df,hue='Surv_status',height=4).map(sns.distplot,feature).add_legend()

    plt.show()
#CDF

plt.figure(figsize=(20,5))

for index, feature in enumerate(list(df.columns)[:-1]):

    plt.subplot(1, 3, index+1)

    print("\n********* "+feature+" *********")

    counts, bin_edges = np.histogram(df[feature], bins=10, density=True)

    print("Bin Edges: {}".format(bin_edges))

    pdf = counts/sum(counts)

    print("PDF: {}".format(pdf))

    cdf = np.cumsum(pdf)

    print("CDF: {}".format(cdf))

    plt.plot(bin_edges[1:], pdf, bin_edges[1:], cdf)

    plt.xlabel(feature)
counts,bins=np.histogram(df['axil_nodes'],bins=10,density=True)

pdf=counts/(sum(counts))

print(pdf)

print(bins)

cdf=np.cumsum(pdf)

plt.plot(bins[1:],pdf,label='pdf')

plt.plot(bins[1:],cdf,label='cdf')

plt.xlabel('axil_nodes')

plt.ylabel('probability')

plt.title("CDF AND PDF OF THE NODES FOR THE DIDNT SURVIVE")

plt.legend()

plt.show()
#Box Plot

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, feature in enumerate(list(df.columns)[:-1]):

    sns.boxplot( x='Surv_status', y=feature, data=df, ax=axes[idx])

plt.show()
#Violin Plots

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, feature in enumerate(list(df.columns)[:-1]):

    sns.violinplot( x='Surv_status', y=feature, data=df, ax=axes[idx])

plt.show()
plt.plot(positive["Age"], np.zeros_like(positive["Age"]), 'o', label = "surv_status\n" "Positive")

plt.plot(negative["Age"], np.zeros_like(negative["Age"]), 'o', label = "Negative")

plt.title("1-D scatter plot for age")

plt.xlabel("Age")

plt.legend()

plt.show()
sns.set_style("whitegrid")

sns.pairplot(df,hue='Surv_status',height=4)

plt.show()
df_small = df[['Age', 'Op_Year', 'axil_nodes', 'Surv_status']]





df_copy = pd.get_dummies(df_small)





df1 = df_copy

df1.head()

y = np.asarray(df1['Surv_status_positive'], dtype="|S6")

df1 = df1.drop(['Surv_status_positive','Surv_status_negative'], axis=1)

X = df1.values

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.30)



radm = RandomForestClassifier()

radm.fit(Xtrain, ytrain)



clf = radm

indices = np.argsort(radm.feature_importances_)[::-1]



# Print the feature ranking

print('Feature ranking:')



for f in range(df1.shape[1]):

   print('%d. feature %d %s (%f)' % (f + 1, indices[f], df1.columns[indices[f]], radm.feature_importances_[indices[f]]))

    
