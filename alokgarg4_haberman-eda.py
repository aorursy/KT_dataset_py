import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings 

warnings.filterwarnings("ignore") 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
haberman = pd.read_csv("/kaggle/input/habermans-survival-data-set/haberman.csv")
haberman.columns =  ['age', 'year', 'nodes', 'status']
haberman.tail()
def details(dataset):
    print("Shape of data set:",haberman.shape)
    print(f"Number of rows:{haberman.shape[0]}\nNumber of Columns:{haberman.shape[1]}")
    print("Columns:",haberman.columns)
    print("data types:\n",haberman.dtypes)
details(haberman)
haberman['status'].value_counts()
haberman['status'] = haberman['status'].apply(lambda x : 0 if x== 2 else 1)
haberman.head()

sns.set_style('darkgrid')
sns.distplot(haberman['age'],hist=True,kde = True,color= 'g')
plt.show()

sns.boxplot(haberman['age'],data = haberman,orient='v')
plt.title("Box plot for age")
print("Mean age is:",haberman['age'].mean())
import numpy as np
#haberman['age'].loc[haberman['age']>= np.percentile()]
Q1 = np.percentile(haberman['age'],q=25)
print(Q1)
Q3 = np.percentile(haberman['age'],q=75)
print(Q3)
IQR = Q3-Q1
print(IQR)
low = Q1- 1.5 * IQR
high = Q3 + 1.5 * IQR
print(low,high)

print(haberman['age'].loc[(haberman['age'] <= low) | (haberman['age'] > high )])
import matplotlib.pyplot as plt
sns.FacetGrid(haberman,size=5) \
   .map(sns.distplot, "age") \
   .add_legend()
plt.show()

haberman.describe()
haberman.isnull().sum()
print("nodes and it's count:",haberman['nodes'].value_counts().to_dict())
print("percentage of each nodes:",list(round(haberman['nodes'].value_counts()/len(haberman['nodes'])*100,2)))
sns.set_style('darkgrid')
sns.distplot(haberman['nodes'],hist=True,kde = False,bins = [0,5,10,15,20,25,30,35,40,45,50,55,60],color= 'g' )
plt.figure(figsize=(10,6))
sns.lineplot(x = haberman['nodes'],y = haberman['age'])
plt.title("age vs nodes lineplot")
from collections import Counter
print(Counter(haberman.loc[haberman['nodes'] <=4]['age'].tolist()))
print(haberman.loc[haberman['nodes'] <=4]['age'].tolist())
print(len(haberman.loc[haberman['nodes'] <=4]['age'])/len(haberman)* 100)
plt.figure(figsize=(10,6))
sns.boxplot(haberman['nodes'],data = haberman,orient='v')
plt.title("box plot for nodes")
haberman['nodes'].mean()
Q1 = np.percentile(haberman['nodes'],q=25)
print(Q1)
Q3 = np.percentile(haberman['nodes'],q=75)
print(Q3)
IQR = Q3-Q1
print(IQR)
low = Q1- 1.5 * IQR
high = Q3 + 1.5 * IQR
print(low,high)

print(len(haberman['nodes'].loc[(haberman['nodes'] <= low) | (haberman['nodes'] > high )]))
haberman['year'].value_counts()
sns.set_style("darkgrid")
import matplotlib.pyplot as plt
sns.FacetGrid(haberman,hue = 'status',size = 5).map(sns.scatterplot,'age','nodes').add_legend()
plt.show()
sns.FacetGrid(haberman,hue = 'status',height = 5).map(sns.scatterplot,'year','age').add_legend()
sns.FacetGrid(haberman,hue = 'status',height = 5).map(sns.scatterplot,'year','nodes').add_legend()
#fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, feature in enumerate(list(haberman.columns)[:-1]):
    
    fg = sns.FacetGrid(haberman, hue='status', height=5)
    fg.map(sns.distplot, feature).add_legend()
    plt.show()
    print("*" * 50)
haberman.loc[haberman['nodes'] <=4]['status'].value_counts()/len(haberman.loc[haberman['nodes'] <=4]['status']) * 100 
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, feature in enumerate(list(haberman.columns)[:-1]):
    sns.boxplot( x='status', y=feature, data=haberman, ax=axes[idx])
plt.show()  
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, feature in enumerate(list(haberman.columns)[:-1]):
    sns.violinplot( x='status', y=feature, data=haberman, ax=axes[idx])
plt.show() 
plt.figure(figsize=(20,5))
for idx, feature in enumerate(list(haberman.columns)[:-1]):
    plt.subplot(1, 3, idx+1)
    counts, bin_edges = np.histogram(haberman[feature], bins=10, density=True)
    pdf = counts/sum(counts)
    cdf = np.cumsum(pdf)
    plt.plot(bin_edges[1:], pdf, bin_edges[1:], cdf)
    plt.legend(['pdf','cdf'])
    plt.xlabel(feature)
sns.pairplot(haberman,hue= 'status',corner= True)
plt.title("pairplot")
sns.heatmap(data= haberman.corr()[['age','nodes','status']],annot = True)
haberman.corr()[['age','nodes','status']]