import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns

%matplotlib inline
df=pd.read_csv('/kaggle/input/eval-lab-3-f464/train.csv')

fd=pd.read_csv('/kaggle/input/eval-lab-3-f464/test.csv')
df['TotalCharges'].replace(" ",0,inplace=True)

fd['TotalCharges'].replace(" ",0,inplace=True)
df['TotalCharges']=df['TotalCharges'].astype('float64')

fd['TotalCharges']=fd['TotalCharges'].astype('float64')
df.dtypes
df=df.replace( { 'Married': { 'No': 0, 'Yes':1 } } )

df=df.replace( { 'Children': { 'No': 0, 'Yes':1 } } )

df=df.replace( { 'Channel1': { 'No': 0, 'Yes':1, 'No tv connection':2 } } )

df=df.replace( { 'Channel2': { 'No': 0, 'Yes':1, 'No tv connection':2 } } )

df=df.replace( { 'Channel3': { 'No': 0, 'Yes':1, 'No tv connection':2 } } )

df=df.replace( { 'Channel4': { 'No': 0, 'Yes':1, 'No tv connection':2 } } )

df=df.replace( { 'Channel5': { 'No': 0, 'Yes':1, 'No tv connection':2 } } )

df=df.replace( { 'Channel6': { 'No': 0, 'Yes':1, 'No tv connection':2 } } )

df=df.replace( { 'Internet': { 'No': 0, 'Yes':1 } } )

df=df.replace( { 'HighSpeed': { 'No': 0, 'Yes':1, 'No internet':2 } } )

df=df.replace( { 'AddedServices': { 'No': 0, 'Yes':1 } } )

df=df.replace( { 'gender': { 'Male': 0, 'Female':1 } } )

df=df.replace( { 'Subscription': { 'Monthly': 0, 'Biannually':1, 'Annually':2 } } )

df=df.replace( { 'PaymentMethod': { 'Net Banking': 0, 'Cash':1, 'Bank transfer':2, 'Credit card':3 } } )

df=df.replace( { 'TVConnection': { 'Cable': 0, 'DTH':1, 'No':2 } } )
fd=fd.replace( { 'Married': { 'No': 0, 'Yes':1 } } )

fd=fd.replace( { 'Children': { 'No': 0, 'Yes':1 } } )

fd=fd.replace( { 'Channel1': { 'No': 0, 'Yes':1, 'No tv connection':2 } } )

fd=fd.replace( { 'Channel2': { 'No': 0, 'Yes':1, 'No tv connection':2 } } )

fd=fd.replace( { 'Channel3': { 'No': 0, 'Yes':1, 'No tv connection':2 } } )

fd=fd.replace( { 'Channel4': { 'No': 0, 'Yes':1, 'No tv connection':2 } } )

fd=fd.replace( { 'Channel5': { 'No': 0, 'Yes':1, 'No tv connection':2 } } )

fd=fd.replace( { 'Channel6': { 'No': 0, 'Yes':1, 'No tv connection':2 } } )

fd=fd.replace( { 'Internet': { 'No': 0, 'Yes':1 } } )

fd=fd.replace( { 'HighSpeed': { 'No': 0, 'Yes':1, 'No internet':2 } } )

fd=fd.replace( { 'AddedServices': { 'No': 0, 'Yes':1 } } )

fd=fd.replace( { 'gender': { 'Male': 0, 'Female':1 } } )

fd=fd.replace( { 'Subscription': { 'Monthly': 0, 'Biannually':1, 'Annually':2 } } )

fd=fd.replace( { 'PaymentMethod': { 'Net Banking': 0, 'Cash':1, 'Bank transfer':2, 'Credit card':3 } } )

fd=fd.replace( { 'TVConnection': { 'Cable': 0, 'DTH':1, 'No':2 } } )
df.dtypes
fd.dtypes
missing_count = df.isnull().sum(axis=0)

missing_count[missing_count > 0]
missing_count = fd.isnull().sum(axis=0)

missing_count[missing_count > 0]
corr = df.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# # Set up the matplotlib figure

f, ax = plt.subplots(figsize=(20, 15))



# # Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)

cmap

# # Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr,annot=True, mask=mask, cmap=cmap, vmax=0.5, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.show()
X=df.copy()

X.drop(columns=['custId','Satisfied'],inplace=True)

y=df['Satisfied']
X_test=fd.copy()

X_test.drop(columns=['custId'],inplace=True)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()

X = lda.fit_transform(X, y)

X_test=lda.transform(X_test)
from collections import Counter

from imblearn.over_sampling import SMOTE

print('Original dataset shape %s' % Counter(y))

sm = SMOTE(random_state=42)

X, y = sm.fit_resample(X, y)

print('Resampled dataset shape %s' % Counter(y))
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X=scaler.fit_transform(X)

X_test=scaler.transform(X_test)
X
from sklearn.cluster import KMeans

wcss = []

for i in range(1,11):

    kmeans = KMeans(n_clusters = i,init = 'k-means++',random_state = 0)    

    kmeans.fit(X)    

    wcss.append(kmeans.inertia_) 

plt.plot(range(1,11),wcss) 

plt.title('The Elbow Method') 

plt.xlabel('Number of cluster') 

plt.ylabel('WCSS') 

plt.show()

cl=19

kmeans = KMeans(n_clusters = cl,init = 'k-means++',random_state =0) 

y_kmeans = kmeans.fit_predict(X)

y_test_kmeans = kmeans.predict(X_test)
unique, counts = np.unique(y_kmeans, return_counts=True)

print(np.asarray((unique, counts)).T)
counts_n={}

counts_y={}

labels={}

for i in range(cl):

    counts_n.update({i:0})

    counts_y.update({i:0})

    labels.update({i:i})



for i in range(len(X)):

    if y[i]==0:

        counts_n[y_kmeans[i]]+=1

    else:

        counts_y[y_kmeans[i]]+=1



for i in labels:

    if counts_n[i]>counts_y[i]:

        labels[i]=0

    else:

        labels[i]=1

for i in range(len(y_kmeans)):

    y_kmeans[i]=labels[y_kmeans[i]]

    

for i in range(len(y_test_kmeans)):

    y_test_kmeans[i]=labels[y_test_kmeans[i]]
from sklearn.metrics import roc_auc_score

roc_auc_score(y,y_kmeans)
submission = pd.DataFrame({'custId':fd['custId'],'Satisfied':y_test_kmeans})

# submission['class']=submission['class'].astype("int")

submission['Satisfied'].dtype
filename = 'submission.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)