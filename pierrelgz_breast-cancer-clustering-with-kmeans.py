import numpy as np 

import pandas as pd 



import seaborn as sns

import matplotlib.pyplot as plt
data = pd.read_csv("/kaggle/input/breast-cancer-wisconsin-data/data.csv")

data.head()
data.columns
data.describe()
data.dtypes
decision = data['diagnosis'].value_counts()

sns.barplot(decision.index[0:],decision[0:])

decision
data_nolabel = data

data['diagnosis'].replace(['B','M'],[0,1], inplace=True) 
#https://www.kaggle.com/amiiiney/price-prediction-regularization-stacking

num=data.select_dtypes(exclude='object')

numcorr=num.corr()

Table_corr=numcorr['diagnosis'].sort_values(ascending=False).head(10).to_frame()



cm = sns.light_palette("cyan", as_cmap=True)



s = Table_corr.style.background_gradient(cmap=cm)

s
columns = ['diagnosis','concave points_worst', 'perimeter_worst',

       'concave points_mean', 'radius_worst', 'perimeter_mean', 'area_worst',

       'radius_mean', 'area_mean', 'concavity_mean']



#scatterplot

sns.set()

sns.pairplot(data[columns], height = 2.5, hue="diagnosis",diag_kind="hist")

plt.show();

#Delete diagnosis from the list

del columns[0]

columns
for x in columns:

    plt.figure(figsize=(7.5,5))

    sns.boxplot(data['diagnosis'],data[x])
bef = data.shape[0]

data_no_outliers = data.copy()

print(data.shape)



for x in columns:

    Q1=data[x].loc[(data['diagnosis'] == 1)].quantile(0.25)

    Q3=data[x].loc[(data['diagnosis'] == 1)].quantile(0.75)

    IQR=Q3-Q1

    Lower_Whisker = Q1-1.5*IQR

    Upper_Whisker = Q3+1.5*IQR



    d1 = data[(data['diagnosis'] == 1) & (data[x] < Upper_Whisker)]

    

    Q1=data[x].loc[(data['diagnosis'] == 0)].quantile(0.25)

    Q3=data[x].loc[(data['diagnosis'] == 0)].quantile(0.75)

    IQR=Q3-Q1

    Lower_Whisker = Q1-1.5*IQR

    Upper_Whisker = Q3+1.5*IQR



    d0 = data[(data['diagnosis'] == 0) & (data[x] < Upper_Whisker)]

    data_no_outliers = d0.append(d1)

    

aft = data_no_outliers.shape[0]

print(data_no_outliers.shape)

print("We removed", bef - aft, "outliers.")



data = data_no_outliers
from scipy.stats import norm

from scipy import stats



for x in columns:

    plt.figure(figsize=(7.5,30))

    p2 = 1

    plt.subplot(len(columns), 2, p2)

    sns.distplot(data[x], fit=norm)

    p2=p2+1

    plt.subplot(len(columns), 2, p2)

    res = stats.probplot(data[x], plot=plt)

    plt.show()
sk = pd.DataFrame({'Skew':data[columns].skew(),'Kurt':data[columns].kurt()})

sk
for x in columns:

    data[x] = np.log1p(data[x])

data
for x in columns:

    plt.figure(figsize=(7.5,30))

    p2 = 1

    plt.subplot(len(columns), 2, p2)

    sns.distplot(data[x], fit=norm)

    p2=p2+1

    plt.subplot(len(columns), 2, p2)

    res = stats.probplot(data[x], plot=plt)

    plt.show()
sk = pd.DataFrame({'Skew':data[columns].skew(),'Kurt':data[columns].kurt()})

sk
#If we didn't knew the number of clusters, we should have done that :



from sklearn.cluster import KMeans



K_range=range(1,10)

inertia = []

for k in K_range:

    kmeans = KMeans(n_clusters=k).fit(data[columns])

    inertia.append(kmeans.inertia_) # Inertia: Sum of distances of samples to their closest cluster center

    

plt.figure()

plt.plot(K_range,inertia)

plt.xlabel("Number of cluster")

plt.ylabel("inertia")

plt.show()
from sklearn.preprocessing import scale



y=data['diagnosis']

X = scale(data[columns]) #to center all the variables

X[0:10,]
model = KMeans(n_clusters=2, random_state=0)

model.fit(X)
X = pd.DataFrame(X)

X.columns = ['concave points_worst', 'perimeter_worst',

       'concave points_mean', 'radius_worst', 'perimeter_mean', 'area_worst',

       'radius_mean', 'area_mean', 'concavity_mean']
color_theme = np.array(['cornflowerblue','darkorange'])



plt.subplot(1,2,1)

plt.scatter(x=X['concave points_worst'], y =X['perimeter_worst'],c=color_theme[data['diagnosis']])

plt.title('Real classification')



plt.subplot(1,2,2)

plt.scatter(x=X['concave points_worst'], y =X['perimeter_worst'],c=color_theme[model.labels_])

plt.title('KMeans classification')
#relabel = np.choose(model.labels_,[1,0]) #0 become 1 and  1 become 0
from sklearn.metrics import confusion_matrix,classification_report

print(classification_report(y,model.labels_)) #if the labels are invert, model.labels_ become relabel (just upward)



#high precision + high recall = highly accurate model results

#the more it is close to 1 and the better it is