# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings('ignore')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/diabetes.csv')
data['index'] = pd.Series(data.index)
data = data[[ 'index','Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']]
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
positive = data[data['Outcome']==1]
negative = data[data['Outcome']==0]
pd.options.display.max_columns = None
print("Overall Description of the Dataset")
pos = positive['index'].count()
neg = negative['index'].count()
lis = [pos,neg]
label = ['Positive','Negative']
f,ax = plt.subplots(figsize=(10,10))
plt.pie(lis,labels=label,autopct='%1.1f%%',shadow=True, startangle=90)
plt.show()
print(data.describe())
print("*******************************************************************************************************")
print("Number of Empty Fields")
for i in range(8):
    print('{0}:{1}'.format(data.columns[i],np.sum(np.isnan(data.iloc[:,i]))))

print("*****************************************************************************")
print("Statistical Desrcription for Women who are Diabetic")
print("*****************************************************************************")
print(positive.describe())
print("")
print("*****************************************************************************")
print("Statistical Desrcription for Women who are Not-Diabetic")
print("*****************************************************************************")
print(negative.describe())
plt.subplots(figsize = (10,10))
sns.distplot(positive['Age'], hist = False, kde = True,
                 kde_kws = {'linewidth': 3},
                 label = 'Positive Outcome')
sns.distplot(negative['Age'], hist = False, kde = True,
                 kde_kws = {'linewidth': 3},
                 label = 'Negative Outcome')
plt.show()
d = data.iloc[:,1:-1]
plt.subplots(figsize = (10,10))
sns.heatmap(d.corr())
plt.show()
f, axes = plt.subplots(2, 4,figsize=(20,10))
count=0

for i in range(2):
    for j in range(4):
        count=count+1
        sns.distplot(positive.iloc[:,count],kde=True,hist=False,kde_kws = {'linewidth': 3},label = 'Positive Outcome',ax = axes[i][j])
        sns.distplot(negative.iloc[:,count],kde=True,hist=False,kde_kws = {'linewidth': 3},label = 'Negative Outcome',ax = axes[i][j])
        axes[i][j].legend()
        
f.suptitle('PDFs')
plt.show()
      
    

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
y_train = data['Outcome']
x_train_scaled = sc.fit_transform(d)
x_train_scaled = pd.DataFrame(data=x_train_scaled)
x_train_scaled.columns = d.columns
temp_data = pd.concat((y_train,x_train_scaled.loc[:,:]),axis=1)
temp_data = pd.melt(temp_data,id_vars='Outcome',value_name='value',var_name='features')
plt.figure(figsize=(12,12))
sns.boxenplot(data=temp_data,x='features',y='value',hue='Outcome')
plt.figure(figsize=(12,12))
sns.violinplot(data=temp_data,x='features',y='value',hue='Outcome',split=True)
plt.show()
f,axes = plt.subplots(2,4,figsize=(16,8))
count=0
for i in range(2):
    for j in range(4):
        axes[i][j].set_title(d.columns[count])
        counts,bins = np.histogram(positive.iloc[:,count],bins=20)
        cdf = np.cumsum(counts)
        sns.lineplot(bins[1:],cdf/cdf[-1],ax = axes[i][j],label = 'positive')
        counts,bins = np.histogram(negative.iloc[:,count],bins=20)
        cdf = np.cumsum(counts)
        sns.lineplot(bins[1:],cdf/cdf[-1],ax = axes[i][j],label='negative')
        count=count+1
plt.suptitle('Cumulative Distribution Frequency ')
plt.show()
from sklearn.model_selection import train_test_split
xTrain,xTest,yTrain,yTest = train_test_split(d,y_train,test_size=0.3,stratify=y_train,random_state=0)
from sklearn.neighbors import KNeighborsClassifier 
acc_vals=[]
kvalue=None
for i in range(1,10):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(xTrain,yTrain)
    output = knn.predict(xTest)
    from sklearn.metrics import accuracy_score
    acc_vals.append(accuracy_score(output,yTest))
    if max(acc_vals)==accuracy_score(output,yTest):
        kvalue=i
        
print("K-Value:{0} and Accuracy Score:{1}%".format(kvalue,max(acc_vals)*100))
  

from sklearn.linear_model import LogisticRegression 
lr = LogisticRegression()
lr.fit(xTrain,yTrain)
ouput = lr.predict(xTest)
print(accuracy_score(output,yTest)*100,'%')

