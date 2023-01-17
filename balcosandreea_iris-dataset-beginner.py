import numpy as np 

import pandas as pd 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style='darkgrid')
iris=pd.read_csv('../input/iris/Iris.csv')
iris.columns
iris.shape
iris.info()



# We don't have null values in our data, all seems fine
iris.rename(columns={'SepalLengthCm':'sepal_length','SepalWidthCm':'sepal_width',

                     'PetalLengthCm':'petal_length','PetalWidthCm':'petal_width'},inplace=True)
iris.head()
iris.describe()  
# We can see the descriptive statistics grouped by species

iris.groupby('Species').describe()
sns.heatmap(iris.drop('Id',axis=1).corr(),annot=True)
sns.pairplot(iris.drop('Id',axis=1),hue='Species',palette='bright')
# For visualizing the summary statistics for each species, we use boxplot 

species=iris.Species.unique()

k=1

plt.figure(figsize=(21,8))

for i in species:

    plt.subplot(1,3,k)

    k=k+1

    sns.boxplot(data=iris.drop('Id',axis=1)[iris.Species==i],width=0.5,fliersize=5)

    plt.title(str('Boxplot'+i.upper()))
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier



X=iris.drop(['Species','Id'],axis=1) # This is also called the independent value (training data)

y=iris.Species # This is also called the dependent value (the target), the value obtained through the independent values
def prediction(k,train,target):

    knn=KNeighborsClassifier(n_neighbors=k)

    knn.fit(train,target) 

    new=pd.DataFrame([[4.5,2.3,3.1,2],[6,2.1,3,4.8]]) # I added 2 random observations 

    new_obsv=knn.predict(new)

    X_train, X_test, y_train, y_test= train_test_split(train,target,test_size=0.3,random_state=21,stratify=target)

    knn.fit(X_train,y_train)

    knn.predict(X_test)

    print('This observations belong to :', new_obsv,'with an accuracy of:',knn.score(X_test,y_test))

    
prediction(6,X,y)

prediction(9,X,y)



# The accuracy for these 2 are different, so a plot will be useful for visualizing the influence of the neighbors 
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.3,random_state=21,stratify=y)

neighbors=np.arange(1,100)

accuracy_list=np.empty(len(neighbors))

for i,k in enumerate(neighbors):

    knn=KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train,y_train)

    knn.predict(X_test)

    accuracy_list[i]=knn.score(X_test,y_test)
from bokeh.io import output_notebook, output_file, show

from bokeh.plotting import figure

output_notebook()



p1=figure(plot_height=500,plot_width=900,title='The influence of neighbors numbers on accuracy',

          x_axis_label='Number of neighbors',y_axis_label='Accuracy of KNN model')

p1.line(x=neighbors,y=accuracy_list)

p1.circle(x=neighbors,y=accuracy_list)

show(p1)
neigh = [i for i, x in enumerate(accuracy_list) if x == max(accuracy_list)]

accuracy_list=list(accuracy_list)
for i in neigh:

    print('The maxim accuracy is :',max(accuracy_list),' number of neighbours:',i+1)