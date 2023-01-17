# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from IPython.display import Image



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure

import plotly.offline as ply

import plotly.graph_objs as gob
mushrooms = pd.read_excel("../input/mushroomdata/mushroom.xlsx")

mushrooms
mushrooms.describe().T # Let's take a rapid glance at count, frequencies, uniqueness, etc. of the dataset.
#Arranging attributes and classes.



x = mushrooms.drop(columns='class')

y = mushrooms['class']



# Renaming classes

y=y.replace('p','poisonous')

y=y.replace('e','edible')



# Creating a pie chart to see frequency of classes. Frequency of classes is important, because if our dataset is imbalanced dataset then we need to analyze the dataset in a different way.



pie_values = pd.Series(y).value_counts()

trace = gob.Pie(labels=['edible','poisonous'],values=pie_values)

ply.iplot([trace])
mushrooms.isnull().any() # This code shows that there is a column contains a null value or not, as you can see below the dataset has a columns contains "nan" value.

# So, this code returns true for stalk-root column. 
# Also, we can write a simple for loop to see column(s) which contains(s) non-sense or null value(s).



for i in mushrooms.columns:

    print(i,mushrooms[i].unique())
# Let's use second step to fill non-sense data with more reasonable one.

x['stalk-root'].value_counts()
x['stalk-root'] = x['stalk-root'].fillna('b') # '?' in stalk-root attribute filled with 'b' has most frequency.

# Let's see whether non-sense data exists, or not.

x['stalk-root'].value_counts()
from sklearn.preprocessing import LabelEncoder

le_encoder = LabelEncoder()



y = le_encoder.fit_transform(y) # Label Encoding for class.



x = pd.get_dummies(x) # One-Hot Encoding for attributes. There are columns has more than two unique values. So, for encoding I use One-Hot Encoding.



# Let's see new looking of our attributes

x
# Maybe, K-Means is one of the simpliest algorithms in ML, but there is some challenging situtations like determining the optimal K value which is diretly related to clustering quality.



from sklearn.cluster import KMeans



WCSS=[] # I created a free list to append WCSS values.



for K in range(1,11):

    KM = KMeans(n_clusters=K, init='k-means++', max_iter=300, n_init=10)

    KM.fit(x.values)

    WCSS.append(KM.inertia_)



# Choosing the right K value for a propoer KMeans model may be a challenging situation. In this point, The Elbow Method helps us. This method is based on interpretation. As WCSS value 

# decreases, we can say that clustering process is getting better because similarity of values in a cluster increases and similarity between clusters decreases. This is what we need to do.

# But some can say that if the number of cluster value is selected too much, the clustering will be so good. This is COMPLETELY WRONG! As Number of Cluster value converges to n (number 

# of data of a dataset) the model may be overfitted. Briefly, choosing the highest value of K does not mean a proper clustering model. Just think about it, if you choose K = n, this is 

# not clustering, this is assigning each data point as a cluster. So, we need to choose optimal K value. We can use The Elbow Method for this.



figure(figsize=(6,5))

plt.plot(range(1,11),WCSS)

plt.title('Elbow Method for Mushroom Dataset')

plt.xlabel('Number of Cluster')

plt.ylabel('WCSS')

plt.show()
# The function coded below shows that differences between WCSS values. We should choose the optimal K value which returns more smoother slope. In this dataset we can choose K=2.

np.diff(WCSS)
# Since we will work with many models, let us keep the names of these models in a list to ease coding.

model_predicted = ['y_KM','y_KNN','y_DT','y_NB','y_LR','y_SVM']



KM = KMeans(n_clusters=2, init='k-means++',max_iter = 300, n_init = 10)

model_predicted[0] = KM.fit_predict(x)
# İmporting libraries for k-cross validation

from sklearn.model_selection import KFold,cross_val_score,cross_val_predict



KFCV = KFold(n_splits = 10, random_state = 42) # Here, random_state value is an important parameter. If the random_state parameter is defined with a fix value like 42, 

# then no matter how many times your code is run,same data will be filled into the train and test test. That means the output will be the same. Otherwise, if random_state parameter is not

# defined with a fix value, the output will change every time the code is run.
# Loading KNN classifier



from sklearn.neighbors import KNeighborsClassifier



# Like KMeans, KNN has the same problem with choosing the best K value to perform a proper classifier.

# Let' s choose the best K value.



values =[]



for K in range(1,20):  

    KNN=KNeighborsClassifier(n_neighbors=K,metric='euclidean')

    accuracies=cross_val_score(estimator=KNN,X=x,y=y,cv=KFCV,scoring='accuracy').mean()

    values.append(accuracies)

    

plt.plot(range(1,20),values)

plt.ylabel('Accuracy')

plt.xlabel('K Value')



print('The K value which is the most accuracy', values.index(max(values)),',','Accuracy:', max(values))
# KNN is performed with the best K found above.

KNN = KNeighborsClassifier(n_neighbors=values.index(max(values)),metric='euclidean') # It can be choosen minkowski or manhattan as metric.
# Loading decision tree library



from sklearn.tree import DecisionTreeClassifier



# Performing the model with criterion = 'entropy'. Branching criteria can be chosen as gini. Branching selection depends on your model.



DT = DecisionTreeClassifier(criterion = 'entropy')
# Loading Naive Bayes library



from sklearn.naive_bayes import GaussianNB



NB = GaussianNB()
# Loading Logistic Regression library



from sklearn.linear_model import LogisticRegression



LogR = LogisticRegression(solver = 'saga')    # The solver component is important. If you work on a large dataset, it would be better to assign solver = 'saga'. 

# This selection is directly related to model performance. In addition mushroom dataset has only two different class, we don't have to use multi_class component of LogisticRegression.
# Loading SVM Library



from sklearn import svm

L_SVM = svm.LinearSVC()  # In this work, I use linear SVM.
# Loading Libraries



from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

from sklearn.metrics import cohen_kappa_score,confusion_matrix



# A list has been created to store some performance metrics.



scoring = ['accuracy' ,'precision','recall','f1']



# accuracy : Contrary to the popular belief, accuracy does not make sense alone. To obtain better result It should be evaluated with other metrics.



# precision and recall: In a machine learning model with a high accuracy, precision or recall may be dramatically low. This is the critical point! In a imbalanced dataset accuracy can be 

# satisfied value like 80 percent or 90, but precision and recall may be dramatically low. Let's consider the classic example, cancer disease. Consider that the data contains 1000 people, 

# 100 of them are truly cancer, rest of all is not. So, this is an imbalanced data set. Let's consider a model correctly predict 880 of 900 truly not cancer people. On the other hand,

# the model correctly predict 10 of 100 truly cancer people. So, the model accuracy is 89 percent. This is a huge, but deceptive. Predicting wrong the people with cancer is not acceptable!

# Although the model accuracy is 89 percent, the percent of corretly predicting of the truly cancer people is 10 percent. This is unacceptable! So, precision and recall help us to determine

# these situations.



# F-Score: Harmonic mean of recall and precision. It shows tradeoff of recall and precision.



model_estimator_list=[KNN,DT,NB,LogR,L_SVM]

model_name_list=['K-Nearest Neighbors','Decision Tree','Naive Bayes','Logistic Regression','SVM']





# Confusion matrix of the models.



print('K-Means Confusion Matrix: \n' ,confusion_matrix(y,model_predicted[0]))



for i in range(1,6):

  model_predicted[i]=cross_val_predict(model_estimator_list[i-1],x,y,cv=KFCV)

  print(model_name_list[i-1],'Confusion Matrix: \n',confusion_matrix(y,model_predicted[i]))

 



# Let's see the performance metrics scores resulting from classification.



# Cross-validation was not applied on KMeans, because it is an unsupervised method. Thats why, I calculate performance metrics of K-Means seperately.



print('K-Means accuracy', accuracy_score(y,model_predicted[0]))

print('K-Means precision', precision_score(y,model_predicted[0]))

print('K-Means recall', recall_score(y,model_predicted[0]))

print('K-Means f1', f1_score(y,model_predicted[0]))





for i in range(0,5):

    for j in range(0,4):

      metrics_scores=cross_val_score(estimator=model_estimator_list[i],X=x,y=y,cv=KFCV,scoring=scoring[j])

      print(model_name_list[i],'model',scoring[j],'on each fold',metrics_scores)

      print(model_name_list[i],'model',scoring[j],'on each fold',metrics_scores.mean())



        

# Let's calculate Cohen Kappa Score.



print('K-Means model kappa',cohen_kappa_score(y,model_predicted[0]))

for i in range(0,5):

   print(model_name_list[i],'kappa',cohen_kappa_score(y,model_predicted[i]))
