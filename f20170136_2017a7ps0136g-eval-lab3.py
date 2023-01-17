import pandas as pd  

import numpy as np  

import matplotlib.pyplot as plt  

import seaborn as seabornInstance 

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LinearRegression

from sklearn import metrics

%matplotlib inline



import seaborn as sns

sns.set()



import os

for dirname, _, filenames in os.walk("/kaggle/input"):

    for filename in filenames:

        print (os.path.join(dirname, filename))
train = pd.read_csv("../input/eval-lab-3-f464/train.csv")

train.head(6)

train = train.drop(['gender', 'TotalCharges', "custId"], axis=1)

y_train = pd.DataFrame(train["Satisfied"], columns=["Satisfied"])

train = pd.get_dummies(train)
X = train

X.columns
from sklearn.preprocessing import Normalizer



transformer = Normalizer().fit(X)  # fit does nothing.

transformer.transform(X)

X_data = pd.DataFrame(transformer.transform(X))

X_data

# X
test = pd.read_csv("../input/eval-lab-3-f464/test.csv")

test = test.drop(['gender', 'TotalCharges', "custId"], axis=1)

test = pd.get_dummies(test)
test_points = test

from sklearn.preprocessing import Normalizer



transformer = Normalizer().fit(test_points)  # fit does nothing.

transformer.transform(test_points)

test_points_normalize = pd.DataFrame(transformer.transform(test_points))

test_points_normalize
from sklearn.cluster import KMeans

from sklearn.cluster import MiniBatchKMeans

from sklearn.cluster import AffinityPropagation

from sklearn.cluster import AgglomerativeClustering



agglomerativeclust = AgglomerativeClustering(n_clusters = 2).fit(X_data)

train_pred = agglomerativeclust.fit_predict(X_data)

test_pred = agglomerativeclust.fit_predict(test_points_normalize)



if (accuracy_score(y_train, train_pred) < 0.5):

    test_pred = np.where(test_pred == 0, 1, 0)

test_pred
final = pd.DataFrame(pd.read_csv("../input/eval-lab-3-f464/test.csv")["custId"], columns = ['custId'])

final["Satisfied"] = test_pred

# final.to_csv("final.csv", index=False)

final
from sklearn.cluster import KMeans

from sklearn.cluster import MiniBatchKMeans

from sklearn.cluster import AffinityPropagation



minikmeans = MiniBatchKMeans (n_clusters = 2).fit(X_data)

test_pred = minikmeans.fit_predict(test_points_normalize)

test_pred
from sklearn.cluster import KMeans

from sklearn.cluster import MiniBatchKMeans

from sklearn.cluster import AgglomerativeClustering



aggkomerativeclust = AgglomerativeClustering(n_clusters= 2).fit(X_data)

test_pred = aggkomerativeclust.fit_predict(test_points_normalize)
test_pred
from sklearn.cluster import KMeans

from sklearn.cluster import MiniBatchKMeans

from sklearn.cluster import Birch



birch = Birch(n_clusters= 2).fit(X_data)

test_pred = birch.fit_predict(test_points_normalize)
test_pred
train= pd.read_csv("../input/eval-lab-3-f464/train.csv")

#drop init



# drop_init = ['gender', 'PaymentMethod', 'TotalCharges']

# for i in drop_init:

#     train = train.drop(i, axis = 1)

     

for i in range(len(train.TotalCharges)):

    if (len(train.TotalCharges[i].strip(" ")) == 0):

        train.TotalCharges[i] = str(train.tenure[i] * train.MonthlyCharges[i])

        print (i, end = " ")

        print (train.TotalCharges[i], end=" ")

        print (train.tenure[i], end=" ")

        print (train.MonthlyCharges[i])

train.TotalCharges = pd.to_numeric(train.TotalCharges)

print (train.TotalCharges.dtypes)

train.head()



# for one-hot-encoding

count = 1

for i in train.columns:

    if (train[i].dtypes == "object"):

        onehot = pd.get_dummies(train[i])

        for column in onehot.columns:

            train[column+str(count)] = onehot[column]

#             print (column)

            count = count+1

        train = train.drop(i,axis = 1)

train.head()



#drop custId

train.head()

train = train.drop("custId", axis = 1)

train.head()

    

# #remove opposite    

# remove_opposite = ['Male2','Yes4', 'Yes6', 'Yes29', 'Yes34']

# for i in remove_opposite:

#     train = train.drop(i, axis = 1)





# #repeat drop

# repeat_drop = ["No tv connection11", "No tv connection14", "No tv connection17",  "No tv connection20", "No tv connection23",

#                  "No tv connection26", "No internet31"]

# for i in repeat_drop:

#     train = train.drop(i, axis = 1)

    

    

# #drop these insignificant

# drop_insignificant = ['SeniorCitizen','MonthlyCharges','Female1','No3','No5','DTH8','No10','Yes12','No13',

#                       'Yes15','Yes18','Yes21','Yes24','Yes27','No28','No30','Yes32','No33','Annually35','Bank transfer38',

#                       'Cash39','Credit card40']

# for i in drop_insignificant:

#     train = train.drop(i, axis = 1)

    

#create X and y

a = list(train.columns)

a.remove("Satisfied")

X = train[a]

y = train["Satisfied"]



test= pd.read_csv("../input/eval-lab-3-f464/test.csv")





for i in range(len(test.TotalCharges)):

    if (len(test.TotalCharges[i].strip(" ")) == 0):

        test.TotalCharges[i] = str(test.tenure[i] * test.MonthlyCharges[i])

        print (i, end = " ")

        print (test.TotalCharges[i], end=" ")

        print (test.tenure[i], end=" ")

        print (test.MonthlyCharges[i])

test.TotalCharges = pd.to_numeric(test.TotalCharges)

print (test.TotalCharges.dtypes)

test.head()





# for one-hot-encoding

count = 1

for i in test.columns:

    if (test[i].dtypes == "object"):

        onehot = pd.get_dummies(test[i])

        for column in onehot.columns:

            test[column+str(count)] = onehot[column]

#             print (column)

            count = count+1

        test = test.drop(i,axis = 1)

test.head()



#drop custId

test.head()

test = test.drop("custId", axis = 1)

test.head()



    

#remove opposite    

# remove_opposite = ['Male2','Yes4', 'Yes6', 'Yes29', 'Yes34']

# for i in remove_opposite:

#     test = test.drop(i, axis = 1)





#repeat drop

# repeat_drop = ["No tv connection11", "No tv connection14", "No tv connection17",  "No tv connection20", "No tv connection23",

#                  "No tv connection26", "No internet31"]

# for i in repeat_drop:

#     test = test.drop(i, axis = 1)   

    

# #drop these insignificant

# drop_insignificant = ['SeniorCitizen','MonthlyCharges','Female1','No3','No5','DTH8','No10','Yes12','No13',

#                       'Yes15','Yes18','Yes21','Yes24','Yes27','No28','No30','Yes32','No33','Annually35','Bank transfer38',

#                       'Cash39','Credit card40']

# for i in drop_insignificant:

#     test = test.drop(i, axis = 1)





#create X and y

a = list(test.columns)

test_points = test[a]

test_points
from sklearn.cluster import KMeans

from sklearn.metrics import accuracy_score



kmeans = KMeans(n_clusters = 2, random_state=2).fit(test_points)

test_pred = kmeans.fit_predict(test_points)

# accuracy_list.append(accuracy_score(y, train_pred))

custId = pd.read_csv("../input/eval-lab-3-f464/test.csv")

final = pd.DataFrame(custId["custId"],columns = ["custId"])

final["Satisfied"] = test_pred

final.to_csv("dropped2.csv", index=False)
import seaborn as sns



plt.figure(figsize=(20, 10))

corr = train.corr()

ax = sns.heatmap(

    corr, 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 220, n=200),

    square=True

)

ax.set_xticklabels(

    ax.get_xticklabels(),

    rotation=45,

    horizontalalignment='right'

);