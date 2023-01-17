import pandas as pd
import numpy as np
add = "../input/mushrooms.csv"
data = pd.read_csv(add)

# seperating X vaules from y values
X= data.iloc[:,1:]
y = data.iloc[:,0]
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
d = defaultdict (LabelEncoder)
Xfit = X.apply(lambda x: d[x.name].fit_transform(x))
le_y = LabelEncoder()
yfit = le_y.fit_transform(y)
# for x in Xfit.columns:
#     print(x)
#     print(Xfit[x].value_counts())
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import OneHotEncoder
ohc = defaultdict (OneHotEncoder)
# Xfit_ohc = Xfit.apply(lambda x: ohc[x.name].fit_transform(x))
final = pd.DataFrame()

for i in range(22):
    # transforming the columns using One hot encoder
    Xtemp_i = pd.DataFrame(ohc[Xfit.columns[i]].fit_transform(Xfit.iloc[:,i:i+1]).toarray())
   
    #Naming the columns as per label encoder
    ohc_obj  = ohc[Xfit.columns[i]]
    labelEncoder_i= d[Xfit.columns[i]]
    Xtemp_i.columns= Xfit.columns[i]+"_"+labelEncoder_i.inverse_transform(ohc_obj.active_features_)
    
    # taking care of dummy variable trap
    X_ohc_i = Xtemp_i.iloc[:,1:]
    
    #appending the columns to final dataframe
    final = pd.concat([final,X_ohc_i],axis=1)
final.shape
final.head(20)
final[1:4]
data[1:4]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(final, yfit, test_size = 0.1, random_state = 0)
from sklearn.neighbors import KNeighborsClassifier
classifier =  KNeighborsClassifier(n_neighbors=30,p=2, metric='minkowski')
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test,y_pred)
cm
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)
classif =  KNeighborsClassifier(n_neighbors=200,p=2, metric='minkowski')
classif.fit(X_train,y_train)
y_pred = classif.predict(X_test)
accuracy_score(y_test,y_pred)
from sklearn.model_selection import cross_val_score

# creating odd list of K for KNN
myList = list(range(1,200))

# empty list that will hold cv scores
cv_scores = []

# perform 10-fold cross validation
for k in myList[::2]:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

from matplotlib import pyplot as plt
# changing to misclassification error
MSE = [1 - x for x in cv_scores]

# determining best k
optimal_k = myList[::2][MSE.index(min(MSE))]
print ("The optimal number of neighbors is %d" % optimal_k)

# plot misclassification error vs k
plt.plot(myList[::2], MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()
n_features = final.shape[1]
clf = KNeighborsClassifier()
feature_score = []

for i in range(n_features):
    X_feature= np.reshape(final.iloc[:,i:i+1],-1,1)
    scores = cross_val_score(clf, X_feature, yfit)
    feature_score.append(scores.mean())
    print('%40s        %g' % (final.columns[i], scores.mean()))

feat_imp = pd.Series(data = feature_score, index = final.columns)
feat_imp.sort_values(ascending=False, inplace=True)
feat_imp[feat_imp>0.7]


columns_imp = feat_imp[feat_imp>0.7].index.values
final_Xy= pd.concat([final,pd.DataFrame(yfit,columns=['class'])], axis=1)
grouped = final_Xy.groupby('class')
# Edible group of mushrooms
grouped.get_group(0)[columns_imp].sum()
# Poisonous group of mushrooms
grouped.get_group(1)[columns_imp].sum()