import numpy as np

train_array = np.genfromtxt("../input/train_numeric.csv",delimiter=',',filling_values=np.NaN,skip_header=1,max_rows=100000)
#Check if defect percentage are propotionate in the chosen subset!

test_array= np.genfromtxt("../input/train_numeric.csv",delimiter=',',filling_values=np.NaN,skip_header=100001,max_rows=30000)

test_rate = 100*sum(test_array[:,-1] ==1 )/len(test_array[:,-1])

train_rate= 100*sum(train_array[:,-1] ==1 )/len(train_array[:,-1])

print(test_rate,train_rate)
#Set up test/train features, Impute NaN's to values that scikit learn can work with!

train_features = train_array[:,1:-1]

test_features = test_array[:,1:-1]

from sklearn.preprocessing import Imputer

imp = Imputer(missing_values='NaN', strategy='mean', axis=0) #Scikit-learn cant deal with NaN's

imp.fit(train_features)

train_features= imp.transform(train_features)

#Test set

imp.fit(test_features)

test_features=imp.transform(test_features)
#Feature reduction using tree model

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.feature_selection import SelectFromModel

clf = ExtraTreesClassifier()

clf = clf.fit(train_features, train_array[:,-1])
#Analyze reduced datasets

len(clf.feature_importances_ ) 

model = SelectFromModel(clf, prefit=True)

train_new = model.transform(train_features)

test_new= model.transform(test_features)

print(train_new.shape) #Shape of the reduced array 968 --> 266
#Extract Feature ranks

importances = clf.feature_importances_

std = np.std([tree.feature_importances_ for tree in clf.estimators_],

             axis=0)



indices = np.argsort(importances)[::-1] #Sort features on the basis of rank
import matplotlib.pyplot as plt

# Print & plotthe feature ranking

print("Feature ranking:")



for f in range(train_features.shape[1]):

    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))



# Plot the feature importances of the forest

plt.figure()

plt.title("Feature importances")

plt.bar(range(train_features.shape[1]), importances[indices],

       color="r", yerr=std[indices], align="center")

plt.xticks(range(train_features.shape[1]), indices)

plt.xlim([-1, train_features.shape[1]])

plt.show()
#Fit the new model with reduced features 

clf_rev = ExtraTreesClassifier()

clf_rev = clf_rev.fit(train_new, train_array[:,-1])



#make predictions using reduced model

prediction = clf_rev.predict(test_new)

print(prediction == test_array[:,-1]) 

success = 100.*sum(prediction == test_array[:,-1])/float(len(test_array[:,-1]))

print("\n The  predictions are  %.4f %% accurate" %(success) )
#Compute MCC to check the quality of prediction

from sklearn.metrics import matthews_corrcoef



matthews_corrcoef(test_array[:,-1],prediction )  
clear