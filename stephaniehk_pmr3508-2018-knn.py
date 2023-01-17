#import main libraries
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
from sklearn.model_selection import cross_val_score
#dataframe to train
poverty = pd.read_csv("../input/train.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
poverty.head()
poverty.shape
#Columns that have more than 50% of missing datas will be deleted
nulls = (poverty.isnull().sum()/9557).tolist()
delet_nulls = [] #Find the columns with more than 50% nulls and delete
correct_nulls =[] #Find the columns that have missing datas and correct

for i in range(len(nulls)):
    if nulls[i]>0 and nulls[i]<0.5:
        correct_nulls.append(i)
    if nulls[i]>=0.5:
        delet_nulls.append(i)
        
print(delet_nulls)
print(correct_nulls)
#Delete columns!
povertytrain = poverty
for md in range(len(delet_nulls)):
    povertytrain = povertytrain.drop(poverty.columns[delet_nulls[md]], axis=1)
povertytrain.head()
 #Replace missing datas for a integer
povertytrain = povertytrain.fillna(20)

#Replace strings datas for a integer
povertytrain = povertytrain.replace("no",0)
povertytrain = povertytrain.replace("yes",1)
povertytrain
Xpovertytrain = povertytrain.drop(["Id","Target","idhogar"], axis=1)
Ypovertytrain = povertytrain.Target
# Creating the model. Find the best k
# mean = 0
# for k in tqdm(range(50,150)):
#    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=4)
#    scores = cross_val_score(knn, Xpovertytrain, Ypovertytrain, cv=10, n_jobs=4)
#    if scores.mean() > mean:
#        mean = scores.mean()
#        bestk = k
# print(bestk,mean)

# After this, we found that the best k is 150!
knn = KNeighborsClassifier(n_neighbors=150)
knn.fit(Xpovertytrain,Ypovertytrain)
# Cross Validation
scores = cross_val_score(knn, Xpovertytrain, Ypovertytrain, cv=10)
scores
Test = pd.read_csv("../input/test.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
#Delet columns
povertytest = Test
for md in range(len(delet_nulls)):
    povertytest = povertytest.drop(poverty.columns[delet_nulls[md]], axis=1)
    
#Replace missing datas for a integer
povertytest = povertytest.fillna(20)

#Replace strings datas for a integer
povertytest= povertytest.replace("no",0)
povertytest = povertytest.replace("yes",1)
Xpovertytest = povertytest.drop(["Id","idhogar"], axis=1)
# Create the Target prediction based on the model learned in the trainig part
targetTest = knn.predict(Xpovertytest)
# Table [[Id][Taregt predict]]
Final = Test.Id
Final_sub = targetTest
outputDataFrame = pd.DataFrame({'Id':Final,'Target':Final_sub[:]})

outputDataFrame.to_csv('submission.csv', index=False)

