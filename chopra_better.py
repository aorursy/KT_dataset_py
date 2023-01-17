#This code helps to find out feature multipliers for KNN.
#This is shown using some features derived by me but this method can be extended for other features as well.
#One needs to derive his own features and then apply similar approach to get the correct weights.


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

recent_train = pd.read_csv("../input/train.csv")

#select a single x_y_grid at random
recent_train = recent_train[(recent_train["x"]>4.5) &(recent_train["x"]<5) &(recent_train["y"]>2) &(recent_train["y"]<2.3)]


#derive some features
recent_train["x"],recent_train["y"] = recent_train["x"]*1000,recent_train["y"]*1000
recent_train["hour"] = recent_train["time"]//60
recent_train["hour_of_day"] = recent_train["hour"]%24 + 1

recent_train["day"] = recent_train["hour"]//24
recent_train["day_of_week"] = recent_train["day"]%7 + 1

recent_train["month"] = recent_train["day"]//30 + 1
recent_train["month_of_year"] = (recent_train["month"]-1)%12 + 1 

recent_train["accuracy"] = np.log(recent_train["accuracy"])

print("recent_train created")
#creating arbitrary test
test = recent_train.sample(axis = 0,frac = 0.05)
print ("selected_part and test created")
features = ["x","y","hour_of_day","day_of_week","month_of_year","accuracy"]
fw = [1,1,0,0,0,0] #at first iteration
fw = [ 1.  ,      1.43696871,  1.26585057,  0.78156837,  1.28182383,  7.01014412] #at second iteration

print (len(test))
colname = str(features)
test[colname] = list
index = iter(test.index)
test["done"] = 0
count = 0
for i in index:
    new_ld = pd.DataFrame(columns = features)
    for j in range(15):
        new_ld1 = abs(recent_train[features] - test.loc[i][features])
        new_ld1 = new_ld1.drop(i)
        new_ld1["target"] = (recent_train["place_id"] != test.loc[i]["place_id"]) + 0
        new_ld1["x+y"] = np.sum(new_ld1[features]*fw,axis = 1)#(new_ld1["x"])+(new_ld1["y"])#
        new_ld1 = new_ld1.sort("x+y")[0:50]
        count += 1
        i = next(index)
        true = new_ld1[new_ld1["target"] == 0]
        false = new_ld1[new_ld1["target"] != 0]
        if (len(true)< 10) | (len(false)< 10):
            continue
        new_ld = new_ld.append(new_ld1)
    lr.fit(new_ld[features],new_ld["target"])
    test.set_value(i,colname,lr.coef_.ravel())
    test.set_value(i,"done",1)
    print ("current status: sample number",count)
#average or sum all the multipliers to get overall multiplier
actual_test2 = test[test["done"]==1]
final_weights = np.array([0,0,0,0,0,0])
for lists in actual_test2[colname]:
    final_weights = final_weights + lists


print (features) 
print ("corresponding weights")
print (final_weights/final_weights[0])
