import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.optim import Adam,SGD,Adagrad
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import LinearSVC
import xgboost as xgb
import os
print(os.listdir("../input"))
%matplotlib inline
train_data = pd.read_csv('../input/train.csv')

def prep_test():
    test_csv = pd.read_csv('../input/test.csv')
    temp_data = train_data
    ####################### Test data #############################################
    test_csv['HF1'] = test_csv['Horizontal_Distance_To_Hydrology'] + test_csv['Horizontal_Distance_To_Fire_Points']
    test_csv['HF2'] = abs(test_csv['Horizontal_Distance_To_Hydrology']-test_csv['Horizontal_Distance_To_Fire_Points'])
    test_csv['HR1'] = abs(test_csv['Horizontal_Distance_To_Hydrology']+test_csv['Horizontal_Distance_To_Roadways'])
    test_csv['HR2'] = abs(test_csv['Horizontal_Distance_To_Hydrology']-test_csv['Horizontal_Distance_To_Roadways'])
    test_csv['FR1'] = abs(test_csv['Horizontal_Distance_To_Fire_Points']+test_csv['Horizontal_Distance_To_Roadways'])
    test_csv['FR2'] = abs(test_csv['Horizontal_Distance_To_Fire_Points']-test_csv['Horizontal_Distance_To_Roadways'])
    test_csv['ele_vert'] = test_csv.Elevation - test_csv.Vertical_Distance_To_Hydrology

    test_csv['slope_hyd'] = (test_csv['Horizontal_Distance_To_Hydrology']**2+test_csv['Vertical_Distance_To_Hydrology']**2)**0.5
    test_csv.slope_hyd=test_csv.slope_hyd.map(lambda x: 0 if np.isinf(x) else x) # remove infinite value if any

    #Mean distance to Amenities 
    test_csv['Mean_Amenities']=(test_csv.Horizontal_Distance_To_Fire_Points + test_csv.Horizontal_Distance_To_Hydrology + test_csv.Horizontal_Distance_To_Roadways) / 3 
    #Mean Distance to Fire and Water 
    test_csv['Mean_Fire_Hyd']=(test_csv.Horizontal_Distance_To_Fire_Points + test_csv.Horizontal_Distance_To_Hydrology) / 2
    test_csv.drop(['Soil_Type15' , "Soil_Type7"], inplace = True, axis = 1 )
    for x in to_normalize:
        mean = temp_data[x].mean()
        std = temp_data[x].std()
        test_csv[x]= test_csv[x].apply(lambda y: (y-mean) / std)
    return test_csv



####################### Train data #############################################
train_data['HF1'] = train_data['Horizontal_Distance_To_Hydrology']+train_data['Horizontal_Distance_To_Fire_Points']
train_data['HF2'] = abs(train_data['Horizontal_Distance_To_Hydrology']-train_data['Horizontal_Distance_To_Fire_Points'])
train_data['HR1'] = abs(train_data['Horizontal_Distance_To_Hydrology']+train_data['Horizontal_Distance_To_Roadways'])
train_data['HR2'] = abs(train_data['Horizontal_Distance_To_Hydrology']-train_data['Horizontal_Distance_To_Roadways'])
train_data['FR1'] = abs(train_data['Horizontal_Distance_To_Fire_Points']+train_data['Horizontal_Distance_To_Roadways'])
train_data['FR2'] = abs(train_data['Horizontal_Distance_To_Fire_Points']-train_data['Horizontal_Distance_To_Roadways'])
train_data['ele_vert'] = train_data.Elevation-train_data.Vertical_Distance_To_Hydrology

train_data['slope_hyd'] = (train_data['Horizontal_Distance_To_Hydrology']**2+train_data['Vertical_Distance_To_Hydrology']**2)**0.5
train_data.slope_hyd=train_data.slope_hyd.map(lambda x: 0 if np.isinf(x) else x) # remove infinite value if any

#Mean distance to Amenities 
train_data['Mean_Amenities']=(train_data.Horizontal_Distance_To_Fire_Points + train_data.Horizontal_Distance_To_Hydrology + train_data.Horizontal_Distance_To_Roadways) / 3 
#Mean Distance to Fire and Water 
train_data['Mean_Fire_Hyd']=(train_data.Horizontal_Distance_To_Fire_Points + train_data.Horizontal_Distance_To_Hydrology) / 2 
train_data.drop(['Soil_Type15' , "Soil_Type7"], inplace = True, axis = 1 )
to_normalize = list(train_data.drop(["Id", "Cover_Type"], inplace=False, axis=1).columns)


test_csv = prep_test()
train_data.head(5)
test_csv.head(5)

for x in to_normalize:
    mean = train_data[x].mean()
    std = train_data[x].std()
    train_data[x]= train_data[x].apply(lambda y: (y-mean)/ std)
    
train_data.head(5)
def print_acc(acc,model_name):
    print("{} validation accuracy is {:.4f}%".format(model_name, acc))
X_train, X_test, y_train, y_test = train_test_split(train_data.drop(["Id", "Cover_Type"], inplace=False, axis=1).as_matrix(), list(train_data["Cover_Type"].values), test_size=0.2)
neigh = KNeighborsClassifier(n_neighbors=10, weights='distance', p=1)
et = ExtraTreesClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=0)
clf2 = RandomForestClassifier(n_estimators=250,random_state=1)
clf3 = GaussianNB(var_smoothing=True)
clf4 = LinearSVC(random_state=5)
gbm =  xgb.XGBClassifier(max_depth=5, n_estimators=250, learning_rate=0.5)
estms = [('rf', clf2), ('xgb', gbm), ('neigh',  neigh), ('et', et)]
eclf1 = VotingClassifier(estimators=estms, voting='hard')
#for tag, voter in estms:
#    voter = voter.fit(X_train, y_train)
#    print_acc(float(np.array(voter.predict(X_test) == y_test, dtype=np.int).sum() * 100) / len(y_test), tag)
eclf1 = eclf1.fit(X_train, y_train)
print_acc(float(np.array(eclf1.predict(X_test) == y_test, dtype=np.int).sum() * 100) / len(y_test), "Voting")
preds = eclf1.predict(test_csv.drop(["Id"], axis=1).as_matrix())
result = pd.DataFrame(data={'Id': test_csv['Id'], 'Cover_Type': preds})
result.to_csv(path_or_buf='soft_voting_submittion.csv', index = False, header = True)