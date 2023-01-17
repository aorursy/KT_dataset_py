# Quick load dataset and check
import pandas as pd

path = "/home/michael/Dokumente/Uni/semester4/Machine_learning/projekt/"

filename = path + "train_set.csv"
data_train = pd.read_csv(filename)
filename = path + "test_set.csv"
data_test = pd.read_csv(filename)
#Max fill function for categorical columns

def maxFillColumn(colName, dataFrame):
    dataFrame[colName].fillna(dataFrame[colName].value_counts()
    .idxmax(), inplace=True)
    
     
def fillColumnsMax(dataFrame, ftype):  #cat, bin, ...
    for col in dataFrame.columns: 
        if ftype in col:
            #print(col)
            maxFillColumn(col, dataFrame)
            
            
def fillContColumnsMedian(dataFrame):
    for col in dataFrame.columns: 
        if ("cat" not in col) and ("bin" not in col) and (col != "target") and (col != "id"):
            #print(col)
            dataFrame[col].fillna(dataFrame[col].median(), inplace=True)

    
import numpy as np
from sklearn.utils import resample
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


#fill missing values

data_train = data_train.mask(data_train<0) 
data_test = data_test.mask(data_test<0)

#filling categorical and bin data with maxium occurance value
fillColumnsMax(data_train, "cat")
fillColumnsMax(data_train, "bin")

fillColumnsMax(data_test, "cat")
fillColumnsMax(data_test, "bin")


#fill missing continous data with median
fillContColumnsMedian(data_train)
fillContColumnsMedian(data_test)

#upsample training data
target_0 = data_train[data_train.target == 0]    #majority
target_1 = data_train[data_train.target == 1]    #minority
num_tar0 = len(target_0)
data_min_upsampled = resample(target_1,                           # upsample
                                 replace=True,            # sample with replacement
                                 n_samples=num_tar0,   # to match majority class
                                 random_state=123)        # reproducible results
data_train = pd.concat([target_0, data_min_upsampled])





######################

#split up features with different type

data_train_cat = data_train.filter(regex='cat')
data_train_cont = data_train.filter(regex='^(?!.*cat)(?!.*bin)(?!.*id)(?!.*target).*$')
data_train_bin = data_train.filter(regex='bin')


data_test_cat = data_test.filter(regex='cat')
data_test_cont = data_test.filter(regex='^(?!.*cat)(?!.*bin)(?!.*id)(?!.*target).*$')
data_test_bin = data_test.filter(regex='bin')




from sklearn.feature_selection import VarianceThreshold
#prepare categorical datac


#1-hot encoding for categorical data
enc = OneHotEncoder(handle_unknown='ignore')
data_train_cat = enc.fit_transform(data_train_cat).toarray()
data_test_cat = enc.transform(data_test_cat).toarray()





#pca to reduce columns with less informative data
pca = PCA()  #diagramm anschaun und einstellen n_components=20
pca.fit(data_train_cat)
data_train_cat_red =pca.transform(data_train_cat)
data_test_cat_red =pca.transform(data_test_cat)


from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt



#scale continuous data

scaler = MinMaxScaler()
scaler.fit(data_train_cont)
data_train_cont_scaled = scaler.transform(data_train_cont)
#same scaler for test data
data_test_cont_scaled = scaler.transform(data_test_cont)


#assemble prepared data

data_train_prep = np.concatenate((data_train_cat_red,data_train_cont_scaled, data_train_bin.to_numpy()),axis=1)
data_train_prep = pd.DataFrame(data_train_prep)

data_test_prep = np.concatenate((data_test_cat_red,data_test_cont_scaled, data_test_bin.to_numpy()),axis=1)
data_test_prep = pd.DataFrame(data_test_prep)


#data_train_prep.describe()
#data_test_prep.describe()
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

#split data into labeled train and testdata
data_Y = data_train['target']        #targets
data_X = data_train_prep         #features


x_train, x_val, y_train, y_val = train_test_split(data_X, data_Y, test_size = 0.3, shuffle = True)

#parameter tuning: https://medium.com/all-things-ai/in-depth-parameter-tuning-for-random-forest-d67bb7e920d
#tested for our data


###test with weigths
#ratio = (len(data_train['target']) - sum(data_train['target']))/sum(data_train['target'])
#print(ratio)
#, class_weight={0:1,1:1}


rfc = RandomForestClassifier(max_depth = 25, n_estimators=10, min_samples_leaf=0.2, min_samples_split=0.2)  
rfc = rfc.fit(x_train, y_train)



from sklearn.metrics import confusion_matrix, classification_report
y_pred = rfc.predict(x_val)

print(confusion_matrix(y_val, y_pred))
print(classification_report(y_val, y_pred))


data_test_X =  data_test_prep #data_test.drop(columns=['id'])
y_target = rfc.predict(data_test_X)
sum(y_target==0)
data_out = pd.DataFrame(data_test['id'].copy())
data_out.insert(1, "target", y_target, True) 
data_out.to_csv('submission.csv',index=False)
data_out

print(sum(y_target))




