import warnings

warnings.filterwarnings('ignore')







import pandas

dataset = pandas.read_csv("../input/train.csv") 

dataset_test = pandas.read_csv("../input/test.csv")

ID = dataset_test['id']



dataset_test.drop('id',axis=1,inplace=True)

pandas.set_option('display.max_rows', None)

pandas.set_option('display.max_columns', None)



print(dataset.head(5))
import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt



%matplotlib inline

labels = []

values = []

for col in dataset_test.columns:

    labels.append(col)

    values.append(dataset[col].isnull().sum())

    print(col, values[-1])

ind = np.arange(len(labels))

width = 0.9

fig, ax = plt.subplots(figsize=(12,50))

rects = ax.barh(ind, np.array(values), color='y')

ax.set_yticks(ind+((width)/2.))

ax.set_yticklabels(labels, rotation='horizontal')

ax.set_xlabel("Count of missing values")

ax.set_title("Number of missing values in each column")

#autolabel(rects)

plt.show()
dataset.head()
dataset = dataset.iloc[:,1:]
#cont data

data=dataset.iloc[:,116:] 

size = 15

data_corr = data.corr()



# Set the threshold to select only highly correlated attributes

threshold = 0.5



# List of pairs along with correlation above threshold

corr_list = []



#Search for the highly correlated pairs

for i in range(0,size): #for 'size' features

    for j in range(i+1,size): #avoid repetition

        if (data_corr.iloc[i,j] >= threshold and data_corr.iloc[i,j] < 1) or (data_corr.iloc[i,j] < 0 and data_corr.iloc[i,j] <= -threshold):

            corr_list.append([data_corr.iloc[i,j],i,j]) #store correlation and columns index



#Sort to show higher ones first            

s_corr_list = sorted(corr_list,key=lambda x: -abs(x[0]))



#Print correlations and column names

for v,i,j in s_corr_list:

    print ("%s and %s = %.2f" % (data.columns[i],data.columns[j],v))
import scipy.stats as scs

from sklearn.preprocessing import LabelEncoder as LE



def categories(series):

    return range(int(series.min()), int(series.max()) + 1)





def chi_square_of_df_cols(df, col1, col2):

    df_col1, df_col2 = df[col1], df[col2]

    le1 = LE()

    df_col1 = le1.fit_transform(df_col1)

    df_col2 = le1.fit_transform(df_col2)



    result = [[sum((df_col1 == cat1) & (df_col2 == cat2))

               for cat2 in categories(df_col2)]

              for cat1 in categories(df_col1)]



    return scs.chi2_contingency(result)

#colsCat = dataset.columns[:116]

#corr_list2 = []

#for i in range(0,116): #for 'size' features

#    for j in range(i+1,116): #avoid repetition

#        ch = chi_square_of_df_cols(dataset,colsCat[i],colsCat[j])[0]

#        print("Correlation between " + colsCat[i] + " " + colsCat[j] +" " + str(ch))

#        corr_list2.append([ch,i,j]) #store correlation and columns index



#Sort to show higher ones first            

#s_corr_list2 = sorted(corr_list2,key=lambda x: -abs(x[0]))



#Print correlations and column names

#for v,i,j in s_corr_list2:

    #print ("%s and %s = %.2f" % (data.columns[i],data.columns[j],v))
y = dataset.loss

dataset.drop(["loss"], axis=1, inplace=True)
dataset.head()
shapeTrain = dataset.shape
data = dataset.append(dataset_test)
catCols = data.columns[:116]

WasteCat = []

for col in catCols:

    if data[col].value_counts()[0] > 310000:

        print(col)

        WasteCat.append(col)

print(len(WasteCat))
print(data.shape[0]*0.99)
data.drop(WasteCat,axis=1,inplace=True)

data.drop(["cont1","cont11"], axis=1, inplace=True)
data.head()
catCols = data.columns[:84]

for col in catCols:

    print(col)

    data[col] = LE().fit_transform(data[col])
data.shape
y = np.log1p(y)
Xtrain = data.iloc[:shapeTrain[0],:]
from sklearn.ensemble import GradientBoostingRegressor as GBR, RandomForestRegressor as RFG

model = RFG(n_estimators=40,n_jobs=4)

model2 = GBR(n_estimators=120)
from sklearn.cross_validation import train_test_split as tts

X_train,X_test,y_train,y_test = tts(Xtrain,y,test_size=0.0,random_state = 1)
model2.fit(X_train,y_train)

print("Gradient Fitted")

model.fit(X_train,y_train)

print("Random Fitted")


from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse

Xtest = data.iloc[shapeTrain[0]:,:]

#y_test = np.expm1(y_test)

#y_train = np.expm1(y_train)

#predTestRandom = model.predict(X_test)

#predTestGrad = model2.predict(X_test)

#predTrainRandom = model.predict(X_train)

#predTrainGrad = model2.predict(X_train)
#predTest =  (predTestRandom*0.43 + 0.57*predTestGrad)

#predTest = (np.expm1(predTest))

#predTrain = (predTrainRandom*0.5 + predTrainGrad*0.5)

#predTrain = (np.expm1(predTrain))
#print(mae(y_test,pred))

#print(mae(y_test,predTest))

#print(mae(y_train,predTrain))

#print(predTestRandom[0])
from xgboost import XGBRegressor as XGBR

X_train,X_test,y_train,y_test = tts(Xtrain,y,test_size=0.0,random_state=1)

model3 = XGBR(n_estimators=1000,nthread=4)

model3.fit(X_train,y_train)
#y_test = np.expm1(y_test)

#y_train = np.expm1(y_train)

#predTrainXGB= model3.predict(X_train)

#predTestXGB = model3.predict(X_test)
#predTest =  (predTestRandom*0.23 + 0.03*predTestGrad + 0.74*predTestXGB)

#predTest = (np.expm1(predTest))

#predTrain = (predTrainRandom*0.23 + predTrainGrad*0.07 + predTrainXGB*0.7)

#predTrain = (np.expm1(predTrain))
#print(mae(y_test,predTestXGB))

#print(mae(y_train,predTrainXGB))
predAns = model3.predict(Xtest)
predAns = np.expm1(predAns)
output = pandas.DataFrame({"id":ID,"loss":predAns})
output.to_csv("output.csv",index=False)