import pandas as pd
import numpy as np 
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from catboost import Pool, CatBoostClassifier, cv
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
%matplotlib inline
train_df = pd.read_csv('../input/train_8wry4cB.csv')
test_df = pd.read_csv('../input/test_Yix80N0.csv')

target = train_df['gender'].copy() ## Target !
display(train_df.head())
display(test_df.head())
display(target.head())


#Displaying Train Data !
#Displaying Test Data !
#Displaying Target Column !
train_df.dtypes
test_df.dtypes
train_df.describe()
test_df.describe()
X = train_df.copy()
X_test = test_df.copy()
#Dropping Gender and Session_id Columns.

X.drop(['session_id', 'gender'], axis = 1, inplace = True)
X_test.drop(['session_id'], axis = 1, inplace = True)
display(X.head())
display(X_test.head())

# call this to get group of product codes for each user_id

def product_group(x):
    prod_list = []
    for i in x.split(';'):
        prod_list.append(i.split('/')[-2])
    return prod_list


# call this to fetch the most frequent D code used.
def frequency(List): 
    return max(set(List), key = List.count)


# Extracting the D codes. 
def final_items(x):
    prod_code = []
    for i in x:
        prod_code.append(i[:4])
    return frequency(prod_code)
def CleanIt(data):

    #PRODUCTS
    data['Category_A'] = data['ProductList'].apply(lambda x : x.split(';')[0].split('/')[0])
    data['Category_B'] = data['ProductList'].apply(lambda x : x.split(';')[0].split('/')[1])
    data['Category_C'] = data['ProductList'].apply(lambda x : x.split(';')[0].split('/')[2])
    
    #Calling Function on ProductList to populate Product table.
    data['Items'] =  data['ProductList'].apply(lambda x: product_group(x))
    data['Product Code'] = data['Items'].apply(lambda x: final_items(x))
    data['Total_Products_Viewed'] = data['ProductList'].apply(lambda x: len(x.split(';'))) 
    
    #TIME
                
    data['startTime'] = pd.to_datetime(data['startTime'], format = "%d/%m/%y %H:%M")
    data['endTime']   = pd.to_datetime(data['endTime'], format = "%d/%m/%y %H:%M")
    data['StartHour'] = data['startTime'].dt.hour
    data['StartMinute'] = data['startTime'].dt.minute
    data['EndHour'] = data['endTime'].dt.hour
    data['EndMinute'] = data['endTime'].dt.minute
    data['Duration_InSeconds']  = (data['endTime']-data['startTime'])/np.timedelta64(1,'s')
    
    
    return data 
    
    


trainset = X.copy()
testset = X_test.copy()
trainset = CleanIt(trainset)
testset = CleanIt(testset)
print("----------------------------- Info on Processed Training Data--------------------")
display(trainset.shape)
display(trainset.dtypes)
display(trainset.describe())
display(trainset.info())
display(trainset.head())
print("******************************Info on Processed Testing Data***********************")
display(testset.shape)
display(testset.dtypes)
display(testset.describe())
display(testset.info())
display(testset.head())
#Dropping Unecessary Columns now - ProductList and Items.

trainset.drop(['ProductList', 'Items', 'startTime', 'endTime'], axis = 1 , inplace = True)
testset.drop(['ProductList', 'Items', 'startTime','endTime'], axis = 1, inplace = True)
display(trainset.head())
display(testset.head())
trainset1 = trainset.copy()
trainset1['gender'] = train_df['gender'].copy()
plt.figure(figsize=(25,15))
sns.countplot('Category_A', data = trainset1, hue = 'gender')
plt.legend(loc = 'center')
plt.figure(figsize=(25,15))
sns.countplot('Category_B', data = trainset1.head(30), hue = 'gender')
plt.legend(loc = 'center')
plt.figure(figsize=(15,10))
sns.countplot('StartHour', data = trainset1, hue = 'gender')
plt.legend(loc = 'upper left')
plt.figure(figsize=(20,5))
sns.lineplot(x='StartHour', y = 'Total_Products_Viewed', data = trainset1, hue = 'gender')
sns.lmplot(x='StartHour', y = 'Total_Products_Viewed', data = trainset1, hue = 'gender')
display(trainset1.Duration_InSeconds.describe())
display(trainset1.Duration_InSeconds.max())
display(trainset1.Duration_InSeconds.min())
#You will Get a warning message if you try to index this group with Multiple Keys. Avoid it by passing keys into a list.

group_AvgTimeSpend = trainset1.groupby('gender')[['Duration_InSeconds','Total_Products_Viewed']]

#For Males.
group_male = pd.DataFrame(group_AvgTimeSpend.get_group('male'))
display(group_male.sort_values('Total_Products_Viewed', ascending=False))
t1 = group_male.sort_values('Total_Products_Viewed', ascending=False)
plt.figure(figsize=(20,10))
sns.lineplot(x='Duration_InSeconds', y = 'Total_Products_Viewed', sort = False, data = t1)
plt.title('For Males')
display(group_male.sort_values('Duration_InSeconds', ascending = True))
t2 = group_male.sort_values('Duration_InSeconds', ascending = True)
sns.lineplot(x='Duration_InSeconds', y = 'Total_Products_Viewed', sort = False, data = t2)
group_female = pd.DataFrame(group_AvgTimeSpend.get_group('female'))
display(group_female.sort_values('Total_Products_Viewed', ascending=False))
t3 = group_female.sort_values('Total_Products_Viewed', ascending=False)
plt.figure(figsize=(20,10))
sns.lineplot(x='Duration_InSeconds', y = 'Total_Products_Viewed', sort = False, data = t3)
plt.title('For Females')
display(group_female.sort_values('Duration_InSeconds', ascending = True))
t4 = group_female.sort_values('Duration_InSeconds', ascending = True)
sns.lineplot(x='Duration_InSeconds', y = 'Total_Products_Viewed', sort = False, data = t4)

plt.figure(figsize=(10,5))
sns.lineplot(x='Duration_InSeconds', y = 'Total_Products_Viewed', data = trainset1, hue = 'gender')
print(trainset.shape, testset.shape)
display(trainset.head())
display(trainset.dtypes)
display(testset.head())
display(trainset.dtypes)

#### MODEL 

cate_features_index = np.where(trainset.dtypes != float)[0]
X1_train, X1_test, y1_train, y1_test = train_test_split(trainset, target, train_size=0.85,random_state=1200)
from catboost import CatBoostClassifier


cat = CatBoostClassifier(eval_metric='Accuracy',
                         use_best_model=True,random_seed=40,loss_function='MultiClass',
                         learning_rate = 0.674 ,iterations = 700,depth = 4,
                         bagging_temperature=3,one_hot_max_size=2)



#Parameters to be tuned :- 
#1. learning Rate
#2. Training Size(As data is biased)
#3. Iterations
#4. OHE size
#5. depth of tree

cat.fit(X1_train,y1_train ,cat_features=cate_features_index,eval_set=(X1_test,y1_test),use_best_model=True)
print('the test accuracy is :{:.6f}'.format(accuracy_score(y1_test,cat.predict(X1_test))))
predcat = cat.predict(X1_test)
print("----------------------------------------------------------------------------")
print('Training set score: {:.4f}'.format(cat.score(X1_train, y1_train)))
print('Test set score: {:.4f}'.format(cat.score(X1_test, y1_test)))


matrix = confusion_matrix(y1_test, predcat)
print("--------------------------------------------------------------------------------------------")
print('Confusion matrix\n\n', matrix)
print('\nTrue Positives(TP) Females  = ', matrix[0,0])
print('\nTrue Negatives(TN)  Males = ', matrix[1,1])
print('\nFalse Positives(FP) = ', matrix[0,1])
print('\nFalse Negatives(FN) = ', matrix[1,0])
preds = cat.predict(testset)
pred1 = preds.flatten()
predlst = pred1.tolist()
output = pd.DataFrame({'session_id': test_df.session_id,'gender': predlst})
output.to_csv('cleaned.csv', index=False)
sns.countplot(predlst)
pd.Series(predlst).value_counts()
