import numpy as np

import pandas as pd 

import seaborn as sns

from sklearn.model_selection import train_test_split,KFold

from sklearn.preprocessing import LabelEncoder

from lightgbm import LGBMClassifier

from xgboost import XGBClassifier

import eli5

from eli5.sklearn import PermutationImportance

from sklearn.metrics import accuracy_score



#Importing Train and Test

train_df=pd.read_csv('/kaggle/input/customer-segmentation/Train_aBjfeNk.csv')

test_df=pd.read_csv('/kaggle/input/customer-segmentation/Test_LqhgPWU.csv')
train_df.info()
test_df.info()
sns.pairplot(train_df)
combine_set=pd.concat([train_df,test_df],ignore_index=True)

combine_set['Ever_Married'].fillna('unknown',inplace=True)

combine_set['Graduated'].fillna('unknown',inplace=True)

combine_set['Profession'].fillna('unknown',inplace=True)

combine_set['Work_Experience'].fillna(combine_set['Work_Experience'].mode()[0], inplace=True)

combine_set['Family_Size'].fillna(0,inplace=True)

combine_set['Var_1'].fillna('Cat_0',inplace=True)

combine_set.head(5)
#Adding more Features

combine_set['Unique_profession_per_agegroup']=combine_set.groupby(['Age'])['Profession'].transform('nunique')

combine_set['Unique_agegroup_per_profession']=combine_set.groupby(['Profession'])['Age'].transform('nunique')

combine_set['Age_Family_size']=combine_set.groupby(['Age'])['Family_Size'].transform('nunique')

combine_set.head(5)
combine_set['Var_1'].value_counts()
combine_set_ann=pd.get_dummies(combine_set,columns=['Gender','Ever_Married','Graduated','Spending_Score','Profession','Var_1'],drop_first=True)

print(combine_set_ann.shape)

combine_set_ann.head(10)
#Encoding Category Variables

def frequency_encoding(col):

    fe=combine_set.groupby(col).size()/len(combine_set)

    combine_set[col]=combine_set[col].apply(lambda x: fe[x])

#     le=LabelEncoder()

#     combine_set[col]=le.fit_transform(combine_set[col])
for col in list(combine_set.select_dtypes(include=['object']).columns):

    if col!='Segmentation':

        frequency_encoding(col)

    

    

combine_set.head(5)

    
train_df=combine_set[combine_set['Segmentation'].isnull()==False]

train_ann=combine_set_ann[combine_set_ann['Segmentation'].isnull()==False]

test_df=combine_set[combine_set['Segmentation'].isnull()==True]

test_ann=combine_set_ann[combine_set_ann['Segmentation'].isnull()==True]





# 90% Train Data is repeated in Test set so seperating the ID's which are common both in test and train set

submission_df=pd.merge(train_df,test_df,on='ID',how='inner')

submission_df=submission_df[['ID','Segmentation_x']]

submission_df.columns=['ID','Segmentation']

submission_ann=pd.merge(train_ann,test_ann,on='ID',how='inner')

submission_ann=submission_ann[['ID','Segmentation_x']]

submission_ann.columns=['ID','Segmentation']





# le=LabelEncoder()

# train_df['Segmentation']=le.fit_transform(train_df['Segmentation'])

print(submission_df.shape)

submission_df.head(5)

# Creating Train and Test Data

X=train_df.drop(['Segmentation'],axis=1)

X_ann=train_ann.drop(['ID','Segmentation'],axis=1)

Y=train_df['Segmentation']

Y_ann=pd.get_dummies(Y)



md_df=pd.concat([pd.DataFrame(submission_df['ID']),pd.DataFrame(test_df['ID'])]).drop_duplicates(keep=False)



test_df=pd.merge(md_df,test_df,on='ID',how='inner')



md_ann=pd.concat([pd.DataFrame(submission_ann['ID']),pd.DataFrame(test_ann['ID'])]).drop_duplicates(keep=False)



test_ann=pd.merge(md_ann,test_ann,on='ID',how='inner')

X_main_test=test_df.drop(['Segmentation'],axis=1)

X_main_ann=test_ann.drop(['ID','Segmentation'],axis=1)





X_train,X_val,Y_train,Y_val=train_test_split(X,Y,test_size=0.2,random_state=100)
X_ann.head(5)
from keras.models import Sequential

from keras.layers import Dense



classifier=Sequential()

classifier.add(Dense(256,activation='relu',input_shape=(X_ann.shape[1],)))

classifier.add(Dense(128,activation='relu'))

classifier.add(Dense(32,activation='relu'))

classifier.add(Dense(4,activation='softmax'))

classifier.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])



classifier.fit(X_ann,Y_ann,epochs=50,batch_size=32)



# preds=classifier.predict(X_main_test)
preds=classifier.predict(X_main_ann)



lis=[]

for item in preds:

    index,value=max(enumerate(item),key=lambda x: x[1])

    if index==0:

        lis.append('A')

    elif index==1:

        lis.append('B')

    elif index==2:

        lis.append('C')

    else:

        lis.append('D')

lg=LGBMClassifier(boosting_type='gbdt', max_depth=10, learning_rate=0.09, objective='multiclass', reg_alpha=0,

                  reg_lambda=1, n_jobs=-1, random_state=100, n_estimators=1000)



lg.fit(X,Y)



# print(accuracy_score(Y_val,lg.predict(X_val)))
from catboost import CatBoostClassifier



cb=CatBoostClassifier(learning_rate=0.05,depth=8,boosting_type='Plain',eval_metric='Accuracy',n_estimators=1000,random_state=294)

cb.fit(X,Y)

# print(accuracy_score(Y_val,xg.predict(X_val)))
xg=XGBClassifier(booster='gbtree',verbose=0,learning_rate=0.07,max_depth=8,objective='multi:softmax',

                  n_estimators=1000,seed=294)

xg.fit(X,Y)

# print(accuracy_score(Y_val,xg.predict(X_val)))

perm = PermutationImportance(xg,random_state=100).fit(X_val, Y_val)

eli5.show_weights(perm,feature_names=X_val.columns.tolist())
d=pd.DataFrame()

d=pd.concat([d,pd.DataFrame(cb.predict(X_main_test)),pd.DataFrame(xg.predict(X_main_test)),pd.DataFrame(lg.predict(X_main_test))],axis=1)

d.columns=['1','2','3']



re=d.mode(axis=1)[0]

re.head(5)
submission_dataframe=pd.DataFrame()



submission_dataframe['ID']=test_df['ID']

submission_dataframe['Segmentation']=np.array(re)

submission_dataframe=pd.concat([submission_df,submission_dataframe])

submission_dataframe.to_csv('/kaggle/working/main_test.csv', index=False)