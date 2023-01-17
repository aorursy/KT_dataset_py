import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')



import os

print(os.listdir("../input"))
IS_LOCAL = False



if IS_LOCAL:

    PATH="Not yet"

else:

    PATH="../input/"
train = pd.read_csv(PATH+"train.csv")

test = pd.read_csv(PATH+"test.csv")
print("The Dataset's shape for train is {}, for test is {}".format(train.shape,test.shape))
train.head()
train.describe()
train.info()
def draw_kdeplot(column):



    plt.figure(figsize=[8,6])

    

    sns.kdeplot(train[column],bw=0.5,label='train')

    sns.kdeplot(test[column],bw=0.5,label='test')

    

    plt.xlabel(column,fontsize=12)

    plt.title(f"Distribution of {column}",fontsize=20)

    plt.show()
def make_count_df(df,column):

    dummy = df.copy()

    result_df = dummy[column].value_counts().sort_index().to_frame().reset_index().rename(columns={"index":column,column:"counts"})

    return result_df
def compare_categorical_ratio(count_train,count_test,column,adjust_x_annotate=5,fontsize=14):

    fig, ax = plt.subplots(1,2,figsize=[12,6])

    

    ax1 = plt.subplot(1,2,1)

    sns.barplot(x=column,y='counts',data=count_train,label='train')



    for p in ax1.patches:

        ax1.annotate('{:.2f}%'.format(p.get_height()/count_train["counts"].sum()) , (p.get_x()+p.get_width()/adjust_x_annotate, p.get_height()),fontsize=fontsize)



    ax2 = plt.subplot(1,2,2)

    sns.barplot(x=column,y='counts',data=count_test,label='test')



    for p in ax2.patches:

        ax2.annotate('{:.2f}%'.format(p.get_height()/count_test["counts"].sum()) , (p.get_x()+p.get_width()/adjust_x_annotate, p.get_height()),fontsize=fontsize)



    plt.suptitle(f"Comparing btw train and test about {column}")

    plt.show()
train.date.head(10)
train.date.apply(lambda x:str(x)[-7:]).value_counts()
print("Minimum value of Price is {}, Maximum value of Price is {}".format(train.price.min(),train.price.max()))
plt.figure(figsize=[8,4])

sns.distplot(train.price,hist=False,label='train',color='blue')

plt.xticks(rotation=60)

plt.title("Distribution of Price value")
train.bedrooms.value_counts().sort_index()
bedroom_train = make_count_df(train,"bedrooms")

bedroom_test = make_count_df(test,"bedrooms")
plt.figure(figsize=[8,6])



# ax = train.bedrooms.value_counts().sort_index().to_frame().plot(kind='bar',linewidth=2,figsize=[8,6])

# for p in ax.patches:

#     ax.annotate(p.get_height(), (p.get_x()-0.05, p.get_height()))



sns.barplot(x='bedrooms',y='counts',data=bedroom_train,label='train',color='red')

sns.barplot(x='bedrooms',y='counts',data=bedroom_test,label='test',color='blue')

plt.legend()

plt.ylabel("# of Bedrooms",fontsize=12)

plt.xlabel("Bedrooms",fontsize=12)



plt.title("Number of Bedrooms",fontsize=20)
train.bathrooms.value_counts().head()
plt.figure(figsize=[8,6])



sns.kdeplot(train.bathrooms,bw=0.5,label='train')

sns.kdeplot(test.bathrooms,bw=0.5,label='test')



plt.xlabel("Bathrooms(# of Bathrooms / # of Bedrooms)",fontsize=12)

plt.title("Distribution of Bathrooms(# of Bathrooms / # of Bedrooms)",fontsize=20)
train.bathrooms.mul(train.bedrooms).head()
print("The min number of real bathroom is {}, max number of real bathroom is {}".format(train.bathrooms.mul(train.bedrooms).min(),train.bathrooms.mul(train.bedrooms).max()))
fig,ax = plt.subplots(1,2,figsize=[12,6])



ax1 = plt.subplot(1,2,1)

sns.kdeplot(train.sqft_living,bw=0.5,label="train")

sns.kdeplot(test.sqft_living,bw=0.5,label='test')

ax1.set_xlabel("sqft_living",fontsize=12)



ax2 = plt.subplot(1,2,2)

sns.kdeplot(train.sqft_lot,bw=0.5,label="train")

sns.kdeplot(test.sqft_lot,bw=0.5,label='test')

ax2.set_xlabel("sqft_lot",fontsize=12)



plt.suptitle("Distribution of sqft_living and sqft_lot")
train.floors.value_counts()
draw_kdeplot("floors")
waterfront_train = make_count_df(train,"waterfront")

waterfront_test = make_count_df(test,"waterfront")
compare_categorical_ratio(waterfront_train,waterfront_test,"waterfront",3)
train.view.value_counts().sort_index()
view_train= make_count_df(train,"view")

view_test = make_count_df(test,"view")
compare_categorical_ratio(view_train,view_test,"view",10)
condition_train = make_count_df(train,"condition")

condition_test = make_count_df(test,"condition")
compare_categorical_ratio(condition_train,condition_test,"condition",8)
grade_train = make_count_df(train,"grade") 

grade_test = make_count_df(test,"grade")
compare_categorical_ratio(grade_train,grade_test,"grade",adjust_x_annotate=20,fontsize=10)
fig,ax = plt.subplots(1,2,figsize=[12,6])



ax1 = plt.subplot(1,2,1)

sns.kdeplot(train.sqft_above,bw=0.5,label="train")

sns.kdeplot(test.sqft_above,bw=0.5,label='test')

ax1.set_xlabel("sqft_above",fontsize=12)



ax2 = plt.subplot(1,2,2)

sns.kdeplot(train.sqft_basement,bw=0.5,label="train")

sns.kdeplot(test.sqft_basement,bw=0.5,label='test')

ax2.set_xlabel("sqft_basement",fontsize=12)



plt.suptitle("Distribution of sqft_above and sqft_basement")
print("Ratio of 0 in sqft_basement of train_set {:.2f}% among {}".format(sum(train.sqft_basement==0)/len(train)*100,len(train)))

print("Ratio of 0 in sqft_basement of test_set {:.2f}% among {}".format(sum(test.sqft_basement==0)/len(test)*100,len(test)))
train["is_basement"] = ~(train.sqft_basement==0)

test["is_basement"] = ~(test.sqft_basement==0)
fig,ax = plt.subplots(1,2,figsize=[12,6])



ax1 = plt.subplot(1,2,1)

sns.kdeplot(train.yr_built,bw=0.5,label="train")

sns.kdeplot(test.yr_built,bw=0.5,label='test')

ax1.set_xlabel("yr_built",fontsize=12)



ax2 = plt.subplot(1,2,2)

sns.kdeplot(train.yr_renovated,bw=0.5,label="train")

sns.kdeplot(test.yr_renovated,bw=0.5,label='test')

ax2.set_xlabel("yr_renovated",fontsize=12)



plt.suptitle("Distribution of yr_built and yr_renovated")
plt.figure(figsize=[6,6])



sns.kdeplot(train.loc[train["yr_renovated"]!= 0,"yr_renovated"],bw=0.5,label="train")

sns.kdeplot(test.loc[test["yr_renovated"]!= 0,"yr_renovated"],bw=0.5,label="test")

plt.xlabel("yr_renovated")

plt.title("yr_renovated except for 0")
print("Ratio of 0 in yr_renovated of train_set {:.2f}% among {}".format(sum(train.yr_renovated==0)/len(train)*100,len(train)))

print("Ratio of 0 in yr_renovated of test_set {:.2f}% among {}".format(sum(test.yr_renovated==0)/len(test)*100,len(test)))
train["is_renovated"] = ~(train.yr_renovated==0)

test["is_renovated"] = ~(test.yr_renovated==0)
train.zipcode.head()
str(train.zipcode[0])
import re



re1='(\\d{5})'

rg = re.compile(re1)



dummy_train = train.zipcode.apply(lambda x :rg.search(str(x)))

dummy_test = test.zipcode.apply(lambda x :rg.search(str(x)))
print("The number of unexpected form about zipcode of train_set {}".format(sum(dummy_train == 0)))

print("The number of unexpected form about zipcode of test_set {}".format(sum(dummy_test == 0)))
plt.scatter(x=train.long,y=train.lat,color='red',label='train',alpha=0.7)

plt.scatter(x=test.long,y=test.lat,color='blue',label='test',alpha=0.7)

plt.legend()

plt.xlabel("longitude",fontsize=14)

plt.ylabel("latitude",fontsize=14)

plt.title("Distribution of lat and long about train and test set")
sns.jointplot(x='long',y='lat',data=train,kind="hex")

plt.suptitle("Longitude and Latitude Distribution of train_set")
sns.jointplot(x='long',y='lat',data=test,kind="hex")

plt.suptitle("Longitude and Latitude Distribution of test_set")
print("The number of house which sqft_living is equal with sqft_living15 is {}".format(sum(train["sqft_living"] == train["sqft_living15"])))

print("The number of house which sqft_lot is equal with sqft_lot15 is {}".format(sum(train["sqft_lot"] == train["sqft_lot15"])))
print("The situation when sqft_living and sqft_lot are equal with 2015 at the same time is {} cases".format(sum(np.logical_and(train["sqft_living"]==train["sqft_living15"],train["sqft_lot"]==train["sqft_lot15"]))))
renovated_train = train.loc[train.is_renovated == 1]

print("The number of renovated house is {}".format(len(renovated_train)))
not_renovated_train = train.loc[train.is_renovated == 0]



print("The number that sqft_living is same with 2015 is {} among {}".format(sum(not_renovated_train.sqft_living == not_renovated_train.sqft_living15),len(not_renovated_train)))

print("The number that sqft_lot is same with 2015 is {} among {}".format(sum(not_renovated_train.sqft_lot == not_renovated_train.sqft_lot15),len(not_renovated_train)))
b4_2015_renovated_train = renovated_train.loc[renovated_train["yr_renovated"] < 2015]

print("The number that sqft_living is same with 2015 is {} among {} for renovated houses".format(sum(b4_2015_renovated_train.sqft_living == b4_2015_renovated_train.sqft_living15),len(b4_2015_renovated_train)))

print("The number that sqft_lot is same with 2015 is {} among {} for renovated houses".format(sum(b4_2015_renovated_train.sqft_lot == b4_2015_renovated_train.sqft_lot15),len(b4_2015_renovated_train)))
after_2015_renovated_train = train.loc[train["yr_renovated"] > 2015]

after_2015_renovated_train.head()
train["date"]= pd.to_datetime(train.date)

test["date"]= pd.to_datetime(test.date)
def decomposition_date(df):

    dummy = df.copy()

    

    dummy["year"] = dummy.date.apply(lambda x: str(x).split("-")[0]).astype('int')

    dummy["month"] = dummy.date.apply(lambda x:str(x).split("-")[1]).astype('int')

    dummy["day"] = dummy.date.apply(lambda x:str(x).split("-")[-1]).apply(lambda x:x.split(" ")[0]).astype('int')

    

    return dummy
decom_train = decomposition_date(train)

decom_test = decomposition_date(test)
decom_train.groupby('year')['price'].agg(['mean','median'])
decom_train.groupby('year')['price'].agg(['mean','median']).plot(kind='bar',linewidth=2)

plt.title("Mean and Median by Year")



decom_train.groupby('month')['price'].agg(['mean','median']).plot(kind='bar',linewidth=1,figsize=[8,6])

plt.title("Mean and Median by Month")



decom_train.groupby('day')['price'].agg(['mean','median']).plot(kind='bar',linewidth=1,figsize=[20,6])

plt.title("Mean and Median by Day")
ax = decom_train.groupby('bedrooms')['price'].agg(['mean','median']).plot(kind='bar',linewidth=2,figsize=[10,6])



for i,p in enumerate(ax.patches):

    if i < 11:

        ax.annotate(decom_train.bedrooms.value_counts().sort_index()[i],(p.get_x()+p.get_width()*0.5, p.get_y()+p.get_height()*1.01),fontsize=15,rotation=45)



plt.title("Mean, Median value by Bedrooms",fontsize=20)

plt.xlabel("Bedrooms",fontsize=12)

plt.ylabel("Price",fontsize=12)
plt.figure(figsize=[12,6])

sns.barplot(x='bathrooms',y='price',data=decom_train)
def float_with_price(xlabel,df):

    fig,ax = plt.subplots(1,2,figsize=[14,6])



    ax1 = plt.subplot(1,2,1)

    sns.scatterplot(x=xlabel,y='price',data=df,ci=0.95)

    ax1.set_title(f"Scatterplot about {xlabel} with price",fontsize=14)

    ax2 = plt.subplot(1,2,2)

    sns.regplot(x=xlabel,y='price',data=df,ci=0.95)

    ax2.set_title(f"Regplot about {xlabel} with price",fontsize=14)

    plt.xticks(rotation=60)

    plt.suptitle(f"Relationship about {xlabel} with price",fontsize=20)
float_with_price("sqft_living",train)
# fig,ax = plt.subplots(1,2,figsize=[14,6])



# ax1 = plt.subplot(1,2,1)

# sns.scatterplot(x='sqft_living',y='price',data=train,ci=0.95)

# ax1.set_title("Scatterplot about sqft_living with price",fontsize=14)

# ax2 = plt.subplot(1,2,2)

# sns.regplot(x='sqft_living',y='price',data=train,ci=0.95)

# ax2.set_title("Regplot about sqft_living with price",fontsize=14)

# plt.xticks(rotation=60)

# plt.suptitle("Relationship about sqft_living with price",fontsize=20)
float_with_price("sqft_lot",train)
# fig,ax = plt.subplots(1,2,figsize=[14,6])



# ax1 = plt.subplot(1,2,1)

# sns.scatterplot(x='sqft_lot',y='price',data=train,ci=0.95)

# ax1.set_title("Scatterplot about sqft_lot with price",fontsize=14)

# ax2 = plt.subplot(1,2,2)

# sns.regplot(x='sqft_lot',y='price',data=train,ci=0.95)

# ax2.set_title("Regplot about sqft_lot with price",fontsize=14)

# plt.xticks(rotation=60)

# plt.suptitle("Relationship about sqft_lot with price",fontsize=20)
train.columns.values
def ordinal_with_price(xlabel,df,rotation=0):

    

    fig,ax = plt.subplots(1,2,figsize=[14,6])



    ax1 = plt.subplot(1,2,1)

    sns.barplot(x=xlabel,y="price",data=train)

    ax1.set_xlabel(xlabel,fontsize=12)

    ax1.set_ylabel("price",fontsize=12)

    ax1.set_title(f"Barplot about {xlabel} with price",fontsize=18)

    

    for i,p in enumerate(ax1.patches):



        ax1.annotate(s=train[xlabel].value_counts().sort_index().values[i],xy= (p.get_x()+p.get_width()/len(train[xlabel].value_counts()), p.get_height()*1.05),fontsize=15,rotation=rotation)

        

    ax2 = plt.subplot(1,2,2)

    sns.boxplot(x=xlabel,y='price',data=train)

    ax2.set_xlabel(xlabel,fontsize=12)

    ax2.set_ylabel("price",fontsize=12)

    ax2.set_title(f"Boxplot about {xlabel} with price",fontsize=18)

    

    plt.suptitle(f"Relationship about {xlabel} with price",fontsize=20)
ordinal_with_price("floors",train)
# fig,ax = plt.subplots(1,2,figsize=[14,6])



# ax1 = plt.subplot(1,2,1)

# sns.barplot(x="floors",y="price",data=train)

# ax1.set_xlabel("floors",fontsize=12)

# ax1.set_ylabel("price",fontsize=12)

# ax1.set_title("Barplot about floors with price",fontsize=18)

# for i,p in enumerate(ax1.patches):

#     if i == 5:

#         ax1.annotate(s=train.floors.value_counts().sort_index().values[i],xy= (p.get_x()+p.get_width()/4, p.get_height()*1.2),fontsize=15)

#     else:

#         ax1.annotate(s=train.floors.value_counts().sort_index().values[i],xy= (p.get_x()+p.get_width()/5, p.get_height()*1.2),fontsize=15)

        

# ax2 = plt.subplot(1,2,2)

# sns.boxplot(x='floors',y='price',data=train)

# ax2.set_xlabel("floors",fontsize=12)

# ax2.set_ylabel("price",fontsize=12)

# ax2.set_title("Boxplot about floors with price",fontsize=18)

# plt.suptitle("Relationship about floors with price",fontsize=20)
ordinal_with_price("waterfront",train)
# plt.subplots(1,2,figsize=[14,6])



# ax1 = plt.subplot(1,2,1)

# ax1 = train.groupby('waterfront')['price'].mean().plot(kind='bar',yerr=train.groupby('waterfront')['price'].std())

# ax1.set_xlabel("waterfront",fontsize=14)

# ax1.set_ylabel("price",fontsize=14)

# ax1.set_title("Barplot for waterfront with price")

# # for i,p in enumerate(ax1.patches):

# #     ax1.annotate(s=train.groupby('waterfront')['price'].mean().apply(lambda x:int(np.floor(x))).values[i],xy=(p.get_x()+p.get_width()/8,p.get_y()+p.get_height()*1.2),fontsize=16)



# ax2 = plt.subplot(1,2,2)

# sns.boxplot(x="waterfront",y="price",data=train)

# ax2.set_xlabel("waterfront",fontsize=14)

# ax2.set_ylabel("price",fontsize=14)

# ax2.set_title("Boxplot for waterfront with price")



# plt.suptitle("Relationship about waterfront with price",fontsize=20)
ordinal_with_price("view",train)
# fig,ax = plt.subplots(1,2,figsize=[14,6])



# ax1 = plt.subplot(1,2,1)

# sns.barplot(x="view",y="price",data=train)

# ax1.set_xlabel("view",fontsize=12)

# ax1.set_ylabel("price",fontsize=12)

# ax1.set_title("Barplot about view with price",fontsize=18)

# for i,p in enumerate(ax1.patches):

# #     if i == 5:

# #         ax1.annotate(s=train.view.value_counts().sort_index().values[i],xy= (p.get_x()+p.get_width()/4, p.get_height()*1.2),fontsize=15

#     ax1.annotate(s=train.view.value_counts().sort_index().values[i],xy= (p.get_x()+p.get_width()/5, p.get_height()*1.1),fontsize=15)

        

# ax2 = plt.subplot(1,2,2)

# sns.boxplot(x='view',y='price',data=train)

# ax2.set_xlabel("view",fontsize=12)

# ax2.set_ylabel("price",fontsize=12)

# ax2.set_title("Boxplot about view with price",fontsize=18)

# plt.suptitle("Relationship about view with price",fontsize=20)
ordinal_with_price("condition",train)
ordinal_with_price("grade",train,rotation=60)
float_with_price("sqft_above",train)
float_with_price("sqft_basement",train.loc[train["is_basement"]])
float_with_price("yr_built",train)
float_with_price("yr_renovated",train.loc[train["is_renovated"]])
float_with_price("sqft_living15",train)
float_with_price("sqft_lot15",train)
ordinal_with_price("is_basement",train)
ordinal_with_price("is_renovated",train)
# 앞에서 나눈 년,월,일과 is_renovate, is_basement등이 포함된 decom_train과 decom_test를 각각 train과 test로 할당합니다.

train = decom_train

test = decom_test
plt.figure(figsize=[12,12])



#id는 우리의 모델에서 크게 상관없는 모델이기 때문에 상관관계를 분석함에 있어서 배제하고 heatmap을 그립니다.

sns.heatmap(train.drop("id",axis=1).corr(),annot=True,square=True,cmap=plt.cm.summer)
# from sklearn.preprocessing import LabelEncoder



# #모든 컬럼간의 plot을 그려보기 위해 pairplot을 사용하는데 is_basement와 is_renovated가 boolean값이라서 그리지 못한다는 에러 해결을 위한 라벨인코딩.

# def labelEncoding(train,test,cols):

    

#     dup_train = train.copy()

#     dup_test = test.copy()

    

#     for col in cols:

#         encoder = LabelEncoder()

#         dup_train[f"{col}"]= encoder.fit_transform(train[f"{col}"])

#         dup_test[f"{col}"] = encoder.transform(test[f"{col}"])



#     return dup_train,dup_test



# train_encoded, test_encoded = labelEncoding(train,test,["is_basement","is_renovated"])
# train 데이터프레임에서 id,date,price를 제외한 것들에 대해서 상관관계를 구하고 절댓값을 취해서 0.9보다 높은 상관관계를 가지는 컬럼들을 추출해봅시다.

corr_train = train[train.columns.values[3:]].corr().abs()

# 아래의 코드를 통해 훈련 데이터프레임에서 상관관계의 값은 주대각성분들을 제외한 상삼각행렬에 대해서만 존재하게 됩니다.

triu_corr_train = corr_train.where(np.triu(np.ones(corr_train.shape),k=1).astype(np.bool))

# 0.9보다 높은 임의의 컬럼을가지는 컬럼을 도출합니다.

del_cols = [col for col in triu_corr_train.columns if any(triu_corr_train[col] > 0.9)]

print("delcols: ",del_cols)

triu_corr_train[del_cols]
train = train.drop(del_cols,axis=1)

test = test.drop(del_cols,axis=1)

train.head()
corr_train = train[train.columns.values[3:]].corr().abs()

triu_corr_train = corr_train.where(np.triu(np.ones(corr_train.shape),k=1).astype(np.bool))

corr_df = triu_corr_train.stack().sort_values(ascending=False)

corr_df = corr_df.reset_index().rename(columns={"level_0":"col_1","level_1":"col_2",0:"correaltion"})

corr_df.head()
corr_df[corr_df["correaltion"] > 0.5]
unique_cols_1 = corr_df["col_1"].unique()

unique_cols_2 = corr_df["col_2"].unique()



total_cols = []

dummy_cols = []



for col1 in unique_cols_1:

    

    if col1 in unique_cols_2:

        dummy_cols.append(col1)

    else:

        total_cols.append(col1)



for col2 in unique_cols_2:

    

    total_cols.append(col2)
from sklearn.preprocessing import PolynomialFeatures



poly = PolynomialFeatures(degree=2,interaction_only=True,include_bias=False)

train_polyed_array = poly.fit_transform(train[total_cols])

test_polyed_array = poly.transform(test[total_cols])





train_polyed = pd.DataFrame(train_polyed_array,columns=poly.get_feature_names(total_cols))

test_polyed = pd.DataFrame(test_polyed_array,columns=poly.get_feature_names(total_cols))



assert train.shape[0] == train_polyed.shape[0]

assert test.shape[0] == test_polyed.shape[0]



train_polyed = pd.concat([train,train_polyed[[col for col in train_polyed.columns if col not in total_cols]]],axis=1)

test_polyed = pd.concat([test,test_polyed[[col for col in test_polyed.columns if col not in total_cols]]],axis=1)



print("The shape of train_polyed is {}, test_polyed is {}".format(train_polyed.shape,test_polyed.shape))
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_val_score

from sklearn.metrics import make_scorer



def rmse(y,preds):

    return np.sqrt(mean_squared_error(y,preds))



rmse_scorer = make_scorer(rmse,greater_is_better=False)



X_train = train[train.columns.values[3:]]

y_train = train["price"]



X_test = test[test.columns.values[2:]]



X_train_polyed = train_polyed[train_polyed.columns.values[3:]]

y_train_polyed = train_polyed["price"]



X_test_polyed = test_polyed[test_polyed.columns.values[2:]]



rf = RandomForestRegressor(random_state=101)



origin = np.mean(-cross_val_score(estimator=rf,X=X_train,y=y_train,cv=5,scoring=rmse_scorer))

polyed = np.mean(-cross_val_score(estimator=rf,X=X_train_polyed,y=y_train_polyed,cv=5,scoring=rmse_scorer))

print("The score of origin is {}, polyed is {}".format(origin,polyed))
# Polyed dataset을 선정

rf.fit(X_train_polyed,y_train_polyed)

preds = rf.predict(X_test_polyed)
#제출

submission = pd.read_csv(PATH+"sample_submission.csv")

assert len(submission) == len(preds)

submission["price"] = preds

submission.to_csv("./submission.csv",index=False)
from sklearn.linear_model import LinearRegression,Ridge,Lasso

from sklearn.model_selection import GridSearchCV



lr = LinearRegression()

ridge = Ridge()

lasso = Lasso()



def model_cross_val_score(model,name):

    origin = np.mean(-cross_val_score(model,X_train,y_train,scoring=rmse_scorer,cv=5))

    polyed = np.mean(-cross_val_score(model,X_train_polyed,y_train_polyed,scoring=rmse_scorer,cv=5))

    

    print(f"{name} model's rmse evaluation value is {origin} for origin, {polyed} for ployed")

    

model_cross_val_score(lr,"LinearRegression")

model_cross_val_score(ridge,"Ridge")

model_cross_val_score(lasso,"Lasso")
ridge_params = {"alpha":[0.01,0.1,1,10,100,1000]}

lasso_params = {"alpha":[0.01,0.1,1,10,100,1000]}



ridge_grid = GridSearchCV(ridge,ridge_params,cv=5,n_jobs=4,scoring=rmse_scorer)

lasso_grid = GridSearchCV(lasso,lasso_params,cv=5,n_jobs=4,scoring=rmse_scorer)



def model_grid_search(grid_model,name):

    grid_model.fit(X_train,y_train)

    print(f"{name}'s origin best_params_ is {grid_model.best_params_}, best_score_ is {-grid_model.best_score_}")

    grid_model.fit(X_train_polyed,y_train_polyed)

    print(f"{name}'s polyed best_params_ is {grid_model.best_params_}, best_score_ is {-grid_model.best_score_}")

    

model_grid_search(ridge_grid,"RidgeGrid")

model_grid_search(lasso_grid,"LassoGrid")
# import lightgbm as lgb

# from sklearn.model_selection import StratifiedKFold



# def lgb_rmse(preds,dtrain):

    

# #     print(preds)

# #     print(len(preds))

    

#     y= list(dtrain.get_label())

# #     print(len(y))

# #     print(y)

#     score = np.sqrt(-mean_squared_error(y,preds))

    

# #     print(pd.concat([pd.Series(preds),pd.Series(y)],axis=1))

    

    

#     return "lgb_rsme",score,False



# def model_lgb(X_train,y_train,X_test,nfolds=5):

    

#     feature_names = train.columns.values

    

#     valid_scores = np.zeros(len(X_train))

#     predictions = np.zeros(len(X_test))

    

#     feature_importance_df = pd.DataFrame()

    

#     params = {

#         "obejctive":"regression",

#         "boosting":"gbrt",

#         "learning_rate":0.2,

#         "num_leaves":31,

#         "seed":1121,

#         "max_depth":10,

#         "min_data_in_leaf":20,

#         "min_sum_hessian_in_leaf":1.0,

#         "bagging_fraction":0.8,

#         "bagging_freq":6,

#         "feature_fraction":0.8,

#         "metric":"neg_mean_squared_error"

#     }

    

#     strkfold = StratifiedKFold(n_splits=nfolds,shuffle=True,random_state=12)

    

#     for i,(train_indices,valid_indices) in enumerate(strkfold.split(X_train.values,y_train.values)):

        

#         print("{} fold processing".format(i+1),"#"*20)

        

#         d_train = lgb.Dataset(X_train.values[train_indices,:],label=y_train[train_indices])

#         d_valid = lgb.Dataset(X_train.values[valid_indices,:],label=y_train[valid_indices])

        

#         n_rounds = 1000

        

#         lgb_model = lgb.train(params,d_train,num_boost_round=n_rounds,valid_sets=[d_train,d_valid],valid_names=["train","valid"],feval="neg_mean_squared_error",verbose_eval=250,early_stopping_rounds=100)
import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold



def lgb_rmse(y,preds):

    

    score = np.sqrt(mean_squared_error(y,preds))

  

    return "lgb_rmse",score,False





def model_lgb(X_train,y_train,X_test,nfolds=5):

    

    feature_names = X_train.columns.values

    

    valid_scores = np.zeros(len(X_train))

    predictions = np.zeros(len(X_test))

    

    valid_scores_list = []

    

    importances = np.zeros(len(feature_names))

    

    feature_importance_df = pd.DataFrame()

    feature_importance_df["features"] = feature_names

    

    params = {'boosting_type': 'gbdt',

             'class_weight': None,

             'colsample_bytree': 1.0,

             'importance_type': 'split',

             'learning_rate': 0.1,

             'max_depth': -1,

             'min_child_samples': 20,

             'min_child_weight': 0.001,

             'min_split_gain': 0.0,

             'n_estimators': 1000,

             'n_jobs': -1,

             'num_leaves': 31,

             'objective': None,

             'random_state': 101,

             'reg_alpha': 0.0,

             'reg_lambda': 0.0,

             'silent': True,

             'subsample': 1.0,

             'subsample_for_bin': 200000,

             'subsample_freq': 0}

    

    lgbr = lgb.LGBMRegressor(**params)

    

    strkfold = StratifiedKFold(n_splits=nfolds,shuffle=True,random_state=12)

    

    for i,(train_indices,valid_indices) in enumerate(strkfold.split(X_train.values,y_train.values)):

        

        X = X_train.values[train_indices]

        y = y_train.values[train_indices]

        X_valid = X_train.values[valid_indices]

        y_valid = y_train.values[valid_indices]

        

        print("{} fold processing".format(i+1),"#"*20)

        

        lgbr.fit(X,y,eval_set=[(X,y),(X_valid,y_valid)],eval_names=["train","valid"],eval_metric=lgb_rmse,verbose=250,early_stopping_rounds=100)

        

#         fi_df = pd.DataFrame(lgbr.feature_importances_)

#         fi_df["folds"] = i+1

        

#         feature_importance_df = pd.concat([feature_importance_df,fi_df],axis=0)

    

#         importances += lgbr.feature_importances_ / nfolds



        valid_scores_list.append(lgbr.best_score_["valid"]["lgb_rmse"])



        feature_importance_df[f"{i+1}"] = lgbr.feature_importances_

    

        valid_score = lgbr.predict(X_valid)

        prediction = lgbr.predict(X_test)

        

        valid_scores[valid_indices] += valid_score

        predictions += prediction / nfolds

        

#     feature_importance_df= pd.DataFrame({"features":feature_names,"importances":importances})

    print(f"mean_valid_score is {np.mean(valid_scores_list)} at {nfolds}")

        

    return feature_importance_df,predictions 
lgb_fi,predictions = model_lgb(X_train,y_train,X_test)
lgb_fi["mean"] = lgb_fi[lgb_fi.columns.values[1:]].mean(axis=1)

lgb_fi["std"] = lgb_fi[lgb_fi.columns.values[1:]].std(axis=1)

# lgb_fi.head()



lgb_fi_sorted = lgb_fi.sort_values("mean",ascending=False)

lgb_fi_sorted.head()
plt.figure(figsize=[6,40])

sns.barplot(x='mean',y='features',data=lgb_fi_sorted,xerr=lgb_fi_sorted["std"])

plt.title("Feature Importances of lightgbm",fontsize=12)
#제출

submission = pd.read_csv(PATH+"sample_submission.csv")

assert len(submission) == len(predictions)

submission["price"] = predictions

submission.to_csv("./lgb_origin_submission.csv",index=False)
lgb_fi,predictions = model_lgb(X_train_polyed,y_train,X_test_polyed)
lgb_fi["mean"] = lgb_fi[lgb_fi.columns.values[1:]].mean(axis=1)

lgb_fi["std"] = lgb_fi[lgb_fi.columns.values[1:]].std(axis=1)

# lgb_fi.head()



lgb_fi_sorted = lgb_fi.sort_values("mean",ascending=False)

lgb_fi_sorted.head()
plt.figure(figsize=[6,40])

sns.barplot(x='mean',y='features',data=lgb_fi_sorted,xerr=lgb_fi_sorted["std"])

plt.title("Feature Importances of lightgbm",fontsize=12)
#제출

submission = pd.read_csv(PATH+"sample_submission.csv")

assert len(submission) == len(predictions)

submission["price"] = predictions

submission.to_csv("./lgb_ploy_submission.csv",index=False)