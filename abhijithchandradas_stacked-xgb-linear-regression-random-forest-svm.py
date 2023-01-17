import os

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

sns.set(style='darkgrid')



from sklearn.linear_model import LinearRegression

from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import LabelEncoder



from xgboost import XGBRegressor



from warnings import filterwarnings

filterwarnings('ignore')
df_test=pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

df_train=pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

print("Train shape:",df_train.shape)

print("Test Shape:",df_test.shape)
X_trainfull=df_train.drop(["SalePrice"], axis=1)

y=df_train.SalePrice
plt.figure(figsize=(8,4))

plt.title("Distribution of Sales Price (y)")

sns.distplot(y)

plt.show()
y=np.log1p(y)



plt.figure(figsize=(8,4))

plt.title("Distribution of log Sales Price (y)")

sns.distplot(y)

plt.xlabel("Log of Sales Price")

plt.show()
d_temp=X_trainfull.isna().sum().sort_values(ascending=False)

d_temp=d_temp[d_temp>0]

d_temp=d_temp/df_train.shape[0]*100



plt.figure(figsize=(8,5))

plt.title("Features Vs Percentage Of Null Values")

sns.barplot(y=d_temp.index,x=d_temp, orient='h')

plt.xlim(0,100)

plt.xlabel("Null Values (%)")

plt.show()
na_index=(d_temp[d_temp>20]).index

X_trainfull.drop(na_index, axis=1, inplace=True)
num_cols=X_trainfull.corrwith(y).abs().sort_values(ascending=False).index

X_num=X_trainfull[num_cols]

X_cat=X_trainfull.drop(num_cols,axis=1)
X_num.sample(5)
high_corr_num=X_num.corrwith(y)[X_num.corrwith(y).abs()>0.5].index

X_num=X_num[high_corr_num]
plt.figure(figsize=(10,6))

sns.heatmap(X_num.corr(), annot=True, cmap='coolwarm')

plt.show()



print("Correlation of Each feature with target")

X_num.corrwith(y)
X_num=X_num[high_corr_num]

X_num.drop(['TotRmsAbvGrd','GarageArea','1stFlrSF','GarageYrBlt'],axis=1, inplace=True)
#function to handle NA

def handle_na(df, func):

    """

    Input dataframe and function 

    Returns dataframe after filling NA values

    eg: df=handle_na(df, 'mean')

    """

    na_cols=df.columns[df.isna().sum()>0]

    for col in na_cols:

        if func=='mean':

            df[col]=df[col].fillna(df[col].mean())

        if func=='mode':

            df[col]=df[col].fillna(df[col].mode()[0])

    return df
X_num=handle_na(X_num, 'mean')
# Function to scale df 

def scale_df(df):

    """

    Input: data frame

    Output: Returns minmax scaled Dataframe 

    eg: df=scale_df(df)

    """

    scaler=MinMaxScaler()

    for col in df.columns:

        df[col]=scaler.fit_transform(np.array(df[col]).reshape(-1,1))

    return df
X_num=scale_df(X_num)
X_train, X_val, y_train, y_val=train_test_split(X_num,y, test_size=0.2)

model=LinearRegression()

model.fit(X_train,y_train)

print(f"Train score : {model.score(X_train,y_train)}")

print(f"Validation score : {model.score(X_val,y_val)}")
model=SVR()

model.fit(X_train,y_train)

print(f"Train score : {model.score(X_train,y_train)}")

print(f"Validation score : {model.score(X_val,y_val)}")
model=RandomForestRegressor(n_estimators=100)

model.fit(X_train,y_train)

print(f"Train score : {model.score(X_train,y_train)}")

print(f"Validation score : {model.score(X_val,y_val)}")
model=XGBRegressor(learning_rate=0.1)

model.fit(X_train,y_train)

print(f"Train score : {model.score(X_train,y_train)}")

print(f"Validation score : {model.score(X_val,y_val)}")
num_features=X_num.columns
X_cat.sample(5)
X_cat.describe()
for feature in X_cat.columns:

    print(

        f"{feature} :{len(X_cat[feature].unique())}: {X_cat[feature].unique()}"

    )
cat_na=X_cat.isna().sum().sort_values(ascending=False)

cat_na=cat_na[cat_na>30]

X_cat.drop(cat_na.index, axis=1, inplace=True)
for feature in X_cat.columns:

    plt.figure(figsize=(4,6))

    plt.title(f"{str(feature)} vs log Sale Price")

    sns.boxplot(X_cat[feature],y)

    plt.show()
X_cat=handle_na(X_cat, 'mode')
le=LabelEncoder()

X_cat_le=pd.DataFrame()

for col in X_cat.columns:

    X_cat_le[col] = le.fit_transform(X_cat[col])
Xc_train, Xc_test, yc_train,yc_test=train_test_split(X_cat_le,y, test_size=0.2)
model=RandomForestRegressor()

model.fit(Xc_train,yc_train)
print(f"Train score : {model.score(Xc_train,yc_train)}")

print(f"Test score : {model.score(Xc_test,yc_test)}")
feat_imp=pd.DataFrame({"Feature":Xc_train.columns,"imp":model.feature_importances_})

feat_imp=feat_imp.sort_values('imp', ascending=False)



plt.figure(figsize=(10,4))

plt.title("Feature Importance", fontsize=16)

sns.barplot('Feature', 'imp', data=feat_imp)

plt.xticks(rotation=80)

plt.show()
feat=[]

score_train=[]

score_test=[]

for i in range(29):

    imp_ft=feat_imp.head(i+1).Feature.unique()



    X_cat_imp=pd.DataFrame()

    for col in imp_ft:

        X_cat_imp[col] = le.fit_transform(X_cat[col])



    Xc_train, Xc_test, yc_train,yc_test=train_test_split(X_cat_imp,y, test_size=0.2)



    model=RandomForestRegressor(n_estimators=100)

    model.fit(Xc_train,yc_train)

    feat.append(i+1)

    score_train.append(model.score(Xc_train,yc_train))

    score_test.append(model.score(Xc_test,yc_test))

    

acc_feat_df=pd.DataFrame({"Feature":feat,"TrainAcc":score_train,"ValAcc":score_test})
plt.figure(figsize=(10,5))

sns.lineplot('Feature', 'TrainAcc', data=acc_feat_df, label="Training Accuracy")

sns.lineplot('Feature', 'ValAcc', data=acc_feat_df, label="Validation Accuracy")

plt.xlabel("Number of Features")

plt.ylabel("R2 Score")

plt.xticks(rotation=80)

plt.xlim(1,29)

plt.show()
cat_features=list(feat_imp.iloc[:17,0])
# Selecting only important features

X_cat=X_cat[cat_features]

# OHE features

X_cat=pd.get_dummies(X_cat)

# Scaling the data

X_cat=scale_df(X_cat)
X_train, X_val, y_train, y_val=train_test_split(X_cat,y, test_size=0.2)
model=LinearRegression()

model.fit(X_train,y_train)

print(f"Train score : {model.score(X_train,y_train)}")

print(f"Validation score : {model.score(X_val,y_val)}")
model=SVR()

model.fit(X_train,y_train)

print(f"Train score : {model.score(X_train,y_train)}")

print(f"Validation score : {model.score(X_val,y_val)}")
model=RandomForestRegressor(n_estimators=100)

model.fit(X_train,y_train)

print(f"Train score : {model.score(X_train,y_train)}")

print(f"Validation score : {model.score(X_val,y_val)}")
model=XGBRegressor(learning_rate=0.1)

model.fit(X_train,y_train)

print(f"Train score : {model.score(X_train,y_train)}")

print(f"Validation score : {model.score(X_val,y_val)}")
#Combine train and test data

Xtt=pd.concat([X_trainfull,df_test])



#Split into Numeric and categoric features

Xtt_num= Xtt[num_features]

Xtt_cat= Xtt[cat_features]



#Handling null values

Xtt_cat=handle_na(Xtt_cat, 'mode')

Xtt_num=handle_na(Xtt_num,'mean')



#OHE Categoric features

Xtt_cat=pd.get_dummies(Xtt_cat,drop_first=True)



#Combine Numeric and Categorical features

Xtt=pd.concat([Xtt_num,Xtt_cat], axis=1)



#Scale Features

Xtt=scale_df(Xtt)



#Training and Testing Features after Feature Engineering

X=Xtt.iloc[:df_train.shape[0],:]

X_test=Xtt.iloc[df_train.shape[0]:,:]



#Training and Validation features and target

X_train, X_val, y_train, y_val=train_test_split(X,y, test_size=0.2)
model_LR=LinearRegression()

model_LR.fit(X_train,y_train)

print(f"Train score : {model_LR.score(X_train,y_train)}")

print(f"Validation score : {model_LR.score(X_val,y_val)}")

yv_LR=model_LR.predict(X_val)

yt_LR=model_LR.predict(X_test)
model_SV=SVR()

model_SV.fit(X_train,y_train)

print(f"Train score : {model_SV.score(X_train,y_train)}")

print(f"Validation score : {model_SV.score(X_val,y_val)}")

yv_SVR=model_SV.predict(X_val)

yt_SVR=model_SV.predict(X_test)
model_RF=RandomForestRegressor(n_estimators=100)

model_RF.fit(X_train,y_train)

print(f"Train score : {model_RF.score(X_train,y_train)}")

print(f"Validation score : {model_RF.score(X_val,y_val)}")

yv_RF=model_RF.predict(X_val)

yt_RF=model_RF.predict(X_test)
model_XGB=XGBRegressor(learning_rate=0.1)

model_XGB.fit(X_train,y_train)

print(f"Train score : {model_XGB.score(X_train,y_train)}")

print(f"Validation score : {model_XGB.score(X_val,y_val)}")

yv_XGB=model_XGB.predict(X_val)

yt_XGB=model_XGB.predict(X_test)
yv_stacked=np.column_stack((yv_SVR,yv_RF,yv_XGB))

yt_stacked=np.column_stack((yt_SVR,yt_RF,yt_XGB))



meta_model=LinearRegression()

meta_model.fit(yv_stacked,y_val)

print(meta_model.score(yv_stacked,y_val))



y_final=np.expm1(meta_model.predict(yt_stacked))
sub = pd.DataFrame()

sub["Id"] = df_test.Id

sub["SalePrice"] = y_final

sub.to_csv("submission.csv", index=False)