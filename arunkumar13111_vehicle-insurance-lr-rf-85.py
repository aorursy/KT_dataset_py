import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, MinMaxScaler, OrdinalEncoder
from sklearn.model_selection import StratifiedKFold
# read data 
train = pd.read_csv("/kaggle/input/health-insurance-cross-sell-prediction/train.csv")
test  = pd.read_csv("/kaggle/input/health-insurance-cross-sell-prediction/test.csv")
print(train.info(),test.info())
#split numerical and categorical feature
num_feature = ["Age","Vintage","Annual_Premium"]
cat_feature = ["Gender","Driving_License","Region_Code","Previously_Insured","Vehicle_Age","Vehicle_Damage","Policy_Sales_Channel"]

# convert into integer
train["Policy_Sales_Channel"] = train["Policy_Sales_Channel"].astype("int")
train["Region_Code"] = train["Region_Code"].astype("int")
psc_tot = train["Policy_Sales_Channel"].value_counts()
train["psc_count"] = train["Policy_Sales_Channel"].map(psc_tot)
psc_scount = train[train["Response"]==1].groupby("Policy_Sales_Channel")["Response"].count()
train["psc_scount"] = train["Policy_Sales_Channel"].map(psc_scount)
train["psc_success_rate"] = (train["psc_scount"]/train["psc_count"])*100
train["psc_success_rate"].fillna(0,inplace=True)
reg_tot = train["Region_Code"].value_counts()
train["reg_count"] = train["Region_Code"].map(reg_tot)
psc_scount = train[train["Response"]==1].groupby("Region_Code")["Response"].count()
train["reg_scount"] = train["Region_Code"].map(psc_scount)
train["reg_success_rate"] = (train["reg_scount"]/train["reg_count"])*100

#Binning the Age 
train["Age_Cat"]= pd.cut(train["Age"],bins=[10,20,30,40,50,60,70,80,90,100],labels=[1,2,3,4,5,6,7,8,9])
# encode the categorical varriable (encoding policy sales channel and Region Code makes high cardinality so i avoid encoding for this fields )
ohe = OneHotEncoder(sparse=False)        
transformed_train_data = ohe.fit_transform(train[["Age_Cat","Gender","Vehicle_Age","Previously_Insured","Driving_License","Vehicle_Damage"]])

# # the above transformed_data is an array so convert it to dataframe
encoded_train_data = pd.DataFrame(transformed_train_data, index=train.index)        
encoded_train_data.columns = ohe.get_feature_names(["Age_Cat","Gender","Vehicle_Age","Previously_Insured","Driving_License","Vehicle_Damage"])
train_data = pd.concat([train, encoded_train_data], axis=1)
sc =  StandardScaler()
vintage_scaled_array = sc.fit_transform(train_data[["Vintage","Region_Code","Policy_Sales_Channel","psc_success_rate","reg_success_rate"]])

minmaxsc = MinMaxScaler()
annual_premium_scaled_array = minmaxsc.fit_transform(train_data[["Annual_Premium"]])

train_data[["Vintage","Region_Code","Policy_Sales_Channel","psc_success_rate","reg_success_rate"]] = vintage_scaled_array
train_data["Sc_Annual_Premium"] = annual_premium_scaled_array
train_data.head()
train_data.columns
trainX = train_data.drop(["id","Age","Age_Cat","Gender","Vehicle_Age","psc_count","psc_scount","reg_count","reg_scount","Vintage","Annual_Premium","Previously_Insured","Driving_License","Vehicle_Damage","Response"], axis=1)
trainY = train_data["Response"]
trainX.head()
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score,confusion_matrix
logreg = LogisticRegression(random_state=0,max_iter=1000)
models = {
    "Logistic Regression":logreg,
}
skf = StratifiedKFold(n_splits=5, shuffle=True)
for model_name,model in models.items():
    scores = []
    train_scores = []
    for train_indx,val_indx in skf.split(X=trainX,y=trainY):
        #get train and test data     
        X_train,X_val = trainX.loc[train_indx],trainX.loc[val_indx]
        Y_train,Y_val = trainY[train_indx],trainY[val_indx]
        model.fit(X_train,Y_train)
        #make a prediction
        Y_predict = model.predict_proba(X_val)

        accuracy = roc_auc_score(Y_val,Y_predict[:,1])
        Ytrain_predict = model.predict_proba(X_train)
        train_accuracy = roc_auc_score(Y_train,Ytrain_predict[:,1])        
        print("train",train_accuracy)
        print("test",accuracy)
        scores.append(accuracy)
        train_scores.append(train_accuracy)
    print("Mean Accurracy of test {0} is {1}".format(model_name,np.mean(scores)))
    print("Mean Accurracy of train {0} is {1}".format(model_name,np.mean(train_scores)))
train
oe = OrdinalEncoder()
train[["Gender","Vehicle_Damage","Vehicle_Age"]] = oe.fit_transform(train[["Gender","Vehicle_Damage","Vehicle_Age"]])
train
trainX = train.drop(["id","Age","Vintage","psc_count","psc_scount","reg_count","reg_scount","Response"], axis=1)
trainY = train["Response"]
trainX.columns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score,confusion_matrix
rf = RandomForestClassifier(n_estimators=7,max_depth=10)
models = {
    "Random Forest":rf,
#     "RandomForestClassifier":rf
}
skf = StratifiedKFold(n_splits=5, shuffle=True)
for model_name,model in models.items():
    scores = []
    train_scores = []
    for train_indx,val_indx in skf.split(X=trainX,y=trainY):
        #get train and test data     
        X_train,X_val = trainX.loc[train_indx],trainX.loc[val_indx]
        Y_train,Y_val = trainY[train_indx],trainY[val_indx]
        model.fit(X_train,Y_train)
        #make a prediction
        Y_predict = model.predict_proba(X_val)

        accuracy = roc_auc_score(Y_val,Y_predict[:,1])
        Ytrain_predict = model.predict_proba(X_train)
        train_accuracy = roc_auc_score(Y_train,Ytrain_predict[:,1])        
        print("train",train_accuracy)
        print("test",accuracy)
        scores.append(accuracy)
        train_scores.append(train_accuracy)
    print("Mean Accurracy of test {0} is {1}".format(model_name,np.mean(scores)))
    print("Mean Accurracy of train {0} is {1}".format(model_name,np.mean(train_scores)))
