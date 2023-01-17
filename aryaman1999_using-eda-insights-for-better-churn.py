

import pandas as pd

import numpy as np

import seaborn as sns



data = pd.read_csv("../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")

data.head()

data.columns
data.info()
count_null = 0

indices = []

for i in range(len(data.TotalCharges)):

    if data.TotalCharges[i]==" ":

        count_null+=1

        indices.append(i)

    

print(count_null)
print(100*count_null/len(data))
daa=data.drop(indices,axis=0)

test = daa.reset_index()

data = test.drop(["index"],axis=1)
def changeService(data,original_var="No phone service",feature_list=["MultipleLines"]):

    

    for feature in feature_list:

        ls = list(data[feature][data[feature]==original_var].index)

        data[feature].iloc[ls]="No"

    return data
df = changeService(data)
feature_list = ["OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]

df = changeService(df,original_var="No internet service",feature_list=feature_list)

df.head()
sns.countplot(df.StreamingTV.value_counts())
## See distribution of target variable ###

ac = sns.countplot(df.Churn)

for p in ac.patches:

    height = p.get_height()

    ac.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}'.format(height/len(df)),

            ha="center")
### Countplots for categorical features ###

import seaborn as sns

import matplotlib.pyplot as plt

fig,ax = plt.subplots(5,3,figsize=(20,20))

sns.set_style("dark")

categorical = ["gender","SeniorCitizen","Partner","Dependents","MultipleLines","InternetService",\

               "OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies",\

               "Contract","PaperlessBilling","PaymentMethod","PhoneService","Churn"]

k = 0

for i in range(5):

    for j in range(3):

        ac = sns.countplot(df[categorical[k]],ax=ax[i][j])

        for p in ac.patches:

            height = p.get_height()

            ac.text(p.get_x()+p.get_width()/2.,

                    height + 3,

                    '{:1.2f}'.format(height/len(df)),

                    ha="center") 

        k+=1
##Visualize probability distribution of continuous variables

cont = ["tenure","MonthlyCharges","TotalCharges"]

fig,ax = plt.subplots(1,3,figsize=(20,10))

sns.set_style("dark")

for i in range(3):

    sns.distplot(df[cont[i]],ax=ax[i])

### Visualize cdf ###

kwargs = {'cumulative': True}

fig,ax = plt.subplots(1,3,figsize=(20,10))

sns.set_style("dark")

for i in range(3):

    x = df[cont[i]]

    sns.distplot(x, hist_kws=kwargs, kde_kws=kwargs,ax = ax[i])





## Visualize categorical variables per churn label ###

fig,ax = plt.subplots(5,3,figsize=(20,20))

sns.set_style("dark")

k = 0

for i in range(5):

    for j in range(3):

        sns.countplot(data = df,x="Churn",hue=categorical[k],ax=ax[i][j])

        k+=1

### Visualize reverse of above plot ###

fig,ax = plt.subplots(5,3,figsize=(20,20))

sns.set_style("dark")

k = 0

for i in range(5):

    for j in range(3):

        ac = sns.countplot(data = df,x=categorical[k],hue="Churn",ax=ax[i][j])

         

        k+=1

### Do the same for the continuous variable distributions ###





def Viz(df,feat = "tenure"):

    df_yes = df[df.Churn == "Yes"]

    df_no = df[df.Churn =="No"]

    #fig,ax = plt.subplots(1,2,figsize=(20,20))



    tenure_yes = df_yes[feat]

    tenure_no = df_no[feat]

    

    sns.kdeplot(tenure_yes,label=feat+"_yes")

    sns.kdeplot(tenure_no,label=feat+"_no")

    plt.xlabel(feat)

    plt.show()



#fig,ax = plt.subplots(1,3,figsize=(20,20))

    

Viz(df,feat=cont[0])

Viz(df,feat=cont[1])

Viz(df,feat=cont[2])



## Comparative visualization of CDF for the following ###

def VizCDF(df,feat="tenure"):

    kwargs = {'cumulative': True}

    df_yes = df[df.Churn == "Yes"]

    df_no = df[df.Churn =="No"]

    tenure_yes = df_yes[feat]

    tenure_no = df_no[feat]



    #fig,ax = plt.subplots(1,3,figsize=(20,10))



    sns.distplot(tenure_yes, hist_kws=kwargs, kde_kws=kwargs,hist=False,label=feat+"_yes")

    sns.distplot(tenure_no, hist_kws=kwargs, kde_kws=kwargs,hist=False,label=feat+"_no")

    plt.show()



VizCDF(df,feat=cont[0])

VizCDF(df,feat=cont[1])

VizCDF(df,feat=cont[2])

### Boxplot to find outliers###

#sns.boxplot(df.tenure)

fig,ax = plt.subplots(3,2,figsize=(10,10))

df["TotalCharges"] = df["TotalCharges"].astype("float")

sns.violinplot(x=df.Churn,y=df.tenure,ax=ax[0][0])

sns.violinplot(x=df.Churn,y=df.MonthlyCharges,ax=ax[1][0])

sns.violinplot(x=df.Churn,y=df.TotalCharges,ax=ax[2][0])





sns.boxplot(x=df.Churn,y=df.tenure,ax=ax[0][1])

sns.boxplot(x=df.Churn,y=df.MonthlyCharges,ax=ax[1][1])

sns.boxplot(x=df.Churn,y=df.TotalCharges,ax=ax[2][1])

## Convert Outliers to mean ###



df[df.tenure>65].tenure = df.tenure.mean()

charge_yes = df[df.Churn=="Yes"].TotalCharges

charge_yes[charge_yes>5000] = charge_yes.mean()

charge_no = df[df.Churn=="No"].TotalCharges

df["TotalCharges"] = pd.concat([charge_yes,charge_no])
### Label encode to convert strings to integers###

from sklearn.preprocessing import LabelEncoder

lec = LabelEncoder()

dct = {}

classes = []

for col in df.columns:

    if col in categorical:

        dct[col] = list(lec.fit_transform(df[col]))

        #print(lec.classes_)

        if len(lec.classes_)>2:

        

            classes.append(lec.classes_)

    else:

        dct[col] = list(df[col].values)



## Visualize correlation heatmap ###

test = pd.DataFrame(dct)

fig = plt.subplots(figsize=(15,15))

sns.heatmap(test.corr(),annot=True)

plt.show()
### One hot encode variables with more than 2 categories ###

from sklearn.preprocessing import OneHotEncoder

onehot = OneHotEncoder()



## first column ###

features = ["InternetService","Contract","PaymentMethod"]

d = onehot.fit_transform(test[features[0]].values.reshape(-1,1)) 

onehot_df = pd.DataFrame(d.todense())

onehot_df.columns = ["DSL","Fibre Optic","No Internet service"]



## other 3 columns ##

for i in range(1,len(features)):

    d = onehot.fit_transform(test[features[i]].values.reshape(-1,1)) 

    temp = pd.DataFrame(d.todense())

    cols = []



    

    for j in range(len(classes[i])):

        cols.append(classes[i][j])

    temp.columns = cols

    onehot_df = pd.concat([onehot_df,temp],axis=1)

    

for feat in features:

    test = test.drop(feat,axis=1)

test = pd.concat([test,onehot_df],axis=1)

test.head()
## View Columns ##

test.columns
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,VotingClassifier

from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier

from sklearn.metrics import classification_report

from sklearn.feature_selection import SelectKBest,chi2

from imblearn.over_sampling import RandomOverSampler,SMOTE
## Separate data and labels ###

Y = test.Churn

X = test.drop("Churn",axis=1)

X = X.drop("customerID",axis=1)

## Drop total charges since its a redundant feature ###

#X = X.drop("TotalCharges",axis=1)
## Split train and test data

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,random_state=42,test_size=0.43)
## Set up machine learning models###



## Logistic Regression ###

clf_1 = LogisticRegression(random_state=42,max_iter=500)

clf_1.fit(X_train,Y_train)

pred = clf_1.predict(X_test)



## Random Forest ###

clf_forest = RandomForestClassifier(n_estimators=590)

clf_forest.fit(X_train,Y_train)

pred_forest = clf_forest.predict(X_test)



## Decision Tree ###

clf_tree = DecisionTreeClassifier(min_samples_split=5)

clf_tree.fit(X_train,Y_train)

pred_tree = clf_tree.predict(X_test)



## XGB Classifier ###

clf_xgb = XGBClassifier()

clf_xgb.fit(X_train,Y_train)

pred_xgb = clf_xgb.predict(X_test)





## GradientBoosting Classifier ##

clf_gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.65,\

                                 max_depth=1, random_state=0)

clf_gb.fit(X_train, Y_train)

pred = clf_gb.predict(X_test)



## Voting Classifier ##

clf_vote = VotingClassifier(estimators=[('lr', clf_1), ('xgb', clf_xgb),("gb",clf_gb)],

                         voting='soft')

clf_vote.fit(X_train,Y_train)

pred_vote = clf_vote.predict(X_test)
## Set up neural network ###

from keras.models import Sequential

from keras.layers import Dense

from keras import Input

from keras.optimizers import Adam



model = Sequential()

model.add(Dense(128,activation="relu",input_shape=(26,),use_bias=True))

model.add(Dense(128,activation="relu",use_bias=True))



model.add(Dense(32,activation="relu",use_bias=True))

model.add(Dense(32,activation="relu",use_bias=True))



model.add(Dense(1,activation="sigmoid"))
## Set up opt and loss ###

opt = Adam(learning_rate=1e-5)

model.compile(optimizer=opt,metrics=["accuracy"],loss="binary_crossentropy")
model.summary()
## Generate classification report ###

print("======== Logistic Regression ========")

print(classification_report(Y_test,pred))

print("======= Random Forest ======")

print(classification_report(Y_test,pred_forest))

print("==== Decision tree ======")

print(classification_report(Y_test,pred_tree))

print("========= XGB =========")

print(classification_report(Y_test,pred_xgb))

print("=========GradientBoosting======")

print(classification_report(Y_test,pred))

print("=========Voting======")

print(classification_report(Y_test,pred_vote))



### Training Loop ####

history = model.fit(x=X_train,y=Y_train,batch_size=32,epochs=200,validation_data=(X_test,Y_test))
fig,ax = plt.subplots(figsize=(8,8))

plt.plot(history.history["loss"],label = "Train Loss")

plt.plot(history.history["val_loss"],label="Val Loss")

plt.legend(['train', 'val'], loc='upper left')

plt.xlabel("epoch")

plt.ylabel("Loss")

plt.title("Training History")

plt.show()
## See the logistic regression weights ###

clf_1.coef_[0]
## Plot feature weights of logistic regression ###

import plotly.express as px

fig = px.bar(x=X.columns,y=clf_1.coef_[0],template="ggplot2",title="Logistic Regression weight visualization")

fig.update_xaxes(title="Features")

fig.update_yaxes(title="Weight")

fig.show()

## Please hover over the plot to get value
## See the xgb weights ##

clf_xgb.feature_importances_
## Plot feature importances of XGB

import plotly.express as px

fig = px.bar(x=X.columns,y=clf_xgb.feature_importances_,template="ggplot2")

fig.update_xaxes(title="Features")

fig.update_yaxes(title="Weight")

fig.show()
