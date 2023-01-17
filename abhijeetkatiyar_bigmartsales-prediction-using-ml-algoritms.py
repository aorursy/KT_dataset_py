import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np



%matplotlib inline

sns.set(color_codes=True)
Train_data=pd.read_csv("../input/Train_UWu5bXk.csv")

Test_data=pd.read_csv("../input/Test_u94Q5KV.csv")
Train_data.head()
Test_data.head()
print(Train_data.shape)

print(Test_data.shape)
Train_data.info()
['Item_Fat_Content','Item_Type','Outlet_Size','Outlet_Location_Type','Outlet_Type']
Train_data.isnull().sum()
Data=Train_data.append(Test_data,sort=False)
Data.isnull().sum()
Train_data.describe()
Train_data.describe().columns
for i in Train_data.describe().columns:

    sns.distplot(Data[i].dropna())

    plt.show()
for i in Train_data.describe().columns:

    sns.boxplot(Data[i].dropna())

    plt.show()
sns.boxplot(Data['Item_Visibility'])
Data['Item_Visibility'].describe()
sns.boxplot(y=Data['Item_Weight'],x=Data['Outlet_Identifier'])

plt.xticks(rotation='vertical')
Data['Item_Fat_Content'].value_counts()
Data['Item_Fat_Content'] = Data['Item_Fat_Content'].replace({'LF':'Low Fat','reg':'Regular','low fat':'Low Fat'})
Data.groupby('Item_Identifier')['Item_Weight'].mean().head(5)
for i in Data.groupby('Item_Identifier')['Item_Weight'].mean().index:

    Data.loc[Data.loc[:,'Item_Identifier']==i,'Item_Weight']=Data.groupby('Item_Identifier')['Item_Weight'].mean()[i]
Data['Outlet_Type'].value_counts()
Data.Outlet_Size[Data['Outlet_Type']=='Grocery Store'].value_counts()
Data.Outlet_Size[Data['Outlet_Type']=='Supermarket Type1'].value_counts()
Data.Outlet_Size[Data['Outlet_Type']=='Supermarket Type2'].value_counts()
Data.Outlet_Size[Data['Outlet_Type']=='Supermarket Type3'].value_counts()
#Data['Outlet_Size'].fillna(Data['Outlet_Size'].mode()[0],inplace=True)

Data.Outlet_Size.fillna(Data.Outlet_Size[Data['Outlet_Type']=='Grocery Store'].mode()[0],inplace=True)
Data.Outlet_Size.fillna(Data.Outlet_Size[Data['Outlet_Type']=='Supermarket Type1'].mode()[0],inplace=True)
Data.Outlet_Size.fillna(Data.Outlet_Size[Data['Outlet_Type']=='Supermarket Type2'].mode()[0],inplace=True)
Data.Outlet_Size.fillna(Data.Outlet_Size[Data['Outlet_Type']=='Supermarket Type3'].mode()[0],inplace=True)
for i in Data.groupby('Item_Identifier')['Item_Visibility'].mean().index:

    Data.loc[Data.loc[:,'Item_Identifier']==i,'Item_Visibility']=Data.groupby('Item_Identifier')['Item_Visibility'].mean()[i]
Data['Outlet_Establishment_Year']=2013-Data['Outlet_Establishment_Year']
Data
Data.isnull().sum()
Train_data=Data.dropna()
Test_Data=Data[Data['Item_Outlet_Sales'].isnull()]

Test_Data.drop('Item_Outlet_Sales',axis=1,inplace=True)
sns.boxplot(Train_data['Item_Visibility'])



# Remove outliers from Item visiblity
Train_data['Item_Visibility'].describe()


print(Test_Data.shape)

print(Train_data.shape)
len(Train_data)
len(Test_data)
from sklearn.preprocessing import LabelEncoder
categorical_list=['Item_Fat_Content','Item_Type','Outlet_Identifier','Outlet_Size','Outlet_Location_Type','Outlet_Type','Outlet_Establishment_Year']
'''le = LabelEncoder()

for i in categorical_list:

    Data[i]=le.fit_transform(Data[i])

    Data[i]=Data[i].astype('category')'''
le = LabelEncoder()

for i in categorical_list:

    Train_data[i]=le.fit_transform(Train_data[i])

    Train_data[i]=Train_data[i].astype('category')

    Test_Data[i]=le.fit_transform(Test_Data[i])

    Test_Data[i]=Test_Data[i].astype('category')
Data
Test_Data.head()
Train_data.head()
Train_data.corr()
#from sklearn.model_selection import train_test_split
#X_train, x_test, Y_train, y_test = train_test_split(Data.drop(['Item_Outlet_Sales','Item_Identifier'],axis=1), Data['Item_Outlet_Sales'], test_size = 0.3)
#Data.Item_Visibility[Data['Item_Visibility']==0].value_counts()
from sklearn.linear_model import LinearRegression as LR
Lm=LR(normalize=True)
Lm.fit(Train_data.drop(['Item_Outlet_Sales','Outlet_Identifier','Item_Identifier'],axis=1),Train_data['Item_Outlet_Sales'])
#Lm.score(x_test,y_test)
#pred_train=Lm.predict(X_train)
#pred_test=Lm.predict(x_test)
#from sklearn import metrics
#metrics.mean_squared_error(Y_train,pred_train)
#metrics.mean_squared_error(y_test,pred_test)
#metrics.mean_squared_error(Y_train,pred_train)-metrics.mean_squared_error(y_test,pred_test)
Train_data
Y_train=Train_data['Item_Outlet_Sales']

X_train=Train_data.drop('Item_Outlet_Sales',axis=1)
train=Train_data.drop(['Item_Outlet_Sales'],axis=1)

predictions=Train_data['Item_Outlet_Sales']

out=[]

LM_model=LR(normalize=True)

for i in range(len(Test_Data)):

    LM_fit=LM_model.fit(train.drop(['Outlet_Identifier','Item_Identifier'],axis=1),predictions)

    Output=LM_fit.predict(Test_Data.drop(['Item_Identifier','Outlet_Identifier'],axis=1)[Test_Data.index==i])

    out.append(Output)

    train.append(Test_Data[Test_Data.index==i])

    predictions.append(pd.Series(Output))

    

    
len(out)
len(Test_Data)
outp=np.vstack(out)
ansp=pd.Series(data=outp[:,0],index=Test_Data.index,name='Item_Outlet_Sales')
Outp_df=pd.DataFrame([Test_data['Item_Identifier'],Test_data['Outlet_Identifier'],ansp]).T
Outp_df.to_csv('UploadLMP.csv',index=False)
mod1_train_pred=Lm.predict(Train_data.drop(['Item_Identifier','Outlet_Identifier','Item_Outlet_Sales'],axis=1))
from sklearn import metrics

from math import sqrt
sqrt(metrics.mean_squared_error(Train_data['Item_Outlet_Sales'],mod1_train_pred))/np.mean(Train_data['Item_Outlet_Sales'])
# analytics vidhya score

#1273.7483459686
ans=Lm.predict(Test_Data.drop(['Item_Identifier','Outlet_Identifier'],axis=1))
answer=pd.Series(data=ans,index=Test_Data.index,name='Item_Outlet_Sales')
#len(ans)
#pd.DataFrame(np.array([[Test_Data['Item_Identifier']],[Test_Data['Outlet_Identifier']],[answer]]),index=Test_Data.index,columns=['Item_Identifier','Outlet_Identifier','Item_Outlet_Sales'])
#dict(ItemTest_Data['Item_Identifier'],Test_Data['Outlet_Identifier'],ans)
#(pd.DataFrame(Test_data['Item_Identifier'],Test_data['Outlet_Identifier'],answer).T).to_csv("upload.csv",encoding='utf-8', index=False)
#Out_df=pd.DataFrame([Test_data['Item_Identifier'],Test_data['Outlet_Identifier'],answer]).T
#Out_df
#Out_df.to_csv('Upload2.csv',index=False)
from sklearn.linear_model import Ridge
#predictors = [x for x in train.columns if x not in [target]+IDcol]
rr=Ridge(alpha=0.5,fit_intercept=True,normalize=True)
rr.fit(Train_data.drop(['Item_Outlet_Sales','Outlet_Identifier','Item_Identifier'],axis=1),Train_data['Item_Outlet_Sales'])
rr_pred=rr.predict(Test_Data.drop(['Item_Identifier','Outlet_Identifier'],axis=1))
rr_pred_train=rr.predict(Train_data.drop(['Item_Identifier','Outlet_Identifier','Item_Outlet_Sales'],axis=1))
rr_ans=pd.Series(data=rr_pred,index=Test_Data.index,name='Item_Outlet_Sales')
len(ans)
rr_out=pd.DataFrame([Test_data['Item_Identifier'],Test_data['Outlet_Identifier'],rr_ans]).T
#rr_out.to_csv('Uploadrr.csv',index=False)
sqrt(metrics.mean_squared_error(Train_data['Item_Outlet_Sales'],rr_pred_train))/np.mean(Train_data['Item_Outlet_Sales'])
from sklearn.linear_model import Lasso
lasso=Lasso(alpha=0.5,fit_intercept=True,normalize=True)
lasso.fit(Train_data.drop(['Item_Outlet_Sales','Outlet_Identifier','Item_Identifier'],axis=1),Train_data['Item_Outlet_Sales'])
lasso_pred=lasso.predict(Test_Data.drop(['Item_Identifier','Outlet_Identifier'],axis=1))
lasso_pred_train=lasso.predict(Train_data.drop(['Item_Outlet_Sales','Outlet_Identifier','Item_Identifier'],axis=1))
lasso_ans=pd.Series(data=lasso_pred,index=Test_Data.index,name='Item_Outlet_Sales')
len(lasso_pred)
lasso_out=pd.DataFrame([Test_data['Item_Identifier'],Test_data['Outlet_Identifier'],rr_ans]).T
#lasso_out.to_csv('Uploadlasso.csv',index=False)
sqrt(metrics.mean_squared_error(Train_data['Item_Outlet_Sales'],lasso_pred_train))/np.mean(Train_data['Item_Outlet_Sales'])
from sklearn.svm import SVR
svr=SVR(kernel='linear',gamma='auto',C=5,epsilon=1.2)
svr.fit(Train_data.drop(['Item_Outlet_Sales','Outlet_Identifier','Item_Identifier'],axis=1),Train_data['Item_Outlet_Sales'])
svr_pred=svr.predict(Test_Data.drop(['Item_Identifier','Outlet_Identifier'],axis=1))
svr_pred_train=svr.predict(Train_data.drop(['Item_Outlet_Sales','Outlet_Identifier','Item_Identifier'],axis=1))
svr_ans=pd.Series(data=svr_pred,index=Test_Data.index,name='Item_Outlet_Sales')
svr_out=pd.DataFrame([Test_data['Item_Identifier'],Test_data['Outlet_Identifier'],svr_ans]).T
#svr_out.to_csv('Uploadsvr.csv',index=False)
sqrt(metrics.mean_squared_error(Train_data['Item_Outlet_Sales'],svr_pred_train))/np.mean(Train_data['Item_Outlet_Sales'])
sqrt(metrics.mean_squared_error(Train_data['Item_Outlet_Sales'],svr_pred_train))
svr_pred_train
#Train_data['Item_Outlet_Sales']
from sklearn.tree import DecisionTreeRegressor
DTR=DecisionTreeRegressor()

DTR.fit(Train_data.drop(['Item_Outlet_Sales','Outlet_Identifier','Item_Identifier'],axis=1),Train_data['Item_Outlet_Sales'])
dtr_pred_train=DTR.predict(Train_data.drop(['Item_Outlet_Sales','Outlet_Identifier','Item_Identifier'],axis=1))
sqrt(metrics.mean_squared_error(Train_data['Item_Outlet_Sales'],dtr_pred_train))/np.mean(Train_data['Item_Outlet_Sales'])
sqrt(metrics.mean_squared_error(Train_data['Item_Outlet_Sales'],dtr_pred_train))
dtr_pred_train
dtr_pred=DTR.predict(Test_Data.drop(['Item_Identifier','Outlet_Identifier'],axis=1))
dtr_ans=pd.Series(data=dtr_pred,index=Test_Data.index,name='Item_Outlet_Sales')
dtr_out=pd.DataFrame([Test_data['Item_Identifier'],Test_data['Outlet_Identifier'],dtr_ans]).T
#dtr_out.to_csv('Uploaddtr.csv',index=False)
from sklearn.metrics import r2_score
r2_score(Train_data['Item_Outlet_Sales'],dtr_pred_train)
from sklearn.neural_network import MLPRegressor
ann=MLPRegressor(activation='relu',alpha=2.0,learning_rate='adaptive',warm_start=True,hidden_layer_sizes=(2500,),max_iter=1000)
ann.fit(Train_data.drop(['Item_Outlet_Sales','Outlet_Identifier','Item_Identifier'],axis=1),Train_data['Item_Outlet_Sales'])
ann_train_pred=ann.predict(Train_data.drop(['Item_Outlet_Sales','Outlet_Identifier','Item_Identifier'],axis=1))
ann_train_pred
r2_score(Train_data['Item_Outlet_Sales'],ann_train_pred)
sqrt(metrics.mean_squared_error(Train_data['Item_Outlet_Sales'],ann_train_pred))/np.mean(Train_data['Item_Outlet_Sales'])
ann_pred=ann.predict(Test_Data.drop(['Item_Identifier','Outlet_Identifier'],axis=1))
ann_ans=pd.Series(data=ann_pred,index=Test_Data.index,name='Item_Outlet_Sales')
ann_out=pd.DataFrame([Test_data['Item_Identifier'],Test_data['Outlet_Identifier'],ann_ans]).T
ann_out.to_csv('Uploadann.csv',index=False)
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline
X_train=np.array(Train_data.drop(['Item_Outlet_Sales','Outlet_Identifier','Item_Identifier'],axis=1))

y_train=np.array(Train_data['Item_Outlet_Sales'])
#X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))

#X_test = np.reshape(X_test, (X_test.shape[0], 1,))
# define base model

def baseline_model():

    # create model

    model = Sequential()

    model.add(Dense(91, input_dim=9, kernel_initializer='Orthogonal', activation='elu'))

    model.add(Dense(78,activation='relu',kernel_initializer='Orthogonal'))

    model.add(Dense(65,activation='relu',kernel_initializer='Orthogonal'))

    model.add(Dense(52,activation='relu',kernel_initializer='Orthogonal'))

    model.add(Dense(39,activation='relu',kernel_initializer='Orthogonal'))    

    model.add(Dense(26,activation='relu',kernel_initializer='Orthogonal'))

    model.add(Dense(1,activation='relu',kernel_initializer='Orthogonal'))   

    # Compile model

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model
seed = 7

np.random.seed(seed)

estimator = KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=20, verbose=0)
estimator.fit(X_train,y_train, batch_size = 50, epochs = 200)
ann1_pred_train=estimator.predict(X_train)
r2_score(Train_data['Item_Outlet_Sales'],ann1_pred_train)
sqrt(metrics.mean_squared_error(Train_data['Item_Outlet_Sales'],ann1_pred_train))/np.mean(Train_data['Item_Outlet_Sales'])
#X_test=np.array(Test_Data.drop(['Item_Identifier','Outlet_Identifier'],axis=1))
#X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
X_train
Test_Data.head()

Test_Data.shape
Train_data.shape
ann1_pred=estimator.predict(np.array(Test_Data.drop(['Outlet_Identifier','Item_Identifier'],axis=1)))
ann1_ans=pd.Series(data=ann1_pred,index=Test_Data.index,name='Item_Outlet_Sales')
ann1_out=pd.DataFrame([Test_data['Item_Identifier'],Test_data['Outlet_Identifier'],ann1_ans]).T
ann1_out.to_csv('Uploadann1.csv',index=False)