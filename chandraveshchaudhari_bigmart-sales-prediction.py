import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

plt.style.use('fivethirtyeight')

#import itertools
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
os.listdir()
traindata= pd.read_csv("/kaggle/input/big-mart-sales-prediction/Train.csv")

testdata= pd.read_csv("/kaggle/input/big-mart-sales-prediction/Test.csv")
traindata
testdata
corr = traindata.corr()

plt.subplots(figsize=(12,9))

sns.heatmap(corr,cmap="YlGnBu")
print(traindata.info())

print('*'*20)

print(testdata.info())

fig,axes=plt.subplots(1,1,figsize=(12,8))

sns.scatterplot(x='Item_MRP',y='Item_Outlet_Sales',hue='Item_Fat_Content',size='Item_Weight',data=traindata)
traindatacol= ['Item_Identifier', 'Item_Weight', 'Item_Fat_Content', 'Item_Visibility',

       'Item_Type', 'Item_MRP', 'Outlet_Identifier',

       'Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type',

        'Outlet_Type']

for i in traindatacol:

    print("print(traindata.{}.unique())".format(i))

    



    
traindata.Item_Type.unique()
print(traindata.Item_Identifier.unique())

print(traindata.Item_Weight.unique())

print(traindata.Item_Fat_Content.unique())

print(traindata.Item_Visibility.unique())

print(traindata.Item_Type.unique())

print(traindata.Item_MRP.unique())

print(traindata.Outlet_Identifier.unique())

print(traindata.Outlet_Establishment_Year.unique())

print(traindata.Outlet_Size.unique())

print(traindata.Outlet_Location_Type.unique())

print(traindata.Outlet_Type.unique())
traindata.hist(figsize=(15,12))


traindatacol= ['Item_Identifier', 'Item_Weight', 'Item_Fat_Content', 'Item_Visibility',

       'Item_Type', 'Item_MRP', 'Outlet_Identifier',

       'Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type',

        'Outlet_Type']

for i in traindatacol:

    print("traindata.{}.value_counts()".format(i))

hel = [traindata.Item_Identifier.value_counts(),

traindata.Item_Weight.value_counts(),

traindata.Item_Fat_Content.value_counts(),

traindata.Item_Visibility.value_counts(),

traindata.Item_Type.value_counts(),

traindata.Item_MRP.value_counts(),

traindata.Outlet_Identifier.value_counts(),

traindata.Outlet_Establishment_Year.value_counts(),

traindata.Outlet_Size.value_counts(),

traindata.Outlet_Location_Type.value_counts(),

traindata.Outlet_Type.value_counts()]

for i in range(11):

    print(hel[i])
traindata.apply(lambda x: len(x.unique()))
testdata.apply(lambda x: len(x.unique()))
traindata=traindata.drop(columns=['Item_Identifier','Item_Visibility'])
testdata=testdata.drop(columns=['Item_Identifier','Item_Visibility'])
traindata.Item_Weight.describe()
testdata.Item_Weight.describe()
traindata.loc[traindata.Item_Weight.isnull()]
# we can remove Item_Weight due to low correlation but here is how we can fill nan values and use

traindata.Item_Weight.fillna(12.857645, inplace=True)

testdata.Item_Weight.fillna(12.695633, inplace=True)
traindata.Item_Weight.isnull().any()
traindata.loc[traindata.Outlet_Size.isnull()]
traindata.groupby(['Outlet_Location_Type','Outlet_Type'])['Outlet_Size'].value_counts()
traindata.groupby('Outlet_Size').Outlet_Identifier.value_counts()
traindata.loc[(traindata.Outlet_Size.isnull())&(traindata.Outlet_Identifier=='OUT013'),'Outlet_Size']='High'

traindata.loc[(traindata.Outlet_Size.isnull())&(traindata.Outlet_Identifier=='OUT027'),'Outlet_Size']='Medium'

traindata.loc[(traindata.Outlet_Size.isnull())&(traindata.Outlet_Identifier=='OUT049'),'Outlet_Size']='Medium'

traindata.loc[(traindata.Outlet_Size.isnull())&(traindata.Outlet_Identifier=='OUT018'),'Outlet_Size']='Medium'



traindata.loc[(traindata.Outlet_Size.isnull())&(traindata.Outlet_Identifier=='OUT035'),'Outlet_Size']='Small'

traindata.loc[(traindata.Outlet_Size.isnull())&(traindata.Outlet_Identifier=='OUT046'),'Outlet_Size']='Small'

traindata.loc[(traindata.Outlet_Size.isnull())&(traindata.Outlet_Identifier=='OUT019'),'Outlet_Size']='Small'
testdata.loc[(testdata.Outlet_Size.isnull())&(testdata.Outlet_Identifier=='OUT013'),'Outlet_Size']='High'

testdata.loc[(testdata.Outlet_Size.isnull())&(testdata.Outlet_Identifier=='OUT027'),'Outlet_Size']='Medium'

testdata.loc[(testdata.Outlet_Size.isnull())&(testdata.Outlet_Identifier=='OUT049'),'Outlet_Size']='Medium'

testdata.loc[(testdata.Outlet_Size.isnull())&(testdata.Outlet_Identifier=='OUT018'),'Outlet_Size']='Medium'



testdata.loc[(testdata.Outlet_Size.isnull())&(testdata.Outlet_Identifier=='OUT035'),'Outlet_Size']='Small'

testdata.loc[(testdata.Outlet_Size.isnull())&(testdata.Outlet_Identifier=='OUT046'),'Outlet_Size']='Small'

testdata.loc[(testdata.Outlet_Size.isnull())&(testdata.Outlet_Identifier=='OUT019'),'Outlet_Size']='Small'
traindata.loc[traindata.Outlet_Size.isnull()]
traindata.loc[(traindata.Outlet_Size.isnull())&(traindata.Outlet_Type=='Grocery Store'),'Outlet_Size']='Small'

testdata.loc[(testdata.Outlet_Size.isnull())&(testdata.Outlet_Type=='Grocery Store'),'Outlet_Size']='Small'
traindata.loc[traindata.Outlet_Size.isnull()]
traindata.loc[(traindata.Outlet_Size.isnull())&(traindata.Outlet_Type=='Supermarket Type1')&(traindata.Outlet_Location_Type=='Tier 2'),'Outlet_Size']='Small'

testdata.loc[(testdata.Outlet_Size.isnull())&(testdata.Outlet_Type=='Supermarket Type1')&(testdata.Outlet_Location_Type=='Tier 2'),'Outlet_Size']='Small'
traindata.loc[traindata.Outlet_Size.isnull()]
traindata.isnull().sum()
testdata.isnull().sum()
traindata['Item_Fat_Content'].replace(['low fat','LF ','reg'],['Low Fat','Low Fat','Regular'],inplace=True)

testdata['Item_Fat_Content'].replace(['low fat','LF ','reg'],['Low Fat','Low Fat','Regular'],inplace=True)
traindata['Item_Fat_Content'].value_counts()
fig,axes=plt.subplots(1,1,figsize=(12,8))

sns.scatterplot(x='Item_MRP',y='Item_Outlet_Sales',hue='Item_Fat_Content',size='Item_Weight',data=traindata)
traindata
testdata
Y_train=traindata.pop('Item_Outlet_Sales')

traindata
Y_train.head()
traindata.Outlet_Establishment_Year= traindata.Outlet_Establishment_Year.astype(object)


testdata.Outlet_Establishment_Year= testdata.Outlet_Establishment_Year.astype(object) 
traindata= pd.get_dummies(traindata)
testdata= pd.get_dummies(testdata)
traindata.head()
testdata.head()
X_train=traindata
from sklearn.model_selection import train_test_split
Y_train
x_train1, x_test1, y_train1, y_test1 = train_test_split( X_train, Y_train, test_size=0.09, random_state=42)
x_test1.shape
y_train1
x_train1
Y_test= testdata
Y_test
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_val_score

import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge

from sklearn.tree import DecisionTreeRegressor

from sklearn.neural_network import MLPRegressor   

from sklearn.ensemble import  GradientBoostingRegressor
import warnings

warnings.filterwarnings('ignore')
import warnings

warnings.filterwarnings('ignore')

model= [LinearRegression(), DecisionTreeRegressor() ,   Lasso(), Ridge(),  MLPRegressor(), GradientBoostingRegressor()  ]

name = ['LinearRegression','DecisionTreeRegressor','Lasso','Ridge','MLPRegressor','GradientBoostingRegressor']

SCORE= []

TESTING=[]

RSME=[]

for ku in model:

    #ku will be replaced with each model like as first one is LogisticRegression()

    algorithm = ku.fit(x_train1,y_train1)

    print(ku)

    #now 'algorithm' will be fitted by API with above line and next line will check score with data training and testing

    predict_ku=ku.predict(x_test1)

    print('RSME: {:.4f}'.format(np.sqrt(mean_squared_error(y_test1,predict_ku))))

    score=cross_val_score(ku,x_train1,y_train1,cv=10,scoring='neg_mean_squared_error')

    ku_score_cross=np.sqrt(-score)

    

    print('mean: {:.2f} and std:{:.2f}'.format(np.mean(ku_score_cross),np.std(ku_score_cross)))

    print('---'*10)

    print('training set accuracy: {:.2f}'.format(algorithm.score(x_train1,y_train1)))

    print('test set accuracy: {:.2f}'.format(algorithm.score(x_test1,y_test1)))

    print('---'*30)

    #Now we are making a dataframe where by each loop the dataframe is added by SCORE,TESTING

    RSME.append(np.sqrt(mean_squared_error(y_test1,predict_ku)))

    SCORE.append(algorithm.score(x_train1,y_train1))

    TESTING.append(algorithm.score(x_test1,y_test1))

models_dataframe=pd.DataFrame({'training score':SCORE,'testing score':TESTING,'RSME':RSME},index=name)
models_dataframe
asendingtraining = models_dataframe.sort_values(by='RSME', ascending=False)

asendingtraining 
asendingtraining['RSME'].plot.barh(width=0.8)

plt.title('RSME')

fig=plt.gcf()

fig.set_size_inches(8,8)

plt.show()
model = GradientBoostingRegressor()

model.fit(X_train,Y_train)

prediction=model.predict(Y_test)
sample=pd.read_csv('/kaggle/input/big-mart-sales-prediction/Submission.csv')
del sample['Item_Outlet_Sales']
df=pd.DataFrame({'Item_Outlet_Sales':prediction})

corr_ans=pd.concat([sample,df],axis=1)

del corr_ans['Unnamed: 0']

corr_ans
corr_ans.to_csv('correct.csv',index=None)
#corr_ans.to_csv('C:\\Users\\ernag\\Desktop\\ML Projects\BigMart Sales Prediction\corr_submission.csv',index=False)