import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix



train = pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')

train.shape

test.shape



train.sample(frac=0.1)

test.sample(frac=0.1)


plt.style.use(style='ggplot')

plt.rcParams['figure.figsize']=(10,6)



train.SalePrice.describe()

#describe the count mean min,max of the data

train.SalePrice.skew()

#describes the linearity of data

#+ve means tails towards right and vice versa

#a value closure to zrero is appreciable



plt.hist(train.SalePrice)



plt.scatter(train.Id,train.SalePrice)
#gives an idea about outliers



'''np.log()  and np.exp()'''



log_trans =np.log(train.SalePrice)



log_trans.skew()



plt.hist(log_trans)

plt.scatter(train.Id,log_trans)



exp_trans = np.exp(log_trans)

exp_trans.skew()



numeric_features=train.select_dtypes(include=[np.number])

corr=numeric_features.corr()



corr['SalePrice'].sort_values(ascending=False)[:5]#also use np.corr here

corr['SalePrice'].sort_values(ascending=False)[-5:]





train=train[train.GarageArea <1200]



plt.scatter(train.GarageArea,np.log(train.SalePrice))



plt.xlim(-200,1200)



nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:25])

nulls.columns=['Null Count']

nulls.index.name='Features'



categoricals=train.select_dtypes(exclude=np.number)

categoricals.describe()



train['enc_street']=pd.get_dummies(train.Street,drop_first=True)

test['enc_street']=pd.get_dummies(test.Street,drop_first=True)



train.enc_street.value_counts()


'''pivot table'''



def encode(x): return 1 if x=='Partial' else 0

train['enc_condition']=train.SaleCondition.apply(encode)

test['enc_condition']=test.SaleCondition.apply(encode)



pivot=train.pivot_table(index='SaleCondition',values='SalePrice',aggfunc=np.median)

pivot.plot(kind='Bar')



data=train.select_dtypes(include=[np.number]).interpolate().dropna()



sum(data.isnull().sum() !=0)



Y=np.log(train.SalePrice)



X=data.drop(['SalePrice','Id'],axis=1)



from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.33,random_state=42)



from sklearn.linear_model import LinearRegression

clf=LinearRegression()

clf.fit(X_train,Y_train)



clf.score(X_train,Y_train)



clf.score(X_test,Y_test)



pred=clf.predict(X_test)

print(pred)
from sklearn import linear_model



for i in range(-2,3):

    alpha=10**i

rm=linear_model.Ridge(alpha=alpha)

rm=rm.fit(X_train,Y_train)

pred=rm.predict(X_test)

print(pred)

# Submission



my_submission=pd.DataFrame({"Id":pred})

my_submission.to_csv('submission.csv', index=False)
