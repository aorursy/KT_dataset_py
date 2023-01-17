import pandas as pd

data = pd.read_csv('../input/Narrativedata.csv',index_col=0)

#index_col=0以第0列為索引，如果不寫，會被當作是特徵之一

data.head()
data.info() #有年齡缺失值
Age = data.loc[:,"Age"].values.reshape(-1,1)

#因為數據是dataframe形式，可以進行提取，取值（索引消失）

#要被導入模型中，不能是一維矩陣，所以要用reshape(-1,1)改變維度

Age[:20]
#填補數據

from sklearn.impute import SimpleImputer

imp_mean = SimpleImputer()      #均值填補

imp_median = SimpleImputer(strategy="median") #中位數填補

imp_0 = SimpleImputer(strategy="constant",fill_value=0)  #用0填補

imp_mean = imp_mean.fit_transform(Age) #實例化，fit調出結果

imp_median = imp_median.fit_transform(Age) #實例化fit調出結果

imp_0 = imp_0.fit_transform(Age)         #實例化fit調出結果

#在这里我们使用中位数填补Age 

data.loc[:,"Age"] = imp_median

data.info()
#使用众数填补Embarked(屬於字符資料)

Embarked = data.loc[:,"Embarked"].values.reshape(-1,1)

#sklearn当中特征矩阵必须是二维

imp_mode = SimpleImputer(strategy = "most_frequent") 

#实例化，默认均值填补 #用中位数填补 #用0填补

#fit_transform一步完成调取结果

data.loc[:,"Embarked"] = imp_mode.fit_transform(Embarked) 

data.info()
import pandas as pd

data_ = pd.read_csv("../input/Narrativedata.csv",index_col=0)

data_.head()

data_.loc[:,"Age"] = data_.loc[:,"Age"].fillna(data_.loc[:,"Age"].median())

#.fillna 在DataFrame里面直接进行填补

#用中位數填補

data_.info()
data_ = data_.dropna(axis=0,inplace=False)
data_.info()
from sklearn.preprocessing import LabelEncoder



y = data.iloc[:,-1] #要输入的是标签，不是特征矩阵，所以允许一维

le = LabelEncoder()

le = le.fit(y)

label = le.transform(y)

#label是一維矩陣（891,)

le.classes_
le.fit_transform(y) 

le.inverse_transform(label)

data.iloc[:,-1] = label #让标签等于我们运行出来的结果 data.head()

data.head() #已經將字符改成數值資料
from sklearn.preprocessing import LabelEncoder

data.iloc[:,-1] = LabelEncoder().fit_transform(data.iloc[:,-1])

from sklearn.preprocessing import OrdinalEncoder 

#接口categories_对应LabelEncoder的接口classes_，一模一样的功能

#幫你看特徵中有多少個類別

data_ = data.copy() #保護原始資料

data_.head()

OrdinalEncoder().fit(data_.iloc[:,1:-1]).categories_

#所有的行，從索引為1開始，取到最後一列之前，-1代表是最後一列之前

#也就是從Sex到Emabarked  （也可以直接data_.iloc[:,1:3]前面會取到，後面不會取到

data_.iloc[:,1:-1] = OrdinalEncoder().fit_transform(data_.iloc[:,1:-1]) 

data_.head()
data.head()
#啞變量（有你沒我的概念）

from sklearn.preprocessing import OneHotEncoder

X = data.iloc[:,1:-1]#中間兩列

enc = OneHotEncoder(categories='auto').fit(X)   

result = enc.transform(X).toarray()   #兩個特徵五個類別（Sex	Embarked）

result
enc.get_feature_names() #每個矩陣（已經fit過的）名稱



result
result.shape

#axis=1,表示跨行进行合并，也就是将量表左右相连，如果是axis=0，就是将量表上下相连

# 變成二維矩陣
newdata = pd.concat([data,pd.DataFrame(result)],axis=1)#合併dataframe

#result 本來是array   ＃將result 加到有邊（因為axis=1)

newdata.head()

newdata.drop(["Sex","Embarked"],axis=1,inplace=True)#把原來的sex、embarked調

newdata.columns = ["Age","Survived","Female","Male","Embarked_C","Embarked_Q","Embarked_S"]

newdata.head()