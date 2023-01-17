import pandas as pd
from sklearn.model_selection import train_test_split

data=pd.read_csv("../input/GenderIdentification.csv")
print("\n Complete Data set Twitter PanAP16 : ")
print(data.head(10))
X= data.loc[:, data.columns != 'Gender']
y= data.loc[:, data.columns == 'Gender']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
train_data=pd.DataFrame()
train_data["Comment"] =X_train["Comment"]
train_data['Gender'] = y_train['Gender']

print("\n Train Dataset Twitter PanAP16 : ")
print(train_data.head(10))
test_data = pd.DataFrame()
test_data["Comment"] =X_test["Comment"]
test_data['Gender'] = y_test['Gender']

print("\n Test Dataset Twitter PanAP16 : ")
print(test_data.head(10))
print("\n Train Dataset Columns : ")
print(train_data.columns)
print("\n Number of Instances in Train Dataset : ",len(train_data))
print("\n Test Dataset Columns : ")
print(test_data.columns)
print("\n Number of Instances in Test Dataset : ", len(test_data))

