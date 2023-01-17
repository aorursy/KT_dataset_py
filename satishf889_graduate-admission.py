import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
graduation_data=pd.read_csv("../input/graduate-admissions/Admission_Predict.csv")
graduation_data.head()
#Getting all the information of our dataset
graduation_data.info()
#We will find out mean,max,count of all the columns in our dataframe
graduation_data.describe()
graduation_data.drop(['Serial No.'],axis=1,inplace=True)

#Now we will be declare features and output
feature_X=graduation_data.drop(['Chance of Admit '],axis=1)
feature_Y=graduation_data['Chance of Admit ']
feature_X.head()
#Initialize sklearn MinMaxScalar
scaler =MinMaxScaler()
feature_to_normalize=feature_X.values
normalized_feature=scaler.fit_transform(feature_to_normalize)
# Create dataframe of normalized feature
df_normalized = pd.DataFrame(normalized_feature)
df_normalized.columns=feature_X.columns
df_normalized.head()
f,ax = plt.subplots(figsize=(14, 14))
sns.heatmap(df_normalized.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
df_normalized.drop(['TOEFL Score','CGPA'],axis=1,inplace=True)
x=df_normalized
y=feature_Y
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.20,shuffle='false')
model_LR=LinearRegression()
model_LR.fit(X_train,Y_train)

prediction=model_LR.predict(X_test)
print(f"Mean Square Error using Linear Regressor is {(np.sqrt(mean_squared_error(Y_test, prediction)))}")
from sklearn.tree import DecisionTreeRegressor

model_DT=DecisionTreeRegressor()
model_DT.fit(X_train,Y_train)

prediction=model_DT.predict(X_test)
print(f"Mean Square Error using Decison Tree is {(np.sqrt(mean_squared_error(Y_test, prediction)))}")
model_RF=RandomForestRegressor()
model_RF.fit(X_train,Y_train)

prediction=model_RF.predict(X_test)
print(f"Mean Square Error using RandomForestRegressor is {(np.sqrt(mean_squared_error(Y_test, prediction)))}")
model_KN=KNeighborsRegressor()
model_KN.fit(X_train,Y_train)

prediction=model_KN.predict(X_test)
print(f"Mean Square Error using Kneighbors is {(np.sqrt(mean_squared_error(Y_test, prediction)))}")
model_SVR=SVR()
model_SVR.fit(X_train,Y_train)
prediction=model_SVR.predict(X_test)
print(f"Mean Square Error using SVR is {(np.sqrt(mean_squared_error(Y_test, prediction)))}")
import pickle
# print(os.listdir())
filename='admission_model.pkl'
# pickle.dump(model_LR, open("./Model/"+filename, 'wb'))