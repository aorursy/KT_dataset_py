import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import linear_model
from sklearn.metrics import r2_score,mean_squared_error
filepath = r"../input/graduate-admissions/Admission_Predict.csv"
df = pd.read_csv(filepath,index_col='Serial No.')
df.isnull().sum()
df.columns=['gre','tofel','ranking','sop','lor','cgpa','research','admit']
sns.heatmap(df.corr(),annot=True)
df.corr().admit
new_df = df[['gre','cgpa','tofel','admit']]
sns.heatmap(new_df.corr(),annot=True)
def fourplots(listofplots,row,col,comment,data):
    fig, axs = plt.subplots(1, 4,figsize=(15,3))
    sub = []
    n = 0
    fig.suptitle(comment,fontsize=20,color='Gray')
    for i in listofplots:
        fig = sns.distplot(data[i],ax=axs[n])
        sub.append(axs)
        n+=1
list_of_plots = ['gre','cgpa','tofel','admit']
row = 1
col = 3
comment = 'Before removing outliers'
fourplots(list_of_plots,row,col,comment,new_df)
z = np.abs(stats.zscore(new_df))
clean = new_df[(z < 3).all(axis=1)]
comment = 'After removing outliers'
fourplots(list_of_plots,row,col,comment,clean)
X = clean.drop(['admit'],axis=1).values
y = clean['admit'].values
y.shape
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33, random_state=42)
reg = linear_model.LinearRegression()
reg.fit(X_train,y_train)
y_test.shape
pred = reg.predict(X_test)
reg.score(X_train,y_train)
plt.figure(figsize=(20,8))
y=pred
y1=y_test
x=np.arange(1, 133, 1)
x1=np.arange(0,133,10)
plt.plot(x,y,color='r',marker='o',label='Predicted',linestyle='dashed')
plt.plot(x,y1,color='g',label='Actual')
plt.xticks(x1)
plt.gca().legend(('Predicted','Test'))
plt.xlabel('Cases',fontsize=20)
plt.ylabel('Chance of getting admitted',fontsize=20)
plt.title('Predictions Vs True Values',fontsize=25)
plt.grid()
plt.ioff()
print("R2 score of our Model: ", end=" ")
print(r2_score(y_test,pred))
print("RMSE of our model: " ,end=' ')
print(np.sqrt(mean_squared_error(y_test,pred)))