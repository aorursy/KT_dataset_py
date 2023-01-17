import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("white")
raw_data = pd.read_csv("/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv")
raw_data.head(10)
raw_data.describe(include='all')
data = raw_data.drop("Serial No.",axis=1)
data.head(5)
data.info()
data.isnull().any().any()
data.columns
for i, column in enumerate(data.columns):
    if(i<8):
        plt.subplot(4,2,i+1)
        plt.hist(data[column],color = "blue",bins = 15)
        plt.xlabel(column)
        plt.ylabel("Frequancy")
        fig, ax = plt.gcf(), plt.gca()
        fig.set_size_inches(13, 12)
        plt.tight_layout()
        plt.grid(True)
correlation = data.corr()
correlation
mask = np.zeros_like(correlation)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(9, 7))
    ax = sns.heatmap(correlation,mask=mask,square=True,annot=True,fmt='0.2f',linewidths=.8,cmap="YlOrRd")
data = data.rename(columns={'Chance of Admit ': "Chance of Admit","LOR ": "LOR"})
plt.scatter(data["GRE Score"],data["Chance of Admit"],color = "blue")
plt.xlabel("GRE Score")
plt.ylabel("Chance of Admit")
plt.grid(True)
plt.show()
plt.scatter(data["TOEFL Score"],data["Chance of Admit"],color = "blue")
plt.xlabel("TOEFL Score")
plt.ylabel("Chance of Admit")
plt.grid(True)
plt.show()
plt.scatter(data["CGPA"],data["Chance of Admit"],color = "blue")
plt.xlabel("CGPA")
plt.ylabel("Chance of Admit")
plt.grid(True)
plt.show()
# Assuming those who chance grater than 0.8 got admission
Cutoff = 0.80
data['Chance of Admission'] = data['Chance of Admit'].map(lambda x : 1 if x >= Cutoff else 0)
data.head()
print("No. of students got admission based on above cutoff :",list(data["Chance of Admission"].value_counts())[1])
fig,ax = plt.subplots(1,3,sharey=True,figsize = (16,5))
data.plot(kind = "scatter",x = 'GRE Score' , y = 'Chance of Admit' , c ='Chance of Admission',ax = ax[0],grid = True,cmap = "bwr")
data.plot(kind = "scatter",x = 'TOEFL Score' ,y = 'Chance of Admit',c = 'Chance of Admission' ,ax = ax[1],grid = True,cmap = "bwr")
data.plot(kind = "scatter",x = 'CGPA' ,y = 'Chance of Admit' ,c = 'Chance of Admission' ,ax = ax[2],grid = True ,cmap = "bwr")
ax[0].set_title('GRE Score vs Chance of Admit')
ax[1].set_title('TOEFL Score vs Chance of Admit')
ax[2].set_title('CGPA vs Chance of Admit')
plt.show()
fig,ax = plt.subplots(2,2,sharey=False,figsize = (16,12))
data.plot(kind = "scatter",x = 'GRE Score' , y = 'Chance of Admit'  , c ='Research',ax = ax[0,0],grid = True,cmap = "bwr")
data.plot(kind = "scatter",x = 'TOEFL Score' ,y = 'Chance of Admit' ,c = 'Research' ,ax = ax[0,1],grid = True,cmap = "bwr")
data.plot(kind = "scatter",x = 'GRE Score' ,y = 'TOEFL Score' ,c = 'Research' ,ax = ax[1,0],grid = True ,cmap = "bwr")
data.plot(kind = "scatter",x = 'CGPA' ,y = 'Chance of Admit' ,c = 'Research' ,ax = ax[1,1],grid = True ,cmap = "bwr")
ax[0,0].set_title('GRE Score vs Chance of Admit')
ax[0,1].set_title('TOEFL Score vs Chance of Admit')
ax[1,0].set_title('GRE Score vs TOEFL Score')
ax[1,1].set_title('CGPA vs Chance of Admit')
ax[0,0].axhline(Cutoff,color = "green",linestyle='dashed')
ax[0,1].axhline(Cutoff,color = "green",linestyle='dashed')
ax[1,0].axhline(105,color = "green",linestyle='dashed')
ax[1,1].axhline(Cutoff,color = "green",linestyle='dashed')
ax[0,0].axvline(310,color = "black",linestyle='dashed')
ax[0,1].axvline(105,color = "black",linestyle='dashed')
ax[1,0].axvline(310,color = "black",linestyle='dashed')
ax[1,1].axvline(8.5,color = "black",linestyle='dashed')
plt.show()
data1 = data[(data['GRE Score']>=310) & (data['Research']==1) & (data["TOEFL Score"]>=105)]
data2 = data[(data['GRE Score']>=310) & (data['Research']==0) & (data["TOEFL Score"]>=105)]
data3 = data[(data['Chance of Admission']==1)]
#data5 = data[(data["Research"] == 1)]
#data6 = data[(data['GRE Score']>=310) & (data["TOEFL Score"]>=105)]
#data5.shape
#data6.shape
print("% of students who have scored greater than 310,105 in GRE and TOEFL respectively and does have research experience:"
      ,(data1.shape[0]/data.shape[0])*100)
print("% of students who have scored greater than 310,105 in GRE and TOEFL respectively and does NOT have research experience: ",
      (data2.shape[0]/data.shape[0])*100)
#print((data5.shape[0]/data.shape[0])*100)
fig,ax = plt.subplots(1,2,figsize = (16,5))
sns.swarmplot(y="SOP", x="University Rating", hue="Chance of Admission",palette=["b", "r"],ax = ax[0], data=data)
sns.swarmplot(y="LOR", x="University Rating", hue="Chance of Admission",palette=["b", "r"],ax = ax[1], data=data)
ax[0].set_title('University Rating vs SOP Rating')
ax[1].set_title('University Rating vs LOR Rating')
ax[0].grid(True)
ax[1].grid(True)
plt.show()
data_SOP = data[(data['University Rating']>=3) & (data["SOP"]<3)]
data_LOR = data[(data['University Rating']>=3) & (data["LOR"]<3)]
print("% of students having university rating above or equal to 3  have got SOP rating below 3 is: "
      ,(data_SOP.shape[0]/data.shape[0])*100)
print("% of students having university rating above or equal to 3  have got LOR rating below 3 is: "
      ,(data_LOR.shape[0]/data.shape[0])*100)
print("Average rating of SOP of students who got admission is : ",data3["SOP"].mean())
print("Average rating of LOR of students who got admission is : ",data3["LOR"].mean())
fig,ax = plt.subplots(1,figsize = (8,5))
sns.countplot("University Rating", hue="Chance of Admission",palette=["b", "r"], data=data)
plt.title('University Rating vs Chance of Admit')
plt.grid(True)
plt.show()
data4 = data[(data['Chance of Admit'] == 0.9)]
data4.sort_values(by = ["GRE Score"],ascending = False)
print(data4.mean())
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import statsmodels.api as sm
x1 = data.iloc[:,:-2]
y = data.iloc[:,7:8]
y[0:5]
x = sm.add_constant(x1)
# Fit the model, according to the OLS (ordinary least squares) method with a dependent variable y and an idependent x
results = sm.OLS(y,x).fit()
print(results.summary())
x1 = data[["GRE Score","TOEFL Score","CGPA","LOR","Research"]]
y = data.iloc[:,7:8]
x = sm.add_constant(x1)
results = sm.OLS(y,x).fit()
print(results.summary())
x1 = data[["GRE Score","TOEFL Score","CGPA","LOR","Research"]]
y = data.iloc[:,7:8]
x_train,x_test,y_train,y_test = train_test_split(x1 , y , test_size = 0.2,random_state = 42)
print(x_train.shape)
print(x_test.shape)
print(y_test.shape)
print(y_train.shape)
x = sm.add_constant(x_train)
results = sm.OLS(y_train,x).fit()
print(results.summary())
lreg = LinearRegression()
lreg.fit(x_train,y_train)
print("Coeffecient:",lreg.coef_)
print("Intercept: ",lreg.intercept_)

## Prediction with OLS
x_test_ols =  x_test.copy() 
x_test_ols = sm.add_constant(x_test_ols)
y_pred_ols = results.predict(x_test_ols)
print(y_pred_ols[0:5])
## Prediction with Sklearn
ypred = lreg.predict(x_test)
print(ypred[0:5] )
## From sklearn R2 score
from sklearn.metrics import r2_score
print("Mean absolute error: %.2f" % np.mean(np.absolute(ypred - y_test)))
print("Residual sum of squares (MSE): %.2f" % np.mean((ypred - y_test) ** 2))
print("R2-score: %.2f" % r2_score(ypred, y_test))
