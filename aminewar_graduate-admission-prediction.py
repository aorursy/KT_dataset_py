import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error

import statsmodels.api as sm

from statsmodels.tools.eval_measures import mse, rmse



%matplotlib inline

pd.options.display.float_format = '{:.3f}'.format



import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv("../input/Admission_Predict_Ver1.1.csv")

df = df.rename(columns={"Chance of Admit ":"Chance of Admit",

                        "LOR ":"LOR"}) 

df=df[['Chance of Admit','GRE Score', 'TOEFL Score', 'University Rating', 'SOP','LOR', 'CGPA','Research']]
df.head()
df.info()
plt.figure(figsize=(15,10))



col_list=df.columns

 

for i in range(8):

    plt.subplot(2,4,i+1)

    plt.boxplot(df[col_list[i]])

    plt.title(col_list[i],color="g",fontsize=15)





plt.show()
plt.figure(figsize=(15,10))



plt.subplot(2, 2, 1)

sns.distplot(df['GRE Score'],color='red')

plt.xlabel("GRE Score",fontsize=14,color='red')

plt.subplot(2, 2, 2)

sns.distplot(df['TOEFL Score'],color='green')

plt.xlabel("TOEFL Score",fontsize=14,color='green')

plt.subplot(2,2,3)

sns.distplot(df['CGPA'],color='purple')

plt.xlabel("CGPA",fontsize=14,color='purple')

plt.subplot(2,2,4)

sns.distplot(df['Chance of Admit'],color='blue')

plt.xlabel("Chance of Admit",fontsize=14,color='blue')



plt.show()
plt.figure(figsize=(20,10))



deg=["LOR","SOP","University Rating","Research"]



for i in range(4):

    plt.subplot(2,2,i+1)

    sns.countplot(x=deg[i],data=df)

    plt.xlabel(deg[i],color="purple",fontsize=20)

    plt.ylabel("Count",color="purple",fontsize=15)

    plt.yticks(fontsize=13)

    plt.xticks(fontsize=13)

plt.show() 
plt.figure(figsize=(18,7))

plt.subplot(1,3,1)

sns.scatterplot(x="GRE Score",y="Chance of Admit",data=df,hue="Research",palette="prism_r")

plt.subplot(1,3,2)

sns.scatterplot(x="TOEFL Score",y="Chance of Admit",data=df,hue="Research",palette="prism_r")

plt.subplot(1,3,3)

sns.scatterplot(x="CGPA",y="Chance of Admit",data=df,hue="Research",palette="prism_r")

plt.show()
plt.figure(figsize=(18,7))



plt.subplot(131)

sns.scatterplot('GRE Score','Chance of Admit', data = df,palette='viridis',hue = 'University Rating')

plt.subplot(132)

sns.scatterplot('TOEFL Score','Chance of Admit', data = df,palette='viridis',hue = 'University Rating')

plt.subplot(133)

sns.scatterplot('CGPA','Chance of Admit', data = df,palette='viridis',hue = 'University Rating')

plt.show()
plt.figure(figsize=(15,5))

sns.swarmplot(x="University Rating",y="Chance of Admit",hue="Research",data=df,palette="prism_r")

plt.xticks(fontsize=10)

plt.yticks(fontsize=10)

plt.xlabel("University Rating",color="red",fontsize=16)

plt.ylabel("Chance of Admit",color="red",fontsize=15)

plt.show()
plt.figure(figsize=(15,5))

sns.swarmplot(x="SOP",y="Chance of Admit",hue="Research",data=df,palette="nipy_spectral")

plt.xticks(fontsize=10)

plt.yticks(fontsize=10)

plt.xlabel("SOP",color="darkcyan",fontsize=16)

plt.ylabel("Chance of Admit",color="darkcyan",fontsize=15)

plt.show()
plt.figure(figsize=(15,5))

sns.swarmplot(x="LOR",y="Chance of Admit",hue="Research",data=df,palette="spring")

plt.xticks(fontsize=10)

plt.yticks(fontsize=10)

plt.xlabel("LOR",color="deeppink",fontsize=16)

plt.ylabel("Chance of Admit",color="deeppink",fontsize=15)

plt.show()
plt.figure(figsize=(15,25))

plt.subplot(4,2,1)

sns.boxplot(x="LOR", y="Chance of Admit", data=df)

plt.title("LOR and Chance of Admit",fontsize=15,color="m")

plt.subplot(4,2,2)

sns.boxplot(x="SOP", y="Chance of Admit", data=df)

plt.title("SOP and Chance of Admit",fontsize=15,color="m")

plt.subplot(4,2,3)

sns.boxplot(x="LOR", y="GRE Score", data=df)

plt.title("LOR and GRE Score",fontsize=15,color="m")

plt.subplot(4,2,4)

sns.boxplot(x="SOP", y="GRE Score", data=df)

plt.title("SOP and GRE Score",fontsize=15,color="m")

plt.subplot(4,2,5)

sns.boxplot(x="LOR", y="TOEFL Score", data=df)

plt.title("LOR and TOEFL Score",fontsize=15,color="m")

plt.subplot(4,2,6)

sns.boxplot(x="SOP", y="TOEFL Score", data=df)

plt.title("SOP and TOEFL Score",fontsize=15,color="m")

plt.subplot(4,2,7)

sns.boxplot(x="LOR", y="CGPA", data=df)

plt.title("LOR and CGPA",fontsize=15,color="m")

plt.subplot(4,2,8)

sns.boxplot(x="SOP", y="CGPA", data=df)

plt.title("SOP and CGPA",fontsize=15,color="m")



plt.show()
plt.figure(figsize=(10,10))

corr=df.corr()

sns.heatmap(corr,square=True,annot=True,linewidths=.5,cmap='viridis')

plt.title("Graduate Admissions Correlation Matrix",color="purple",fontsize=15)



plt.show()
df.describe().T
corr["Chance of Admit"].sort_values(ascending=False)
X_dogru = df["CGPA"]

y = df["Chance of Admit"]
import statsmodels.api as sm

X_dogru = sm.add_constant(X_dogru)

model_dogrusal = sm.OLS(y,X_dogru).fit()

model_dogrusal.summary()
print("Chance of Admit = "+str("%.2f"%model_dogrusal.params[0] +" + CGPA * "+str("%.2f"%model_dogrusal.params[1])))
import matplotlib.pyplot as plt

plt.figure(figsize=(7,5))



baslik_font = {'family': 'arial','color':  'darkblue','weight': 'bold','size': 14 }

eksen_font  = {'family': 'arial','color':  'darkblue','weight': 'bold','size': 13 }



sns.regplot(df["CGPA"],df["Chance of Admit"],ci=None,scatter_kws={"color":"r","s":9})

plt.xlabel("CGPA",fontdict=eksen_font)

plt.ylabel("Chance of Admit",fontdict=eksen_font)

plt.title("Model Equation : Chance of Admit = -1.04 + CGPA * 0.21",fontdict=baslik_font)

plt.show()
X = df [["CGPA","University Rating","SOP","LOR","Research"]]

y = df ["Chance of Admit"]
X_train_one,X_test_one,y_train_one,y_test_one = train_test_split(X,y,test_size=0.2,random_state=465)



X_train_one = sm.add_constant(X_train_one)

X_test_one  = sm.add_constant(X_test_one)



model_1 = sm.OLS(y_train_one,X_train_one).fit()



y_preds_train_one = model_1.predict (X_train_one)

y_preds_test_one = model_1.predict(X_test_one)

model_1.summary()
def calc_adj_rsq(r2,n,p):

    return ((1-(1-r2)*((n-1)/(n-p-1))))
plt.figure(figsize=(15,7))



baslik_font = {'family': 'arial','color':  'darkblue','weight': 'bold','size': 15 }

eksen_font = {'family': 'arial','color':  'darkblue','weight': 'bold','size': 13 }





plt.subplot(1,2,1)

plt.scatter(y_test_one,y_preds_test_one,alpha=0.7,color="purple")

plt.scatter(y_train_one,y_preds_train_one,alpha=0.4,color="green")

plt.plot(y_test_one,y_test_one,color="blue")

plt.xlabel("Actual Values",fontdict=eksen_font)

plt.ylabel("Estimated Values(With Test Set Values)",fontdict=eksen_font)

plt.title("Chance of Admit : Actual and Estimated Values",fontdict=baslik_font)



plt.subplot(1,2,2)

artik_test_one  = y_preds_test_one - y_test_one

artik_train_one = y_preds_train_one - y_train_one

plt.scatter(y_test_one,artik_test_one,alpha=0.7,color="purple")

plt.scatter(y_train_one,artik_train_one,alpha=0.4,color="green")

plt.hlines(y = 0, xmin = 0, xmax = 1, color = "blue")

plt.xticks(rotation=90)

plt.xlabel("Estimated",fontdict=eksen_font)

plt.ylabel("Residual",fontdict=eksen_font)

plt.title("Residual and Estimation",fontdict=baslik_font)





plt.show()



r2_train_one = r2_score(y_train_one,y_preds_train_one)

r2_test_one  = r2_score(y_test_one,y_preds_test_one)

adj_rsq_train_one = calc_adj_rsq(r2_train_one,X_train_one.shape[0],len(X_train_one.columns))

adj_rsq_test_one  = calc_adj_rsq(r2_test_one,X_test_one.shape[0],len(X_test_one.columns))

rmse_train_one = rmse(y_train_one, y_preds_train_one)

rmse_test_one  = rmse(y_test_one, y_preds_test_one)



print("Train Set Adjusted R-Squared            : {:.4f}".format(calc_adj_rsq(r2_train_one,X_train_one.shape[0],len(X_train_one.columns))))

print("Mean Absolute Error (MAE)               : {:.4f}".format(mean_absolute_error(y_train_one, y_preds_train_one)))

print("Mean Square Error (MSE) (MSE)           : {:.4f}".format(mse(y_train_one, y_preds_train_one)))

print("Root Mean Square Error (RMSE)           : {:.4f}".format(rmse(y_train_one, y_preds_train_one)))

print("Mean Absolute Percentage Error (MAPE)   : {:.4f}".format(np.mean(np.abs((y_train_one - y_preds_train_one) / y_train_one)) * 100),"\n")



print("Test Set Adjusted R-Squared             : {:.4f}".format(calc_adj_rsq(r2_test_one,X_test_one.shape[0],len(X_test_one.columns))))

print("Mean Absolute Error(MAE)                : {:.4f}".format(mean_absolute_error(y_test_one, y_preds_test_one)))

print("Mean Square Error (MSE)                 : {:.4f}".format(mse(y_test_one, y_preds_test_one)))

print("Root Mean Square Error (RMSE)           : {:.4f}".format(rmse(y_test_one, y_preds_test_one)))

print("Mean Absolute Percentage Error (MAPE)   : {:.4f}".format(np.mean(np.abs((y_test_one - y_preds_test_one) / y_test_one)) * 100))

X = df[["CGPA","GRE Score","TOEFL Score"]]

y = df ["Chance of Admit"]
X_train_two,X_test_two,y_train_two,y_test_two = train_test_split(X,y,test_size=0.2,random_state=465)



X_train_two = sm.add_constant(X_train_two)

X_test_two  = sm.add_constant(X_test_two)



model_2 =sm.OLS(y_train_two,X_train_two).fit()



y_preds_train_two = model_2.predict (X_train_two)

y_preds_test_two  = model_2.predict(X_test_two)

model_2.summary()

plt.figure(figsize=(15,7))



baslik_font = {'family': 'arial','color':  'darkblue','weight': 'bold','size': 15 }

eksen_font = {'family': 'arial','color':  'darkblue','weight': 'bold','size': 13 }





plt.subplot(1,2,1)

plt.scatter(y_test_two,y_preds_test_two,alpha=0.7,color="purple")

plt.scatter(y_train_two,y_preds_train_two,alpha=0.4,color="green")

plt.plot(y_test_two,y_test_two,color="blue")

plt.xlabel("Actual Values",fontdict=eksen_font)

plt.ylabel("Estimated Values (With Test Set Values)",fontdict=eksen_font)

plt.title("Chance of Admit : Actual and Estimated Values",fontdict=baslik_font)



plt.subplot(1,2,2)

artik_test_two  = y_preds_test_two - y_test_two

artik_train_two = y_preds_train_two - y_train_two

plt.scatter(y_test_two,artik_test_two,alpha=0.7,color="purple")

plt.scatter(y_train_two,artik_train_two,alpha=0.4,color="green")

plt.hlines(y = 0, xmin = 0, xmax = 1, color = "blue")

plt.xticks(rotation=90)

plt.xlabel("Estimated",fontdict=eksen_font)

plt.ylabel("Residual",fontdict=eksen_font)

plt.title("Residual and Estimation",fontdict=baslik_font)



plt.show()



r2_train_two = r2_score(y_train_two,y_preds_train_two)

r2_test_two  = r2_score(y_test_two,y_preds_test_two)

adj_rsq_train_two = calc_adj_rsq(r2_train_two,X_train_two.shape[0],len(X_train_two.columns))

adj_rsq_test_two  = calc_adj_rsq(r2_test_two,X_test_two.shape[0],len(X_test_two.columns))

rmse_train_two = rmse(y_train_two, y_preds_train_two)

rmse_test_two  = rmse(y_test_two, y_preds_test_two)





print("Train Set Adjusted R-Squared            : {:.4f}".format(calc_adj_rsq(r2_train_two,X_train_two.shape[0],len(X_train_two.columns))))

print("Mean Absolute Error (MAE)               : {:.4f}".format(mean_absolute_error(y_train_two, y_preds_train_two)))

print("Mean Square Error (MSE)                 : {:.4f}".format(mse(y_train_two, y_preds_train_two)))

print("Root Mean Square Error (RMSE)           : {:.4f}".format(rmse(y_train_two, y_preds_train_two)))

print("Mean Absolute Percentage Error (MAPE)   : {:.4f}".format(np.mean(np.abs((y_train_two - y_preds_train_two) / y_train_two)) * 100),"\n")



print("Test Set Adjusted R-Squared             : {:.4f}".format(calc_adj_rsq(r2_test_two,X_test_two.shape[0],len(X_test_two.columns))))

print("Mean Absolute Error (MAE)               : {:.4f}".format(mean_absolute_error(y_test_two, y_preds_test_two)))

print("Mean Square Error (MSE)                 : {:.4f}".format(mse(y_test_two, y_preds_test_two)))

print("Root Mean Square Error (RMSE)           : {:.4f}".format(rmse(y_test_two, y_preds_test_two)))

print("Mean Absolute Percentage Error (MAPE)   : {:.4f}".format(np.mean(np.abs((y_test_two - y_preds_test_two) / y_test_two)) * 100))

uni_dummies = pd.get_dummies(df['SOP'],prefix='SOP')

df = pd.concat([df,uni_dummies],axis=1)

df.head()
y = df["Chance of Admit"]

X_dummy_sop = df[['GRE Score', 'TOEFL Score','LOR', 'CGPA', 'Research','SOP_1.0',

       'SOP_1.5', 'SOP_2.0', 'SOP_2.5', 'SOP_3.0', 'SOP_3.5', 'SOP_4.0',

       'SOP_4.5', 'SOP_5.0']]
X_train_dum, X_test_dum, y_train_dum, y_test_dum = train_test_split(X_dummy_sop,y,test_size = 0.2,random_state =465)



X_train_dum = sm.add_constant(X_train_dum)

X_test_dum  = sm.add_constant(X_test_dum)



model_dummy_top_1 = sm.OLS(y_train_dum, X_train_dum).fit()



y_preds_test_dum  = model_dummy_top_1.predict(X_test_dum)

y_preds_train_dum = model_dummy_top_1.predict(X_train_dum)



model_dummy_top_1.summary()
plt.figure(figsize=(15,7))



baslik_font = {'family': 'arial','color': 'darkblue' , 'weight': 'bold','size': 15 }

eksen_font =  {'family': 'arial','color': 'darkblue' , 'weight': 'bold','size': 13 }





plt.subplot(1,2,1)

plt.scatter(y_test_dum,y_preds_test_dum,alpha=0.7,color="purple")

plt.scatter(y_train_dum,y_preds_train_dum,alpha=0.4,color="green")

plt.plot(y_test_dum,y_test_dum,color="blue")

plt.xlabel("Actual Values",fontdict=eksen_font)

plt.ylabel("Estimated Values (With Test Set Values)",fontdict=eksen_font)

plt.title("Chance of Admit : Actual and Estimated Values",fontdict=baslik_font)



plt.subplot(1,2,2)

artik_test_dum  = y_preds_test_dum - y_test_dum

artik_train_dum = y_preds_train_dum - y_train_dum

plt.scatter(y_test_dum,artik_test_dum,alpha=0.7,color="purple")

plt.scatter(y_train_dum,artik_train_dum,alpha=0.4,color="green")

plt.hlines(y = 0, xmin = 0, xmax = 1, color = "blue")

plt.xticks(rotation=90)

plt.xlabel("Estimated",fontdict=eksen_font)

plt.ylabel("Residual",fontdict=eksen_font)

plt.title("Residual and Estimation",fontdict=baslik_font)



plt.show()



r2_train_dum = r2_score(y_train_dum,y_preds_train_dum)

r2_test_dum  = r2_score(y_test_dum,y_preds_test_dum)

adj_rsq_train_sop = calc_adj_rsq(r2_train_dum,X_train_dum.shape[0],len(X_train_dum.columns))

adj_rsq_test_sop  = calc_adj_rsq(r2_test_dum,X_test_dum.shape[0],len(X_test_dum.columns))

rmse_train_sop = rmse(y_train_dum, y_preds_train_dum)

rmse_train_sop = rmse(y_test_dum, y_preds_test_dum)



print("Train Set Adjusted R-Squared            : {:.4f}".format(calc_adj_rsq(r2_train_dum,X_train_dum.shape[0],len(X_train_dum.columns))))

print("Mean Absolute Error (MAE)               : {:.4f}".format(mean_absolute_error(y_train_dum, y_preds_train_dum)))

print("Mean Square Error (MSE)                 : {:.4f}".format(mse(y_train_dum, y_preds_train_dum)))

print("Root Mean Square Error (RMSE)           : {:.4f}".format(rmse(y_train_dum, y_preds_train_dum)))

print("Mean Absolute Percentage Error (MAPE)   : {:.4f}".format(np.mean(np.abs((y_train_dum - y_preds_train_dum) / y_train_dum)) * 100),"\n")



print("Test Set Adjusted R-Squared             : {:.4f}".format(calc_adj_rsq(r2_test_dum,X_test_dum.shape[0],len(X_test_dum.columns))))

print("Mean Absolute Error (MAE)               : {:.4f}".format(mean_absolute_error(y_test_dum, y_preds_test_dum)))

print("Mean Square Error (MSE)                 : {:.4f}".format(mse(y_test_dum, y_preds_test_dum)))

print("Root Mean Square Error (RMSE)           : {:.4f}".format(rmse(y_test_dum, y_preds_test_dum)))

print("Mean Absolute Percentage Error (MAPE)   : {:.4f}".format(np.mean(np.abs((y_test_dum - y_preds_test_dum) / y_test_dum)) * 100))

uni_dummies = pd.get_dummies(df['University Rating'],prefix='University Rating')

df = pd.concat([df,uni_dummies],axis=1)

df.head()
y = df["Chance of Admit"]

X_dummy_uni = df[['GRE Score', 'TOEFL Score','LOR', 'CGPA', 'Research','University Rating_1', 'University Rating_2', 'University Rating_3',

       'University Rating_4', 'University Rating_5']]
X_train_dumu, X_test_dumu, y_train_dumu, y_test_dumu = train_test_split(X_dummy_uni,y,test_size = 0.2,random_state =465)



X_train_dumu = sm.add_constant(X_train_dumu)

X_test_dumu  = sm.add_constant(X_test_dumu)



model_dummy_uni_1 = sm.OLS(y_train_dumu, X_train_dumu).fit()





y_preds_test_dumu  = model_dummy_uni_1.predict(X_test_dumu)

y_preds_train_dumu = model_dummy_uni_1.predict(X_train_dumu)



model_dummy_uni_1.summary()
plt.figure(figsize=(15,7))



baslik_font = {'family': 'arial','color': 'darkblue','weight': 'bold','size': 15 }

eksen_font = {'family': 'arial','color':  'darkblue','weight': 'bold','size': 13 }





plt.subplot(1,2,1)

plt.scatter(y_test_dumu,y_preds_test_dumu,alpha=0.7,color="purple")

plt.scatter(y_train_dumu,y_preds_train_dumu,alpha=0.4,color="green")

plt.plot(y_test_dumu,y_test_dumu,color="blue")

plt.xlabel("Actual Values",fontdict=eksen_font)

plt.ylabel("Estimated Values (With Test Set Values)",fontdict=eksen_font)

plt.title("Chance of Admit : Actual and Estimated Values",fontdict=baslik_font)



plt.subplot(1,2,2)

artik_test_dumu  = y_preds_test_dumu - y_test_dumu

artik_train_dumu = y_preds_train_dumu - y_train_dumu

plt.scatter(y_test_dumu,artik_test_dumu,alpha=0.7,color="purple")

plt.scatter(y_train_dumu,artik_train_dumu,alpha=0.4,color="green")

plt.xlabel("Estimated",fontdict=eksen_font)

plt.ylabel("Residual",fontdict=eksen_font)

plt.title("Residual and Estimation",fontdict=baslik_font)

plt.hlines(y = 0, xmin = 0, xmax = 1, color = "blue")

plt.xticks(rotation=90)



plt.show()



r2_train_dumu = r2_score(y_train_dumu,y_preds_train_dumu)

r2_test_dumu  = r2_score(y_test_dumu,y_preds_test_dumu)

adj_rsq_train_uni= calc_adj_rsq(r2_train_dumu,X_train_dumu.shape[0],len(X_train_dumu.columns))

adj_rsq_test_uni = calc_adj_rsq(r2_test_dumu,X_test_dumu.shape[0],len(X_test_dumu.columns))

rmse_train_uni   = rmse(y_train_dumu, y_preds_train_dumu)

rmse_test_uni    = rmse(y_test_dumu, y_preds_test_dumu)





print("Train Set Adjusted R-Squared             : {:.4f}".format(calc_adj_rsq(r2_train_dumu,X_train_dumu.shape[0],len(X_train_dumu.columns))))

print("Mean Absolute Error (MAE)                : {:.4f}".format(mean_absolute_error(y_train_dumu, y_preds_train_dumu)))

print("Mean Square Error (MSE)                  : {:.4f}".format(mse(y_train_dumu, y_preds_train_dumu)))

print("Root Mean Square Error (RMSE)            : {:.4f}".format(rmse(y_train_dumu, y_preds_train_dumu)))

print("Mean Absolute Percentage Error (MAPE)    : {:.4f}".format(np.mean(np.abs((y_train_dumu - y_preds_train_dumu) / y_train_dumu)) * 100),"\n")



print("Test Set Adjusted R-Squared              : {:.4f}".format(calc_adj_rsq(r2_test_dumu,X_test_dumu.shape[0],len(X_test_dumu.columns))))

print("Mean Absolute Error (MAE)                : {:.4f}".format(mean_absolute_error(y_test_dumu, y_preds_test_dumu)))

print("Mean Square Error (MSE)                  : {:.4f}".format(mse(y_test_dumu, y_preds_test_dumu)))

print("Root Mean Square Error (RMSE)            : {:.4f}".format(rmse(y_test_dumu, y_preds_test_dumu)))

print("Mean Absolute Percentage Error (MAPE)    : {:.4f}".format(np.mean(np.abs((y_test_dumu - y_preds_test_dumu) / y_test_dumu)) * 100))



y = df["Chance of Admit"]

X_dummy_top = df[['GRE Score', 'TOEFL Score',

              'LOR', 'CGPA', 'Research', 'SOP_1.0',

              'SOP_1.5', 'SOP_2.0', 'SOP_2.5', 'SOP_3.0', 'SOP_3.5', 'SOP_4.0',

              'SOP_4.5', 'SOP_5.0', 'University Rating_1', 'University Rating_2',

              'University Rating_3', 'University Rating_4', 'University Rating_5']]
X_train_dum_top, X_test_dum_top, y_train_dum_top, y_test_dum_top = train_test_split(X_dummy_top,y,test_size = 0.2,random_state =465)



X_train_dum_top = sm.add_constant(X_train_dum_top)

X_test_dum_top  = sm.add_constant(X_test_dum_top)



model_dummy_top_1 = sm.OLS(y_train_dum_top, X_train_dum_top).fit()



y_preds_test_dum_top  = model_dummy_top_1.predict(X_test_dum_top)

y_preds_train_dum_top = model_dummy_top_1.predict(X_train_dum_top)



model_dummy_top_1.summary()

plt.figure(figsize=(15,7))



baslik_font = {'family': 'arial','color': 'darkblue','weight': 'bold','size': 15 }

eksen_font =  {'family': 'arial','color': 'darkblue','weight': 'bold','size': 13 }





plt.subplot(1,2,1)

plt.scatter(y_test_dum_top,y_preds_test_dum_top,alpha=0.7,color="purple")

plt.scatter(y_train_dum_top,y_preds_train_dum_top,alpha=0.4,color="green")

plt.plot(y_test_dum_top,y_test_dum_top,color="blue")

plt.xlabel("Actual Values",fontdict=eksen_font)

plt.ylabel("Estimated Values (With Test Set Values)",fontdict=eksen_font)

plt.title("Chance of Admit : Actual and Estimated Values",fontdict=baslik_font)



plt.subplot(1,2,2)

artik_test_dum_top  = y_preds_test_dum_top - y_test_dum_top

artik_train_dum_top = y_preds_train_dum_top - y_train_dum_top

plt.scatter(y_test_dum_top,artik_test_dum_top,alpha=0.7,color="purple")

plt.scatter(y_train_dum_top,artik_train_dum_top,alpha=0.4,color="green")

plt.xlabel("Estimated",fontdict=eksen_font)

plt.ylabel("Residual",fontdict=eksen_font)

plt.title("Residual and Estimation",fontdict=baslik_font)

plt.hlines(y = 0, xmin = 0, xmax = 1, color = "blue")

plt.xticks(rotation=90)



plt.show()



r2_train_dum_top   = r2_score(y_train_dum_top,y_preds_train_dum_top)

r2_test_dum_top    = r2_score(y_test_dum_top,y_preds_test_dum_top)

adj_rsq_dum_top    = calc_adj_rsq(r2_train_dum_top,X_train_dum_top.shape[0],len(X_train_dum_top.columns))

adj_rsq_dum_top    = calc_adj_rsq(r2_test_dum_top,X_test_dum_top.shape[0],len(X_test_dum_top.columns))

rmse_train_dum_top = rmse(y_train_dum_top, y_preds_train_dum_top)

rmse_test_dum_top  = rmse(y_test_dum_top, y_preds_test_dum_top)



print("Train Set Adjusted R-Squared            : {:.4f}".format(calc_adj_rsq(r2_train_dum_top,X_train_dum_top.shape[0],len(X_train_dum_top.columns))))

print("Mean Absolute Error (MAE)               : {:.4f}".format(mean_absolute_error(y_train_dum_top, y_preds_train_dum_top)))

print("Mean Square Error (MSE)                 : {:.4f}".format(mse(y_train_dum_top, y_preds_train_dum_top)))

print("Root Mean Square Error (RMSE)           : {:.4f}".format(rmse(y_train_dum_top, y_preds_train_dum_top)))

print("Mean Absolute Percentage Error (MAPE)   : {:.4f}".format(np.mean(np.abs((y_train_dum_top - y_preds_train_dum_top) / y_train_dum_top)) * 100),"\n")



print("Test Set Adjusted R-Squared             : {:.4f}".format(calc_adj_rsq(r2_test_dum_top,X_test_dum_top.shape[0],len(X_test_dum_top.columns))))

print("Mean Absolute Error (MAE)               : {:.4f}".format(mean_absolute_error(y_test_dum_top, y_preds_test_dum_top)))

print("Mean Square Error (MSE)                 : {:.4f}".format(mse(y_test_dum_top, y_preds_test_dum_top)))

print("Root Mean Square Error (RMSE)           : {:.4f}".format(rmse(y_test_dum_top, y_preds_test_dum_top)))

print("Mean Absolute Percentage Error (MAPE)   : {:.4f}".format(np.mean(np.abs((y_test_dum_top - y_preds_test_dum_top) / y_test_dum_top)) * 100))



def poly_hesapla(X,y,n):

    from sklearn.preprocessing import PolynomialFeatures

    import statsmodels.api as sm

    poly = PolynomialFeatures(n)

    X_poly=poly.fit_transform(X)

    X_poly = sm.add_constant(X_poly)

    model_poly=sm.OLS(y,X_poly).fit()

    print("anlamlı p değer sayısı : {} ".format(len(model_poly.pvalues[model_poly.pvalues < 0.1])),"\n")

    print("R-squared      : {}".format(model_poly.rsquared))

    print("Adj. R-squared : {}".format(model_poly.rsquared_adj))

def model_poly_summary_hesapla(X,y,n):

    from sklearn.preprocessing import PolynomialFeatures

    import statsmodels.api as sm

    poly = PolynomialFeatures(n)

    X_poly=poly.fit_transform(X)

    X_poly=pd.DataFrame(X_poly,columns = poly.get_feature_names(X.columns))

   

    

    X_train, X_test, y_train, y_test = train_test_split(X_poly,y,test_size = 0.2,random_state =465)



    X_train = sm.add_constant(X_train)



    model = sm.OLS(y_train, X_train).fit()



    return model.summary()
def grafik_ciz(X,y,n):

    

    from sklearn.preprocessing import PolynomialFeatures

    import statsmodels.api as sm

    poly = PolynomialFeatures(n)

    X_poly=poly.fit_transform(X)

    X_poly=pd.DataFrame(X_poly,columns = poly.get_feature_names(X.columns))

   

    

    X_train, X_test, y_train, y_test = train_test_split(X_poly,y,test_size = 0.2,random_state =465)



    X_train = sm.add_constant(X_train)



    model = sm.OLS(y_train, X_train).fit()

 

    y_preds_test = model.predict(X_test)

    y_preds_train = model.predict(X_train)





    plt.figure(figsize=(15,7))



    baslik_font = {'family': 'arial','color':  'darkblue','weight': 'bold','size': 15 }

    eksen_font  = {'family': 'arial','color':  'darkblue','weight': 'bold','size': 13 }





    plt.subplot(1,2,1)

    plt.scatter(y_test,y_preds_test,alpha=0.7,color="purple")

    plt.scatter(y_train,y_preds_train,alpha=0.4,color="green")

    plt.plot(y_test,y_test,color="blue")

    plt.xlabel("Actual Values",fontdict=eksen_font)

    plt.ylabel("Estimated Values (With Test Set Values)",fontdict=eksen_font)

    plt.title("Chance of Admit : Actual and Estimated Values ",fontdict=baslik_font)



    plt.subplot(1,2,2)

    artik_test  = y_preds_test - y_test

    artik_train = y_preds_train - y_train

    plt.scatter(y_test,artik_test,alpha=0.7,color="purple")

    plt.scatter(y_train,artik_train,alpha=0.4,color="green")

    plt.xlabel("Estimated",fontdict=eksen_font)

    plt.ylabel("Residual",fontdict=eksen_font)

    plt.title("Residual and Estimation",fontdict=baslik_font)

    plt.hlines(y = 0, xmin = 0, xmax = 1, color = "blue")

    plt.xticks(rotation=90)



    plt.show()

    

    r2_train = r2_score(y_train,y_preds_train)

    r2_test  = r2_score(y_test,y_preds_test)

    

    print("Train Set Adjusted R-Squared             : {:.4f}".format(calc_adj_rsq(r2_train,X_train.shape[0],len(X_train.columns))))

    print("Mean Absolute Error (MAE)                : {:.4f}".format(mean_absolute_error(y_train, y_preds_train)))

    print("Mean Square Error (MSE)                  : {:.4f}".format(mse(y_train, y_preds_train)))

    print("Root Mean Square Error (RMSE)            : {:.4f}".format(rmse(y_train, y_preds_train)))

    print("Mean Absolute Percentage Error (MAPE)    : {:.4f}".format(np.mean(np.abs((y_train - y_preds_train) / y_train)) * 100),"\n")

    print("Test Set Adjusted R-Squared              : {:.4f}".format(calc_adj_rsq(r2_test,X_test.shape[0],len(X_test.columns))))

    print("Mean Absolute Error(MAE)                 : {:.4f}".format(mean_absolute_error(y_test, y_preds_test)))

    print("Mean Square Error (MSE)                  : {:.4f}".format(mse(y_test, y_preds_test)))

    print("Root Mean Square Error (RMSE)            : {:.4f}".format(rmse(y_test, y_preds_test)))

    print("Mean Absolute Percentage Error (MAPE)    : {:.4f}".format(np.mean(np.abs((y_test - y_preds_test) / y_test)) * 100))

          

                
y = df["Chance of Admit"]

X = df[['GRE Score', 'TOEFL Score',

        'LOR', 'CGPA', 'Research','SOP','University Rating']]
poly_hesapla(X,y,1)
model_poly_summary_hesapla(X,y,1)
grafik_ciz(X,y,1)
adj_rsq_train_poly_1 = 0.8326

adj_rsq_test_poly_1  = 0.7305

rmse_train_poly_1    = 0.0577

rmse_test_poly_1     = 0.0667
poly_hesapla(X,y,2)
model_poly_summary_hesapla(X,y,2)
grafik_ciz(X,y,2)
adj_rsq_train_poly_2 = 0.8426

adj_rsq_test_poly_2  = 0.5556

rmse_train_poly_2    = 0.0539

rmse_test_poly_2     = 0.0713
poly_hesapla(X,y,3)
model_poly_summary_hesapla(X,y,3)
grafik_ciz(X,y,3)
adj_rsq_train_poly_3 = 0.8468

adj_rsq_test_poly_3  = 3.4978

rmse_train_poly_3    = 0.0467

rmse_test_poly_3     = 0.0975
X = df [["CGPA","University Rating","SOP","LOR","Research","GRE Score","TOEFL Score"]]

y = df ["Chance of Admit"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=465)
lambdalar = 10**np.linspace(10,-2,100)*0.5
from sklearn.linear_model import RidgeCV

ridge_cv = RidgeCV(alphas = lambdalar,

                   scoring ="neg_mean_squared_error",

                   normalize=True)
ridge_cv.fit(X_train,y_train)
ridge_cv.alpha_
from sklearn.linear_model import Ridge

ridge_tuned = Ridge(alpha=ridge_cv.alpha_,

                   normalize = True).fit(X_train,y_train)
y_preds_train_ridge = ridge_tuned.predict(X_train)

y_preds_test_ridge = ridge_tuned.predict(X_test)
plt.figure(figsize=(15,7))



baslik_font = {'family': 'arial','color': 'darkblue','weight': 'bold','size': 15 }

eksen_font =  {'family': 'arial','color': 'darkblue','weight': 'bold','size': 13 }





plt.subplot(1,2,1)

plt.scatter(y_test,y_preds_test_ridge,alpha=0.7,color="purple")

plt.scatter(y_train,y_preds_train_ridge,alpha=0.4,color="green")

plt.plot(y_test,y_test,color="blue")

plt.xlabel("Actual Values",fontdict=eksen_font)

plt.ylabel("Estimated Values (With Test Set Values)",fontdict=eksen_font)

plt.title("Chance of Admit : Actual and Estimated Values",fontdict=baslik_font)



plt.subplot(1,2,2)

artik_test_ridge  = y_preds_test_ridge  - y_test

artik_train_ridge = y_preds_train_ridge - y_train

plt.scatter(y_test,artik_test_ridge,alpha=0.7,color="purple")

plt.scatter(y_train,artik_train_ridge,alpha=0.4,color="green")

plt.xlabel("Estimated",fontdict=eksen_font)

plt.ylabel("Residual",fontdict=eksen_font)

plt.title("Residual and Estimation",fontdict=baslik_font)

plt.hlines(y = 0, xmin = 0, xmax = 1, color = "blue")

plt.xticks(rotation=90)

r2_train_ridge = r2_score(y_train,y_preds_train_ridge)

r2_test_ridge  = r2_score(y_test,y_preds_test_ridge)



plt.show()



print("Train Set Adjusted R-Squared             : {:.4f}".format(calc_adj_rsq(r2_train_ridge,X_train.shape[0],len(X_train.columns))))

print("Mean Absolute Error (MAE)                : {:.4f}".format(mean_absolute_error(y_train,ridge_tuned.predict(X_train))))

print("Mean Square Error (MSE)                  : {:.4f}".format(mse(y_train,ridge_tuned.predict(X_train))))

print("Root Mean Square Error (RMSE)            : {:.4f}".format(rmse(y_train,ridge_tuned.predict(X_train))))

print("Mean Absolute Percentage Error (MAPE)    : {:.4f}".format(np.mean(np.abs((y_train - ridge_tuned.predict(X_train)) / y_train)) * 100),"\n")

print("Test Set Adjusted R-Squared              : {:.4f}".format(calc_adj_rsq(r2_test_ridge,X_test.shape[0],len(X_test.columns))))

print("Mean Absolute Error(MAE)                 : {:.4f}".format(mean_absolute_error(y_test,ridge_tuned.predict(X_test))))

print("Mean Square Error (MSE)                  : {:.4f}".format(mse(y_test,ridge_tuned.predict(X_test))))

print("Root Mean Square Error (RMSE)            : {:.4f}".format(rmse(y_test,ridge_tuned.predict(X_test))))

print("Mean Absolute Percentage Error (MAPE)    : {:.4f}".format(np.mean(np.abs((y_test -ridge_tuned.predict(X_test)) / y_test)) * 100))

    

r2_train_ridge = r2_score(y_train,y_preds_train_ridge)

r2_test_ridge  = r2_score(y_test,y_preds_test_ridge)

adj_rsq_train_ridge = calc_adj_rsq(r2_train_ridge,X_train.shape[0],len(X_train.columns))

adj_rsq_test_ridge  = calc_adj_rsq(r2_test_ridge,X_test.shape[0],len(X_test.columns))

rmse_train_ridge = rmse(y_train,ridge_tuned.predict(X_train))

rmse_test_ridge  = rmse(y_test,ridge_tuned.predict(X_test))
from sklearn.linear_model import Lasso

from sklearn.linear_model import LassoCV
lasso_cv_model = LassoCV(alphas=None,

                         cv =10,

                         max_iter = 10000,normalize=True)
lasso_cv_model.fit(X_train,y_train)
lasso_cv_model.alpha_
lasso_tuned = Lasso(alpha = lasso_cv_model.alpha_)
lasso_tuned.fit(X_train,y_train)

y_preds_train_lasso = lasso_tuned.predict(X_train)

y_preds_test_lasso = lasso_tuned.predict(X_test)
plt.figure(figsize=(15,7))



baslik_font = {'family': 'arial','color': 'darkblue','weight': 'bold','size': 15 }

eksen_font =  {'family': 'arial','color': 'darkblue','weight': 'bold','size': 13 }





plt.subplot(1,2,1)

plt.scatter(y_test,y_preds_test_lasso,alpha=0.7,color="purple")

plt.scatter(y_train,y_preds_train_lasso,alpha=0.4,color="green")

plt.plot(y_test,y_test,color="blue")

plt.xlabel("Actual Values",fontdict=eksen_font)

plt.ylabel("Estimated Values (With Test Set Values)",fontdict=eksen_font)

plt.title("Chance of Admit : Actual and Estimated Values",fontdict=baslik_font)



plt.subplot(1,2,2)

artik_test_lasso  = y_preds_test_lasso  - y_test

artik_train_lasso = y_preds_train_lasso - y_train

plt.scatter(y_test,artik_test_lasso,alpha=0.7,color="purple")

plt.scatter(y_train,artik_train_lasso,alpha=0.4,color="green")

plt.xlabel("Estimated",fontdict=eksen_font)

plt.ylabel("Residual",fontdict=eksen_font)

plt.title("Residual and Estimation",fontdict=baslik_font)

plt.hlines(y = 0, xmin = 0, xmax = 1, color = "blue")

plt.xticks(rotation=90)

plt.show()

r2_train_lasso = r2_score(y_train,y_preds_train_lasso)

r2_test_lasso  = r2_score(y_test,y_preds_test_lasso)





print("Train Set Adjusted R-Squared            : {:.4f}".format(calc_adj_rsq(r2_train_lasso,X_train.shape[0],len(X_train.columns))))

print("Mean Absolute Error (MAE)               : {:.4f}".format(mean_absolute_error(y_train, y_preds_train_lasso)))

print("Mean Square Error (MSE)                 : {:.4f}".format(mse(y_train, y_preds_train_lasso)))

print("Root Mean Square Error (RMSE)           : {:.4f}".format(rmse(y_train, y_preds_train_lasso)))

print("Mean Absolute Percentage Error (MAPE)   : {:.4f}".format(np.mean(np.abs((y_train - y_preds_train_lasso) / y_train)) * 100),"\n")

print("Test Set Adjusted R-Squared             : {:.4f}".format(calc_adj_rsq(r2_test_lasso,X_test.shape[0],len(X_test.columns))))

print("Mean Absolute Error(MAE)                : {:.4f}".format(mean_absolute_error(y_test, y_preds_test_lasso)))

print("Mean Square Error (MSE)                 : {:.4f}".format(mse(y_test, y_preds_test_lasso)))

print("Root Mean Square Error (RMSE)           : {:.4f}".format(rmse(y_test, y_preds_test_lasso)))

print("Mean Absolute Percentage Error (MAPE)   : {:.4f}".format(np.mean(np.abs((y_test - y_preds_test_lasso) / y_test)) * 100))

r2_train_lasso = r2_score(y_train,y_preds_train_lasso)

r2_test_lasso  = r2_score(y_test,y_preds_test_lasso)

adj_rsq_train_lasso = calc_adj_rsq(r2_train_lasso,X_train.shape[0],len(X_train.columns))

adj_rsq_test_lasso  = calc_adj_rsq(r2_test_lasso,X_test.shape[0],len(X_test.columns))

rmse_train_lasso = rmse(y_train, y_preds_train_lasso)

rmse_test_lasso  = rmse(y_test, y_preds_test_lasso)
from sklearn.linear_model import ElasticNet,ElasticNetCV
enet_cv_model = ElasticNetCV(cv = 10,random_state=0).fit(X_train,y_train)
enet_cv_model.alpha_
enet_tuned = ElasticNet(alpha = enet_cv_model.alpha_).fit(X_train,y_train)
y_preds_train_elastic = enet_tuned.predict(X_train)

y_preds_test_elastic  = enet_tuned.predict(X_test)
plt.figure(figsize=(15,7))



baslik_font = {'family': 'arial','color': 'darkblue','weight': 'bold','size': 15 }

eksen_font =  {'family': 'arial','color': 'darkblue','weight': 'bold','size': 13 }





plt.subplot(1,2,1)

plt.scatter(y_test,y_preds_test_elastic,alpha=0.7,color="purple")

plt.scatter(y_train,y_preds_train_elastic,alpha=0.4,color="green")

plt.plot(y_test,y_test,color="blue")

plt.xlabel("Actual Values",fontdict=eksen_font)

plt.ylabel("Estimated Values (With Test Set Values)",fontdict=eksen_font)

plt.title("Chance of Admit : Actual and Estimated Values",fontdict=baslik_font)



plt.subplot(1,2,2)

artik_test_elastic  = y_preds_test_elastic  - y_test

artik_train_elastic = y_preds_train_elastic - y_train

plt.scatter(y_test,artik_test_elastic,alpha=0.7,color="purple")

plt.scatter(y_train,artik_train_elastic,alpha=0.4,color="green")

plt.xlabel("Estimated",fontdict=eksen_font)

plt.ylabel("Residual",fontdict=eksen_font)

plt.title("Residual and Estimation",fontdict=baslik_font)

plt.hlines(y = 0, xmin = 0, xmax = 1, color = "blue")

plt.xticks(rotation=90)

r2_train_elastic = r2_score(y_train,y_preds_train_elastic)

r2_test_elastic  = r2_score(y_test,y_preds_test_elastic)



plt.show()



print("Train Set Adjusted R-Squared            : {:.4f}".format(calc_adj_rsq(r2_train_elastic,X_train.shape[0],len(X_train.columns))))

print("Mean Absolute Error (MAE)               : {:.4f}".format(mean_absolute_error(y_train, y_preds_train_elastic)))

print("Mean Square Error (MSE)                 : {:.4f}".format(mse(y_train, y_preds_train_elastic)))

print("Root Mean Square Error (RMSE)           : {:.4f}".format(rmse(y_train, y_preds_train_elastic)))

print("Mean Absolute Percentage Error (MAPE)   : {:.4f}".format(np.mean(np.abs((y_train - y_preds_train_elastic) / y_train)) * 100),"\n")

print("Test Set Adjusted R-Squared             : {:.4f}".format(calc_adj_rsq(r2_test_elastic,X_test.shape[0],len(X_test.columns))))

print("Mean Absolute Error (MAE)               : {:.4f}".format(mean_absolute_error(y_test, y_preds_test_elastic)))

print("Mean Square Error (MSE)                 : {:.4f}".format(mse(y_test, y_preds_test_elastic)))

print("Root Mean Square Error (RMSE)           : {:.4f}".format(rmse(y_test, y_preds_test_elastic)))

print("Mean Absolute Percentage Error (MAPE)   : {:.4f}".format(np.mean(np.abs((y_test - y_preds_test_elastic) / y_test)) * 100))



r2_train_elastic = r2_score(y_train,y_preds_train_elastic)

r2_test_elastic  = r2_score(y_test,y_preds_test_elastic)

adj_rsq_train_elastic = calc_adj_rsq(r2_train_elastic,X_train.shape[0],len(X_train.columns))

adj_rsq_test_elastic  = calc_adj_rsq(r2_test_elastic,X_test.shape[0],len(X_test.columns))

rmse_train_elastic    = rmse(y_train, y_preds_train_elastic)

rmse_test_elastic     = rmse(y_test, y_preds_test_elastic)
result = pd.DataFrame(columns=["Models","Train_Adjusted_R_Sq","Test_Adjusted_R_Sq","Train_Set_RMSE","Test_Set_RMSE"])

result["Models"] = ["Model 1","Model 2","Model Dummy SOP","Model Dummy UNİ","Model Dummy TOP ","Poly 1","Poly 2","Poly 3",

                    "Model Ridge","Model Lasso","Model Elasticnet"] 

result["Train_Adjusted_R_Sq"] = [adj_rsq_train_one,adj_rsq_train_two,adj_rsq_train_sop,adj_rsq_train_uni,adj_rsq_dum_top,

                                adj_rsq_train_poly_1,adj_rsq_train_poly_2,adj_rsq_train_poly_3,adj_rsq_train_ridge,

                                adj_rsq_train_lasso,adj_rsq_train_elastic]

result["Test_Adjusted_R_Sq"] = [adj_rsq_test_one,adj_rsq_test_two,adj_rsq_test_sop,adj_rsq_test_uni,adj_rsq_dum_top,

                                adj_rsq_test_poly_1,adj_rsq_test_poly_2,adj_rsq_test_poly_3,adj_rsq_test_ridge,

                                adj_rsq_test_lasso,adj_rsq_test_elastic]

result["Train_Set_RMSE"] = [rmse_train_one,rmse_train_two,rmse_train_sop,rmse_train_uni,rmse_train_dum_top,

                           rmse_train_poly_1,rmse_train_poly_2,rmse_train_poly_3,rmse_train_ridge,

                           rmse_train_lasso,rmse_train_elastic]

result["Test_Set_RMSE"]  = [rmse_test_one,rmse_test_two,rmse_train_sop,rmse_test_uni,rmse_test_dum_top,

                           rmse_test_poly_1,rmse_test_poly_2,rmse_test_poly_3,rmse_test_ridge,

                           rmse_test_lasso,rmse_test_elastic]
result
model_labels = ["Model 1","Model 2","Model Dummy SOP","Model Dummy UNİ","Model Dummy TOP","Model Poly 1",

                "Model Poly 2","Model Poly 3","Model Ridge","Model Lasso","Model Elasticnet"]



model_adjusted_train =[adj_rsq_train_one,adj_rsq_train_two,adj_rsq_train_sop,adj_rsq_train_uni,adj_rsq_dum_top,

                       adj_rsq_train_poly_1,adj_rsq_train_poly_2,adj_rsq_train_poly_3,adj_rsq_train_ridge,

                       adj_rsq_train_lasso,adj_rsq_train_elastic]



model_adjusted_test = [adj_rsq_test_one,adj_rsq_test_two,adj_rsq_test_sop,adj_rsq_test_uni,adj_rsq_dum_top,

                       adj_rsq_test_poly_1,adj_rsq_test_poly_2,adj_rsq_test_poly_3,adj_rsq_test_ridge,

                       adj_rsq_test_lasso,adj_rsq_test_elastic]



model_rmse_train =[rmse_train_one,rmse_train_two,rmse_train_sop,rmse_train_uni,rmse_train_dum_top,

                   rmse_train_poly_1,rmse_train_poly_2,rmse_train_poly_3,rmse_train_ridge,

                   rmse_train_lasso,rmse_train_elastic]



model_rmse_test =[rmse_test_one,rmse_test_two,rmse_train_sop,rmse_test_uni,rmse_test_dum_top,

                  rmse_test_poly_1,rmse_test_poly_2,rmse_test_poly_3,rmse_test_ridge,

                  rmse_test_lasso,rmse_test_elastic]





plt.figure(figsize = (20,20))

plt.subplot(2,1,1)

n_groups = 11

index = np.arange(n_groups)

bar_width = 0.3

opacity = 0.7

 

rects1 = plt.bar(index, model_rmse_train, bar_width,

alpha=opacity,

color='green',

label='Train Set RMSE.')

 

rects2 = plt.bar(index + bar_width, model_rmse_test, bar_width,

alpha=opacity,

color='purple',

label='Test Set RMSE.')

 

plt.xlabel('Models',fontsize =18)

plt.ylabel('RMSE Values',fontsize =18)

plt.title('Train and Test RMSEs',fontsize =20)

plt.xticks(index + bar_width/2, ("Model 1","Model 2","Model Dummy SOP","Model Dummy UNİ","Model Dummy TOP","Model Poly 1",

                "Model Poly 2","Model Poly 3","Model Ridge","Model Lasso","Model Elasticnet"),rotation=90,fontsize=15)

plt.yticks(fontsize=15)

plt.legend(fontsize='xx-large')



plt.subplot(2,1,2)

rects1 = plt.bar(index, model_adjusted_train, bar_width,

alpha=opacity,

color='green',

label='Train Set Adjusted R Sq.')

 

rects2 = plt.bar(index + bar_width, model_adjusted_test, bar_width,

alpha=opacity,

color='purple',

label='Test Set Adjusted R Sq.')

 

plt.xlabel('Models',fontsize =18)

plt.ylabel('Adjusted R Squareds',fontsize =18)

plt.title('Train and Test Adjusted Rs',fontsize =20)

plt.xticks(index + bar_width/2, ("Model 1","Model 2","Model Dummy SOP","Model Dummy UNİ","Model Dummy TOP","Model Poly 1",

                "Model Poly 2","Model Poly 3","Model Ridge","Model Lasso","Model Elasticnet"),rotation=90,fontsize=15)

plt.yticks(fontsize=15)

plt.legend(fontsize='xx-large')



plt.tight_layout()

plt.show()
