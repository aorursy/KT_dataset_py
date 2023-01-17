import seaborn as sns

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import statsmodels.api as sm

from statsmodels.stats.diagnostic import het_breuschpagan

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score
fish = pd.read_csv("../input/fish-market/Fish.csv")
fish.head(5)
fish.info()
fish.columns = ['Species', 'Weight', 'LengthV', 'LengthD', 'LengthC', 'Height','Width']
fish.describe().round(1)
fish.iloc[40,:]
fish = fish.loc[fish["Weight"]>0,:]
sns.set()

plt.figure(figsize=[12,6])



sns.distplot(fish["Weight"],kde_kws={"bw":40})

plt.show()
sns.pairplot(fish,hue="Species")

plt.show()
sns.lmplot(x="Height", y="Weight", hue="Species", data=fish, fit_reg=False)

sns.regplot(x="Height", y="Weight", data=fish, scatter=False)

plt.gcf().set_size_inches(14, 8)

plt.show()
sns.lmplot(x="Width", y="Weight", hue="Species", data=fish, fit_reg=False)

sns.regplot(x="Width", y="Weight", data=fish, scatter=False)

plt.gcf().set_size_inches(14, 8)

plt.show()
sns.lmplot(x="LengthV", y="Weight", hue="Species", data=fish, fit_reg=False)

sns.regplot(x="LengthV", y="Weight", data=fish, scatter=False)

plt.gcf().set_size_inches(14, 8)

plt.show()
fish["Weightlog"] = np.log(fish["Weight"])

fish["Widthlog"] = np.log(fish["Width"])

fish["Heightlog"] = np.log(fish["Height"])

fish["LengthVlog"] = np.log(fish["LengthV"])
sns.lmplot(x="Heightlog", y="Weightlog", hue="Species", data=fish, fit_reg=False)

sns.regplot(x="Heightlog", y="Weightlog", data=fish, scatter=False)

plt.gcf().set_size_inches(14, 8)

plt.show()
sns.lmplot(x="Widthlog", y="Weightlog", hue="Species", data=fish, fit_reg=False)

sns.regplot(x="Widthlog", y="Weightlog", data=fish, scatter=False)

plt.gcf().set_size_inches(14, 8)

plt.show()
sns.lmplot(x="LengthVlog", y="Weightlog", hue="Species", data=fish, fit_reg=False)

sns.regplot(x="LengthVlog", y="Weightlog", data=fish, scatter=False,ci=None)

plt.gcf().set_size_inches(14, 8)

plt.show()
fish_x = fish.loc[:,["LengthV","LengthD","LengthC","Height","Width"]]



fish_x.corr()
y = fish["Weight"]

x = fish.loc[:,["LengthV","LengthD","LengthC","Height","Width"]]



x = sm.add_constant(x)

fish_reg = sm.OLS(y,x).fit()

fish_reg.summary()  
y = fish["Weight"]

x = fish.loc[:,["LengthV","Height","Width"]]



x = sm.add_constant(x)

fish_reg_red = sm.OLS(y,x).fit()

fish_reg_red.summary()  
print( "P-Value of the F-Test:\t\t\t",round(fish_reg.compare_f_test(fish_reg_red)[1],4),"\n")





print("AIC for the non-Restricted model:\t",round(fish_reg.aic,4))

print("AIC for the non-Restricted model:\t",round(fish_reg_red.aic,4),"\n")



print("BIC for the non-Restricted model:\t",round(fish_reg.bic,4))

print("BIC for the non-Restricted model:\t",round(fish_reg_red.bic,4))
plt.figure(figsize=(16,10))

plt.hlines(0,xmin=0,xmax=1700,linestyle="dashed",alpha=0.6)

sns.scatterplot(y,fish_reg_red.resid,s=100)



plt.show()
y_log = fish["Weightlog"]

x_log = fish.loc[:,["LengthVlog","Heightlog","Widthlog"]]



x_log = sm.add_constant(x_log)

fish_reg_ll = sm.OLS(y_log,x_log).fit()

fish_reg_ll.summary()  
plt.figure(figsize=(16,10))

plt.hlines(0,xmin=1.5,xmax=8,linestyle="dashed",alpha=0.6)

sns.scatterplot(y_log,fish_reg_ll.resid,s=100)



plt.show()
print("P-Value of Breusch & Pagan Test: ",round(het_breuschpagan(fish_reg_ll.resid,x_log)[1],4))
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=5)



X_log_train, X_log_test, y_log_train, y_log_test = train_test_split(x_log, y_log, test_size=0.3,random_state=5)
X_train = sm.add_constant(X_train)

fish_reg_red = sm.OLS(y_train,X_train).fit()



X_log_train = sm.add_constant(X_log_train)

fish_reg_ll = sm.OLS(y_log_train,X_log_train).fit()
pred_lin = fish_reg_red.predict(X_test)

print("R2 of the Non-Transformed Linear Model:",round(r2_score(y_test,pred_lin),4))



pred_ll = np.exp(fish_reg_ll.predict(X_log_test))

print("R2 of the LogLog Linear Model:",round(r2_score(y_test,pred_ll),4))


unit =np.r_[1:49]





fig, ax = plt.subplots()

sns.scatterplot(unit,y_test,color="red",s=250)

sns.scatterplot(unit,pred_ll,color="blue",s=150)

sns.scatterplot(unit,pred_lin,color="green",s=150)

ax.set_xticks(range(1,49))

plt.gcf().set_size_inches(20, 10)

plt.show()
pd.DataFrame({"Pred_Lin":round(pred_lin[pred_lin<0],2),"Pred_LogLog":round(pred_ll[pred_lin<0],2),"Real_Values":y_test[pred_lin<0]})