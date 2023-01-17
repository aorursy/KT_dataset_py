import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
admsn = pd.read_csv("../input/graduate-admissions/Admission_Predict.csv")
admsn.head()
admsn = admsn.set_index('Serial No.')
admsn.head()
admsn.describe()
full_score_gre = admsn[admsn["GRE Score"] == 340].count()["SOP"]
full_score_toefl = admsn[admsn['TOEFL Score'] == 120].count()["SOP"]
full_score_both = admsn[(admsn["GRE Score"] == 340) & (admsn['TOEFL Score'] == 120)].count()["SOP"]
print("Number of students who scored full markes in GRE :",full_score_gre)
print("Number of students who scored full markes in TOEFL :",full_score_toefl)
print("Number of students who scored full markes in both GRE and TOEFL :",full_score_both)
sns.distplot(admsn['GRE Score'])
sns.distplot(admsn["TOEFL Score"])
sns.distplot(admsn['CGPA'])
sns.jointplot(x = 'GRE Score', y = 'TOEFL Score',data = admsn)
sns.heatmap(admsn.corr(), annot=True,fmt=".2f",annot_kws={'size':16},cbar=False)
sns.pairplot(admsn)
sns.pairplot(admsn,hue = 'Research')
g = sns.FacetGrid(admsn,col ='University Rating')
g = g.map(sns.distplot,'Chance of Admit ')
from sklearn.model_selection import train_test_split
X = admsn.drop('Chance of Admit ',axis = 1)
Y = admsn['Chance of Admit ']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.4,random_state = 101)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,Y_train)
predictions = lm.predict(X_test)
plt.scatter(predictions,Y_test)
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df
sns.distplot(Y_test - predictions)
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(Y_test, predictions))
print('MSE:', metrics.mean_squared_error(Y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(Y_test, predictions)))
data = {"Serial No." : [1,2,3],"GRE Score" : [340,317,290],"TOEFL Score" : [120,107,90],'University Rating' : [3,3,3],'SOP' : [5,3,1],'LOR':[5,3,1],'CGPA':[10,8.5,6.8],"Research" : [1,1,0]}
testcase = pd.DataFrame(data)
testcase = testcase.set_index('Serial No.')
testcase
pred = lm.predict(testcase)*100
for i in range(3):
    print("Prediction for candidate "+str(i+1)+ ":",str(int(pred[i]))+"%")