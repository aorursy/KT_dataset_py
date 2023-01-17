# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
data = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')

data.head()
columns = data.columns

data[columns[14]] = data[columns[14]].fillna(0)

val = [1,3,5,6,8,9,11,13]

columns
for i in range (0,len(val)):

    k = (val[i])

    col_name = columns[k]

    #print(col_name)

    k2 = data[col_name].value_counts()

    print("--")

    print(k2)

    print("####################")
sns.countplot("gender",data = data,hue="status")
print("Total Boys = ", len(data[data['gender'] == 'M'])," , and Boys placed = ",len(data[(data["gender"] =="M") & (data["status"] =="Placed")]))

print("total Girls = " ,len(data[data["gender"] == "F"]) ,"and girls placed = ", len(data[(data["gender"] == "F") & (data["status"] == "Placed")]))

print("Boys percentage",len(data[(data["gender"] =="M") & (data["status"] == "Placed")]) / len(data[data["gender"] == "M"])  * 100)   

print("Girls Percentage", len(data[(data["gender"] == "F")&( data["status"] =="Placed")])/ len(data[data["gender"] =="F"]) * 100)     



print("ssc Central Board Placement Count",len(data[(data["status"] =="Placed")&(data["ssc_b"] == "Central")]))

print("Placed Central school percentage",len(data[(data["status"] =="Placed")&(data["ssc_b"] == "Central")])        /  len(data[data["status"] =="Placed"]) * 100)

print("ssc other Board Placement Count",len(data[(data["status"] =="Placed")&(data["ssc_b"] != "Central")]))

print("Other school placed percentage", len(data[(data["status"] == "Placed")&( data["ssc_b"] !="Central")])    /  len(data[data["status"] == "Placed"]) * 100)



print("Total Central Student",len(data[data["hsc_b"] =="Central"]))

print("ssc Central Board Placement Count",len(data[(data["status"] =="Placed")&(data["hsc_b"] == "Central")]))

print("Placed Central school percentage",len(data[(data["status"] =="Placed")&(data["hsc_b"] == "Central")])/len(data[data["status"] =="Placed"]) * 100)

print("Total Other  Student",len(data[data["hsc_b"] !="Central"]))

print("ssc other Board Placement Count",len(data[(data["status"] =="Placed")&(data["hsc_b"] != "Central")]))

print("Other school placed percentage", len(data[(data["status"] == "Placed")&( data["hsc_b"] !="Central")])    /  len(data[data["status"] == "Placed"]) * 100)

print("Total Arts Students in Central is",len(data[(data['hsc_b']=="Central") & (data["hsc_s"] =="Arts")]))

print("Total Arts Students in Central and placed is",len(data[(data['hsc_b']=="Central") & (data["hsc_s"] =="Arts") & (data["status"] == "Placed")]))





print("Total Science Students in Central is",len(data[(data['hsc_b']=="Central") & (data["hsc_s"] =="Science")]))

print("Total Science Students in Central and placed is",len(data[(data['hsc_b']=="Central") & (data["hsc_s"] =="Science") & (data["status"] == "Placed")]))





print("Total Commerce Students in Central is",len(data[(data['hsc_b']=="Central") & (data["hsc_s"] =="Commerce")]))

print("Total Commerce Students in Central and placed is",len(data[(data['hsc_b']=="Central") & (data["hsc_s"] =="Commerce") & (data["status"] == "Placed")]))





print("========" *5)



print("Total Arts Students in Other is",len(data[(data['hsc_b']!="Central") & (data["hsc_s"] =="Arts")]))

print("Total Arts Students in Other and placed is",len(data[(data['hsc_b']!="Central") & (data["hsc_s"] =="Arts") & (data["status"] == "Placed")]))





print("Total Science Students in Other is",len(data[(data['hsc_b']!="Central") & (data["hsc_s"] =="Science")]))

print("Total Science Students in Other and placed is",len(data[(data['hsc_b']!="Central") & (data["hsc_s"] =="Science") & (data["status"] == "Placed")]))





print("Total Commerce Students in Other is",len(data[(data['hsc_b']!="Central") & (data["hsc_s"] =="Commerce")]))

print("Total Commerce Students in Other and placed is",len(data[(data['hsc_b']!="Central") & (data["hsc_s"] =="Commerce") & (data["status"] == "Placed")]))

print("Students who had Central board in 10th and 12th is , ",len(data[(data["ssc_b"] =="Central")&(data["hsc_b"] =="Central")]))

print("Students who had Central board in10th/12th and placed , ",len(data[(data["ssc_b"] =="Central")&(data["hsc_b"] =="Central")&(data['status'] =="Placed")]))

print("Students who had Cental in 10th and Other in 12th is , ",len(data[(data["ssc_b"] =="Central")&(data["hsc_b"] =="Others")]))

print("Students who changed collage from central to other and placed , ",len(data[(data["ssc_b"] =="Central")&(data["hsc_b"] =="Others")&(data["status"] == "Placed")]))



print("Students who had Others board in 10th and 12th is , ",len(data[(data["ssc_b"] =="Others")&(data["hsc_b"] =="Others")]))

print("Students who had Others board in 10th and 12th and placed =  ",len(data[(data["ssc_b"] =="Others")&(data["hsc_b"] =="Others") &(data["status"] =="Placed")]))



print("Students who had Others in 10th and Central in 12th is , ",len(data[(data["ssc_b"] =="Others")&(data["hsc_b"] =="Central")]))

print("Students who moved from Others to Central and placed , ",len(data[(data["ssc_b"] =="Others")&(data["hsc_b"] =="Central") &(data["status"] =="Placed")]))
print("student count",len(data))

print("Placed student count",len(data[data["status"] == "Placed"]))

print("Total student with 60% in HSC",len(data[(data["hsc_p"] >= 60)]))

print("Total student with below 60% in HSC",len(data[(data["hsc_p"] < 60)]))

print("Student with above and equal to 60 in SSC and placed",len(data[(data["ssc_p"]>=60) & (data["status"] == "Placed")]))

print("Student with below 60% in SSC and placed is",len(data[(data["ssc_p"]<60) & (data["status"] == "Placed")]))

print("Student with above and equal to 60 in SSC and not placed",len(data[(data["ssc_p"]>=60) & (data["status"] != "Placed")]))

print("Student with below 60% in SSC and not placed is",len(data[(data["ssc_p"]<=60) & (data["status"] != "Placed")]))





print("==============" *4)

print("student count",len(data))

print("Placed student count",len(data[data["status"] == "Placed"]))

#print("Total Placed",len(data[data["status"] == "Placed"]))

print("Student with above and equal to 60 in HSC and placed",len(data[(data["hsc_p"]>=60) & (data["status"] == "Placed")]))

print("Student with below 60% in HSC and placed is",len(data[(data["hsc_p"]<=60) & (data["status"] == "Placed")]))

print("Student with above and equal to 60 in HSC and not placed",len(data[(data["hsc_p"]>=60) & (data["status"] != "Placed")]))

print("Student with below 60% in HSC and not placed is",len(data[(data["hsc_p"]<=60) & (data["status"] != "Placed")]))





print("==============" *4)

print("student count",len(data))

print("Placed student count",len(data[data["status"] == "Placed"]))

print("Total student with 60% in Degree",len(data[(data["degree_p"] >= 60)]))

print("Total student with below 60% in Degree",len(data[(data["degree_p"] < 60)]))

print("Student with above and equal to 60 in Degree and placed",len(data[(data["degree_p"]>=60) & (data["status"] == "Placed")]))

print("Student with below 60% in Degree and placed is",len(data[(data["degree_p"]<60) & (data["status"] == "Placed")]))

print("Student with above and equal to 60 in Degree and not placed",len(data[(data["degree_p"]>=60) & (data["status"] != "Placed")]))

print("Student with below 60% in Degree and not placed is",len(data[(data["degree_p"]<=60) & (data["status"] != "Placed")]))







print("==============" *4)

print("student count",len(data))

print("Placed student count",len(data[data["status"] == "Placed"]))

print("Total student with 60% in etest_p",len(data[(data["etest_p"] >= 60)]))

print("Total student with below 60% in etest_p",len(data[(data["etest_p"] < 60)]))

print("Student with above and equal to 60 in etest_p and placed",len(data[(data["etest_p"]>=60) & (data["status"] == "Placed")]))

print("Student with below 60% in etest_p and placed is",len(data[(data["etest_p"]<60) & (data["status"] == "Placed")]))

print("Student with above and equal to 60 in etest_p and not placed",len(data[(data["etest_p"]>=60) & (data["status"] != "Placed")]))

print("Student with below 60% in etest_p and not placed is",len(data[(data["etest_p"]<=60) & (data["status"] != "Placed")]))









print("==============" *4)

print("student count",len(data))

print("Placed student count",len(data[data["status"] == "Placed"]))

print("Total student with 60% in MBA",len(data[(data["mba_p"] >= 60)]))

print("Total student with below 60% in MBA",len(data[(data["mba_p"] < 60)]))

print("Student with above and equal to 60 in MBA and placed",len(data[(data["mba_p"]>=60) & (data["status"] == "Placed")]))

print("Student with below 60% in MBA and placed is",len(data[(data["mba_p"]<60) & (data["status"] == "Placed")]))

print("Student with above and equal to 60 in MBA and not placed",len(data[(data["mba_p"]>=60) & (data["status"] != "Placed")]))

print("Student with below 60% in MBA and not placed is",len(data[(data["mba_p"]<=60) & (data["status"] != "Placed")]))





print("student with above 60 in all exams and placed",len(data[(data["ssc_p"] >=60) & (data["hsc_p"] >=60) & (data["etest_p"] >=60) & (data["degree_p"] >=60)&(data["mba_p"]>=60) & (data["status"] =="Placed")]))



print("student with above 60 in all exams and not placed",len(data[(data["ssc_p"] >=60) & (data["hsc_p"] >=60) & (data["etest_p"] >=60) & (data["degree_p"] >=60)&(data["mba_p"]>=60) & (data["status"] !="Placed")]))



print("student with above 60% till Etest and above 75 in MBA and not placed",len(data[(data["ssc_p"] >=60) & (data["hsc_p"] >=60) & (data["etest_p"] >=60) & (data["degree_p"] >=60)&(data["mba_p"]>=75) & (data["status"] !="Placed")]))
#MBA percentage above 75  and minimum 60% till Degree is enough for placement
print("Mkt& HR Students count",len(data[data["specialisation"] == "Mkt&HR"]))

print("Mkt & HR placed count =",len(data[(data["specialisation"] == "Mkt&HR") & (data["status"] == "Placed")]))

print("Mkt&Hr students placement % = ",len(data[(data["specialisation"] == "Mkt&HR") & (data["status"] == "Placed")]) *100 / len(data[data["specialisation"] == "Mkt&HR"]) )



print("======="*5)



print("Mkt& Fin Students count",len(data[data["specialisation"] == "Mkt&Fin"]))

print("Mkt&Fin placed count =",len(data[(data["specialisation"] == "Mkt&Fin") & (data["status"] == "Placed")]))

print("Mkt&Fin students placement % = ",len(data[(data["specialisation"] == "Mkt&Fin") & (data["status"] == "Placed")]) *100 / len(data[data["specialisation"] == "Mkt&Fin"]) )
print("Students counts with work experience=",len(data[data["workex"] == "Yes"]))

print("Students with work exp and placed = ",len(data[(data["workex"]== "Yes") &(data["status"] == "Placed")] ))

print("Students with work exp and not placed = ",len(data[(data["workex"]== "Yes") &(data["status"] != "Placed")] ))

print("Students with no work exp and placed = ",len(data[(data["workex"]!= "Yes") &(data["status"] == "Placed")] ))

print("Students with no work exp and not placed = ",len(data[(data["workex"]!= "Yes") &(data["status"] != "Placed")] ))
data = data.drop("sl_no",axis = 1)
from scipy import stats

def cramers_stat(confusion_matrix):

    chi2 = stats.chi2_contingency(confusion_matrix)[0]

    n = confusion_matrix.sum()

    return np.sqrt(chi2 / (n*(min(confusion_matrix.shape)-1)))
cat_data = data.drop('status',axis=1).select_dtypes(include = object).columns
for i in cat_data:

    ctest = pd.crosstab(data[i],data['status'])

    print(ctest)

    print([ctest.iloc[0].values,ctest.iloc[1].values])
for i in cat_data:

    ctest = pd.crosstab(data[i],data['status'])

    (chi2,p,dof,_) = stats.chi2_contingency([ctest.iloc[0].values,ctest.iloc[1].values])

    print("Value for Target=",i)

    print("Valuse for chi = ",chi2)

    print(" 'P' Value     =  ",p)

    print("Value for Freedom is", dof)

    print("==================="*3)
#Value for Worex and Specilisation having impact over status. 


plt.figure(figsize = (8,8))

sns.heatmap(data.corr(method = 'spearman'),annot = True)

plt.show()
sns.pairplot(data,diag_kind = 'kde',hue = 'status')

plt.show()
num_data = data.select_dtypes(exclude = object).columns
type(num_data)
for i in num_data:

    if i != 'salary':

        sns.scatterplot('salary',i,data = data)

        plt.show()
data_copy = data.copy()

data.info()
data_fet = data.apply(LabelEncoder().fit_transform)

data_fet.head(5)

data_fet['degree_t'].value_counts()
data['degree_t'].value_counts()
X = data_fet.drop("status",axis = 1)

y = data_fet['status']
rnd = RandomForestClassifier().fit(X,y)
rnd
feature_importances = pd.DataFrame((rnd.feature_importances_*100),index = X.columns,columns=['importance']).sort_values('importance',ascending=False)
feature_importances
plt.figure(figsize=(15,8))

plt.plot(feature_importances)

plt.title("Importance Graph")

plt.show()
used_col = feature_importances[feature_importances["importance"] <=1.5].index
data_imp = data_fet.drop(used_col,axis=1)
data_imp
X_new = data_imp.drop('status',axis = 1)

y_new = data_imp["status"]
x_train,x_test,y_train,y_test = train_test_split(X_new,y_new,test_size = 0.3,random_state = 123)
from sklearn.tree import DecisionTreeClassifier

tree_model = DecisionTreeClassifier().fit(x_train,y_train)

treepred = tree_model.predict(x_test)
from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve,f1_score
print(confusion_matrix(y_test,treepred))

print("Accuracy = ", accuracy_score(y_test,treepred))

print("f1 Score",f1_score(y_test,treepred))