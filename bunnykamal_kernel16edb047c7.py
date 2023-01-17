import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

from pylab import rcParams

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,BaggingClassifier,ExtraTreesClassifier,GradientBoostingClassifier

from sklearn.tree import DecisionTreeClassifier
bank = pd.read_excel("/kaggle/input/bank-telemarketing-analysis/bank-full.xls")

bank.head()
bank.isnull().sum()
bank.shape
bank.dtypes
bank.describe().transpose()
bank["poutcome"].value_counts()
bank["y"].value_counts()
bank["education"].value_counts()
sns.countplot(bank["y"])
sns.countplot(x="poutcome",data=bank,hue="y")
rcParams["figure.figsize"]=17,7

p = sns.countplot(x="age",hue="y",data=bank[bank["y"]=="yes"],palette="Set1")

p.set_xticklabels(p.get_xticklabels(),rotation=90,ha="right")

plt.title("count of target wrt age",size=15)

p
sns.countplot(x="job",hue="y",data=bank)
bank[bank["job"]=="student"]["y"].value_counts()
bank.head()
mar = bank.groupby(by="marital")

job = bank.groupby(by="job")

educ = bank.groupby(by="education")

default = bank.groupby(by="default")

house = bank.groupby(by="housing")

loan = bank.groupby(by="loan")

con = bank.groupby(by="contact")

mon = bank.groupby(by="month")

pout = bank.groupby(by="poutcome")
lis = []

for i in bank["job"].unique():

    b = job.get_group(i)

    lis.append(len(b[b["y"]=="yes"])/len(b))
subscribe_rate_job = pd.DataFrame({"job":bank["job"].unique(),"sub_rate":lis})
subscribe_rate_job.sort_values(by="sub_rate",ascending=False)
mar = bank.groupby(by="marital")
lis1 = []

for i in bank["marital"].unique():

    b = mar.get_group(i)

    lis1.append(len(b[b["y"]=="yes"])/len(b))
subscribe_rate_mar = pd.DataFrame({"marital":bank["marital"].unique(),"sub_rate":lis1})

subscribe_rate_mar.sort_values(by="sub_rate",ascending=False)
lis2 = []

for i in bank["education"].unique():

    b = educ.get_group(i)

    lis2.append(len(b[b["y"]=="yes"])/len(b))

subscribe_rate_educ = pd.DataFrame({"education":bank["education"].unique(),"sub_rate":lis2})

subscribe_rate_educ.sort_values(by="sub_rate",ascending=False)
lis3 = []

for i in bank["default"].unique():

    b = default.get_group(i)

    lis3.append(len(b[b["y"]=="yes"])/len(b))

subscribe_rate_def = pd.DataFrame({"default":bank["default"].unique(),"sub_rate":lis3})

subscribe_rate_def.sort_values(by="sub_rate",ascending=False)
lis4 = []

for i in bank["housing"].unique():

    b = house.get_group(i)

    lis4.append(len(b[b["y"]=="yes"])/len(b))

subscribe_rate_house = pd.DataFrame({"house":bank["housing"].unique(),"sub_rate":lis4})

subscribe_rate_house.sort_values(by="sub_rate",ascending=False)
lis5 = []

for i in bank["loan"].unique():

    b = loan.get_group(i)

    lis5.append(len(b[b["y"]=="yes"])/len(b))

subscribe_rate_loan = pd.DataFrame({"loan":bank["loan"].unique(),"sub_rate":lis5})

subscribe_rate_loan.sort_values(by="sub_rate",ascending=False)
lis5 = []

for i in bank["contact"].unique():

    b = con.get_group(i)

    lis5.append(len(b[b["y"]=="yes"])/len(b))

subscribe_rate_con = pd.DataFrame({"contact":bank["contact"].unique(),"sub_rate":lis5})

subscribe_rate_con.sort_values(by="sub_rate",ascending=False)
lis6 = []

for i in bank["month"].unique():

    b = mon.get_group(i)

    lis6.append(len(b[b["y"]=="yes"])/len(b))

subscribe_rate_mon = pd.DataFrame({"month":bank["month"].unique(),"sub_rate":lis6})

subscribe_rate_mon.sort_values(by="sub_rate",ascending=False)
lis7 = []

for i in bank["poutcome"].unique():

    b = pout.get_group(i)

    lis7.append(len(b[b["y"]=="yes"])/len(b))

subscribe_rate_pout = pd.DataFrame({"poutcome":bank["poutcome"].unique(),"sub_rate":lis7})

subscribe_rate_pout.sort_values(by="sub_rate",ascending=False)
#subscription rate is more persons 
rcParams["figure.figsize"]=5,5

subscribe_rate_job.plot.bar(x="job",y="sub_rate")
rcParams["figure.figsize"]=5,5

subscribe_rate_mar.plot.bar(x="marital",y="sub_rate")
rcParams["figure.figsize"]=5,5

subscribe_rate_educ.plot.bar(x="education",y="sub_rate")
rcParams["figure.figsize"]=5,5

subscribe_rate_def.plot.bar(x="default",y="sub_rate")
rcParams["figure.figsize"]=5,5

subscribe_rate_house.plot.bar(x="house",y="sub_rate")
rcParams["figure.figsize"]=5,5

subscribe_rate_loan.plot.bar(x="loan",y="sub_rate")
rcParams["figure.figsize"]=5,5

subscribe_rate_con.plot.bar(x="contact",y="sub_rate")
rcParams["figure.figsize"]=5,5

subscribe_rate_mon.plot.bar(x="month",y="sub_rate")
rcParams["figure.figsize"]=5,5

subscribe_rate_pout.plot.bar(x="poutcome",y="sub_rate")
k = bank.select_dtypes(include="object").columns
for i in k:

    bank[i] = bank[i].astype("category")

    
cat = bank.select_dtypes(include="category")
rcParams["figure.figsize"]=12,12

i = 1

for col in cat.columns:

    plt.subplot(2,5,i)

    bank[col].value_counts().plot.bar()

    i=i+1

    plt.xlabel(col)

    plt.ylabel("count")

plt.tight_layout()

plt.show()
col2 =bank.select_dtypes(include="integer").columns
rcParams["figure.figsize"]=7,22

i=1

for c in col2:

    plt.subplot(7,1,i)

    sns.distplot(bank[c])

    i = i+1

plt.tight_layout()

plt.show()
sns.distplot(bank["balance"])
rcParams["figure.figsize"] = 5,5

sns.boxplot(bank["balance"])
sns.boxplot(bank["age"])
sns.boxplot(bank["duration"])
b = pd.get_dummies(bank["job"],prefix="job",drop_first=True)
bank = pd.concat([bank,b],axis=1)
c = pd.get_dummies(bank["marital"],prefix="marital",drop_first=True)
bank = pd.concat([bank,c],axis=1)
d = pd.get_dummies(bank["contact"],prefix="contact",drop_first=True)
bank = pd.concat([bank,d],axis=1)
e = pd.get_dummies(bank["poutcome"],prefix="outcome",drop_first=True)
bank = pd.concat([bank,e],axis=1)
bank.head()
val = {"yes":1,"no":0}
bank["loan"]= bank["loan"].map(val)
bank["default"]=bank["default"].map(val)
bank["housing"]=bank["housing"].map(val)
bank["y"]=bank["y"].map(val)
val1 = {"unknown":0,"primary":1,"secondary":"2","tertiary":"3"}
bank["education"]=bank["education"].map(val1)
bank = bank.drop(["marital"],axis=1)
bank = bank.drop(["job"],axis=1)
bank = bank.drop(["contact"],axis=1)
bank = bank.drop(["poutcome"],axis=1)
mont = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,"jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12}
month1 = {1:"moderate_month",2:"moderate_month",4:"moderate_month",11:"moderate_month",6:"busy_month",8:"busy_month",7:"busy_month",5:"busy_month",12:"low_month",3:"low_month",9:"low_month",10:"low_month"}
bank["month"]=bank["month"].map(mont)
bank["education"] = bank["education"].astype("int")
x = bank.drop(["y"],axis=1)

y = bank["y"]
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg = logreg.fit(x_train,y_train)
y_pred = logreg.predict(x_test)
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix,classification_report,r2_score

print("f1_score:",f1_score(y_test,y_pred))
print("f1_score:",f1_score(y_test,y_pred))
print("Train Accuracy:",accuracy_score(y_train,logreg.predict(x_train)))
print("Test Accuracy:",accuracy_score(y_test,y_pred))
confusion_matrix(y_test,y_pred)
def create_conf_mat(test_class_set, predictions):

    if (len(test_class_set.shape) != len(predictions.shape) == 1):

        return print('Arrays entered are not 1-D.\nPlease enter the correctly sized sets.')

    elif (test_class_set.shape != predictions.shape):

        return print('Number of values inside the Arrays are not equal to each other.\nPlease make sure the array has the same number of instances.')

    else:

        # Set Metrics

        test_crosstb_comp = pd.crosstab(index = test_class_set,

                                        columns = predictions)

        # Changed for Future deprecation of as_matrix

        test_crosstb = test_crosstb_comp.values

        return test_crosstb
conf_mat = create_conf_mat(y_test, y_pred)

sns.heatmap(conf_mat, annot=True, fmt='d', cbar=False)

plt.xlabel('Predicted Values')

plt.ylabel('Actual Values')

plt.title('Actual vs. Predicted Confusion Matrix')

plt.show()
TN=7763

TP=350

FN=713

FP=217

sensitivity=TP/float(TP+FN)

specificity=TN/float(TN+FP)
print('The acuuracy of the model = TP+TN / (TP+TN+FP+FN) = ',(TP+TN)/float(TP+TN+FP+FN),'\n\n',



'The Miss-classification = 1-Accuracy = ',1-((TP+TN)/float(TP+TN+FP+FN)),'\n\n',



'Sensitivity or True Positive Rate = TP / (TP+FN) = ',TP/float(TP+FN),'\n\n',



'Specificity or True Negative Rate = TN / (TN+FP) = ',TN/float(TN+FP),'\n\n',



'Positive Predictive value = TP / (TP+FP) = ',TP/float(TP+FP),'\n\n',



'Negative predictive Value = TN / (TN+FN) = ',TN/float(TN+FN),'\n\n',



'Positive Likelihood Ratio = Sensitivity / (1-Specificity) = ',sensitivity/(1-specificity),'\n\n',



'Negative likelihood Ratio = (1-Sensitivity) / Specificity = ',(1-sensitivity)/specificity)

print(classification_report(y_test,y_pred))
bank["age_group"] = pd.cut(bank["age"],bins=[13,29,60,99],labels=["young","adult","old"])
#bank["month"] = bank["month"].map(month1)
#val1 ={"low_month":1,"moderate_month":2,"busy_month":3}

#bank["month"] = bank["month"].map(val1)
val = {"young":1,"adult":2,"old":3}

bank["age_group"] = bank["age_group"].map(val)
bank = bank.drop(["age"],axis=1)
x = bank.drop(["y"],axis=1)

y = bank["y"]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)
logreg = LogisticRegression()

logreg = logreg.fit(x_train,y_train)
y_pred = logreg.predict(x_test)
f1_score(y_test,y_pred)
print(classification_report(y_test,y_pred))
confusion_matrix(y_test,y_pred)
bank.columns
pd.options.display.max_columns=None
bank.shape
from imblearn.over_sampling import SMOTE
bank["duration"] = bank["duration"]//60
bank = bank[bank["balance"]<=20000]
bank = bank[(bank["balance"]<10000) ]
bank = bank[bank["balance"]>-5000]
sns.boxplot(bank["balance"])
def VIF(formula,data):

    import pip #To install packages

    #pip.main(["install","dmatrices"])

    #pip.main(["install","statsmodels"])

    from patsy import dmatrices

    from statsmodels.stats.outliers_influence import variance_inflation_factor

    y , X = dmatrices(formula,data = data,return_type="dataframe")

    vif = pd.DataFrame()

    vif["VIF Factor"] = [variance_inflation_factor(X.values, i) \

       for i in range(X.shape[1])]

    vif["features"] = X.columns

    return(vif.round(1))
bank.columns
bank = bank.rename(columns={"job_self-employed":"job_selfemployed","job_blue-collar":"job_bluecollar"})
VIF("y ~education +default +balance +housing +loan +day +month +duration +campaign +pdays +previous  +job_entrepreneur +job_housemaid +job_selfemployed +job_bluecollar +job_management +job_retired +job_services +job_student +job_technician +job_unemployed +job_unknown +marital_married +marital_single +contact_telephone +contact_unknown +outcome_other +outcome_success +outcome_unknown +age_group",data=bank)


x = bank.drop(["y"],axis=1)

y = bank["y"]

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
from imblearn.under_sampling import TomekLinks

tl = TomekLinks(return_indices=True, ratio='majority')

x1,y1,id1=tl.fit_resample(x_train, y_train)
print(classification_report(y_test,(gb.fit(x1,y1).predict(x_test))))
x = bank.drop(["y"],axis=1)

y = bank["y"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
from imblearn.under_sampling import ClusterCentroids

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

cc = ClusterCentroids(ratio={0: 10})

x_cc, y_cc = cc.fit_resample(x_train, y_train)

lr = LogisticRegression()
print(classification_report(y_test,(lr.fit(x_cc,y_cc).predict(x_test))))
x = bank.drop(["y"],axis=1)

y = bank["y"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
from imblearn.over_sampling import SMOTE



smote = SMOTE(ratio="minority")

X_sm, y_sm = smote.fit_resample(x_train, y_train)

gb = GradientBoostingClassifier()
print(classification_report(y_test,(gb.fit(X_sm,y_sm).predict(x_test))))
x = bank.drop(["y"],axis=1)

y = bank["y"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=0)
from imblearn.combine import SMOTETomek



smt = SMOTETomek(ratio='auto')

x_smt, y_smt = smt.fit_resample(x_train, y_train)
print(classification_report(y_test,(lr.fit(x_smt,y_smt).predict(x_test))))
x = bank.drop(["y"],axis=1)

y = bank["y"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
from imblearn.over_sampling import SMOTE

from imblearn.pipeline import make_pipeline



pipe = make_pipeline(SMOTE(),LogisticRegression())



print(classification_report(y_test,(pipe.fit(x_train,y_train).predict(x_test))))
x = bank.drop(["y"],axis=1)

y = bank["y"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

x_train = sc_X.fit_transform(x_train)

x_test = sc_X.transform(x_test)
from sklearn.preprocessing import MinMaxScaler

mn = MinMaxScaler()

x_train = mn.fit_transform(x_train)

x_test = mn.transform(x_test)
from imblearn.over_sampling import SMOTE

from imblearn.pipeline import make_pipeline



pipe = make_pipeline(SMOTE(),GradientBoostingClassifier())



print(classification_report(y_test,(pipe.fit(x_train,y_train).predict(x_test))))
x = bank.drop(["y","job_unknown","default","job_retired","previous"],axis=1)

y = bank["y"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
from imblearn.combine import SMOTETomek

from imblearn.under_sampling import RandomUnderSampler

x_train3,y_train3 = SMOTETomek().fit_resample(x_train,y_train)

x_train2,y_train2 = SMOTE().fit_resample(x_train,y_train)

x_train4,y_train4 = RandomUnderSampler().fit_resample(x_train,y_train)
rf1 = RandomForestClassifier(n_estimators=80,criterion="gini",max_depth = 13,min_samples_leaf=20)

print(classification_report(y_test,rf1.fit(x_train3,y_train3).predict(x_test)))
rf1 = RandomForestClassifier(n_estimators=80,criterion="gini",max_depth = 13,min_samples_leaf=20)

print(classification_report(y_test,rf1.fit(x_train4,y_train4).predict(x_test)))
rf1 = RandomForestClassifier(n_estimators=80,criterion="gini",max_depth = 13,min_samples_leaf=15)

print(classification_report(y_test,rf1.fit(x_train2,y_train2).predict(x_test)))
rf1 = RandomForestClassifier(n_estimators=70,criterion="gini",max_depth = 13,min_samples_leaf=15)

print(classification_report(y_test,rf1.fit(x_train4,y_train4).predict(x_test)))
rcParams["figure.figsize"] = 6,6

conf_mat = create_conf_mat(y_test, rf1.fit(x_train4,y_train4).predict(x_test))

sns.heatmap(conf_mat, annot=True, fmt='d', cbar=False)

plt.xlabel('Predicted Values')

plt.ylabel('Actual Values')

plt.title('Actual vs. Predicted Confusion Matrix')

plt.show()
accuracy_score(y_test,rf1.fit(x_train4,y_train4).predict(x_test))
accuracy_score(y_train,rf1.fit(x_train4,y_train4).predict(x_train))