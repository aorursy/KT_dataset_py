import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="whitegrid")

from sklearn.model_selection import train_test_split



from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix

from sklearn import metrics

from sklearn.linear_model import LogisticRegression



from sklearn.ensemble import RandomForestClassifier

from sklearn.dummy import DummyClassifier

from sklearn.svm import SVC

from xgboost import XGBClassifier
train = pd.read_csv('/kaggle/input/airplane-accidents-severity-dataset/train.csv')

test = pd.read_csv('/kaggle/input/airplane-accidents-severity-dataset/test.csv')
train.head()
y = train['Severity']

x = train.drop(['Severity','Accident_ID'],axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

x_train, x_cv, y_train, y_cv = train_test_split(x_train, y_train, test_size=0.20,random_state=0)
train.info()
test.info()
train.describe().T
test.describe().T
class_label = train['Severity'].value_counts()

total_points = len(train)

print("Points with class label -> 'Highly fatal and damaging' are = ",class_label.values[0]/total_points*

100,"%")

print("Points with class label -> 'Significant damage and serious injuries' are = ",class_label.values[1]/total_points*

100,"%")

print("Points with class label -> 'Minor damage and injuries' are = ",class_label.values[2]/total_points*

100,"%")

print("Points with class label -> 'Significant damage and fatalities' are = ",class_label.values[3]/total_points*

100,"%")

labels = ['Highly fatal and damaging','Significant damage and serious injuries','Minor damage and injuries','Significant damage and fatalities']

sizes = [30.490000000000002,27.29,25.27,16.950000000000003]

colors = ['yellowgreen', 'gold','orange','green']

plt.figure(figsize=(8,10))

plt.pie(sizes, labels=labels, colors=colors,autopct='%1.1f%%', shadow=True)
sns.FacetGrid(train, hue="Severity",height=8).map(sns.distplot, "Safety_Score").add_legend()
plt.figure(figsize=(20,8))

sns.violinplot(data=train,x = 'Severity', y = 'Safety_Score') 
sns.catplot(x="Days_Since_Inspection",hue="Severity", kind="count", data=train,aspect=2)
sns.catplot(x="Total_Safety_Complaints",hue="Severity", kind="count", data=train,aspect=2,height=8)
plt.figure(figsize=(20,8))

sns.violinplot(data=train,x = 'Severity', y = 'Total_Safety_Complaints') 
sns.FacetGrid(train, hue="Severity",height=8,aspect=2).map(sns.distplot, "Control_Metric").add_legend()
plt.figure(figsize=(20,8))

sns.violinplot(data=train,x = 'Severity', y = 'Control_Metric')
sns.FacetGrid(train, hue="Severity",height=8).map(sns.distplot, "Turbulence_In_gforces").add_legend()
plt.figure(figsize=(20,8))

sns.violinplot(data=train,x = 'Severity', y = 'Turbulence_In_gforces')
sns.FacetGrid(train, hue="Severity",height=8).map(sns.distplot, "Cabin_Temperature").add_legend()
plt.figure(figsize=(20,8))

sns.violinplot(data=train,x = 'Severity', y = 'Cabin_Temperature')
sns.catplot(x="Accident_Type_Code",hue="Severity", kind="count", data=train,aspect=2)
sns.FacetGrid(train, hue="Severity",height=8,aspect=2).map(sns.distplot, "Max_Elevation").add_legend()
MinorDamagesAndInjuries = train[train.Severity == 'Minor_Damage_And_Injuries']

SignificantDamageAndFatalities = train[train.Severity == 'Significant_Damage_And_Fatalities']

SignificantDamageAndSeriousInjuries = train[train.Severity == 'Significant_Damage_And_Serious_Injuries']

HighlyFatalAndDamaging = train[train.Severity == 'Highly_Fatal_And_Damaging']
count1,binedges1 = np.histogram(MinorDamagesAndInjuries['Max_Elevation'], bins=10, density = True)

pdf1=count1/sum(count1)

cdf1=np.cumsum(pdf1)

plt.plot(binedges1[1:],pdf1)

plt.plot(binedges1[1:], cdf1)

plt.xlabel("Max elevation for Minor damages and injuries")

plt.title("Minor damages and injuries - > PDF v/s CDF for Max elevation")
count2,binedges2 = np.histogram(SignificantDamageAndFatalities['Max_Elevation'], bins=10, density = True)

pdf2=count2/sum(count2)

cdf2=np.cumsum(pdf2)

plt.plot(binedges1[1:],pdf2)

plt.plot(binedges1[1:], cdf2)

plt.xlabel("Max elevation for Significant damage and fatalities")

plt.title("Significant damage and fatalities - > PDF v/s CDF for Max elevation")
count3,binedges3 = np.histogram(SignificantDamageAndSeriousInjuries['Max_Elevation'], bins=10, density = True)

pdf3=count3/sum(count3)

cdf3=np.cumsum(pdf3)

plt.plot(binedges1[1:],pdf3)

plt.plot(binedges1[1:], cdf3)

plt.xlabel("Max elevation for Significant damage and serious injuries")

plt.title("Significant damage and serious injuries - > PDF v/s CDF for Max elevation")
count4,binedges4 = np.histogram(HighlyFatalAndDamaging['Max_Elevation'], bins=10, density = True)

pdf4=count4/sum(count4)

cdf4=np.cumsum(pdf4)

plt.plot(binedges1[1:],pdf4)

plt.plot(binedges1[1:], cdf4)

plt.xlabel("Max elevation for Highly fatal and damaging")

plt.title("Highly fatal and damaging - > PDF v/s CDF for Max elevation")
sns.catplot(x="Violations",hue="Severity", kind="count", data=train,aspect=2)
sns.FacetGrid(train, hue="Severity",height=8).map(sns.distplot, "Adverse_Weather_Metric").add_legend()
plt.figure(figsize=(20,8))

sns.violinplot(data=train,x = 'Severity', y = 'Adverse_Weather_Metric') 
sns.catplot(x="Days_Since_Inspection",y="Total_Safety_Complaints",hue="Severity", data=train,aspect=2)
plt.figure(figsize=(20,20))

sns.scatterplot(x="Safety_Score", y="Max_Elevation", hue="Severity",data=train)
plt.figure(figsize=(20,20))

sns.scatterplot(x="Control_Metric", y="Max_Elevation", hue="Severity",data=train)
plt.figure(figsize=(20,10))

sns.scatterplot(x="Turbulence_In_gforces", y="Control_Metric", hue="Severity",data=train)
plt.figure(figsize=(20,10))

sns.scatterplot(x="Days_Since_Inspection", y="Safety_Score", hue="Severity",data=train)
dummy_clf = DummyClassifier(strategy="uniform") # uniform means that the model randomly assigns a class label given a quiery point.

dummy_clf.fit(x_train, y_train)
print("Log loss of random model on 'Training data' = ",metrics.log_loss(y_train,dummy_clf.predict_proba(x_train),labels=class_label.index))

print("Log loss of random model on 'CV data' = ",metrics.log_loss(y_cv,dummy_clf.predict_proba(x_cv),labels=class_label.index))

print("Log loss of random model on 'Testing data' = ",metrics.log_loss(y_test,dummy_clf.predict_proba(x_test),labels=class_label.index))
lg = LogisticRegression()

params = {'C':[0.0001,0.001,0.01,0.1,10,100,1000]}

gs = GridSearchCV(lg,param_grid=params,scoring='neg_log_loss')
gs.fit(x_train['Safety_Score'].values.reshape(-1,1),y_train)

gs.best_params_
lg = LogisticRegression(C=0.001)

lg.fit(x_train['Safety_Score'].values.reshape(-1,1),y_train)

print("log loss on train = ",metrics.log_loss(y_train,lg.predict_proba(x_train['Safety_Score'].values.reshape(-1,1))))

print("log loss on cv = ",metrics.log_loss(y_cv,lg.predict_proba(x_cv['Safety_Score'].values.reshape(-1,1))))

print("log loss on test = ",metrics.log_loss(y_test,lg.predict_proba(x_test['Safety_Score'].values.reshape(-1,1))))
gs.fit(x_train['Days_Since_Inspection'].values.reshape(-1,1),y_train)

gs.best_params_
lg = LogisticRegression(C=0.0001)

lg.fit(x_train['Days_Since_Inspection'].values.reshape(-1,1),y_train)

print("log loss on train = ",metrics.log_loss(y_train,lg.predict_proba(x_train['Days_Since_Inspection'].values.reshape(-1,1))))

print("log loss on cv = ",metrics.log_loss(y_cv,lg.predict_proba(x_cv['Days_Since_Inspection'].values.reshape(-1,1))))

print("log loss on test = ",metrics.log_loss(y_test,lg.predict_proba(x_test['Days_Since_Inspection'].values.reshape(-1,1))))
gs.fit(x_train['Total_Safety_Complaints'].values.reshape(-1,1),y_train)

gs.best_params_
lg = LogisticRegression(C=0.0001)

lg.fit(x_train['Total_Safety_Complaints'].values.reshape(-1,1),y_train)

print("log loss on train = ",metrics.log_loss(y_train,lg.predict_proba(x_train['Total_Safety_Complaints'].values.reshape(-1,1))))

print("log loss on cv = ",metrics.log_loss(y_cv,lg.predict_proba(x_cv['Total_Safety_Complaints'].values.reshape(-1,1))))

print("log loss on test = ",metrics.log_loss(y_test,lg.predict_proba(x_test['Total_Safety_Complaints'].values.reshape(-1,1))))
gs.fit(x_train['Control_Metric'].values.reshape(-1,1),y_train)

gs.best_params_
lg = LogisticRegression(C=0.001)

lg.fit(x_train['Control_Metric'].values.reshape(-1,1),y_train)

print("log loss on train = ",metrics.log_loss(y_train,lg.predict_proba(x_train['Control_Metric'].values.reshape(-1,1))))

print("log loss on cv = ",metrics.log_loss(y_cv,lg.predict_proba(x_cv['Control_Metric'].values.reshape(-1,1))))

print("log loss on test = ",metrics.log_loss(y_test,lg.predict_proba(x_test['Control_Metric'].values.reshape(-1,1))))
gs.fit(x_train['Turbulence_In_gforces'].values.reshape(-1,1),y_train)

gs.best_params_
lg = LogisticRegression(C=10)

lg.fit(x_train['Turbulence_In_gforces'].values.reshape(-1,1),y_train)

print("log loss on train = ",metrics.log_loss(y_train,lg.predict_proba(x_train['Turbulence_In_gforces'].values.reshape(-1,1))))

print("log loss on cv = ",metrics.log_loss(y_cv,lg.predict_proba(x_cv['Turbulence_In_gforces'].values.reshape(-1,1))))

print("log loss on test = ",metrics.log_loss(y_test,lg.predict_proba(x_test['Turbulence_In_gforces'].values.reshape(-1,1))))
gs.fit(x_train['Cabin_Temperature'].values.reshape(-1,1),y_train)

gs.best_params_
lg = LogisticRegression(C=0.01)

lg.fit(x_train['Cabin_Temperature'].values.reshape(-1,1),y_train)

print("log loss on train = ",metrics.log_loss(y_train,lg.predict_proba(x_train['Cabin_Temperature'].values.reshape(-1,1))))

print("log loss on cv = ",metrics.log_loss(y_cv,lg.predict_proba(x_cv['Cabin_Temperature'].values.reshape(-1,1))))

print("log loss on test = ",metrics.log_loss(y_test,lg.predict_proba(x_test['Cabin_Temperature'].values.reshape(-1,1))))
gs.fit(x_train['Accident_Type_Code'].values.reshape(-1,1),y_train)

gs.best_params_
lg = LogisticRegression(C=0.1)

lg.fit(x_train['Accident_Type_Code'].values.reshape(-1,1),y_train)

print("log loss on train = ",metrics.log_loss(y_train,lg.predict_proba(x_train['Accident_Type_Code'].values.reshape(-1,1))))

print("log loss on cv = ",metrics.log_loss(y_cv,lg.predict_proba(x_cv['Accident_Type_Code'].values.reshape(-1,1))))

print("log loss on test = ",metrics.log_loss(y_test,lg.predict_proba(x_test['Accident_Type_Code'].values.reshape(-1,1))))
gs.fit(x_train['Max_Elevation'].values.reshape(-1,1),y_train)

gs.best_params_
lg = LogisticRegression(C=0.0001)

lg.fit(x_train['Max_Elevation'].values.reshape(-1,1),y_train)

print("log loss on train = ",metrics.log_loss(y_train,lg.predict_proba(x_train['Max_Elevation'].values.reshape(-1,1))))

print("log loss on cv = ",metrics.log_loss(y_cv,lg.predict_proba(x_cv['Max_Elevation'].values.reshape(-1,1))))

print("log loss on test = ",metrics.log_loss(y_test,lg.predict_proba(x_test['Max_Elevation'].values.reshape(-1,1))))
gs.fit(x_train['Violations'].values.reshape(-1,1),y_train)

gs.best_params_
lg = LogisticRegression(C=0.01)

lg.fit(x_train['Violations'].values.reshape(-1,1),y_train)

print("log loss on train = ",metrics.log_loss(y_train,lg.predict_proba(x_train['Violations'].values.reshape(-1,1))))

print("log loss on cv = ",metrics.log_loss(y_cv,lg.predict_proba(x_cv['Violations'].values.reshape(-1,1))))

print("log loss on test = ",metrics.log_loss(y_test,lg.predict_proba(x_test['Violations'].values.reshape(-1,1))))
gs.fit(x_train['Adverse_Weather_Metric'].values.reshape(-1,1),y_train)

gs.best_params_
lg = LogisticRegression(C=10)

lg.fit(x_train['Adverse_Weather_Metric'].values.reshape(-1,1),y_train)

print("log loss on train = ",metrics.log_loss(y_train,lg.predict_proba(x_train['Adverse_Weather_Metric'].values.reshape(-1,1))))

print("log loss on cv = ",metrics.log_loss(y_cv,lg.predict_proba(x_cv['Adverse_Weather_Metric'].values.reshape(-1,1))))

print("log loss on test = ",metrics.log_loss(y_test,lg.predict_proba(x_test['Adverse_Weather_Metric'].values.reshape(-1,1))))
cor = train.corr()

plt.figure(figsize=(12,10))

sns.heatmap(cor,annot=True)
temp = x_train['Violations'] + x_train['Total_Safety_Complaints']

x_train['Total_Problems'] = temp



temp = x_test['Violations'] + x_test['Total_Safety_Complaints']

x_test['Total_Problems'] = temp



temp = x_cv['Violations'] + x_cv['Total_Safety_Complaints']

x_cv['Total_Problems'] = temp
clf = RandomForestClassifier()

params = {'max_features': np.arange(1, 11),'criterion' :['gini', 'entropy']}

best_model = GridSearchCV(clf, params,n_jobs=-1)

best_model.fit(x_train,y_train)
best_model.best_params_
rf = RandomForestClassifier(criterion='entropy',max_features=8,n_estimators=1000)

rf.fit(x_train,y_train)

print(rf.score(x_test,y_test))

print(rf.score(x_cv,y_cv))
x = train.drop(['Severity','Accident_ID'],axis=1)

x['Total_Probelms'] = train['Violations'] + train['Total_Safety_Complaints']
rf.fit(x,y)
x1 = test.drop(['Accident_ID'],axis=1)

x1['Total_Probelms'] = x1['Violations'] + x1['Total_Safety_Complaints']
y_pred = rf.predict(x1)

submission = pd.DataFrame()

submission['Accident_ID'] = test['Accident_ID']

submission['Severity'] = y_pred

export_csv = submission.to_csv (r'predictions.csv', index = False, header=True)