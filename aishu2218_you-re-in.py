import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

pd.set_option('display.max_rows',500)

%matplotlib inline

sns.set()
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, confusion_matrix ,accuracy_score,recall_score,precision_score
campus = pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
# Let's see head of our data.

campus.head()
# Let's see how many rows and columns we have.

campus.shape
# Let's dig in some more info.

campus.info()
num = campus.select_dtypes(include=[np.float64,np.int64])

print("Numerical Columns:",num.shape[1])
cat = campus.select_dtypes(include=[np.object])

print("Categorical Columns:",cat.shape[1])
campus.describe()
campus.set_index('sl_no',inplace=True)
plt.figure(figsize=(10,5))

sns.heatmap(campus.isnull(),cbar=False)
campus['salary'].fillna('0',inplace=True)
campus.isnull().sum()
campus.head()
campus.gender.value_counts(normalize=True)*100
plt.figure(figsize=(8,5))

sns.countplot('gender',data=campus,palette='Accent')
campus['hsc_b'].value_counts(normalize=True)*100
plt.figure(figsize=(8,5))

sns.countplot('hsc_b',data=campus,palette='magma')
campus['hsc_s'].value_counts(normalize=True)*100
plt.figure(figsize=(8,5))

sns.countplot('hsc_s',data=campus,palette='mako')
campus['degree_t'].value_counts(normalize=True)*100
plt.figure(figsize=(8,5))

sns.countplot('degree_t',data=campus,palette='plasma')
campus['workex'].value_counts(normalize=True)*100
plt.figure(figsize=(8,5))

sns.countplot('workex',data=campus,palette='summer')
campus['specialisation'].value_counts(normalize=True)*100
plt.figure(figsize=(8,5))

sns.countplot('specialisation',data=campus,palette='autumn')
campus['status'].value_counts(normalize=True)*100
plt.figure(figsize=(8,5))

sns.countplot('status',data=campus,palette='spring')
campus['ssc_b'].value_counts(normalize=True)*100
plt.figure(figsize=(8,5))

sns.countplot('ssc_b',data=campus,palette='GnBu')
campus.head()
plt.figure(figsize=(8,5))

sns.countplot('gender',hue='status',data=campus,palette='winter')
avg_per= campus.groupby(['gender','status'])['ssc_p'].mean()

avg_per
plt.figure(figsize=(8,5))

sns.barplot(avg_per.index,avg_per.values,palette='hot')
plt.figure(figsize=(8,5))

sns.boxplot('status','ssc_p',data=campus,palette="Greys")
plt.figure(figsize=(8,5))

sns.boxplot('ssc_b','ssc_p',data=campus,palette="plasma")
plt.figure(figsize=(8,5))

sns.countplot('ssc_b',hue='status',data=campus,palette="rocket")
plt.figure(figsize=(8,5))

sns.boxplot(y='hsc_p',x='status',data=campus,palette="afmhot")
plt.figure(figsize=(8,5))

sns.boxplot('hsc_b','hsc_p',data=campus,palette="bone")
plt.figure(figsize=(8,5))

sns.countplot('hsc_b',hue='status',data=campus,palette="copper")
plt.figure(figsize=(8,5))

sns.countplot('hsc_s',hue='status',data=campus,palette="Pastel1")
plt.figure(figsize=(8,5))

sns.boxplot(y='degree_p',x='status',data=campus,palette="icefire")
plt.figure(figsize=(8,5))

sns.countplot('degree_t',hue='status',data=campus,palette="tab20")
plt.figure(figsize=(8,5))

sns.boxplot('degree_t','degree_p',hue='status',data=campus,palette="Set2")
plt.figure(figsize=(8,5))

sns.countplot('workex',hue='status',data=campus,palette="brg")
plt.figure(figsize=(8,5))

sns.boxplot(y='etest_p',x='status',data=campus,palette="cubehelix")
plt.figure(figsize=(8,5))

sns.countplot('specialisation',hue='status',data=campus,palette="Dark2")
plt.figure(figsize=(8,5))

sns.boxplot('specialisation','mba_p',hue='status',data=campus,palette="Set2")
#check the dtype of salary

campus['salary'].dtype

# the column dtype needs to be numeric

campus['salary'] = pd.to_numeric(campus['salary'])
# Fetching the row that has maximum salary

campus.iloc[campus['salary'].argmax()]
# Fetching the row that has minimum salary

campus.iloc[campus['salary'].argmin()]
campus[(campus['status']=='Placed') & (campus['specialisation']=='Mkt&Fin')][['ssc_p','hsc_p','degree_p','mba_p','salary']].sort_values(by='salary',ascending=False)
campus[(campus['status']=='Placed') & (campus['specialisation']=='Mkt&HR')][['ssc_p','hsc_p','degree_p','mba_p','salary']].sort_values(by='salary',ascending=False)
campus[campus['status']=='Placed'][['ssc_b','hsc_b','hsc_s','degree_t','specialisation']].value_counts(normalize=True)*100
campus[campus['status']=='Not Placed'][['ssc_b','hsc_b','hsc_s','degree_t','specialisation']].value_counts(normalize=True)*100
plt.figure(figsize=(15,8))

sns.heatmap(campus.corr(),annot=True)
campus.head()
campus.columns
campus['status']= campus['status'].map({'Placed':1,'Not Placed':0})

campus['workex']= campus['workex'].map({'Yes':1,'No':0})
campus= pd.get_dummies(data=campus,columns=['gender','ssc_b','hsc_b','hsc_s','degree_t','specialisation'],drop_first=True)
campus.head()
X= campus.drop(['status','salary'],1)

y= campus['status']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=22)
X_train.columns
scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
# instantiate the model (using the default parameters)

logreg = LogisticRegression()



# fit the model with data

logreg.fit(X_train,y_train)



# making predictions

y_pred=logreg.predict(X_test)
cnf_matrix = confusion_matrix(y_test, y_pred)

cnf_matrix
class_names=[0,1] # name  of classes

fig, ax = plt.subplots()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks, class_names)

plt.yticks(tick_marks, class_names)

# create heatmap

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="summer" ,fmt='g')

ax.xaxis.set_label_position("top")

plt.tight_layout()

plt.title('Confusion matrix', y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')
print("Accuracy:",accuracy_score(y_test, y_pred)*100)

print("Precision:",precision_score(y_test, y_pred)*100)

print("Recall:",recall_score(y_test, y_pred)*100)
#Create a Gaussian Classifier

clf=RandomForestClassifier(n_estimators=100)



#Train the model using the training sets y_pred=clf.predict(X_test)

clf.fit(X_train,y_train)



y_pred = clf.predict(X_test)
# Model Accuracy, how often is the classifier correct?

print("Accuracy:",accuracy_score(y_test, y_pred)*100)
features = ['ssc_p', 'hsc_p', 'degree_p', 'workex', 'etest_p', 'mba_p','gender_M', 'ssc_b_Others', 

            'hsc_b_Others', 'hsc_s_Commerce','hsc_s_Science', 'degree_t_Others', 'degree_t_Sci&Tech','specialisation_Mkt&HR']
feature_imp = pd.Series(clf.feature_importances_,index=features ).sort_values(ascending=False)

feature_imp
plt.figure(figsize=(10,8))

sns.barplot(y=feature_imp.index,x=feature_imp.values,palette='CMRmap')