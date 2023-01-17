import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # visualization
import seaborn as sns; sns.set()

data=pd.read_csv("../input/StudentsPerformance.csv")

print("data shape is:", data.shape,"\n") #There are 1000 records with 8 features each
print("features are:", data.columns.tolist()) #To see the columns names

print(data.isnull().sum(),"\n")
print(data.describe().round(2))
fig, axes = plt.subplots(1,3, sharey=True, figsize=(18,5))

ax1, ax2, ax3 = axes.flatten()
ax1.hist(data['math score'], bins=10, color="red")
ax2.hist(data['reading score'], bins=10, color="blue")
ax3.hist(data['writing score'], bins=10, color="orange")
ax1.set_xlabel('MATH', fontsize="large")
ax1.set_ylabel("SCORE", fontsize="large")
ax2.set_xlabel('READING', fontsize="large")
ax3.set_xlabel('WRITING', fontsize="large")

plt.suptitle('Score Comparison', ha='center', fontsize='x-large')
plt.show()
import warnings
warnings.filterwarnings('ignore') # to ignore some warnings

sns.kdeplot(data['math score'], shade=True, color="red", alpha=0.9)
sns.kdeplot(data['reading score'], shade=True, color="blue", alpha=0.6)
sns.kdeplot(data['writing score'], shade=True, color="orange", alpha=0.4)
plt.show()
col=["gender", "race/ethnicity","parental level of education", "lunch", "test preparation course"]
for item in col:
    print(item.upper(),":")
    print(data[item].value_counts(),"\n") 
fig, axs = plt.subplots(3, 2)
fig.set_figheight(15)
fig.set_figwidth(10)
ax1, ax2, ax3, ax4,ax5 = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1], axs[2, 0]
axs[-1,-1].axis('off')
colors = ["grey", "pink", "yellowgreen", "orange", "violet", "yellow"]
ax1.pie(data['gender'].value_counts(), labels=list(data['gender'].unique()), colors=colors, autopct='%1.1f%%', startangle=90)
ax2.pie(data['race/ethnicity'].value_counts(), labels=list(data['race/ethnicity'].unique()), colors=colors, autopct='%1.1f%%', startangle=90)
ax3.pie(data['lunch'].value_counts(), labels=list(data['lunch'].unique()), colors=colors, autopct='%1.1f%%', startangle=90)
ax4.pie(data['test preparation course'].value_counts(), labels=list(data['test preparation course'].unique()), colors=colors, autopct='%1.1f%%')
ax5.pie(data['parental level of education'].value_counts(), labels=list(data['parental level of education'].unique()), colors=colors, autopct='%1.1f%%', startangle=60)
plt.suptitle('PERCENTAGE DISTRIBUTION', ha='center', fontsize='xx-large',fontweight='bold')
ax1.set_title("GENDER",fontsize='x-large',fontweight='bold' )
ax2.set_title("RACE/ETHNICITY",fontsize='x-large',fontweight='bold')
ax3.set_title("LUNCH",fontsize='x-large',fontweight='bold')
ax4.set_title("TEST PREP. COURSE",fontsize='x-large',fontweight='bold')
ax5.set_title("PARENT EDUCATION",fontsize='x-large',fontweight='bold')
plt.show()
data["average score"]=np.mean(data[['math score', 'reading score', 'writing score']], axis=1).round(1)
data['admitted/rejected']=np.where(data['average score']>70,1,0)
data_old=data.copy()
data_old.head()
sns.set(style="ticks")
g = sns.catplot(data=data, x="parental level of education", y="average score", hue="gender")
g.set_xticklabels(rotation=90)
plt.show()
g=sns.FacetGrid(data, col='admitted/rejected', hue="gender", height=3.5)
g.map(sns.kdeplot, 'math score')
plt.legend()
plt.show()
g=sns.FacetGrid(data, col='admitted/rejected', hue="gender", height=3.5)
g.map(sns.kdeplot, 'reading score')
plt.legend()
plt.show()
g=sns.FacetGrid(data, col='admitted/rejected', hue="gender", height=3.5)
g.map(sns.kdeplot, 'writing score')
plt.legend()
plt.show()
g = sns.pairplot(data.iloc[:, [0,5,6,7]], hue="gender", diag_kind="kde", height=2.5)
plt.show()
from sklearn.preprocessing import LabelEncoder
lbl=LabelEncoder()

data["gender_code"]=lbl.fit_transform(data[["gender"]])
data["race/ethnicity_code"]=lbl.fit_transform(data[["race/ethnicity"]])
data["parental level of education_code"]=lbl.fit_transform(data[["parental level of education"]])
data["lunch_code"]=lbl.fit_transform(data[["lunch"]])
data["test preparation course_code"]=lbl.fit_transform(data[["test preparation course"]])
good_cols=['reading score','writing score','math score','gender_code','race/ethnicity_code','parental level of education_code','lunch_code','test preparation course_code']
target=data['admitted/rejected']
data[good_cols].head()
fig, ax = plt.subplots(figsize=(3,3))
col=['reading score','writing score', 'math score']
corr_matrix=data[col].corr(method="spearman")
ax=sns.heatmap(corr_matrix, center=0, vmax=1, vmin=-1, annot=True, square=True)
from collections import Counter
import math
import scipy.stats as ss

def conditional_entropy(x,y):
    # entropy of x given y
    y_counter = Counter(y)
    xy_counter = Counter(list(zip(x,y)))
    total_occurrences = sum(y_counter.values())
    entropy = 0
    for xy in xy_counter.keys():
        p_xy = xy_counter[xy] / total_occurrences
        p_y = y_counter[xy[1]] / total_occurrences
        entropy += p_xy * math.log(p_y/p_xy)
    return entropy

def theil_u(x,y):
    s_xy = conditional_entropy(x,y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))
    s_x = ss.entropy(p_x)
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x
    
column=['gender_code','race/ethnicity_code','parental level of education_code','lunch_code','test preparation course_code']
data=data[column]
data["admitted/rejected"]=target
columns=data.columns
theilu = pd.DataFrame(index=['admitted/rejected'], columns=data.drop("admitted/rejected", axis=1).columns)


for j in range(0,len(column)):
    u = theil_u(data['admitted/rejected'].tolist(),data[columns[j]].tolist())
    theilu.loc[:,columns[j]] = u
theilu.fillna(value=np.nan,inplace=True)
plt.figure(figsize=(15,1))
sns.heatmap(theilu,annot=True,fmt='.3f')
plt.show()
from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.ensemble import RandomForestClassifier
RF=RandomForestClassifier(max_depth=2, random_state=0)

from sklearn.linear_model import LogisticRegression
LR=LogisticRegression(random_state=0)

from sklearn.svm import SVC
svc=SVC(kernel="sigmoid", random_state=0)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(data_old.iloc[:,5:8], target, test_size=0.2, random_state=0)
models=[RF, LR, svc]
for model in models:
    model.fit(x_train, y_train)
    y_pred=model.predict(x_test)
    print(model)
    print('confusion matrix:',"\n",confusion_matrix(y_test, y_pred))
    print('accuracy score:',accuracy_score(y_test, y_pred),"\n\n") 
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

models=[RF, LR, svc]
for model in models:
    model.fit(x_train, y_train)
    y_pred=model.predict(x_test)
    print(model)
    print('confusion matrix:',"\n",confusion_matrix(y_test, y_pred))
    print('accuracy score:',accuracy_score(y_test, y_pred),"\n\n") 