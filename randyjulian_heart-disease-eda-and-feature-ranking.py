import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
heart=pd.read_csv("../input/heart.csv")
heart.head()
ax= sns.countplot(x="target", data=heart, palette="Set2")

plt.title("Target Count")

print("Percentage of Patients with heart disease: {:.2f}".format(len(heart[heart.target ==1])*100/len(heart.target)),"%")

print("Percentage of Patients without heart disease: {:.2f}".format(len(heart[heart.target ==0])*100/len(heart.target)),"%")

plt.figure(figsize=(20,10))

plt.subplot(2,1,1)

sns.countplot(x="age",data=heart,palette='Reds')

plt.title("Age distribution")



plt.subplot(2,1,2)

sns.countplot(x="age",hue="target",data=heart,palette='Set2')

plt.title("Age distribution by Target")



plt.show()
plt.figure(figsize=(15,6))

plt.subplot(1,2,1)

ax= sns.countplot(x="sex", data=heart, palette="Set2")

plt.title("Gender Count")

plt.xlabel("Sex (1 = male; 0 = female)")



plt.subplot(1,2,2)

sns.countplot(x="sex",hue="target",data=heart,palette='Set2')

plt.title("Gender distribution by Target")

plt.xlabel("Sex (1 = male; 0 = female)")

plt.show()



print("Percentages by Gender")

print("Male: {:.2f}".format(len(heart[heart.sex ==1])*100/len(heart.target)),"%")

print("Female: {:.2f}".format(len(heart[heart.sex ==0])*100/len(heart.target)),"%")

plt.figure(figsize=(15,6))

plt.subplot(1,2,1)

ax= sns.countplot(x="cp", data=heart, palette="Set2")

plt.title("Chest Pain type count")

plt.xlabel("Chest Pain Type")

_0 = mpatches.Patch(color="mediumaquamarine", label='0: typical angina')

_1 = mpatches.Patch(color="sandybrown", label='1: atypical angina')

_2 = mpatches.Patch(color="cornflowerblue", label='2: non-anginal pain')

_3 = mpatches.Patch(color="orchid", label='3: asymptomatic')

plt.legend(handles=[_0,_1,_2,_3])



plt.subplot(1,2,2)

sns.countplot(x="cp",hue="target",data=heart,palette='Set2')

plt.title("Chest Pain type by Target")

plt.xlabel("Chest Pain type")

plt.show()



print("Chest Pain type Percentages:")

print("0 - typical angina: {:.2f}".format(len(heart[heart.cp ==0])*100/len(heart.target)),"%")

print("1 - atypical angina: {:.2f}".format(len(heart[heart.cp ==1])*100/len(heart.target)),"%")

print("2 - non-anginal pain: {:.2f}".format(len(heart[heart.cp ==2])*100/len(heart.target)),"%")

print("3 - asymptomatic: {:.2f}".format(len(heart[heart.cp ==3])*100/len(heart.target)),"%")

plt.figure(figsize=(20,10))

plt.subplot(2,1,1)

sns.countplot(x="trestbps",data=heart,palette='Reds')

plt.title("Resting blood pressure distribution ")



plt.subplot(2,1,2)

sns.countplot(x="trestbps",hue="target",data=heart,palette='Set2')

plt.title("Resting blood pressure distribution by Target")

plt.show()

plt.figure(figsize=(20,8))



sns.distplot(heart['chol'],kde=False)

plt.title("Serum cholestoral in mg/dl distribution ")

plt.show()
plt.figure(figsize=(20,6))

plt.subplot(1,3,1)

ax= sns.countplot(x="fbs", data=heart, palette="Set2")

plt.title("Fasting Blood Sugar Count")

plt.xlabel("fbs (1 = true; 0 = false)")



plt.subplot(1,3,2)

#plt.figure(figsize=(20,8))

sns.countplot(x="fbs",hue="target",data=heart,palette='Set2')

plt.title("FBS distribution by Target")

plt.xlabel("FBS > 120 mg/dl (1 = True; 0 = False)")



plt.subplot(1,3,3)

#plt.figure(figsize=(20,8))

sns.countplot(x="fbs",hue="sex",data=heart,palette='Set2')

plt.title("FBS distribution by Gender")

plt.xlabel("FBS > 120 mg/dl (1 = True; 0 = False)")

plt.show()



print("Percentages of FBS>120 mg/dl")

print("FBS >120 mg/dl: {:.2f}".format(len(heart[heart.fbs ==1])*100/len(heart.target)),"%")

print("FBS <=120 mg/dl: {:.2f}".format(len(heart[heart.fbs ==0])*100/len(heart.target)),"%")

plt.figure(figsize=(15,6))

plt.subplot(1,2,1)

ax= sns.countplot(x="restecg", data=heart, palette="Set2")

plt.title("resting electrocardiographic results count")

plt.xlabel("resting electrocardiographic results")

_0 = mpatches.Patch(color="mediumaquamarine", label='0: normal')

_1 = mpatches.Patch(color="sandybrown", label='1: having ST-T wave abnormality')

_2 = mpatches.Patch(color="cornflowerblue", label='2: probable or definite left ventricular hypertrophy')

plt.legend(handles=[_0,_1,_2])



plt.subplot(1,2,2)

sns.countplot(x="restecg",hue="target",data=heart,palette='Set2')

plt.title("resting electrocardiographic results by Target")

plt.xlabel("resting electrocardiographic results")

plt.show()





print("resting electrocardiographic results Percentages:")

print("0 - normal: {:.2f}".format(len(heart[heart.restecg ==0])*100/len(heart.target)),"%")

print("1 - having ST-T wave abnormality: {:.2f}".format(len(heart[heart.restecg ==1])*100/len(heart.target)),"%")

print("2 - probable or definite left ventricular hypertrophy: {:.2f}".format(len(heart[heart.restecg ==2])*100/len(heart.target)),"%")

plt.figure(figsize=(20,8))



sns.distplot(heart['thalach'],kde=False)

plt.title("maximum heart rate achieved distribution ")

plt.show()
plt.figure(figsize=(20,6))

plt.subplot(1,2,1)

ax= sns.countplot(x="exang", data=heart, palette="Set2")

plt.title("exercise induced angina")

plt.xlabel("exang (1 = true; 0 = false)")



plt.subplot(1,2,2)

#plt.figure(figsize=(20,8))

sns.countplot(x="exang",hue="target",data=heart,palette='Set2')

plt.title("exercise induced angina by Target")

plt.xlabel("exercise induced angina (1 = yes; 0 = no)")



plt.show()



print("Percentages of exercise induced angina ")

print("exang True: {:.2f}".format(len(heart[heart.exang ==1])*100/len(heart.target)),"%")

print("exang False: {:.2f}".format(len(heart[heart.exang ==0])*100/len(heart.target)),"%")

plt.figure(figsize=(20,8))



sns.distplot(heart['oldpeak'],kde=False)

plt.title("ST depression induced by exercise relative to rest distribution")

plt.show()
plt.figure(figsize=(15,6))

plt.subplot(1,2,1)

ax= sns.countplot(x="slope", data=heart, palette="Set2")

plt.title("the slope of the peak exercise ST segment count")

plt.xlabel("slope")

_0 = mpatches.Patch(color="mediumaquamarine", label='0: upsloping')

_1 = mpatches.Patch(color="sandybrown", label='1: flat')

_2 = mpatches.Patch(color="cornflowerblue", label='2: downsloping')

plt.legend(handles=[_0,_1,_2])



plt.subplot(1,2,2)

sns.countplot(x="slope",hue="target",data=heart,palette='Set2')

plt.title("the slope of the peak exercise ST segment by Target")

plt.xlabel("slope")

plt.show()





print("resting electrocardiographic results Percentages:")

print("0 - upsloping: {:.2f}".format(len(heart[heart.slope ==0])*100/len(heart.target)),"%")

print("1 - flat: {:.2f}".format(len(heart[heart.slope ==1])*100/len(heart.target)),"%")

print("2 - downsloping: {:.2f}".format(len(heart[heart.slope ==2])*100/len(heart.target)),"%")

plt.figure(figsize=(15,6))

plt.subplot(1,2,1)

ax= sns.countplot(x="ca", data=heart, palette="Set2")

plt.title("number of major vessels (0-3) colored by flourosopy Count")

plt.xlabel("number of major vessels (0-3) colored by flourosopy")



plt.subplot(1,2,2)

sns.countplot(x="ca",hue="target",data=heart,palette='Set2')

plt.title("number of major vessels (0-3) colored by flourosopy by Target")

plt.xlabel("number of major vessels (0-3) colored by flourosopy")

plt.show()

train_x,test_x, train_y, test_y= train_test_split(heart[heart.columns[~heart.columns.isin(["target"])]]

                                                  , heart.target, test_size=0.2, random_state=123)
rf=RandomForestClassifier(n_estimators = 2000,random_state=123)

rf.fit(train_x,train_y)

print("Random Forest Accuracy : {:.2f}%".format(rf.score(test_x,test_y)*100))
gbm=GradientBoostingClassifier(learning_rate=0.25,n_estimators = 1000,random_state=123)

gbm.fit(train_x,train_y)

print("Gradient Boosting Accuracy : {:.2f}%".format(rf.score(test_x,test_y)*100))
xgb = XGBClassifier(max_depth = 8)

xgb.fit(train_x,train_y)

target = xgb.predict(test_x)

print("XGBoost Accuracy : {:.2f}%".format(accuracy_score(test_y, target)*100))
def feature_rank(train,model,n):

    cols=train.columns

    col_indices = np.argsort(model.feature_importances_)[::-1]

    feature_ranking = pd.DataFrame(columns=['index', 'variable', 'importance'])

    gb_top_col_list = []

    for f in range(n): 

        z = pd.DataFrame([col_indices[f],cols[col_indices[f]],model.feature_importances_[col_indices[f]]]).transpose()

        z.columns = ['index', 'variable', 'importance']

        gb_top_col_list.append(cols[col_indices[f]])

        feature_ranking = feature_ranking.append(z)

    return feature_ranking
feature_rank(train_x,rf,10)
feature_rank(train_x,gbm,10)
feature_rank(train_x,xgb,10)