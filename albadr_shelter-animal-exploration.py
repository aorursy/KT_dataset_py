import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import csv as csv
import matplotlib.pyplot as plt
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/train.csv")
df.head()

# Check the outcome possibility
df["OutcomeType"].unique()
# See the distribution of outcome
sns.countplot(x="OutcomeType",data=df)
# See the distribution of outcome subtype
plt.figure(figsize=(17,6))
sns.countplot(x="OutcomeSubtype",data=df)
# See the outcome based on animal type
plt.figure(figsize=(17,6))
sns.countplot(x="AnimalType",hue="OutcomeType",data=df)
# There is NaN and Unknown in Sex
df["SexuponOutcome"]=df["SexuponOutcome"].fillna("Unknown")
# Plot outcome based on sex
sns.countplot(y="OutcomeType",data=df,hue="SexuponOutcome")
sns.countplot(y="SexuponOutcome",data=df,hue="OutcomeType")
#Time series
df["DateTime"] = pd.to_datetime(df["DateTime"]).dt.date
plt.figure(figsize=(17,6))
df["OutcomeType"].groupby(df["DateTime"]).count().plot(kind="line")
#For each outcome
monthGroup=df["DateTime"].groupby(df["OutcomeType"])
plt.subplots(5, 1, figsize=(15, 20), sharex=True)
plt.subplots_adjust( hspace=0.7)
colors = list('rgbcmyk')
for i, (_, g) in enumerate(monthGroup):
    plt.subplot(5,1,i+1)
    plt.title(_)
    g.groupby(df["DateTime"]).count().plot(kind="line", color=colors[i])

#Monthly time series
df_ym=df.DateTime.map(lambda x: x.strftime('%Y-%m'))
df_ym_outcomeGroup = df_ym.groupby(df["OutcomeType"])

plt.subplots(5, 1, figsize=(15, 20), sharex=True)
plt.subplots_adjust( hspace=0.7)
colors = list('rgbcmyk')
for i, (_, g) in enumerate(df_ym_outcomeGroup):
    plt.subplot(5,1,i+1)
    plt.title(_)
    g.groupby(df_ym).count().plot(kind="line", color=colors[i])


#For each outcome
df_dow=pd.to_datetime(df.DateTime).dt.dayofweek
dayinweekGroup=df["DateTime"].groupby(df["OutcomeType"])
plt.subplots(5, 1, figsize=(15, 20), sharex=True)
plt.subplots_adjust( hspace=0.7)
colors = list('rgbcmyk')
for i, (_, g) in enumerate(dayinweekGroup):
    plt.subplot(5,1,i+1)
    plt.title(_)
    g.groupby(df_dow).count().plot(kind="line", color=colors[i])

#Monthly time series
df_ym=df.DateTime.map(lambda x: x.strftime('%d'))
df_ym_outcomeGroup = df_ym.groupby(df["OutcomeType"])

plt.subplots(5, 1, figsize=(15, 20), sharex=True)
plt.subplots_adjust( hspace=0.7)
colors = list('rgbcmyk')
for i, (_, g) in enumerate(df_ym_outcomeGroup):
    plt.subplot(5,1,i+1)
    plt.title(_)
    g.groupby(df_ym).count().plot(kind="line", color=colors[i])


from __future__ import division
def label_age (row):
  if row['AgeuponOutcome'] == "0 years" :
      return 0
  if row['AgeuponOutcome'] == "1 year" :
      return 1
  if row['AgeuponOutcome'] == "2 years" : 
      return 2
  if row['AgeuponOutcome'] == "3 years" : 
      return 3
  if row['AgeuponOutcome'] == "4 years" : 
      return 4
  if row['AgeuponOutcome'] == "5 years" : 
      return 5
  if row['AgeuponOutcome'] == "6 years" : 
      return 6
  if row['AgeuponOutcome'] == "7 years" : 
      return 7
  if row['AgeuponOutcome'] == "8 years" : 
      return 8
  if row['AgeuponOutcome'] == "9 years" : 
      return 9
  if row['AgeuponOutcome'] == "10 years" : 
      return 10
  if row['AgeuponOutcome'] == "11 years" : 
      return 11
  if row['AgeuponOutcome'] == "12 years" : 
      return 12
  if row['AgeuponOutcome'] == "13 years" : 
      return 13
  if row['AgeuponOutcome'] == "14 years" : 
      return 14
  if row['AgeuponOutcome'] == "15 years" : 
      return 15
  if row['AgeuponOutcome'] == "16 years" :
      return 16
  if row['AgeuponOutcome'] == "17 years" :
      return 17
  if row['AgeuponOutcome'] == "18 years" :
      return 18
  if row['AgeuponOutcome'] == "20 years" :
      return 20
  if row['AgeuponOutcome'] == "1 month" :
      return 1/12
  if row['AgeuponOutcome'] == "2 months" :
      return 2/12
  if row['AgeuponOutcome'] == "3 months" :
      return 3/12
  if row['AgeuponOutcome'] == "4 months" :
      return 4/12
  if row['AgeuponOutcome'] == "5 months" :
      return 5/12
  if row['AgeuponOutcome'] == "6 months" :
      return 6/12
  if row['AgeuponOutcome'] == "7 months" :
      return 7/12
  if row['AgeuponOutcome'] == "8 months" :
      return 8/12
  if row['AgeuponOutcome'] == "9 months" :
      return 9/12
  if row['AgeuponOutcome'] == "10 months" :
      return 10/12
  if row['AgeuponOutcome'] == "11 months" :
      return 11/12
  if row['AgeuponOutcome'] == "1 week" :
      return 1/48
  if row['AgeuponOutcome'] == "1 weeks" :
      return 1/48
  if row['AgeuponOutcome'] == "2 weeks" :
      return 2/48
  if row['AgeuponOutcome'] == "3 weeks" :
      return 3/48
  if row['AgeuponOutcome'] == "4 weeks" :
      return 4/48
  if row['AgeuponOutcome'] == "5 weeks" :
      return 5/48
  if row['AgeuponOutcome'] == "1 day" :
      return 1/336
  if row['AgeuponOutcome'] == "2 days" :
      return 2/336
  if row['AgeuponOutcome'] == "3 days" :
      return 3/336
  if row['AgeuponOutcome'] == "4 days" :
      return 4/336
  if row['AgeuponOutcome'] == "5 days" :
      return 5/336
  if row['AgeuponOutcome'] == "6 days" :
      return 6/336
#Convert age string to be float
df["Agecat"]=df.apply (lambda row: label_age (row),axis=1)
df_age_outcomeGroup = df["Agecat"].groupby(df["OutcomeType"])
plt.subplots(5, 1, figsize=(15, 20), sharex=True)
plt.subplots_adjust( hspace=0.7)
colors = list('rgbcmyk')
for i, (_, g) in enumerate(df_age_outcomeGroup):
    ax=plt.subplot(5,1,i+1)
    ax.set_xscale('log')
    plt.title(_)
    g.groupby(df["Agecat"]).count().plot(kind="line", color=colors[i])
# Prepare for training data
ytrain = df["OutcomeType"]
Xtrain = df.drop(["OutcomeType","OutcomeSubtype","AgeuponOutcome","AnimalID","Name"],axis=1)
Xtrain.head()
# Encode categorical data
from sklearn import preprocessing
le_anima = preprocessing.LabelEncoder()
Xtrain.AnimalType = le_anima.fit_transform(Xtrain.AnimalType)
le_sex = preprocessing.LabelEncoder()
Xtrain.SexuponOutcome = le_sex.fit_transform(Xtrain.SexuponOutcome)
le_breed = preprocessing.LabelEncoder()
Xtrain.Breed = le_breed.fit_transform(Xtrain.Breed)
le_color = preprocessing.LabelEncoder()
Xtrain.Color = le_color.fit_transform(Xtrain.Color)
le_out = preprocessing.LabelEncoder()
ytrain = le_out.fit_transform(ytrain)
#Let's see
Xtrain.head()
## Explode date time
xdt=pd.to_datetime(Xtrain.DateTime)
Xtrain["dow"] = xdt.dt.dayofweek
Xtrain["month"] = xdt.dt.month
Xtrain["year"] = xdt.dt.year
Xtrain=Xtrain.drop(["DateTime"],axis=1)
Xtrain.head()
Xtrain=Xtrain.fillna(-1)
from sklearn.ensemble import RandomForestClassifier
# Do random forest
rf = RandomForestClassifier(n_estimators=1000)
rf.fit(Xtrain, ytrain)
# Let's see the train accuracy
tra_score=rf.score(Xtrain, ytrain)
print("Training accuracy ",tra_score)
from sklearn.ensemble import RandomForestClassifier
m=[6000,8000,10000,12000,14000,16000,18000,20000]
train_err=[]
val_err=[]
perm=np.random.permutation(len(Xtrain))
Xtr=Xtrain.iloc[perm[0:20000]]
ytr=ytrain[perm[0:20000]]
Xval=Xtrain.iloc[perm[20001:]]
yval=ytrain[perm[20001:]]
for i in range(8):
    trainSize = m[i]
    perm=np.random.permutation(len(Xtr))
    XtrNow = Xtr.iloc[perm[0:trainSize]]
    ytrNow = ytr[perm[0:trainSize]]
    # Do random forest
    rf = RandomForestClassifier(n_estimators=1000)
    rf.fit(XtrNow, ytrNow)
    # Let's see the train accuracy
    rScore=rf.score(XtrNow, ytrNow)
    vScore=rf.score(Xval, yval)
    train_err.append(rScore)
    val_err.append(vScore)
plt.plot(m, train_err, 'r', m, val_err, 'b')
#Cross validation
from sklearn.cross_validation import KFold
kf = KFold(len(ytrain), n_folds=10, shuffle=True)
foldScore=[]
for train_index, test_index in kf:
    Xtr, X_test = Xtrain.iloc[train_index], Xtrain.iloc[test_index]
    ytr, y_test = ytrain[train_index], ytrain[test_index]
    rf.fit(Xtr, ytr)
    val_acc=rf.score(X_test,y_test)
    tra_acc=rf.score(Xtr,ytr)
    print(val_acc,tra_acc)
    foldScore.append([val_acc,tra_acc])
print ("Over all Fold", np.mean(foldScore, axis=0))
#Get test data
tt = pd.read_csv("../input/test.csv")
tt.head()
IDtest=tt["ID"]
Xtest=tt.drop(["ID","Name"],axis=1)

Xtest.AnimalType = le_anima.fit_transform(Xtest.AnimalType)
Xtest.SexuponOutcome = le_sex.fit_transform(Xtest.SexuponOutcome)
Xtest.Breed = le_breed.fit_transform(Xtest.Breed)
Xtest.Color = le_color.fit_transform(Xtest.Color)
Xtest["AgeCat"] = Xtest.apply (lambda row: label_age (row),axis=1)
xtt=pd.to_datetime(Xtest.DateTime)
Xtest["dow"] = xdt.dt.dayofweek
Xtest["month"] = xdt.dt.month
Xtest["year"] = xdt.dt.year
Xtest=Xtest.drop(["AgeuponOutcome","DateTime"],axis=1)
Xtest.head()
Xtest=Xtest.fillna(-1)
#Get prediction
ytest=rf.predict(Xtest)
ytestproba=rf.predict_proba(Xtest)
yfin=le_out.inverse_transform(ytest)
yprint = pd.DataFrame()
yprint["ID"]=IDtest
yprint["Adoption"] = (yfin=="Adoption").astype(int)
yprint["Died"] = (yfin=="Died").astype(int)
yprint["Euthanasia"] = (yfin=="Euthanasia").astype(int)
yprint["Return_to_owner"] = (yfin=="Return_to_owner").astype(int)
yprint["Transfer"] = (yfin=="Transfer").astype(int)
yprint.to_csv("submit_randomforest.csv",index=False)
f=open("submit_rfproba.csv","w")
filewrite=csv.writer(f)
filewrite.writerow(["ID","Adoption","Died","Euthanasia","Return_to_owner","Transfer"])
for i in range(len(ytestproba)):
    filewrite.writerow([IDtest[i],ytestproba[i,0],ytestproba[i,1],ytestproba[i,2],ytestproba[i,3],ytestproba[i,4]])
f.close()

# Let us try again but without using Breed and Color feature
Xtrain2 = Xtrain.drop(["Breed","Color"],axis=1)
rf2 = RandomForestClassifier(n_estimators=1000)
rf2.fit(Xtrain2, ytrain)
# How is our training accuracy now?
yt_pred2 = rf2.predict(Xtrain2)
np.mean(ytrain==yt_pred2)
#Get prediction and print!
Xtest2 = Xtest.drop(["Breed","Color"],axis=1);
ytest2 = rf2.predict(Xtest2)
ytestproba2 = rf2.predict_proba(Xtest2)
yfin2 = le_out.inverse_transform(ytest2)

f=open("submit_rfproba2.csv","w")
filewrite=csv.writer(f)
filewrite.writerow(["ID","Adoption","Died","Euthanasia","Return_to_owner","Transfer"])
for i in range(len(ytestproba2)):
    filewrite.writerow([IDtest[i],ytestproba2[i,0],ytestproba2[i,1],ytestproba2[i,2],ytestproba2[i,3],ytestproba2[i,4]])
f.close()

