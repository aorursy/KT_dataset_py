import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv("../input/diabetes/diabetes.csv")
pd.set_option("max_columns",300)
df
df.info()
for i in df.columns:
    plt.scatter(df["Outcome"],df[i])
    plt.title(i)
    plt.show()
sns.pairplot(df)
df[df["SkinThickness"]==0]
df["SkinThickness"].mean()
df["Insulin"].mean()
df.corr()
plt.plot(df["Age"],df["Pregnancies"],"d")
for i in df.columns:
    sns.distplot(df[i])
    plt.title(i)
    plt.show()
for i in df.columns:
    sns.stripplot(df["Outcome"],df[i],jitter=True)
    plt.title(i)
    plt.show()
# replacing zero values with the mean of the column
df['BMI'] = df['BMI'].replace(0,df['BMI'].mean())
df['BloodPressure'] = df['BloodPressure'].replace(0,df['BloodPressure'].mean())
df['Glucose'] = df['Glucose'].replace(0,df['Glucose'].mean())
df['Insulin'] = df['Insulin'].replace(0,df['Insulin'].mean())
df['SkinThickness'] = df['SkinThickness'].replace(0,df['SkinThickness'].mean())
for i in df.columns:
    sns.distplot(df[i])
    plt.title(i)
    plt.show()
x=df.iloc[:,:8]
y=df["Outcome"]
x,y
from sklearn.preprocessing import StandardScaler 
scalar = StandardScaler()
X_scaled = scalar.fit_transform(x)
X_scaled
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size= 0.50, random_state = 355)
x_train
y_train
from sklearn.linear_model  import LogisticRegression
log=LogisticRegression()
log.fit(x_train,y_train)
y_pred = log.predict(x_test)
y_pred
y_prob=log.predict_proba(x_test)
y_prob=y_prob[:,1]
from sklearn.metrics import confusion_matrix , classification_report
print(classification_report(y_test,y_pred))
con_mat=confusion_matrix(y_test,y_pred)
con_mat
true_positive = con_mat[0][0]
false_positive = con_mat[0][1]
false_negative = con_mat[1][0]
true_negative = con_mat[1][1]
recall=true_positive/(true_positive+false_negative)
recall
precision=true_positive/(true_positive+false_positive)
precision
f1_score=2*recall*precision/(recall+precision)
f1_score
