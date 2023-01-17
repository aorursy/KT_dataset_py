import numpy as np
import pandas as pd 
import seaborn as sns
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold


from sklearn.neighbors import LocalOutlierFactor

import warnings
warnings.simplefilter(action = "ignore") 


pd.set_option('display.max_rows', None)

pd.set_option('display.max_columns', None)
diabetes= pd.read_csv("../input/diabetes/diabetes.csv")
df=diabetes.copy()
df.head()
### VERY IMPORTANT INFORMATION !!!!

### There are zero values which are filled with NA based on this article:
### https://www.sciencedirect.com/science/article/pii/S2352914816300016#bib65  --> [66]

### In this article, there is an explanation :  "In 2001, 376 of 786 observations in the PID dataset were shown to
### lack experimental validity [65] because for some attributes, the value of zero was recorded in place of missing
### experimental observations"
df.info()   #768 observations
df.isnull().sum()  #No NA values
df.shape   # dataset dimension : 768-9 
df.groupby('Outcome').size()  # It isn't unbalanced
sns.countplot(diabetes['Outcome'],label="Count");
#Table 1
sns.pairplot(df);
#Table 1 interpretations
# If pregnancy is more than ..., outcome is 1
# If SkınThickness is more than..., outcome is 1
# If DiabetesPedigreeFunction is more than ..., outcome is 1
# If Age is more than ..., outcome is 1
df.groupby("Outcome")["Pregnancies"].describe().T   # If pregnancy is more than 13, the outcome is 1
df.groupby("Outcome")["BloodPressure"].describe().T   # If BloodPressure is more than 114, the outcome is 0
df.groupby("Outcome")["SkinThickness"].describe().T  # If SkınThickness is more than 60, the outcome is 1
df.groupby("Outcome")["DiabetesPedigreeFunction"].describe().T  # If DiabetesPedigreeFunction is more than 2.329, outcome is 1
                                                                # # If DiabetesPedigreeFunction is less than 0.88, outcome is 0
df.groupby("Outcome")["Age"].describe().T      # If Age is more than 70, outcome is 0
df.groupby("Outcome")["BloodPressure"].mean()
#These are not real skewness values because of additional zeros

df.skew(axis = 0) # Skewness (*** "BloodPressure", "Insulin", "DiabetesPedigreeFunction","Age" ***)
df.describe().T
df.corr()
sns.heatmap(df.corr(), xticklabels=df.columns, yticklabels=df.columns); 
# ZEROS IN OBSERVATIONS

print("Pregnancies the value of zero-->",df[df.loc[:,"Pregnancies"].isin([0])].shape[0])  #111 observations
print("Glucose the value of zero-->", df[df.loc[:,"Glucose"].isin([0])].shape[0])  #5 observations
print("BloodPressure the value of zero-->",df[df.loc[:,"BloodPressure"].isin([0])].shape[0])  #35 observations
print("SkinThickness the value of zero-->",df[df.loc[:,"SkinThickness"].isin([0])].shape[0])  #227 observations
print("Insulin the value of zero-->",df[df.loc[:,"Insulin"].isin([0])].shape[0])  #374 observations
print("BMI the value of zero-->",df[df.loc[:,"BMI"].isin([0])].shape[0])  #11 observations
print("DiabetesPedigreeFunction the value of zero-->",df[df.loc[:,"DiabetesPedigreeFunction"].isin([0])].shape[0])  # no 0 value
print("Age-->",df[df.loc[:,"Age"].isin([0])].shape[0])  # no 0 value
# Pregnancies zero values make sense.
# Glucose:
# According to this article : https://www.journalagent.com/scie/pdfs/KEAH_1_2_10_12.pdf, "Plasma glucose concentration 
# after 2 h in an OGTT" can't be zero. Therefore, these 0 values seems to be NA values.
df[df["Glucose"]==0]
# Replace Zero Values with NaN

df['Glucose'] = df['Glucose'].replace(0, np.nan)

df[df["Glucose"]==0]
# BloodPressure
df[df["BloodPressure"]==0].head(3)
df["BloodPressure"] = df["BloodPressure"].replace(0, np.nan)
# SkinThickness
df["SkinThickness"] = df["SkinThickness"].replace(0, np.nan)
#Insulin
df["Insulin"] = df["Insulin"].replace(0, np.nan)
#BMI
df["BMI"] = df["BMI"].replace(0, np.nan)
### CONTROL  --> It is equal to the specified in that article. (376 missing value)
df[df.isnull().any(axis=1)].shape
df1=df.copy()  
df1.isnull().any(axis=1).sum()
df2=df1.dropna()
df2=df2.reset_index(drop=True)
print(df2.shape)
df2.head(2)
# Balance of the data isn't changed so much

print("original dataset",diabetes.groupby('Outcome').size())
print("new df outcome",df2.groupby('Outcome').size())
#df3   (The "Pregnancies" values which are more than 9 are replaced by 9.)
df3=df2.copy()
df3.shape
sns.boxplot(df2['Pregnancies'])
Q3= df2['Pregnancies'].quantile(0.75)
Q1=df2['Pregnancies'].quantile(0.25)
IQR=Q3-Q1
upper=Q3+IQR
print("Q1-->",Q1,"Q3-->",Q3,"upper-->",upper)
df2['Pregnancies'].value_counts()
sns.countplot(df3['Pregnancies'],label="Count");
df3[df3['Pregnancies']>10].shape
16/392   # %4 of the women has more than 10 times pregnant
df3.loc[df3['Pregnancies']>10, 'Pregnancies']=10
df3[df3['Pregnancies']>10].shape
df3.groupby("Outcome")["Pregnancies"].describe().T
df4=df3.copy()
df4.head()
print(df4['Pregnancies'].skew())
print(df4['Glucose'].skew())
print(df4['BloodPressure'].skew())
print(df4['SkinThickness'].skew())
print(df4['Insulin'].skew())
print(df4['BMI'].skew())
print(df4['DiabetesPedigreeFunction'].skew())
print(df4['Age'].skew())

# The max value is 198. If "Glucose" values are equal to 200 or more, the outcome qill be 1.

# According to Wikipedia;
# Blood plasma glucose between 7.8 mmol/L (140 mg/dL) and 11.1 mmol/L (200 mg/dL) indicate 
# "impaired glucose tolerance", and levels at or above 11.1 mmol/L at 2 hours confirm a 
# diagnosis of diabetes.
# https://en.wikipedia.org/wiki/Glucose_tolerance_test#:~:text=Blood%20plasma%20glucose%20between%207.8,confirm%20a%20diagnosis%20of%20diabetes.
df4["Glucose"].describe().T 
sns.distplot(df4['Glucose'], hist=False);
print(df4['Glucose'].skew())
df4["BloodPressure"].describe().T
df4.groupby("Outcome")["BloodPressure"].describe().T    #diastolic blood pressure: 
                                                        # The lowest pressure when your heart relaxes between beats.
                                                        # http://www.bloodpressureuk.org/BloodPressureandyou/Thebasics/Bloodpressurechart
sns.distplot(df4['BloodPressure'], hist=False);
print(df4['BloodPressure'].skew())
# Outliers
print(df4[df4["BloodPressure"]<40].shape)
print(df4[df4["BloodPressure"]>100].shape)
df4[df4["BloodPressure"]<40]
df4[df4["BloodPressure"]>100]
df4.loc[df4['BloodPressure']<40, 'BloodPressure']=40
df4.loc[df4['BloodPressure']>100, 'BloodPressure']=100
#df5
df5=df4.copy()

df5["SkinThickness"].describe().T
df5.groupby("Outcome")["SkinThickness"].describe().T
sns.distplot(df5['SkinThickness'], hist=False);
print(df5['SkinThickness'].skew())
sns.boxplot(df5['SkinThickness'])
df5["Insulin"].describe().T
df5.groupby("Outcome")["Insulin"].describe().T
sns.distplot(df5['Insulin'], hist=False);
print(df5['Insulin'].skew())
sns.distplot((df5['Insulin']), hist=False)
df5['Insulin']=np.log(df5['Insulin'])
#df6
df6=df5.copy()
df6["BMI"].describe().T
df6.groupby("Outcome")["BMI"].describe().T
sns.distplot(df6['BMI'], hist=False);
print(df6['BMI'].skew())
df6["DiabetesPedigreeFunction"].describe().T
df6.groupby("Outcome")["DiabetesPedigreeFunction"].describe().T
sns.distplot(df6['DiabetesPedigreeFunction'], hist=False);
print(df6['DiabetesPedigreeFunction'].skew())
df6['DiabetesPedigreeFunction']=np.log(df6['DiabetesPedigreeFunction'])
#df7
df7=df6.copy()
df7.groupby("Outcome")["Age"].describe().T
sns.distplot(df7['Age'], hist=False);
print(df7['Age'].skew())
df7['Age']=np.log(df7['Age'])
#LOC
df8=df7.copy()
clf=LocalOutlierFactor(n_neighbors=20, contamination=0.1)
clf.fit_predict(df8)
df8_scores= clf.negative_outlier_factor_
np.sort(df8_scores)[0:10]
pd.DataFrame(np.sort(df8_scores)).plot(stacked=True,xlim=[0,20], style=".-");

df8.loc[df8_scores< np.sort(df8_scores)[6]]
df8=df8.loc[df8_scores> np.sort(df8_scores)[6]]
df8.shape
#feature generation
df11=df8.copy()
df11["Pregnancies/Age"]=df11["Pregnancies"]/df11["Age"]


df11["Insulin/Glucose"]=df11["Insulin"]/df11["Glucose"]


#Regression
y = df11["Outcome"]
X = df11.drop(["Outcome"], axis = 1)
log_model = LogisticRegression().fit(X,y)
y_pred = log_model.predict(X)
print(accuracy_score(y, y_pred))
print(classification_report(y, y_pred))

y = df11["Outcome"]
X = df11.drop(["Outcome"], axis = 1)



models = []
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('RF', RandomForestClassifier()))
models.append(('SVM', SVC(gamma='auto')))
models.append(('XGB', GradientBoostingClassifier()))
models.append(("LightGBM", LGBMClassifier()))

# evaluate each model in turn
results = []
names = []



for name, model in models:
    
        kfold = KFold(n_splits = 10, random_state = 123456)
        cv_results = cross_val_score(model, X, y, cv = 10, scoring= "accuracy")
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
        
# Feature Importance



Importance = pd.DataFrame({'Importance':GradientBoostingClassifier().fit(X, y).feature_importances_*100}, 
                          index = X.columns)


Importance.sort_values(by = 'Importance', 
                       axis = 0, 
                       ascending = True).plot(kind = 'barh', 
                                              color = 'r', figsize=(14,12))

plt.xlabel('Variable Importance')
plt.gca().legend_ = None


# References
# https://www.sciencedirect.com/topics/medicine-and-dentistry/diabetes-mellitus
# https://www.sciencedirect.com/science/article/pii/S2352914816300016#bib65