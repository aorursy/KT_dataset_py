import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

import seaborn as sns
Data=pd.read_csv("../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")



Data.rename(columns ={'residual sugar':'residual_sugar'},inplace=True)
Data.head()
Data.info()
print("The Number Of Rows and Columns:",Data.shape)

print("Data Size:",Data.size)
Data.isnull().sum()
Data.describe().T
sns.countplot(Data['quality']);
sns.distplot(Data['quality']);
def plot_corr(Data, size=15):

    corr = Data.corr()

    fig, ax = plt.subplots(figsize=(size, size))

    ax.matshow(corr)

    plt.xticks(range(len(corr.columns)), corr.columns)

    plt.yticks(range(len(corr.columns)), corr.columns)
plot_corr(Data)
plt.figure(figsize= (3,3))

sns.boxplot(x=Data['fixed acidity'],color='orange')

plt.figure(figsize= (3,3))

sns.boxplot(x=Data['pH'],color='orange')

plt.figure(figsize= (3,3))

sns.boxplot(x=Data['alcohol'],color='orange')

plt.figure(figsize= (3,3))

sns.boxplot(x=Data['sulphates'],color='orange')

plt.figure(figsize= (3,3))

sns.boxplot(x=Data['citric acid'],color='orange')
Data.skew()
def func(row):

    if row["quality"] > 6.5:

        return("Good")

    else:

        return("Bad")

Data["quality_change"]=Data.apply(func,axis=1)

Data.groupby('quality_change')['quality'].sum().plot.pie(autopct='%1.2f%%');
sns.scatterplot(x=Data['pH'],y=Data['alcohol'],hue=Data['quality_change']);
import scipy.stats as stats



H0="Alcohol does have an impact on the quality of wine as the P_value is greater than 0.05 :"

Ha="Alcohol does NOT have any significant impact on the quality of wine, as the P_value is less than 0.05 :"

Good_quality_Wine_OH=np.array(Data[Data.quality_change =='Good'].alcohol)

Bad_quality_wine_OH=np.array(Data[Data.quality_change =='Bad'].alcohol)

t, p_value  = stats.ttest_ind(Good_quality_Wine_OH,Bad_quality_wine_OH,axis=0)

p_value

if p_value < 0.5:

    print(Ha,format(p_value))

else:

    print(H0,format(p_value))
Ho="Residual Sugars have a significant role in quality of alchohol"

Ha="Residual Sugars do not have significance on the quality of alcohol" 



Good_Quality_Wine_Sugar=np.array(Data[Data.quality_change=='Good'].residual_sugar)

Bad_Quality_Wine_Sugar=np.array(Data[Data.quality_change=='Bad'].residual_sugar)



f_stat,p_value=stats.f_oneway(Good_Quality_Wine_Sugar,Bad_Quality_Wine_Sugar)

if p_value < 0.05:

    print(Ha,"since P_value is less than 0.05 with a value {}:".format(p_value))

else:

    print(Ho,"since p_value is greater than 0.05 with a value of:{}".format(p_value))

    

sns.scatterplot(x=Data['alcohol'],y=Data['quality'],hue=Data['residual_sugar']);
Data_For_Linear=Data.drop(['quality_change'],axis=1)

Data_For_Linear.head()
X_Linear=Data_For_Linear.drop(['quality'],axis=1)

y_Linear=Data_For_Linear['quality']
X_Linear_train,X_Linear_test,y_Linear_train,y_Linear_test=train_test_split(X_Linear,y_Linear,test_size=0.3, random_state=1)
from sklearn.linear_model import LinearRegression

qual_linear=LinearRegression()
qual_linear.fit(X_Linear_train,y_Linear_train)
#qual_linear.coef_=pd.DataFrame(qual_linear.coef_,X_Linear.columns,columns=['Coefficients'])

qual_linear.coef_

for idx,col_name in enumerate(X_Linear_train.columns):

    print("The coefficient for {} is {}".format(col_name,qual_linear.coef_[idx]))
qual_linear.intercept_
Linear_Pred=qual_linear.predict(X_Linear_test)
df = pd.DataFrame({"Actual": y_Linear_test, "Predicted": Linear_Pred})

Top_25= df.head(25)
Top_25.plot(kind='bar',figsize=(15,10))

plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')

plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
qual_linear.score(X_Linear_train,y_Linear_train)
qual_linear.score(X_Linear_test,y_Linear_test)
from sklearn import metrics



print('Mean Absolute Error:', metrics.mean_absolute_error(y_Linear_test,Linear_Pred))  

print('Mean Squared Error:', metrics.mean_squared_error(y_Linear_test,Linear_Pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_Linear_test, Linear_Pred)))
Data_for_Logistic=Data.copy()

Data_for_Logistic.drop(['quality'],axis=1,inplace=True)

Data_for_Logistic.head()
X=Data_for_Logistic.drop(['quality_change'],axis=1)

y=Data_for_Logistic['quality_change']
X_Train,X_test,y_Train,y_test=train_test_split(X,y,test_size=0.3, random_state=1)
X_test.head()
LOG_REG=LogisticRegression(solver="liblinear")

LOG_REG.fit(X_Train,y_Train)

for idx, col_name in enumerate(X_Train.columns):

    print("The coefficient for {} is {}".format(col_name, LOG_REG.coef_[0][idx]))
LOG_REG.intercept_
y_pred_log=LOG_REG.predict(X_test)



from sklearn.metrics import accuracy_score

print(accuracy_score(y_pred_log, y_test)*100)
from sklearn import metrics

cm=metrics.confusion_matrix(y_test,y_pred_log,labels=["Good", "Bad"])

cm
from sklearn.metrics import classification_report,confusion_matrix,mean_squared_error



print(classification_report(y_test,y_pred_log))