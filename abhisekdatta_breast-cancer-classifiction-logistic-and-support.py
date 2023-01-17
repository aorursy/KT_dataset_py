import pandas as pd 
df=pd.read_csv("/kaggle/input/breast-cancer-prediction-dataset/Breast_cancer_data.csv")

df.head()
#Checking missing values 

df.isnull().sum()
import seaborn as sns
sns.boxplot(df["mean_radius"])
df["mean_radius"].describe()
max_rad=(15.78+(1.5*(15.78-11.7)))

max_rad
df["mean_radius"][df["mean_radius"]>max_rad]=max_rad
sns.boxplot(df["mean_radius"])
# Ditribution and PDF and CDF

#sns.displot(df["mean_radius"])
sns.distplot(df["mean_radius"])
#sns.ecdfplot(df["mean_radius"])
sns.boxplot(df["mean_texture"])
df["mean_texture"].describe()
max_text=(21.8+(1.5*(21.8-16.17)))

max_text
df["mean_texture"][df["mean_texture"]>max_text]=max_text
sns.boxplot(df["mean_texture"])
# Distribution PDF and CDF

#sns.displot(df["mean_texture"])
sns.distplot(df["mean_texture"])
#sns.ecdfplot(df["mean_texture"])
sns.boxplot(df["mean_perimeter"])
df["mean_perimeter"].describe()
max_peri=(104.1+(1.5*(104.1-75.17)))

max_peri
df["mean_perimeter"][df["mean_perimeter"]>max_peri]=max_peri
sns.boxplot(df["mean_perimeter"])
# Distribution, PDF and CDF 

#sns.displot(df["mean_perimeter"])
sns.distplot(df["mean_perimeter"])
#sns.ecdfplot(df["mean_perimeter"])
sns.boxplot(df["mean_area"])
df["mean_area"].describe()
max_area=(782.7+(1.5*(782.7-420.3)))

max_area
df["mean_area"][df["mean_area"]>max_area]=max_area
sns.boxplot(df["mean_area"])
# Distribution PDF and CDF

#sns.displot(df["mean_area"])
sns.distplot(df["mean_area"])
#sns.ecdfplot(df["mean_area"])
sns.boxplot(df["mean_smoothness"])
df["mean_smoothness"].describe()
max_smooth=(0.1053+(1.5*(.1053-.08637)))

max_smooth
df["mean_smoothness"][df["mean_smoothness"]>max_smooth]=max_smooth
sns.boxplot(df["mean_smoothness"])
# Distribution,PDF and CDF

#sns.displot(df["mean_smoothness"])
sns.distplot(df["mean_smoothness"])
#sns.ecdfplot(df["mean_smoothness"])
X=df.iloc[:,[0,1,2,3,4]]

X.head()
y=df.iloc[:,[5]]

y.head()
## Relationship between features 

sns.pairplot(X)
from sklearn.feature_selection import VarianceThreshold

var=VarianceThreshold(threshold=0.8)

var.fit(X)
var.get_support()
var.get_support().sum()
X.columns[var.get_support()]
const_columns=[column for column in X.columns

              if column not in X.columns[var.get_support()]]

print (const_columns)
X=X.drop("mean_smoothness",axis=1)
X.head()
from sklearn.ensemble import ExtraTreesClassifier

imp=ExtraTreesClassifier()

imp.fit(X,y)
imp.feature_importances_
df.features=pd.DataFrame(X.columns,columns=["Features"])

df.importances=pd.DataFrame(imp.feature_importances_,columns=["Importances"])

after_concat=pd.concat([df.features,df.importances],axis=1)

after_concat.nlargest(4,"Importances")
sns.heatmap(X.corr(),annot=True)
def correlation(dataset, threshold): 

    col_corr = set()                                 # Set of all the names of correlated columns

    corr_matrix = dataset.corr()

    for i in range(len(corr_matrix.columns)): 

      for j in range(i): 

        if abs(corr_matrix.iloc[i, j]) > threshold:  # we are interested in absolute coeff value 

          colname = corr_matrix.columns[i]           # getting the name of column

          col_corr.add(colname) 

    return col_corr
correlation(X,0.6)
X=X.drop("mean_area",axis=1)
X.head()
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25,random_state=100)

print(X_train.shape)

print(X_test.shape)
from sklearn.linear_model import LogisticRegression

logreg=LogisticRegression()

logreg.fit(X_train,y_train)
y_predlog_test=logreg.predict(X_test)

y_predlog_test
y_predlog_train=logreg.predict(X_train)

y_predlog_train
from sklearn.metrics import confusion_matrix

cnf_test=confusion_matrix(y_test,y_predlog_test)

cnf_test
cnf_train=confusion_matrix(y_train,y_predlog_train)

cnf_train
from sklearn import metrics

print ("Accuracy for test", metrics.accuracy_score(y_test,y_predlog_test))

print ("Accuracy for train", metrics.accuracy_score(y_train,y_predlog_train))
print ("Precision for test",metrics.precision_score(y_test,y_predlog_test))

print ("Precision for train",metrics.precision_score(y_train,y_predlog_train))
print ("Recall for test",metrics.recall_score(y_test,y_predlog_test))

print ("Recall for train",metrics.recall_score(y_train,y_predlog_train))
from sklearn.svm import SVC

svc=SVC()

svc.fit(X_train,y_train)
y_predsvc_test=svc.predict(X_test)

y_predsvc_test
y_predsvc_train=svc.predict(X_train)

y_predsvc_train
cnfsvc_test=metrics.confusion_matrix(y_test,y_predsvc_test)

cnfsvc_test
cnfsvc_train=metrics.confusion_matrix(y_train,y_predsvc_train)

cnfsvc_train
print ("Accuracy for test", metrics.accuracy_score(y_test,y_predsvc_test))

print ("Accuracy for train", metrics.accuracy_score(y_train,y_predsvc_train))