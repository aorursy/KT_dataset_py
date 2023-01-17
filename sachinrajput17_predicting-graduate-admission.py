import numpy as np 

import pandas as pd 



%matplotlib inline

import matplotlib.pyplot as plt # Visualization 

import seaborn as sns

plt.style.use('fivethirtyeight')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import warnings      # to ignore warnings

warnings.filterwarnings("ignore")





path="/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv"

df=pd.read_csv(path)

#df1=pd.read_csv(path1)

df.head()  # Top 5 rows

df.tail()  # Bottom 5 rows

df.sample(5)  # Random 5 rows

df.sample(5) # random fractional numbers rows  of total no of rows
data=df.copy()
import pandas_profiling as pp

data=df.copy()

report=pp.ProfileReport(data, title='Pandas Profiling Report')  # overview and quick data analysis

report
print("Columns of the data are:",df.columns)


df=df.rename(columns={'Serial No.':'SerialNo', 'GRE Score':'GRE', 'TOEFL Score':'TOEFL',

                      'University Rating':'UniversityRating','LOR ':'LOR','Chance of Admit ':'ChanceOfAdmit'})

df.columns
#Drop the column "Serial No." 

df=df.drop(columns="SerialNo")
df.isnull().sum()
df.info()
df.isnull().values.any() # check the null values in whole of the data set if any.

missing_data=df.isnull()

for column in missing_data.columns.values.tolist():

    print(column)

    print(missing_data[column].value_counts())
def missing_percentage(data):

    total=data.isnull().sum().sort_values(ascending=False)

    percent=np.round(total/len(data)*100,2)

    return pd.concat([total,percent],axis=1,keys=["Total","Percent"])

missing_percentage(df)
#check any duplicates data in dataframe.

df.duplicated().any()
print("Shape of the data",df.shape)
#statistical summary of the data

df.describe()
# Groupby the data by "University rating".

df.groupby("UniversityRating").mean()
print(" Minimum requirements for more than 85% chance to get admission.\n",df[(df['ChanceOfAdmit']>0.85)].min())

df.pivot_table(values=['GRE','TOEFL'],index=['UniversityRating'],columns='Research',aggfunc=np.median)

plt.figure(figsize=(15,15))

df['ChanceOfAdmit'].value_counts().plot.bar()

plt.show()
#relashionship between the variables of the data in scatter form.

pd.plotting.scatter_matrix(df,figsize=(15,20)) # Scatter matrix for the data.

plt.show()
sns.pairplot(df,hue="ChanceOfAdmit")

plt.show()
df.hist(figsize=(10,10),edgecolor="k")

plt.tight_layout()

plt.show()
plt.figure(figsize=(15,12))

col_list=df.columns

 

for i in range(len(df.columns)):

    plt.subplot(3,3,i+1)

    plt.hist(df[col_list[i]],edgecolor="w")

    plt.title(col_list[i],color="g",fontsize=15)





plt.show()


#Boxplot for all variables.

"""for col in df.columns:

    df[[col]].boxplot()

    plt.show()"""

df.plot(kind='box',subplots=True,layout=(3,3),grid=True,figsize=(8,8))

plt.tight_layout()



plt.show()





# Correlation Between the data features.

df.corr()
#heatmap of the correlation of the data variables.

mask = np.zeros_like(df.corr(), dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

plt.figure(figsize=(12,12))

sns.heatmap(df.corr(),mask=mask,annot=True,linewidths=1.0)

plt.show()


sns.pairplot(df,x_vars=['GRE','TOEFL','UniversityRating','CGPA','SOP','LOR','Research'],

             y_vars='ChanceOfAdmit')

plt.tight_layout()
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.metrics import r2_score, mean_squared_error

from sklearn.linear_model import LinearRegression
X=df.iloc[:,:-1]

Y=df.iloc[:,-1]

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=2)
sc=StandardScaler()

X_train=sc.fit_transform(x_train)

X_test=sc.fit_transform(x_test)

lr=LinearRegression()

lr.fit(X_train,y_train)



#Training data score

ytrain_pred=lr.predict(X_train)

r2_score(y_train,ytrain_pred),mean_squared_error(y_train,ytrain_pred)



#Testdata score

y_pred=lr.predict(X_test)

r2_score(y_test,y_pred),mean_squared_error(y_test,y_pred)



print("Intercept of Linear Regression is:\n,",lr.intercept_,"Coefficients of Linear Regression are:\n,",lr.coef_)
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators = 150,max_depth=4,random_state = 42,criterion="mse")

rf_model.fit(X_train,y_train)

y_pred_rf=rf_model.predict(X_test)

r2_score(y_test,y_pred_rf),mean_squared_error(y_test,y_pred_rf)

feature_importance = pd.DataFrame(rf_model.feature_importances_, X.columns)

feature_importance
from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor(random_state = 4,max_depth=4)

dt.fit(X_train,y_train)

y_pred_dt = dt.predict(X_test) 

print(r2_score(y_test,y_pred_dt),mean_squared_error(y_test,y_pred_dt))

    
from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor(random_state = 1)

dt.fit(x_train,y_train)

y_pred_dt = dt.predict(x_test) 

r2_score(y_test,y_pred_dt),mean_squared_error(y_test,y_pred_dt)
#classification

from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC,SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

 



#regression

from sklearn.linear_model import LinearRegression,Ridge,Lasso,RidgeCV

from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor,AdaBoostRegressor

from sklearn.svm import SVR

from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor



#model selection

from sklearn.model_selection import train_test_split,cross_validate

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV



#preprocessing

from sklearn.preprocessing import MinMaxScaler,StandardScaler,Imputer,LabelEncoder



#evaluation metrics

from sklearn.metrics import mean_squared_log_error,mean_squared_error, r2_score,mean_absolute_error # for regression

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score 
models=[LinearRegression(),

        RandomForestRegressor(n_estimators=150,max_depth=4),

        DecisionTreeRegressor(random_state=42,max_depth=4),GradientBoostingRegressor(),AdaBoostRegressor(),

        KNeighborsRegressor(n_neighbors=35),

        BaggingRegressor(),Ridge(alpha=1.0),RidgeCV(),SVR()]

model_names=['LinearRegression','RandomForestRegressor','DecisionTree','GradientBoostingRegressor','AdaBoost','kNN',

             'BaggingReg','Ridge','RidgeCV',"SVR"]



R2_SCORE=[]

MSE=[]

      

for model in range(len(models)):

    print("*"*35,"\n",model_names[model])

    reg=models[model]

    reg.fit(X_train,y_train)

    pred=reg.predict(X_test)

    r=r2_score(y_test,pred)

    mse=mean_squared_error(y_test,pred)

    R2_SCORE.append(r)

    MSE.append(mse)

    print("R2 Score",r)

    print("MSE",mse)

df_model=pd.DataFrame({'Modelling Algorithm':model_names,'R2_score':R2_SCORE,"MSE":MSE})

df_model=df_model.sort_values(by="R2_score",ascending=False).reset_index()

print(df_model)





plt.figure(figsize=(10,10))

sns.barplot(y="Modelling Algorithm",x="R2_score",data=df_model)



plt.xlim(0.35,0.95)

plt.grid()

plt.tight_layout()
df_model.head(5)


lr=LinearRegression()

lr.fit(X_train,y_train)

y_lr_pred=lr.predict(X_test)

print("MSE:",mean_squared_error(y_test,y_lr_pred),"R2 SCORE:",r2_score(y_test,y_lr_pred))
ridge=Ridge()

ridge.fit(X_train,y_train)

y_ridge=ridge.predict(X_test)

print("MSE:",mean_squared_error(y_test,y_ridge),"R2 SCORE:",r2_score(y_test,y_ridge))
r_CV=RidgeCV()

r_CV.fit(X_train,y_train)

y_rCV=r_CV.predict(X_test)

print("MSE:",mean_squared_error(y_test,y_rCV),"R2 SCORE:",r2_score(y_test,y_rCV))

tp=pd.DataFrame({"TEST_value":y_test,"LR_predict_value": y_lr_pred,"RIDGE_predict_value": y_ridge,"RCV_predict_value": y_rCV,"DIFF(TEST_value-LR_predict_value)": (y_test-y_lr_pred)})

tp.head()
plt.figure(figsize=(10,10),dpi=75)

x=np.arange(len(tp["TEST_value"]))

y=tp["TEST_value"]

z=tp["LR_predict_value"]

plt.plot(x,y)

plt.plot(x,z,color='r')

print("Score of Linear Regression:",r2_score(y_test,y_lr_pred))