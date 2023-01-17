import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#Visualization

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

# Plotly for interactive graphics 

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot

import plotly.express as px



from collections import Counter





#Accuracy Score,MSE,ROC_Curve,Confusion Matrix

from sklearn.metrics import accuracy_score,mean_squared_error,roc_curve,roc_auc_score,classification_report,r2_score,confusion_matrix

#train test split, Grid Search CV

from sklearn.model_selection import train_test_split,cross_val_score,ShuffleSplit,GridSearchCV



#Disabling the warnings

import warnings

warnings.filterwarnings("ignore")



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

churn = pd.read_csv("/kaggle/input/churn-modelling/Churn_Modelling.csv")

churn.head()
df = churn.copy()

df.info() # As we can see there is not any NaN values, this sounds perfect but there can be other problems with columns.
columns =["RowNumber","CustomerId","Surname"]

df = df.drop(columns,axis = 1)

df.head()
len(df.columns)
#To change the gender column to 1 and 0:

df.Gender = [1 if each == 'Male' else 0 for each in df.Gender]

df.head()
#To change the Geography column to numerical:

Geography = [None] * len(df['Geography'])

for i in range(len(df['Geography'])):

    if df['Geography'][i] == 'Germany':

        Geography[i] = 0

    elif df['Geography'][i] == 'France':

        Geography[i] = 1

    else:

        Geography[i] = 2

        

df['New_Geography'] = Geography

df.sample(10)
df.describe().T
# Looking at the empty / non values in the columns so that it makes easier  decide and to interpret  which columns can we use

df.isna() 
# Examining amount of non-values 



df.isna().sum()
df[["Gender","Exited"]].groupby(["Gender"],as_index = False).mean().sort_values(by="Exited",ascending = False)

# Examining Exit rates of Gender Exit rates of females are higher than males
df[["New_Geography","Exited"]].groupby(["New_Geography"],as_index = False).mean().sort_values(by="Exited",ascending = False)

# Examining Exit amounts according to countries Germany with the highest exited amount among countries
df[["CreditScore","Exited"]].groupby(["CreditScore"],as_index = False).mean().sort_values(by="Exited",ascending = False)

# Examining Exit according to Credit rates
df[["Age","Exited"]].groupby(["Age"],as_index = False).mean().sort_values(by="Exited",ascending = False)

#Yas gruplarina gore en fazla cikislar orta yaslarda oluyor ve gencler ile yaslilarda cikis orani pek yuksek degil.
df[["Tenure","Exited"]].groupby(["Tenure"],as_index = False).mean().sort_values(by="Exited",ascending = False)

# Tenure does not give clear information
df[["Balance","Exited"]].groupby(["Balance"],as_index = False).mean().sort_values(by="Exited",ascending = False)

# Examining relation of balance and exited
df[["NumOfProducts","Exited"]].groupby(["NumOfProducts"],as_index = False).mean().sort_values(by="Exited",ascending = False)

# Although there is no linear proportion, the exited increases in general as the number of products increases.
df[["HasCrCard","Exited"]].groupby(["HasCrCard"],as_index = False).sum().sort_values(by="Exited",ascending = False)

# Values of the exited column does not change according to whether or not a person has card
df[["IsActiveMember","Exited"]].groupby(["IsActiveMember"],as_index = False).sum().sort_values(by="Exited",ascending = False)

#aktif uye olanlar bir tik cikisa daha meyilli
df[["EstimatedSalary","Exited"]].groupby(["EstimatedSalary"],as_index = False).mean().sort_values(by="Exited",ascending = False)

#maas ile cikis egilimi oranli degil.
def detect_outliers(df,features):

    outlier_indices = []

    

    for c in features:

        # 1st quartile

        Q1 = np.percentile(df[c],25)

        # 3rd quartile

        Q3 = np.percentile(df[c],75)

        # IQR

        IQR = Q3 - Q1

        # Outlier step

        outlier_step = IQR * 1.5

        # detect outlier and their indeces

        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index

        # store indeces

        outlier_indices.extend(outlier_list_col)

    

    outlier_indices = Counter(outlier_indices)

    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)

    

    return multiple_outliers
df.loc[detect_outliers(df,["Age","Gender","CreditScore","Exited"])]
# If we have some drop outliers, we could drop them

df = df.drop(detect_outliers(df,["Age","Gender","CreditScore","Exited"]),axis = 0).reset_index(drop = True)
plt.style.use("ggplot")

f,ax=plt.subplots(figsize = (8,8))

# corr() is actually pearson correlation

sns.heatmap(df.corr(),annot= True,linewidths=0.5,fmt = ".1f",ax=ax)

plt.xticks(rotation=90)

plt.yticks(rotation=0)

plt.title('Correlation Map')

plt.show()
# Tum kolonlara gore uyguladik ve Exited' hangi kolonlar arasi yogun iliski oldugunu gozlemlemeyi amacladik.

ranked_data = df.rank()

spearman_corr = ranked_data.loc[:,:].corr(method= "pearson")

print("Spearman's correlation: ")

print(spearman_corr)
plt.figure(figsize = (12,6)) 

sns.countplot(x="HasCrCard",hue = "New_Geography", data=df, palette="husl");

print(df.groupby('Geography')["HasCrCard"].sum())
#df.groupby('Geography')['EstimatedSalary'].mean()

avarage_salaries = df.groupby("New_Geography").mean()["EstimatedSalary"]

print("Avarage Salaries according to Countries:\n", avarage_salaries)
plt.figure(figsize = (14,8));

sns.catplot(x='Geography',

            y = "EstimatedSalary",

            hue="Exited",

            col="Gender",

            aspect=1.2,height=5,

            kind="swarm", data=df);
# Another visualization about salary effect

fig = px.box(df, x="New_Geography", y = "EstimatedSalary",color = 'Exited');

fig.update_layout(title_text="The country with the mean salary-With Outliers(Exited-Not Exited groups)")

fig.show();
plt.figure(figsize = (14,8)) 

plt.xticks(rotation=90)

plt.title('Credit Card Usage for Ages',color = 'blue',fontsize=15)

sns.countplot(x=df["Age"],hue = 'HasCrCard',data=df);

plt.xlabel('Ages')

plt.ylabel('Number of Credit Card Users');
fig = px.box(df, x="HasCrCard", y = "Age",color = 'Exited');

fig.update_layout(title_text="Credit Card Usage & Age - With Outliers(Exited-Not Exited groups)")

fig.show();
fig = px.parallel_categories(df, dimensions=['Gender', 'Geography', 'Exited'],

                color="Exited", color_continuous_scale=px.colors.sequential.Inferno,

                labels={'Gender':'Gender(Female,Male)', 'Exited':'Exited(0:No,1:Yes)'})

fig.update_layout(title_text="Gender-Geography-Exited-Not Exited Schema")

fig.show();
fig = px.parallel_categories(df, dimensions=['Gender','HasCrCard',"IsActiveMember", 'Exited'],

                color="Exited", color_continuous_scale=px.colors.sequential.Inferno,

                labels={'HasCrCard':'Has Credit Card', 'Gender':'Gender(Female,Male)', 'Exited':'Exited(0:No,1:Yes)'})

fig.update_layout(title_text="Credit Card-Gender-Exited-Not Exited Schema")

fig.show(); 
df.groupby("New_Geography")["CreditScore"].mean()

fig = px.box(df, x="New_Geography", y = "CreditScore",color = 'Exited');

fig.update_layout(title_text="The country with the highest credit score(mean)-With Outliers(Exited-Not Exited groups)")

fig.show();
plt.figure(figsize = (14,8));

sns.catplot(x='New_Geography',

            y = "CreditScore",

            hue="Exited",

            col="IsActiveMember",

            aspect=1.2,height=5,

            kind="swarm", data=df);
plt.figure(figsize = (14,8));

sns.catplot(x='New_Geography',

            y = "CreditScore",

            hue="Exited",

            col="Gender",

            aspect=1.2,height=5,

            kind="swarm", data=df);
plt.figure(figsize = (16,6)) 

plt.xticks(rotation=45)

sns.scatterplot(x=df['Age'],y = df["CreditScore"],hue = "Gender",data=df);
plt.figure(figsize = (16,6)) 

plt.xticks(rotation=45)

sns.scatterplot(x=df['Age'],y = df["CreditScore"],hue = "Exited",data=df);
df[df["CreditScore"]<405]['Exited'].value_counts()
plt.figure(figsize = (16,6)) 

plt.xticks(rotation=45)

sns.countplot(x=df["Age"],hue = 'Exited',data=df, palette="husl");

plt.xlabel('Age')

plt.ylabel('Number of customers (Exited or not)');
below_30 = df[df["Age"]<30]

between_30_40 = df[(df["Age"]>=30) & (df["Age"]<40)]

between_40_50 = df[(df["Age"]>=40) & (df["Age"]<50)]

between_50_60 = df[(df["Age"]>=50) & (df["Age"]<60)]

between_60_70 = df[(df["Age"]>=60) & (df["Age"]<70)]

above_70 = df[(df["Age"]>=70)]







k = below_30["Exited"].sum()

l = between_30_40["Exited"].sum()

m = between_40_50["Exited"].sum()

n = between_50_60["Exited"].sum()

o = between_60_70["Exited"].sum()

p = above_70["Exited"].sum()
f,ax = plt.subplots(figsize=(15, 15))

plt.subplot(6,1,1)

sns.countplot(x=below_30["Age"],hue = 'Exited',data=df, palette="husl");

plt.xlabel('Age')

plt.ylabel('Customers (Exited)');

plt.xticks(rotation= 30)



plt.subplot(6,1,2)

sns.countplot(x=between_30_40["Age"],hue = 'Exited',data=df, palette="husl");

plt.xlabel('Age')

plt.ylabel('Customers (Exited)');

plt.xticks(rotation= 30)





plt.subplot(6,1,3)

sns.countplot(x=between_40_50["Age"],hue = 'Exited',data=df, palette="husl");

plt.xlabel('Age')

plt.ylabel('Customers (Exited)');

plt.xticks(rotation= 30);



plt.subplot(6,1,4)

sns.countplot(x=between_50_60["Age"],hue = 'Exited',data=df, palette="husl");

plt.xlabel('Age')

plt.ylabel('Customers (Exited)');

plt.xticks(rotation= 30);



plt.subplot(6,1,5)

sns.countplot(x=between_60_70["Age"],hue = 'Exited',data=df, palette="husl");

plt.xlabel('Age')

plt.ylabel('Customers (Exited)');

plt.xticks(rotation= 30);



plt.subplot(6,1,6)

sns.countplot(x=above_70["Age"],hue = 'Exited',data=df, palette="husl");

plt.xlabel('Age')

plt.ylabel('Customers (Exited)');

plt.xticks(rotation= 30);
age_list = [('Total Stayed=',below_30['Exited'].value_counts()[:1],"Ages below 30==>",k,"Exited"),

            ('Total stayed=',between_30_40['Exited'].value_counts()[:1],'Ages between 30-40==>',l,"Exited"),

            ('Total stayed=',between_40_50['Exited'].value_counts()[:1],"Ages between 40-50==>",m,"Exited"),

            ('Total stayed=',between_50_60['Exited'].value_counts()[:1],"Ages between 50-60==>",n,"Exited"),

            ('Total stayed=',between_60_70['Exited'].value_counts()[:1],"Ages between 60-70==>",o,"Exited"),

            ('Total stayed=',above_70['Exited'].value_counts()[:1],"Ages above 70==>",p,"Exited")]
##### Plotly Pie Graph for Visualizing Percentage of Age Groups with Working A Bank 



pie_list=[k,l,m,n,o,p]

labels=age_list

fig={

    "data":[

        {

            "values":pie_list,

            "labels":labels,

            "domain": {"x": [.2, 1]},

            "name": "Age Groups-Exit Rate",

            "hoverinfo":"label+percent+name",

            "hole": .4,

            "type": "pie"

        },],

    "layout":{

        "title":"Percentage of Age Groups for Longer Work With Bank",

        "annotations":[

            {

                "font":{"size":20},

                "showarrow": False,

                "text": "Age Group-Exited",

                "x": 0.60,

                "y": 0.50

            },

        ]

    }  

}

iplot(fig)

plt.figure(figsize = (16,6)) 

plt.xticks(rotation=45)

sns.barplot(x=df['New_Geography'],y = df["Exited"],hue = "Gender",data=df, palette="husl");

plt.ylabel('Percetage of people (Exited %)');
plt.figure(figsize = (16,6)) 

plt.xticks(rotation=45)

sns.countplot(x=df["New_Geography"],hue = 'Exited',data=df, palette="husl");

plt.xlabel('Geo')

plt.ylabel('Number of customers (Exited or not)');
print("Total Number of People By Geography\n",df["Geography"].value_counts())

print("Number of People Exited By Geography\n",df[df['Exited']==1]["Geography"].value_counts(),'\n')

print("Number of People Exited By Gender in Germany \n",df[(df['Exited']==1)&(df['Geography']=='Germany')]["Gender"].value_counts())

print("Number of People Exited By Gender in France \n",df[(df['Exited']==1)&(df['Geography']=='France')]["Gender"].value_counts())

print("Number of People Exited By Gender in Spain \n",df[(df['Exited']==1)&(df['Geography']=='Spain')]["Gender"].value_counts())
plt.figure(figsize = (16,6)) 

plt.xticks(rotation=45)

sns.scatterplot(x='Age',y = "EstimatedSalary",hue = "Exited",data=df);
df = churn.copy()
age_group_data = [None] * len(df['Age'])

for i in range(len(df['Age'])):

    if df['Age'][i] < 30:

        age_group_data[i] = 'Young'

    elif df['Age'][i] >=30 and df['Age'][i] < 40:

        age_group_data[i] = 'Young-Adults'

    elif df['Age'][i] >=40 and df['Age'][i] < 50:

        age_group_data[i] = 'Adults'

    elif df['Age'][i] >=50 and df['Age'][i] < 60:

        age_group_data[i] = 'Elderly-Adults'

    elif df['Age'][i] >=60 and df['Age'][i] < 74:

        age_group_data[i] = 'Old'

    else:

        age_group_data[i] = 'Very-Old'



df['age_group'] = age_group_data
Credit = [None] * len(df['CreditScore'])

for i in range(len(df['CreditScore'])):

    if df['CreditScore'][i] < 405:

        Credit[i] = 0

    else:

        Credit[i] = 1

        

df['new_credit'] = Credit
df['new_credit'].value_counts()
g = sns.factorplot(x = "new_credit", y = "Exited", data = df, kind = "bar")

plt.xticks(rotation=45)

g.set_ylabels("Exited")

plt.show()
age74 = df[(df["Age"]>=74)]

age74["Exited"].value_counts()

# We dropped the 2 lines that is outlier of the above 71 ages.

df.drop([3110,3531],axis =0,inplace = True)
g = sns.factorplot(x = "age_group", y = "Exited", data = df, kind = "bar")

plt.xticks(rotation=45)

g.set_ylabels("Exited")

plt.show()
gender_dummies = df.replace(to_replace={'Gender': {'Female': 0,'Male':1}})

a = pd.get_dummies(df['Geography'], prefix = "Geo_dummy")

c = pd.get_dummies(df['age_group'], prefix = "Age_dummy")
frames = [gender_dummies,a,c]  

df = pd.concat(frames, axis = 1)

df = df.drop(["RowNumber","Geography","Surname","CustomerId",'Age','age_group','Geography',"CreditScore"],axis = 1)
x = df.drop(["Exited"],axis = 1) #Independent value

y = df["Exited"] #Depended value 
# data normalization with sklearn

from sklearn.preprocessing import MinMaxScaler



# fit scaler on training data

norm = MinMaxScaler().fit(x)



# transform independent data

x_norm = norm.transform(x)



####Generally code yourself ===>>>        x = (x-np.min(x))/(np.max(x)-np.min(x)).values
x_train,x_test,y_train,y_test = train_test_split(x_norm,y,test_size = 0.3, random_state = 42)
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression().fit(x_train,y_train)

y_pred = log_reg.predict(x_test)

log_model = (accuracy_score(y_test,y_pred)*100)

log_model
y_probs = log_reg.predict_proba(x_test)[:,1]

y_pred = [1 if i >0.53 else 0 for i in y_probs]

log_proba_score = (accuracy_score(y_test,y_pred)*100)

print ("log score=",log_proba_score)
confusion_matrix(y_test,y_pred)
log_params = {"C":np.logspace(-3,3,7),

              "penalty": ["l1","l2"],

              "max_iter":[10,50,500,1000]} #"solver":['lbfgs', 'liblinear', 'sag', 'saga'],

log =LogisticRegression()

log_cv = GridSearchCV(log,log_params,cv = 10)



log_tuned = log_cv.fit(x_train,y_train)

log_tuned.best_params_
log_reg_tuned = LogisticRegression(C=100,max_iter=50,penalty='l2',solver='liblinear').fit(x_train,y_train)

y_probs = log_reg.predict_proba(x_test)[:,1]

y_pred = [1 if i >0.53 else 0 for i in y_probs]
log_tuned_score = (accuracy_score(y_test,y_pred)*100)

print ("log tuned score=",log_tuned_score)
lr_cm = confusion_matrix(y_test,y_pred)

lr_cm
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()

gnb_model = nb.fit(x_train,y_train)

gnb_model
y_pred = gnb_model.predict(x_test)

nb_score = (accuracy_score(y_test,y_pred)*100)

nb_score
nb_params = {'var_smoothing': np.logspace(0,-9, num=100)}
nb =GaussianNB()

nb_cv = GridSearchCV(nb,nb_params,cv = 10)



nb_cv = nb_cv.fit(x_train,y_train)

nb_cv.best_params_
nb_tuned =GaussianNB(var_smoothing=0.43287612810830584).fit(x_train,y_train)

y_pred = nb_tuned.predict(x_test)

nb_tuned = (accuracy_score(y_test,y_pred)*100)

nb_tuned
nb_cm = confusion_matrix(y_test,y_pred)
from sklearn.neighbors import KNeighborsClassifier
knn =KNeighborsClassifier()

knn_model = knn.fit(x_train,y_train)

knn_model
y_pred = knn_model.predict(x_test)

knn_score = (accuracy_score(y_test,y_pred)*100)

knn_score
knn_params = {"n_neighbors":np.arange(1,50),

              "weights": ["uniform","distance"],

              "metric":["euclidean","manhattan"]}
knn =KNeighborsClassifier()

knn_cv = GridSearchCV(knn,knn_params,cv = 10)

knn_cv = knn_cv.fit(x_train,y_train)
print("Best Parameters:"+str(knn_cv.best_params_))
knn_final =KNeighborsClassifier(n_neighbors =15,metric='manhattan',weights='distance')

knn_final = knn_final.fit(x_train,y_train)

y_pred = knn_final.predict(x_test)

knn_tuned = (accuracy_score(y_test,y_pred)*100)

knn_tuned
knn_cm = confusion_matrix(y_test,y_pred)

knn_cm
from sklearn.svm import SVC
svm_model_linear = SVC(kernel='linear').fit(x_train,y_train)

svm_model_poly = SVC(kernel='poly').fit(x_train,y_train)

svm_model_rbf = SVC(kernel='rbf').fit(x_train,y_train)
y_pred_linear = svm_model_linear.predict(x_test)

y_pred_poly = svm_model_poly.predict(x_test)

y_pred_rbf = svm_model_rbf.predict(x_test)
print(accuracy_score(y_test,y_pred_linear))

print(accuracy_score(y_test,y_pred_poly))

print(accuracy_score(y_test,y_pred_rbf))
svc_params = {"C": [1,5,10,50,100,200],

              'kernel':['poly','rbf'],

              "gamma": [0.001, 0.01, 0.1,0.5],}

                 

svc = SVC()

svc_cv_model = GridSearchCV(svc,svc_params,

                            cv = 5,

                           n_jobs = -1,

                           verbose = 2)

svc_cv_model.fit(x_train,y_train)

print("Best Parameters:"+str(svc_cv_model.best_params_))
svc_tuned = SVC(kernel = "poly",C=100,gamma=0.5).fit(x_train,y_train)
y_pred = svc_tuned.predict(x_test)

svc_tuned_score = (accuracy_score(y_test,y_pred)*100)

svc_tuned_score
confusion_matrix(y_test,y_pred)
svc_rbf_tuned = SVC(kernel = "rbf",C=100,gamma=0.1).fit(x_train,y_train)

y_pred = svc_rbf_tuned.predict(x_test)
svc_rbf_score = (accuracy_score(y_test,y_pred)*100)

svc_rbf_score
svm_cm = confusion_matrix(y_test,y_pred)

svm_cm
from sklearn.ensemble import RandomForestClassifier
r_for = RandomForestClassifier().fit(x_train,y_train)

r_for
y_pred = r_for.predict(x_test)

rf_score = accuracy_score(y_test,y_pred)

rf_score
rf_params  = {'max_depth':list(range(1,10)),

             "max_features":["log2","auto","sqrt"],

             "n_estimators":[2,10,20,50,150,300],

             'criterion' : ['gini','entropy'],

             'min_samples_leaf' : [1,3,5,10]}
rf_model = RandomForestClassifier()
rf_cv_model = GridSearchCV(rf_model,

                           rf_params,

                           cv = 5,

                           n_jobs = -1)
rf_cv_model.fit(x_train,y_train)

rf_cv_model.best_params_
rf_tuned = RandomForestClassifier(max_depth = 10,

                                  criterion = 'gini',

                                  max_features = 'log2',

                                  min_samples_leaf = 1,

                                  n_estimators = 150,random_state=42)

rf_tuned = rf_tuned.fit(x_train,y_train)

y_pred  = rf_tuned.predict(x_test)

rf_tuned_score = (accuracy_score(y_test,y_pred)*100)

rf_tuned_score
rf_cm = confusion_matrix(y_test,y_pred)

rf_cm
from sklearn.ensemble import GradientBoostingClassifier
gbm = GradientBoostingClassifier()

gbm_model = gbm.fit(x_train,y_train) 

gbm_model
y_pred = gbm_model.predict(x_test)

gbm_score = accuracy_score(y_test,y_pred)*100

gbm_score
gbm_params = {"learning_rate" : [0.001, 0.01, 0.1, 0.2],

             "n_estimators": [100,200,300,500,1000],

             "max_depth": [1,3,5,10],

             "min_samples_split": [1,2,5,10]}

gbm = GradientBoostingClassifier()

clf = GridSearchCV(gbm,gbm_params,verbose=0,n_jobs=-1,cv=3)

gb = clf.fit(x_train,y_train)

gb.best_params_ 
gbm = GradientBoostingClassifier(n_estimators=100,min_samples_split=5,max_depth=3,learning_rate=0.2,random_state=42)

gbm.fit(x_train,y_train)

y_pred = gbm.predict(x_test)

gbm_tuned_score = accuracy_score(y_test,y_pred)*100

gbm_tuned_score
gbm_cm = confusion_matrix(y_test,y_pred)

gbm_cm
#!pip install xgboost

from xgboost import XGBClassifier
xgb = XGBClassifier(n_estimators=100)

xgb_model = xgb.fit(x_train,y_train) 

xgb_model
y_pred = xgb_model.predict(x_test)

xgb_score = accuracy_score(y_test,y_pred)*100

xgb_score
xgb_params ={

        'n_estimators': [50, 100, 200],

        'subsample': [ 0.6, 0.8, 1.0],

        'max_depth': [1,2,3,4],

        'learning_rate': [0.1,0.2, 0.3, 0.4, 0.5],

        "min_samples_split": [1,2,4,6]}
xgb = XGBClassifier()

xgb = GridSearchCV(xgb,xgb_params,verbose=0,n_jobs=-1,cv=3)

xgb = xgb.fit(x_train,y_train)

xgb.best_params_
xgbm_cv = XGBClassifier(learning_rate=0.3,

                       max_depth=2,

                       min_samples_split=1,

                       n_estimators=100,

                       subsample=1.0,random_state=42).fit(x_train,y_train)
y_pred = xgbm_cv.predict(x_test)

xgbm_score = (accuracy_score(y_test,y_pred)*100)

xgbm_score
xgbm_cm = confusion_matrix(y_test,y_pred)

xgbm_cm
from lightgbm import LGBMClassifier
lgbm = LGBMClassifier().fit(x_train,y_train)

y_pred = lgbm.predict(x_test)
lgbm_score = (accuracy_score(y_test,y_pred)*100)

lgbm_score
lgbm_params = {"learning_rate" : [0.001,0.01, 0.1],

             "n_estimators": [100,200,300,500,1000],

             "max_depth": [2,3,5,7],

             "min_child_samples": [1,3,5,7]}

lgbm = LGBMClassifier()

lgbm_cv = GridSearchCV(lgbm,lgbm_params,verbose=0,n_jobs=-1,cv=5)

lgbm_cv_model = lgbm_cv.fit(x_train,y_train)

lgbm_cv_model.best_params_
lgbm = LGBMClassifier(learning_rate=0.01,max_depth=5,min_child_samples=5,n_estimators=400)

lgbm_tuned = lgbm.fit(x_train,y_train)

y_pred = lgbm_tuned.predict(x_test)

lgbm_tuned_acc = (accuracy_score(y_test,y_pred)*100)

lgbm_tuned_acc
lgbm_cm = confusion_matrix(y_test,y_pred)

lgbm_cm
from catboost import CatBoostClassifier
cat_model = CatBoostClassifier().fit(x_train,y_train)

y_pred = cat_model.predict(x_test)
cat_score = accuracy_score(y_test,y_pred)

cat_score
cat_params = {"iterations":[100,200,500,700],

              'loss_function': ['Logloss', 'CrossEntropy'],

              "learning_rate":[0.01,0.02,0.1],

              "depth":[1,3,5,8]}
catb =  CatBoostClassifier()

catb_cv_model = GridSearchCV(catb,cat_params,cv = 5,n_jobs = -1,verbose = 2)

catb_cv_model.fit(x_train,y_train)

catb_cv_model.best_params_
catb_final = CatBoostClassifier(depth=5,iterations=500,learning_rate=0.02,loss_function= 'Logloss')

catb_final = catb_final.fit(x_train,y_train)
y_pred = catb_final.predict(x_test)

catb_final_score =(accuracy_score(y_test,y_pred)*100)

catb_final_score 
catb_cm = confusion_matrix(y_test,y_pred)

catb_cm
## We will use also scaler for improving the score of ML algorithms

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(x)

x_scaled = scaler.transform(x)
x_train,x_test,y_train,y_test = train_test_split(x_scaled,y,

                                                test_size = 0.30,

                                                random_state = 42)
knn_params = {"n_neighbors":np.arange(1,50),

              "weights": ["uniform","distance"],

              "metric":["euclidean","manhattan"]}



knn =KNeighborsClassifier()

knn_cv = GridSearchCV(knn,knn_params,cv = 5)

knn_cv = knn_cv.fit(x_train,y_train)

print("Best Parameters:"+str(knn_cv.best_params_))
knn_scaled =KNeighborsClassifier(n_neighbors =29,metric='manhattan',weights='distance')

knn_scaled = knn_scaled.fit(x_train,y_train)

y_pred = knn_scaled.predict(x_test)

knn_sscore = (accuracy_score(y_test,y_pred)*100)

knn_sscore
knn_scaled_conf = confusion_matrix(y_test,y_pred)

knn_scaled_conf
svm_scaled_linear = SVC(kernel='linear').fit(x_train,y_train)

svm_scaled_poly = SVC(kernel='poly').fit(x_train,y_train)

svm_scaled_rbf = SVC(kernel='rbf').fit(x_train,y_train)
y_pred_slinear = svm_scaled_linear.predict(x_test)

y_pred_spoly = svm_scaled_poly.predict(x_test)

y_pred_srbf = svm_scaled_rbf.predict(x_test)



print(accuracy_score(y_test,y_pred_slinear))

print(accuracy_score(y_test,y_pred_spoly))

print(accuracy_score(y_test,y_pred_srbf))
svc_params = {"C": [10,50,100,500,700],

              'kernel':['poly','rbf'],

              "gamma": [0.001, 0.01, 0.1]} 

                 

svc = SVC()

svc_cv_model = GridSearchCV(svc,svc_params,

                            cv = 5,

                           n_jobs = -1,

                           verbose = 2)

svc_cv_model.fit(x_train,y_train)

print("Best Parameters:"+str(svc_cv_model.best_params_))
svc_scaled = SVC(kernel = 'rbf',C = 500, gamma = 0.01)

scaled = svc_scaled.fit(x_train,y_train)

y_pred = scaled.predict(x_test)
svm_scaled_score = (accuracy_score(y_test,y_pred)*100)

svm_scaled_score
svc_scaled_conf = confusion_matrix(y_test,y_pred)

svc_scaled_conf
rf_params  = {'max_depth':list(range(1,11)),

             "max_features":["log2","auto","sqrt"],

             "n_estimators":[2,10,20,50,150,300],

             'criterion' : ['gini','entropy'],

             'min_samples_leaf' : [1,3,5,10]}
rf_model = RandomForestClassifier(random_state = 42)

rf_cv_model = GridSearchCV(rf_model,

                           rf_params,

                           cv = 5,

                           n_jobs = -1)

rf_cv_model.fit(x_train,y_train)

rf_cv_model.best_params_
rf_tuned = RandomForestClassifier(max_depth = 10,

                                  criterion = 'gini',

                                  max_features = 'log2',

                                  min_samples_leaf = 1,

                                  n_estimators = 150,random_state = 42)

rf_tuned = rf_tuned.fit(x_train,y_train)

y_pred  = rf_tuned.predict(x_test)

rf_scaled_score = (accuracy_score(y_test,y_pred)*100)

rf_scaled_score
rf_scaled_conf = confusion_matrix(y_test,y_pred)

rf_scaled_conf
lgbm_params = {"learning_rate" : [0.01, 0.02,0.1],

             "n_estimators": [100,200,300,500,1000],

             "max_depth": [2,3,5,7],

             "min_child_samples": [1,2,5,10]}

lgbm = LGBMClassifier()

lgbm_cv = GridSearchCV(lgbm,lgbm_params,verbose=0,n_jobs=-1,cv=5)

lgbm_cv_model = lgbm_cv.fit(x_train,y_train)

lgbm_cv_model.best_params_
lgbm = LGBMClassifier(learning_rate=0.02,max_depth=5,min_child_samples=5,n_estimators=500,random_state = 42)

lgbm_tuned = lgbm.fit(x_train,y_train)

y_pred = lgbm_tuned.predict(x_test)

lgbm_scaled_acc = (accuracy_score(y_test,y_pred)*100)

lgbm_scaled_acc
lgbm_scaled_conf = confusion_matrix(y_test,y_pred)

lgbm_scaled_conf
logit_roc_auc = roc_auc_score(y_test,log_reg_tuned.predict(x_test))



fpr, tpr, tresholds = roc_curve(y_test,log_reg_tuned.predict_proba(x_test)[:,1])

plt.figure(figsize=(6,6))

plt.plot(fpr,tpr,label = "AUC (area = %0.2f)"%logit_roc_auc)

plt.plot([0,1],[0,1],"r--")

plt.xlim([0.0,1.0])

plt.ylim([0.0,1.0])

plt.xlabel("False Positive Ratio")

plt.ylabel("True Positive Ratio")

plt.title('ROC Curve');
from sklearn.metrics import roc_auc_score

roc_auc_score(y_test,y_probs)
fig = plt.figure(figsize=(15,15))



ax1 = fig.add_subplot(4, 4, 1) # row, column, position

ax1.set_title('Logistic Regression Classification')



ax2 = fig.add_subplot(4, 4, 2) # row, column, position

ax2.set_title('KNN Classification')



ax3 = fig.add_subplot(4, 4, 3)

ax3.set_title('SVM Classification')



ax4 = fig.add_subplot(4, 4, 4)

ax4.set_title('Naive Bayes Classification')



ax5 = fig.add_subplot(4, 4, 5)

ax5.set_title('Random Forest Classification')



ax6 = fig.add_subplot(4, 4, 6)

ax6.set_title('GBM Classification')



ax7 = fig.add_subplot(4, 4, 7)

ax7.set_title('LightGBM Classification')



ax8 = fig.add_subplot(4, 4, 8)

ax8.set_title('XGBoost Classification')

ax9 = fig.add_subplot(4, 4, 9)

ax9.set_title('CatBoost Classification')



ax10 = fig.add_subplot(4, 4, 10)

ax10.set_title('KNN Scaled Classification')



ax11 = fig.add_subplot(4,4, 11)

ax11.set_title('SVC Scaled Classification')



ax12 = fig.add_subplot(4,4, 12)

ax12.set_title('Random Forest Scaled Classification')



ax13 = fig.add_subplot(4, 4, 13)

ax13.set_title('LightGBM Scaled Classification')





sns.heatmap(data=lr_cm, annot=True, linewidth=0.7, linecolor='cyan', fmt='.0f', ax=ax1, cmap='magma')

sns.heatmap(data=knn_cm, annot=True, linewidth=0.7, linecolor='cyan', fmt='.0f', ax=ax2, cmap='magma') 

sns.heatmap(data=svm_cm, annot=True, linewidth=0.7, linecolor='cyan', fmt='.0f', ax=ax3, cmap='magma')

sns.heatmap(data=nb_cm, annot=True, linewidth=0.7, linecolor='cyan', fmt='.0f', ax=ax4, cmap='magma')

sns.heatmap(data=rf_cm, annot=True, linewidth=0.7, linecolor='cyan', fmt='.0f', ax=ax5, cmap='magma')

sns.heatmap(data=gbm_cm, annot=True, linewidth=0.7, linecolor='cyan', fmt='.0f', ax=ax6, cmap='magma')

sns.heatmap(data=lgbm_cm, annot=True, linewidth=0.7, linecolor='cyan', fmt='.0f', ax=ax7, cmap='magma')

sns.heatmap(data=xgbm_cm, annot=True, linewidth=0.7, linecolor='cyan', fmt='.0f', ax=ax8, cmap='magma')

sns.heatmap(data=catb_cm, annot=True, linewidth=0.7, linecolor='cyan', fmt='.0f', ax=ax9, cmap='magma')

sns.heatmap(data=knn_scaled_conf, annot=True, linewidth=0.7, linecolor='cyan', fmt='.0f', ax=ax10, cmap='magma')

sns.heatmap(data=svc_scaled_conf, annot=True, linewidth=0.7, linecolor='cyan', fmt='.0f', ax=ax11, cmap='magma')

sns.heatmap(data=rf_scaled_conf, annot=True, linewidth=0.7, linecolor='cyan', fmt='.0f', ax=ax12, cmap='magma')

sns.heatmap(data=lgbm_scaled_conf, annot=True, linewidth=0.7, linecolor='cyan', fmt='.0f', ax=ax13, cmap='magma')

plt.show()
indexx = ["Log","RF","KNN","SVM","NB","GBM","LightGBM","XGBoost",'CatBoost',"KNN Scaled","SVM Scaled", 'RF Scaled',"LightGBM Scaled"]

regressions = [log_tuned_score,rf_tuned_score,knn_tuned,svc_rbf_score,nb_tuned,gbm_tuned_score,

               lgbm_tuned_acc,xgbm_score,catb_final_score,knn_sscore,svm_scaled_score,rf_scaled_score,lgbm_scaled_acc]



plt.figure(figsize=(12,8))

sns.barplot(x=indexx,y=regressions)

plt.xticks(rotation=45)

plt.title('Model Comparision',color = 'green',fontsize=20);
pie_list=regressions

labels=list(zip(indexx,regressions))

fig={

    "data":[

        {

            "values":pie_list,

            "labels":labels,

            "domain": {"x": [.2, 1]},

            "name": "Models-Accuracy Score",

            "hoverinfo":"label+percent+name",

            "hole": .4,

            "type": "pie"

        },],

    "layout":{

        "title":"Accuracy Scores",

        "annotations":[

            {

                "font":{"size":20},

                "showarrow": False,

                "text": "Age Group-Exited",

                "x": 0.60,

                "y": 0.50

            },

        ]

    }  

}

iplot(fig)