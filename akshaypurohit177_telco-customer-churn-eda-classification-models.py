# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
#print(df.head(5))

df.info()
#df['TotalCharges'] = df['TotalCharges'].astype(float)





df['TotalCharges'] = pd.to_numeric(df['TotalCharges'],errors='coerce')



#Checking Data Type after making changes

print(df.info())
#print(df[df['TotalCharges'].isnull()])



sns.distplot(df['TotalCharges'])

plt.show()

print(df['TotalCharges'].skew())
#sns.pairplot(df)

#plt.show()

from scipy.stats import boxcox



#sns.distplot(df['TotalCharges'])

#df['TotalCharges']=df['TotalCharges'].transform('sqrt')

df['TotalCharges'] = boxcox(df['TotalCharges'],0.5)

print(df['TotalCharges'].skew())



sns.distplot(df['TotalCharges'])

plt.show()





sns.distplot(df['MonthlyCharges'])

plt.show()

print("Checking Skewness :- ",df['MonthlyCharges'].skew())

# Creating a function to take DF and identify Categorical Varibles and create a crosstab 

#and plot the same.

def category_rel_y(df):

    X=df.columns

    #print(df[X[1]])

 

    #print(len(X))

 

    for i in range(1,len(X)-2):

        

    

        if (df[X[i]].dtype=='object'):

                fig, axs = plt.subplots(1, 2,figsize=(13,3))

                tab_values=pd.crosstab(df[X[i]],df.iloc[:,20])

                #Creating  % of total values cross tab by using 'normalize=True' in normal tab

                tab_percentage=pd.crosstab(df[X[i]],df.iloc[:,20],normalize=True)

                #print(tab_percentage)

                tab_values.plot(kind='bar', stacked=False ,ax=axs[0] )

                sns.heatmap(tab_percentage,annot=True,cmap='YlGnBu',ax=axs[1])

                plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1, hspace=None)

                #sns.barplot(tab)

                axs[0].set_title('Actual counts for CHURN output versus '+X[i].upper())

              

                plt.title("% of total distribution for CHURN and "+(X[i]).upper()) 

                plt.show()

                 

            

        

        

 #plt.show()     

category_rel_y(df)   
sns.distplot(df['tenure'])

df['tenure'].skew()



def bins(x):

    if x>=0 and x<=15:

        return '0-15'

    elif x>=16 and x<=30:

        return '16-30'

    elif x>=31 and x<=45:

        return '31-45'

    elif x>=46 and x<=60:

        return '46-60'

    elif x>=61 and x<=75:

        return '61-75'

    else:

        return 'Above 80'

df['tenure_bins']=df['tenure'].map(bins)

#print(df[['tenure_bins','tenure']])



x=(pd.crosstab(df['tenure_bins'],df['Churn']))



x.plot(kind='bar', stacked=False)



plt.show()
#sns.heatmap(df.corr(),annot=True,cmap='viridis')

#plt.show()





#Convert Churn into numerical values of 0 and 1 to get correlation with other numerical variables

churn=pd.get_dummies(df["Churn"],drop_first=True)

churn.rename(columns = {'Yes':'Churn_continuous'}, inplace = True) 

df=pd.concat([df,churn],axis=1)







df_corr=df[['MonthlyCharges','TotalCharges','tenure','Churn_continuous']].corr()

#create a mask to avoid duplicates in correlation matrix

mask = np.zeros_like(df_corr, dtype=np.bool)

#print(mask.shape)

mask[np.triu_indices_from(mask)] = True

fig=plt.subplots(figsize=(8,4))

sns.heatmap(df_corr,annot=True,cmap='viridis',mask=mask)

plt.show()
list=["Contract",

"OnlineSecurity",

"InternetService",

"PaymentMethod",

"PaperlessBilling","TechSupport","StreamingMovies"]



for i in range(len(list)):

    fig, ax = plt.subplots(figsize =(8, 4)) 

    sns.violinplot(ax = ax,data=df, x = list[i],  

                  y = "MonthlyCharges", hue="Churn", kind='violin',split=True) 



X=df['Churn'].value_counts()

print(X)

y=X/len(df)

sns.barplot(x=X,y=(X/len(df))*100)









plt.show()

#print(df.columns)

# drop 13 rows with null in "Total charges"

df = df.dropna()



#one-hot encoding



x=df[['gender', 'SeniorCitizen', 'Partner', 'Dependents','tenure',

        'PhoneService', 'MultipleLines', 'InternetService',

       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',

       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',

       'PaymentMethod', 'MonthlyCharges']]

X=pd.get_dummies(x, drop_first=True )

#print(X.columns)

c=X.columns





c=X.columns        

#for i in range(0,len(c)):

#    if (X[c[i]].dtype=='object'):

#        X.drop(c[i],axis=1,inplace=True)

    



y=pd.get_dummies(df['Churn'], drop_first=True)

#print(y)





#print("After ",X)

#print("After ",y)



#Applyting SMOTE after doing one-hot encoding and removing null values



import sklearn.model_selection as model_selection

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.70,test_size=0.30, random_state=101)



        



from imblearn.over_sampling import SMOTE

smt = SMOTE(random_state = 2)

X_train, y_train = smt.fit_sample(X_train, y_train)





smote_class=y_train['Yes'].value_counts()

print(smote_class)

#print(X_train.info())





from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics

from sklearn.metrics import confusion_matrix  

from sklearn.metrics import classification_report  



#from sklearn.preprocessing import StandardScaler

#SC = StandardScaler()

#X_train_MC = SC.fit_transform(X_train['MonthlyCharges'])

#X_train.drop['MonthyCharges']

#X_test = SC.transform(X_test)





clf=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

            max_depth=None, max_features='auto', max_leaf_nodes=None,

            min_impurity_decrease=0.0, min_impurity_split=None,

            min_samples_leaf=1, min_samples_split=2,

            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,

            oob_score=False, random_state=None, verbose=0,

            warm_start=False)

clf.fit(X_train,y_train)



y_pred=clf.predict(X_test)



# we can use predict_proba() as well

#y_pred=clf.predict_proba(X_test)













print("Accuracy:******************",metrics.accuracy_score(y_test, y_pred))





#visualizing confusion matrix

from sklearn.metrics import plot_confusion_matrix



plot_confusion_matrix(clf, X_test, y_test,cmap='viridis')  # doctest: +SKIP

plt.title("*Confusion Matrix*")

plt.show()



print("***************Classification report*************")

print(classification_report(y_test, y_pred))



#sns.heatmap(classification_report(y_test, y_pred))

#plt.show()



feature_imp = pd.Series(clf.feature_importances_,index=X_train.columns).sort_values(ascending=False)

#sns.barplot(x=feature_imp, y=feature_imp.index)

# Add labels to your graph

#print(feature_imp)

visual=feature_imp.reset_index()

   

   

visual.rename(columns = {'index':'features', 0:'Importance_score'}, inplace = True) 

print(visual)

fig=plt.subplots(figsize=(10,15))

sns.barplot(x = 'Importance_score', y = 'features', data = visual)

plt.xlabel('Importance_score')

plt.ylabel('Features')

#plt.title("Visualizing Important Features")



plt.legend()

plt.show()






from lightgbm import LGBMClassifier

model = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,

               importance_type='split', learning_rate=0.1, max_depth=-1,

               min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

               n_estimators=500, n_jobs=-1, num_leaves=31, objective=None,

               random_state=None, reg_alpha=0.0, reg_lambda=0.0, silent=True,

               subsample=1.0, subsample_for_bin=200000, subsample_freq=0)

model.fit(X_train, y_train)

y_pred=model.predict(X_test)



print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from xgboost import XGBClassifier

from sklearn.model_selection import cross_val_score



# define dataset



model = XGBClassifier(silent=False, 

                      scale_pos_weight=1,

                      learning_rate=0.1,  

                      colsample_bytree = 0.4,

                      subsample = 0.8,

                      objective='binary:logistic', 

                      n_estimators=1000, 

                      reg_alpha = 0.3,

                      max_depth=4, 

                      gamma=10)

model.fit(X_train, y_train)

y_pred=model.predict(X_test)



print("Accuracy:",metrics.accuracy_score(y_test, y_pred))





from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dropout,Dense,Flatten



model=Sequential()



model.add(Dense(64,activation='relu'))

model.add(Dense(64,activation='relu'))

model.add(Dense(64,activation='relu'))

model.add(Dropout(.4))

model.add(Dense(64,activation='relu'))







model.add(Dense(1,activation='sigmoid'))



model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])





from tensorflow.keras.callbacks import EarlyStopping

early_stop=EarlyStopping(monitor='val_loss',patience=2)

results=model.fit(X_train,y_train,epochs=50,validation_data=(X_test,y_test),callbacks=[early_stop])




