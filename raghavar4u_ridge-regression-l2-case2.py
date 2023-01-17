%matplotlib inline  

# To make data visualisations display in Jupyter Notebooks 

import numpy as np   # linear algebra

import pandas as pd  # Data processing, Input & Output load

import matplotlib.pyplot as plt # Visuvalization & plotting

import seaborn as sns

import datetime  

from sklearn.linear_model import LogisticRegression #  Logistic Regression (aka logit) classifier in linear model

import joblib  #Joblib is a set of tools to provide lightweight pipelining in Python (Avoid computing twice the same thing)



from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

                                    # GridSearchCV - Implements a “fit” and a “score” method

                                    # train_test_split - Split arrays or matrices into random train and test subsets

                                    # cross_val_score - Evaluate a score by cross-validation

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score, make_scorer, accuracy_score, roc_curve, confusion_matrix, classification_report

                                    # Differnt metrics to evaluate the model 

import pandas_profiling as pp   # simple and fast exploratory data analysis of a Pandas Datafram



import warnings   # To avoid warning messages in the code run

warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder # Labeling the columns with 0 & 1
Tdata = pd.read_csv("../input/TelcoCustomerChurn.csv")
print ("Rows     : " ,Tdata.shape[0])

print ("Columns  : " ,Tdata.shape[1])

#print ("\nMissing values :  ", Tdata.isnull().sum())

#print ("\nUnique values :  \n",'Tdata.nunique()' + 'Tdata.info()')

#Tdata.info()



#def df_summary(df):

#df_U = Tdata.nunique()

#df_M = Tdata.isnull().sum()

#df_I = Tdata.dtypes

#df_U = df_U.to_frame().reset_index()

#df_M = df_M.to_frame().reset_index()

#df_I = df_I.to_frame().reset_index()

#df_U = df_U.rename(columns= {0: 'Unique Data'})

#df_M = df_M.rename(columns= {0: 'Missing Data'})

#df = pd.merge(df_U,df_M,on = 'index')



## Using all the above steps lets create function for summary for any given dataset  



def df_summary(df):

  df_U = df.nunique()

  df_M = df.isnull().sum()

  df_I = df.dtypes

  df_U = df_U.to_frame().reset_index()

  df_M = df_M.to_frame().reset_index()

  df_I = df_I.to_frame().reset_index()

  df_U = df_U.rename(columns= {0: 'Unique Data'})

  df_M = df_M.rename(columns= {0: 'Missing Data'})

  df_I = df_I.rename(columns= {0: 'Data Types'})

  output = pd.merge(pd.merge(df_M,df_U,on='index'),df_I,on='index')

  return output;

df_summary(Tdata)
Tdata.V18.head()
Tdata.V18= pd.to_numeric(Tdata.V18, errors='coerce')

Tdata.V18.describe()
df_summary(Tdata)
Tdata['V18'].fillna((Tdata['V18'].mean()), inplace=True)

Tdata['V18'].isnull().sum()

#df_summary(Tdata)
Tdata.drop('customerID',axis=1, inplace=True)

Tdata.columns
Num_cols = Tdata.select_dtypes(include=['float64','int64']).columns.tolist()

Cat_cols = Tdata.select_dtypes(include=['object']).columns.tolist()

print("Number columns : ",Num_cols , "Catogarical columns :" ,Cat_cols,sep="\n")
Tdata[Num_cols].describe()
Tdata[Num_cols].hist(figsize = (10,10));
def cat_col_desc(data):

    col_list = data.select_dtypes(include=['object']).columns.tolist()

    for i in col_list: 

        print("Variable :", i)

        print("Count of unique values :", len(set( data[i])))

        print("Unique values : " ,set( data[i]))

        print("================================")
cat_col_desc(Tdata)

#cat_col_desc(Tdata.loc[:, Tdata.columns != 'customerID']) # Since we have already dropped ID column no need
sns.countplot(x="Churn", hue="gender", data=Tdata)
Binary_class = Tdata[Cat_cols].nunique()[Tdata[Cat_cols].nunique() == 2].keys().tolist()

Multi_class =  Tdata[Cat_cols].nunique()[Tdata[Cat_cols].nunique() > 2].keys().tolist()

print(Binary_class)

print(Multi_class)
fig, axes = plt.subplots(nrows = 3,ncols = 3,figsize = (15,12))

for i, item in enumerate(Multi_class):

    if i < 3:

        ax = Tdata[item].value_counts().plot(kind = 'bar',ax=axes[i,0],rot = 0)

        

    elif i >=3 and i < 6:

        ax = Tdata[item].value_counts().plot(kind = 'bar',ax=axes[i-3,1],rot = 0)

        

    elif i < 9:

        ax = Tdata[item].value_counts().plot(kind = 'bar',ax=axes[i-6,2],rot = 0)

    ax.set_title(item)
#Tdata.groupby('V2')['Churn'].count().plot(kind = 'barh')

#sns.countplot(x="Churn", hue="V2", data=Tdata)

sns.catplot(x="V2", hue="Churn", col="Churn",data=Tdata, kind="count",height=4, aspect=.7,  facecolor=(0, 0, 0, 0),

linewidth=5,edgecolor=sns.color_palette("dark", 5))
#Tdata.groupby('V3')['Churn'].count().plot(kind = 'barh')

sns.catplot(x="V3", hue="Churn", col="Churn",data=Tdata, kind="count",height=4, aspect=.7,  facecolor=(0, 0, 0, 0),

linewidth=5,edgecolor=sns.color_palette("dark", 5))
#Tdata.groupby('V5')['Churn'].count().plot(kind = 'barh')

sns.catplot(x="V5", hue="Churn", col="Churn",data=Tdata, kind="count",height=4, aspect=.7,  facecolor=(0, 0, 0, 0),

linewidth=5,edgecolor=sns.color_palette("dark", 5))
#Tdata.groupby('V15')['Churn'].count().plot(kind = 'barh')

sns.catplot(x="V15", hue="Churn", col="Churn",data=Tdata, kind="count",height=4, aspect=.7,  facecolor=(0, 0, 0, 0),

linewidth=5,edgecolor=sns.color_palette("dark", 5))
#Label encoding Binary columns

le = LabelEncoder()

for i in Binary_class :

    Tdata[i] = le.fit_transform(Tdata[i])
Tdata[Binary_class].shape
Tdata[Num_cols].shape
# Split multi class catergory columns as dummies  

Tdata_Dummy = pd.get_dummies(Tdata[Multi_class])

Tdata_Dummy.head()

New_df = pd.concat([Tdata[Num_cols],Tdata[Binary_class],Tdata_Dummy], axis=1)

New_df.shape
# Data to plot

labels =New_df['Churn'].value_counts(sort = True).index

sizes = New_df['Churn'].value_counts(sort = True)



colors = ["whitesmoke","red"]

explode = (0.1,0)  # explode 1st slice

 

plt.pie(sizes, explode=explode, labels=labels, colors=colors,autopct='%1.1f%%', shadow=True, startangle=270,)



plt.title('Percent of churn in customer')

plt.show()
#correlation

corr = New_df.corr()
# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

#cmap = sns.diverging_palette(220, 10, as_cmap=True)

cmap=sns.light_palette("seagreen", reverse=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5});
X = New_df.loc[:, New_df.columns != 'Churn']

y = New_df["Churn"]



x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state =1)
params = {'random_state': 0, 'n_jobs': 4, 'n_estimators': 5000, 'max_depth': 8}



# Fit RandomForest Classifier

clf = RandomForestClassifier(**params)

clf = clf.fit(X, y)

# Plot features importances

imp = pd.Series(data=clf.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10,12))

plt.title("Feature importance")

ax = sns.barplot(y=imp.index, x=imp.values, palette="Blues_d", orient='h')
# V4 distibution 

g = sns.kdeplot(New_df.V4[(New_df["Churn"] == 0) ], color="Red", shade = True)

g = sns.kdeplot(New_df.V4[(New_df["Churn"] == 1) ], ax =g, color="Blue", shade= True)

g.set_xlabel("V4")

g.set_ylabel("Frequency")

plt.title('Distribution of tenure comparing with churn feature')

g = g.legend(["Not Churn","Churn"])
print(sorted(New_df["V4"].unique()))
X = New_df.loc[:, New_df.columns != 'Churn']

y = New_df["Churn"]



x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state =1)
print('The number of samples into the Train data is {}.'.format(x_train.shape[0]))

print('The number of samples into the test data is {}.'.format(x_test.shape[0]))
logistic_model = LogisticRegression()

logistic_model.fit(x_train,y_train)
accuracy = logistic_model.score(x_test,y_test)

print("Logistic Regression accuracy is :",accuracy*100)
#from sklearn.metrics import recall_score, confusion_matrix, precision_score, f1_score, accuracy_score, classification_report

#for Logistic Regression

cm_lr = confusion_matrix(y_test,logistic_model.predict(x_test))



# %% confusion matrix visualization

import seaborn as sns

f, ax = plt.subplots(figsize = (5,5))

sns.heatmap(cm_lr, annot = True, linewidths = 0.5, color = "red", fmt = ".0f", ax=ax)

plt.xlabel("y_predicted")

plt.ylabel("y_true")

plt.title("Confusion Matrix of Logistic Regression")

plt.show()
# Define Model parameters to tune

model_parameters = {

        'C': [1,10,100,1000],

        'class_weight': ['balanced', None]

    }
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

# Gridsearch the parameters to find the best parameters. Using L2 penalty

model = LogisticRegression(penalty='l2')

gscv = GridSearchCV(estimator=model, 

                    param_grid=model_parameters, 

                    cv=5, 

                    verbose=1, 

                    n_jobs=-1,

                    scoring='f1')



gscv.fit(x_train, y_train)
print('The best parameter are -', gscv.best_params_)
# Re-fit the model with the best parameters

#final_mod = LogisticRegression(penalty='l2', C=1000, class_weight='balanced')

final_mod = LogisticRegression(**gscv.best_params_)

final_mod.fit(x_train,y_train)



# View the model coefficients

list(zip(x_train.columns, final_mod.coef_[0]))
confusion_matrix(y_test,final_mod.predict(x_test))
from sklearn.metrics import classification_report



print(classification_report(y_test,final_mod.predict(x_test)))
# Generate ROC



import scikitplot as skplt

import matplotlib.pyplot as plt



#y_test = # ground truth labels

#predict_proba = # predicted probabilities generated by sklearn classifier

skplt.metrics.plot_roc_curve(y_test,final_mod.predict_proba(x_test))

plt.show()
auc = roc_auc_score(y_test, final_mod.predict(x_test))

print('AUC: %.3f' % auc)

# calculate roc curve

fpr, tpr, thresholds = roc_curve(y_test, final_mod.predict(x_test))

# plot no skill

plt.plot([0, 1], [0, 1], linestyle='--')

# plot the precision-recall curve for the model

plt.plot(fpr, tpr, marker='.')

# show the plot

plt.show()