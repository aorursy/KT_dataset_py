# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import seaborn as sns



from plotly import __version__

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import cufflinks as cf

cf.go_offline()

init_notebook_mode(connected=True)





from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Activation

from tensorflow.keras.optimizers import Adam

from sklearn.metrics import accuracy_score,classification_report, roc_auc_score, roc_curve, confusion_matrix, auc







# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory'';

'''

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

'''

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/africa-economic-banking-and-systemic-crisis-data/african_crises.csv")
#Always have a copy of the orginal dataframe

df_copy1= df.copy()



#Show few samples

df.sample(5)
#Get a glance at columns and data types 

df.info()
#More info about the dataframe and data

df.describe()
#Define function to return feature information

#create a dictionary {column name : column info}



feature_dict={ 

        'case':"A number which denotes a specific country", 

        'cc3': "A 3-letter country code", 

        'country':"The name of the country", 

        'year':"The year of the observation", 

        'systemic_crisis':" '0' means that no currency crisis  and '1' means that a currency crisis occurred in the year", 

        'exch_usd':"The exchange rate of the country vis-a-vis the USD",

       'domestic_debt_in_default':" '0' means that no sovereign domestic debt  and '1' means that a sovereign domestic debt occurred in the year", 

       'sovereign_external_debt_default':" '0' means that no sovereign external debt  and '1' means that a sovereign external debt occurred in the year",

       'gdp_weighted_default':"The total debt in default vis-a-vis the GDP", 

       'inflation_annual_cpi':"The annual CPI inflation rate", 

       'independence' :" '0' means no independence and '1' means independence",

       'currency_crises':  " '0' means that no currency crisis  and '1' means that an currency crisis occurred in the year", 

       'inflation_crises': " '0' means that no inflation crisis  and '1' means that an inflation crisis occurred in the year", 

       'banking_crisis':" 'no_crisis' means that no banking crisis  and 'crisis' means that a banking crisis occurred in the year"

        

        }

#Define a function that returns column information given column name

def feature_info(text):

    if (feature_dict.get(text)):

        return feature_dict[text]

    else:

        print(text,"column does not exist.Check the spelling")
feature_info('banking_crisis')
#plot the numbers 

sns.countplot(x='banking_crisis',data=df,palette=['red','navy'])

#to get the exact number.Run the line below

df['banking_crisis'].value_counts()
plt.figure(figsize=(12,6))

sns.countplot(x='country', hue='banking_crisis', data=df,palette=['red','navy'])

plt.xticks(rotation=90)
#Get  countries and no of banking_Crisis

nbcrisis=df.query("banking_crisis == 'crisis'")['country'].value_counts()

nbcrisis
plt.figure(figsize=(12,6))

sns.countplot(x='country', hue='banking_crisis',order=nbcrisis.index, data=df,palette=['red','navy'])

plt.xticks(rotation=90)
#Order by no of crisis

nscrisis=df.query("systemic_crisis == 1")['country'].value_counts()



plt.figure(figsize=(12,6))

sns.countplot(x='country',hue='systemic_crisis',order=nscrisis.index ,data=df,palette=['navy','red'])

plt.xticks(rotation=90)
#To get feature information 

feature_info('currency_crises')
#Order by no of crisis

nccrisis=df.query("currency_crises == 1")['country'].value_counts()



plt.figure(figsize=(12,6))

sns.countplot(x='country',hue='currency_crises',order=nccrisis.index ,data=df,palette=['navy','red','olive'])

plt.xticks(rotation=90)
#To get feature information 

feature_info('inflation_crises')
#Order by no of crisis

nicrisis=df.query("inflation_crises == 1")['country'].value_counts()



plt.figure(figsize=(12,6))

sns.countplot(x='country',hue='inflation_crises',order=nicrisis.index ,data=df, palette=['navy','red'])

plt.xticks(rotation=90)
#To get feature information 

feature_info('exch_usd')
plt.figure(figsize=(15,6))

sns.boxplot(x='country', y='exch_usd', data=df[['country','exch_usd','banking_crisis']])

plt.xticks(rotation=90)
df.groupby('country')['exch_usd'].mean().sort_values()
#For low rate countries

plt.figure(figsize=(15,6))                                                 

sns.boxplot(x='country', y='exch_usd', data=df.query(" country in ['Egypt','Zambia','South Africa','Morocco'] ")[['country','exch_usd','banking_crisis']],hue='banking_crisis',palette=['navy','red'])
#For the rest 

plt.figure(figsize=(15,6))                                                 

sns.boxplot(x='country', y='exch_usd', data=df.query(" country not in ['Egypt','Zambia','South Africa','Morocco'] ")[['country','exch_usd','banking_crisis']],hue='banking_crisis', palette=['red','navy',])

plt.xticks(rotation=90)
plt.figure(figsize=(15,10))

plt.title('USD ecxhange rate over the years')

sns.lineplot(x='year',y='exch_usd',data=df[['country','year','exch_usd']],hue='country')

#Play with labels to see individual figures

fig = go.Figure()

for country, details in df.groupby('country'):

    fig.add_scatter(x=details.year,y=details.exch_usd,name=country,mode='lines')



iplot(fig)
feature_info('inflation_annual_cpi')
plt.figure(figsize=(15,10))

plt.title('Inflation rate over the years')

sns.lineplot(x='year',y='inflation_annual_cpi',data=df[['country','year','inflation_annual_cpi']],hue='country')
#Average inflation per country

df.groupby('country')['inflation_annual_cpi'].mean().sort_values(ascending=False)
#Remove countries with high inflation rate like Zimbabwe and Angola

plt.figure(figsize=(10,8))

sns.scatterplot(x='year',y='inflation_annual_cpi',data=df_copy1.query("country not in ['Zimbabwe','Angola'] "), hue='country')
import plotly.express as px



subset_df=df.query("country not in ['Zimbabwe','Angola']")

#Replace subset_df by df if you want to include Zim and Angola

fig=px.scatter(subset_df,x='year',y='inflation_annual_cpi',color="country") 

fig.show()

#Use plotly for interactive graphs

#Click on label of Zim, Angola to remove them from the interactive graph

fig = go.Figure()

#To remove Zim and Angola from the dataset

#subset_df=df.query("country not in ['Zimbabwe','Angola']")



for country, details in df.groupby('country'):   #replace df by subset_df if you wanna remove Zim and Ang

    fig.add_scatter(x=details.year,y=details.inflation_annual_cpi,name=country,mode='lines')

iplot(fig)
feature_info("independence")
#Build a df to compare data before and after independence

df_independence=df.query('independence==1').groupby('country')['year'].min()   #Get independence year for each country

df_min_year=df.groupby('country')['year'].min()                                #Get the min year

df_count_no_ind=df[df['independence']==0].groupby('country')['independence'].count() #No independence data

df_count_ind=df[df['independence']==1].groupby('country')['independence'].count()    #independence data

#concat all 4 dfs

df_compare = pd.concat([df_min_year,df_independence,df_count_no_ind,df_count_ind],axis=1)

df_compare.columns=['Min_year','Ind_year','Data_Ind_0','Data_Ind_1']

#sort by Data_Ind_O to get countries with considerable data before independence

df_compare.sort_values(by='Data_Ind_0',ascending=False)
#In banking crisis

sns.countplot(x='independence',data=df,hue='banking_crisis',palette=['navy','red'])
#Systemic crisis

sns.countplot(x='independence',data=df,hue='systemic_crisis',palette=['navy','red'])
#Currency crisis

sns.countplot(x='independence',data=df,hue='currency_crises',palette=['navy','red','olive'])

#Inflation

#Remove countries with high inflation rate like Zimbabwe and Angola

plt.figure(figsize=(10,8))

sns.scatterplot(x='year',y='inflation_annual_cpi',

                data=df.query("country not in ['Zimbabwe','Angola'] "), hue='independence',palette=['navy','red'])
#Exchange rate

plt.figure(figsize=(10,8))

sns.scatterplot(x='year',y='exch_usd',

                data=df, hue='independence',palette=['navy','red'])
feature_info("domestic_debt_in_default")
sns.countplot(x='domestic_debt_in_default',data=df, palette=['navy','red'])
plt.figure(figsize=(12,4))

plt.xlabel('No of domestic debts')

df.query('domestic_debt_in_default == 1').groupby('country')['domestic_debt_in_default'].count().sort_values().plot(kind='barh')
feature_info("sovereign_external_debt_default")
#No of debts 

sns.countplot(x='sovereign_external_debt_default',data=df,palette=['navy','red'])
plt.figure(figsize=(12,8))

df.query('sovereign_external_debt_default == 1').groupby('country')['sovereign_external_debt_default'].count().sort_values().plot(kind='barh')
feature_info('gdp_weighted_default')
#Use plotly for interactive graphs

#Double click on country name to separate it from the rest

fig = go.Figure()

for country, details in df.groupby('country'):

    fig.add_scatter(x=details.year,y=details.gdp_weighted_default,name=country,mode='lines')

iplot(fig)
df_corr = df.corr()

df_corr
#Add a new column called banking_crisis1 , crisis=1 and no_crisis=0 using lambda expression

df['banking_crisis1']= df.apply(lambda x : 1 if x[-1]=='crisis' else 0, axis=1)

#check both columns

#df.columns

df[['banking_crisis','banking_crisis1']].head(5)
df_corr = df.corr()

plt.figure(figsize=(12,8))

sns.heatmap(df_corr,cmap='YlGnBu')
df_corr['banking_crisis1'].sort_values(ascending=False)
df.isnull().sum()
#Make a copy of dataframe before any transformation

df_copy1 = df.copy()



#print all features 

df.columns
#Analyse case , cc3

#feature_info('case') #return : 'A number which denotes a specifc country'

#feature_info('cc3')  #return : 'A 3-letter country code'

#Delete case, cc3 and duplicate banking_crisis

df = df.drop(['case','cc3','banking_crisis'], axis=1)

#columns deletion

df.columns
#Rename 'banking_crisis1' to banking_crisis

df = df.rename(columns={'banking_crisis1':'banking_crisis'})

df.columns
#Make another copy

df_copy2 = df.copy()



#List all non-numerical columns 

df.select_dtypes(['object']).columns 
country_dummies = pd.get_dummies(df['country'])

country_dummies.head(2)
#Concatenate the dfs

df = pd.concat([df,country_dummies],axis=1)

df = df.drop('country',axis=1)



#check df & columns

#df.head(5)

df.columns
# Something was odd with this column. Unknow value '2'

df['currency_crises'].value_counts()
#Replace value '2' with '1 'in 'currency_crises' column. We suppose it was a mistake!

#.Make another copy of df. Just in case!

df_copy2= df.copy()

df['currency_crises']=df['currency_crises'].apply(lambda x : 1 if x==2 else x)

df['currency_crises'].value_counts()
df.head(5)
#Split training and testing dataset using train_test_split

X= df.drop('banking_crisis', axis=1)

y=df['banking_crisis']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
y_test.value_counts()
#define a dictionary to record models' accuracy

model_accuracy={}
#create model

lr_model = LogisticRegression(max_iter=10000)

#fit & train

lr_model.fit(X_train,y_train)

#predict with test data

predictions_lr = lr_model.predict(X_test)

#Evaluate the model 

print("Confusion matrix : \n {} \n\nAccuracy Score:{}% \n\nClassification Report : \n  {} ".format(

    confusion_matrix(y_test,predictions_lr),

    round(accuracy_score(y_test,predictions_lr),3)*100 ,

    classification_report(y_test,predictions_lr)))
#Save accuracy score in dict 

model_accuracy['Logistic_Regression']= round(accuracy_score(y_test,predictions_lr),3)*100
#create model 

svc_model = SVC()

#fit & train 

svc_model.fit(X_train,y_train)

#Predict with test data

predictions_svm = svc_model.predict(X_test)

#Evaluate the model

print("Confusion matrix : \n {} \n\nAccuracy Score:{}% \n\nClassification Report : \n  {}".format(

      confusion_matrix(y_test,predictions_svm),

      round(accuracy_score(y_test,predictions_svm),3)*100 ,

      classification_report(y_test,predictions_svm)))

#Save accuracy score in dict 

model_accuracy['Support Vector Machine']= round(accuracy_score(y_test,predictions_svm),3)*100
#create the model with k=1

knn_model = KNeighborsClassifier(n_neighbors=1)

#train&fit 

knn_model.fit(X_train,y_train)

#predict with test data

predictions_knn = knn_model.predict(X_test)

#Evaluate the model

print("Confusion matrix : \n {} \n\nAccuracy Score:{}% \n\nClassification Report : \n  {}".format(

      confusion_matrix(y_test,predictions_knn),

      round(accuracy_score(y_test,predictions_knn),3)*100 ,

      classification_report(y_test,predictions_knn)))
error_rate = []

# Up to 30 iterations 

for i in range(1,31):   

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    predictions_knn_i = knn.predict(X_test)

    error_rate.append(np.mean(predictions_knn_i != y_test))



#Plot the error vs k value

plt.figure(figsize=(10,6))

plt.plot(range(1,31),error_rate,color='blue', marker='o',

         markerfacecolor='orange', markersize=10)

plt.title('Error_Rate vs K_Value')

plt.xlabel('K')

plt.ylabel('Error Rate')
#From previous graph, the least error is at k =11 . 

#Error rate difference btw k1&k11 +/- 1.5%.So the model acc won't improve much.

#Retrain model with k=11



knn = KNeighborsClassifier(n_neighbors=11)

knn.fit(X_train,y_train)

predictions_knn_11 = knn.predict(X_test)

#Evaluate the model

print("Confusion matrix : \n {} \n\nAccuracy Score:{}% \n\nClassification Report : \n  {}".format(

      confusion_matrix(y_test,predictions_knn_11),

      round(accuracy_score(y_test,predictions_knn_11),3)*100 ,

      classification_report(y_test,predictions_knn_11)))
#Save both accuracy scores 

model_accuracy['KNN']= round(accuracy_score(y_test,predictions_knn),3)*100

model_accuracy['KNN_tuned']= round(accuracy_score(y_test,predictions_knn_11),3)*100
#create model

dt_model = DecisionTreeClassifier()

#Train & fit 

dt_model.fit(X_train,y_train)

#predict using test data

predictions_dt = dt_model.predict(X_test)

#Evaluate model

print("Confusion matrix : \n {} \n\nAccuracy Score:{}% \n\nClassification Report : \n  {}".format(

      confusion_matrix(y_test,predictions_dt),

      round(accuracy_score(y_test,predictions_dt),3)*100 ,

      classification_report(y_test,predictions_dt)))

#Save  accuracy score 

model_accuracy['Decision Tree']= round(accuracy_score(y_test,predictions_dt),3)*100
#create

rf_model = RandomForestClassifier()

#fit & train

rf_model.fit(X_train,y_train)

#predict using test data

predictions_rf = rf_model.predict(X_test)

print("Confusion matrix : \n {} \n\nAccuracy Score:{}% \n\nClassification Report : \n  {}".format(

      confusion_matrix(y_test,predictions_rf),

      round(accuracy_score(y_test,predictions_rf),3)*100 ,

      classification_report(y_test,predictions_rf)))
#Save both accuracy scores 

model_accuracy['Random Forest']= round(accuracy_score(y_test,predictions_rf),3)*100
#Scale/Normalize the input

#For more info about this , please read [10]

#create scaler

scaler = MinMaxScaler()



#scale both X_train and X_test

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)



#create a sequential model

model_nn = Sequential()



# hidden layer 1, specify input_dim ()

model_nn.add(Dense(46, activation='relu', input_dim=23))

# hidden layer 2

model_nn.add(Dense(23, activation='relu'))

# output layer

model_nn.add(Dense(units=1,activation='sigmoid'))



#compile

model_nn.compile(optimizer='adam',

            loss='binary_crossentropy', metrics=['accuracy'])



#print the model summary 

print(model_nn.summary())
#Train the model 

#You can try various no of epochs to analyse the model

model_history=model_nn.fit(x=X_train_scaled,y=y_train, epochs=40,validation_data=(X_test_scaled, y_test))

model_history
#save model 

model_nn.save('NN_40.h5') 



#predict using test data

predictions_nn = model_nn.predict_classes(X_test_scaled)



#Evaluate model

print("Confusion matrix : \n {} \n\nAccuracy Score:{}% \n\nClassification Report : \n  {}".format(

      confusion_matrix(y_test,predictions_nn),

      round(accuracy_score(y_test,predictions_nn),3)*100 ,

      classification_report(y_test,predictions_nn)))

#Save accuracy acore 

model_accuracy['Neural Network']= round(accuracy_score(y_test,predictions_nn),3)*100
# Data available in history keys

model_history.history.keys()
# Compare training & testing accuracy

plt.plot(model_history.history['accuracy'])

plt.plot(model_history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='lower left')

plt.show()

#Plot loss on training and validation dataset

plt.plot(model_history.history['loss'])

plt.plot(model_history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='lower left')

plt.show()
#top5 most important features 

#Find out top 5 most important features using random forest model

pd.DataFrame(data = rf_model.feature_importances_*100,

                   columns = ["Importances"],

                   index = pd.DataFrame(X_train).columns).sort_values("Importances", ascending = False)[:5].plot(kind = "barh", color = "blue")



plt.xlabel("Features' Importance (%)")
#Compare all precisions 

pd.DataFrame.from_dict(model_accuracy,orient='index',columns=['Accuracy %']).sort_values(by='Accuracy %',ascending=False)