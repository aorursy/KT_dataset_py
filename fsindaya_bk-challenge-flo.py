# importing librairis

import pandas as pd

import plotly.graph_objs as go

import plotly.io as pio

import matplotlib.pyplot as plt

import seaborn as sns 

import numpy as np

#importing machine learing models

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, accuracy_score
# Read the churn_modelling dataset

Train_data=pd.read_csv('../input/predicting-churn-for-bank-customers/Churn_Modelling.csv')
# All the columns are not interesting to use right now, therefore lets remove them in our datasets for the moment

#to have fewer columns.

Train_data=Train_data.drop(['RowNumber','CustomerId','Surname'],axis=1)
Train_data.head()
# I am going to check for unique values in the data attributes

Train_data.nunique()
# I am going to check for how many rows and columns is my dataset

Train_data.shape
# I am going to check for the standard deviation, medium, mean to help me understand how spread out a data set is.

# A high standard deviation will implies that, on average, data points will look spread out (far from the average)

#, and a low standard of deviation, close from the average



Train_data_frame1=Train_data[['CreditScore','Geography','Age','Tenure','Balance','EstimatedSalary']]



# Checking for the standard deviation

Train_data_frame1.std()
# Checking for the mean

Train_data_frame1.mean()
# Checking for the median

Train_data_frame1.median()
#percentages of people per category (exited or stayed)



#def Graph():

perc_people_cat=round(100*Train_data['Exited'].value_counts(normalize=True),2)

text=[]

for i in range (0,2):

    text.append(('{}%'.format(perc_people_cat[i])))

    

colors = ['purple','green']





# Building a graph to vizualize what has been found above



data=go.Bar(y=perc_people_cat,

             text=text,

             textposition='auto',



            marker={'color': colors}







            )



layout=  go.Layout(

                        title = ' Data distribution ',

                        xaxis={'title':'0 for stayed and 1 for exited'},

                        yaxis={'title':'Customers in Percentage'}





                            )



                                



fig = go.Figure(data=data, layout=layout)



pio.show(fig)

Train_data.head()
#Geographical distribution

#Exited

customer_france_ex=Train_data[(Train_data['Geography']=='France') & (Train_data['Exited']==1)]

num_customer_france_ex=[customer_france_ex['Exited'].count()]

customer_Spain_ex=Train_data[(Train_data['Geography']=='Spain') & (Train_data['Exited']==1)]

num_customer_Spain_ex=[customer_Spain_ex['Exited'].count()]

customer_Germany_ex=Train_data[(Train_data['Geography']=='Germany') & (Train_data['Exited']==1)]

num_customer_Germany_ex=[customer_Germany_ex['Exited'].count()]

#Stayed

customer_france_st=Train_data[(Train_data['Geography']=='France') & (Train_data['Exited']==0)]

num_customer_france_st=[customer_france_st['Exited'].count()]

customer_Spain_st=Train_data[(Train_data['Geography']=='Spain') & (Train_data['Exited']==0)]

num_customer_Spain_st=[customer_Spain_st['Exited'].count()]

customer_Germany_st=Train_data[(Train_data['Geography']=='Germany') & (Train_data['Exited']==0)]

num_customer_Germany_st=[customer_Germany_st['Exited'].count()]

#Building the graph for vizualization

trace1=go.Bar(x=['exited'],

              y=num_customer_france_ex,

              name='France',

                  

             text='France',

             textposition='auto'



            #marker={'color': colors}



            )

trace2=go.Bar(x=['exited'],

             y=num_customer_Spain_ex,

              name='Spain',

             text='Spain',

             textposition='auto'



            #marker={'color': colors}



            )

trace3=go.Bar(x=['exited'],

                y=num_customer_Germany_ex,

              name='Germany',

              

             text='Germany',

             textposition='auto'



            #marker={'color': colors}



            )

trace4=go.Bar(x=['Stayed'],

              y=num_customer_france_st,

              name='France',

                  

            text='France',

             textposition='auto'



            #marker={'color': colors}



            )

trace5=go.Bar(x=['Stayed'],

              y=num_customer_Spain_st,

              name='Spain',

            text='Spain',

             textposition='auto'



            #marker={'color': colors}



            )

trace6=go.Bar(x=['Stayed'],

              y=num_customer_Germany_st,

              name='Germany',

                  

            text='Germany',

             textposition='auto'



            #marker={'color': colors}



            )

data = [trace1, trace2,trace3,trace4,trace5,trace6]



layout=  go.Layout(

                        title = 'Geographical distribution ',

                        #xaxis={'title':'0 for stayed and 1 for exited'},

                        yaxis={'title':'Customers'},

                        barmode='stack'





                            )                    



fig = go.Figure(data=data, layout=layout)



pio.show(fig)

#Gender Distribution

#Exited

Exited_Male=Train_data[(Train_data['Gender']=='Male') & (Train_data['Exited']==1)].count()

Exited_Female=Train_data[(Train_data['Gender']=='Female') & (Train_data['Exited']==1)].count()

#Stayed

Stayed_Male=Train_data[(Train_data['Gender']=='Male') & (Train_data['Exited']==0)].count()

Stayed_Female=Train_data[(Train_data['Gender']=='Female') & (Train_data['Exited']==0)].count()

#Building the graph

trace1=go.Bar(x=['exited'],

              y=Exited_Male,

              name='Male',

              text='Male',

             textposition='auto'



            )

trace2=go.Bar(x=['exited'],

              y=Exited_Female,

              name='Female',

              text='Female',

             textposition='auto'



            )

trace3=go.Bar(x=['Stayed'],

              y=Stayed_Male,

              name='Male',

                 text='Male',

             textposition='auto'



            )

trace4=go.Bar(x=['Stayed'],

              y=Stayed_Female,

              name='Female',

                 text='Female',

             textposition='auto'



            )

data = [trace1, trace2,trace3,trace4]

layout=  go.Layout(

                        title = 'Gender distribution ',

                        #xaxis={'title':'0 for stayed and 1 for exited'},

                        yaxis={'title':'Customers'},

                        barmode='stack'





                            )                    



fig = go.Figure(data=data, layout=layout)



pio.show(fig)

Train_data.head()
#Age distribution

#Exited customers

Age_Exited_dataframe=Train_data[['Age','Exited']][Train_data['Exited']==1].groupby('Age').count()

#Stayed customers

Age_Stayed_dataframe=Train_data[['Age','Exited']][Train_data['Exited']==0].groupby('Age').count()

Age_Exited_dataframe['Age'] = Age_Exited_dataframe.index

Age_Stayed_dataframe['Age'] = Age_Stayed_dataframe.index

X1=Age_Exited_dataframe['Age']

Y1=Age_Exited_dataframe['Exited']

X2=Age_Stayed_dataframe['Age']

Y2=Age_Stayed_dataframe['Exited']



#BUilding the graph

trace1=go.Bar(x=X1,

              y=Y1,

             name='Exited',



            )

trace2=go.Bar(x=X2,

              y=Y2,

             name='Stayed'



            )

data = [trace1, trace2]

layout=  go.Layout(

                        title = 'Age distribution ',

                        xaxis={'title':'Age'},

                        yaxis={'title':'Customers'},



                            )                    



fig = go.Figure(data=data, layout=layout)



pio.show(fig)

# I am going to get the correlation matrix so that I can understand the relationship between all this variables

#in our datasets. For me to make sure I strengthen my model, I am going to identify and try

#reducing the features in our datasets that are highly correlated.

correlation=Train_data[Train_data.columns[:10]].corr()



# I am going to vizualize it by using  a  heatmap. 



sns.set()

sns.set(font_scale = 1)

sns.heatmap(correlation,cmap='coolwarm', annot = True,fmt = ".2f",annot_kws={'size':15})

plt.show()





#Below is the function which will be helping us to detect outliers

trace = []

def boxplot(df):

    for element in df:

        trace.append(

            go.Box(

                name = element,

                y = df[element]

            )

        )

#credit Score plot box

credit_Score_df=Train_data[Train_data.columns[:1]]

boxplot(credit_Score_df)

data=trace

layout=  go.Layout(

                        title = 'Credit Score plot box ')

                        



                                

fig = go.Figure(data=data, layout=layout)

pio.show(fig)







#Below is the function which will be helping us to detect outliers

trace = []

def boxplot(df):

    for element in df:

        trace.append(

            go.Box(

                name = element,

                y = df[element]

            )

        )

#Age plot box

age_df=Train_data[Train_data.columns[3:4]]

boxplot(age_df)

data=trace

layout=  go.Layout(

                        title = 'Age plot box ')

                        



                                

fig = go.Figure(data=data, layout=layout)

pio.show(fig)



#Below is the function which will be helping us to detect outliers

trace = []

def boxplot(df):

    for element in df:

        trace.append(

            go.Box(

                name = element,

                y = df[element]

            )

        )

#Tenure plot box

Tenure_df=Train_data[Train_data.columns[4:5]]

boxplot(Tenure_df)

data=trace

layout=  go.Layout(

                        title = 'Tenure plot box ')

                        



                                

fig = go.Figure(data=data, layout=layout)

pio.show(fig)

#Below is the function which will be helping us to detect outliers

trace = []

def boxplot(df):

    for element in df:

        trace.append(

            go.Box(

                name = element,

                y = df[element]

            )

        )

#Balance plot box

Balance_df=Train_data[Train_data.columns[5:6]]

boxplot(Balance_df)

data=trace

layout=  go.Layout(

                        title = 'Tenure plot box ')

                        



                                

fig = go.Figure(data=data, layout=layout)

pio.show(fig)

#Below is the function which will be helping us to detect outliers

trace = []

def boxplot(df):

    for element in df:

        trace.append(

            go.Box(

                name = element,

                y = df[element]

            )

        )

#Estimated Salary plot box

EstimatedSalary_df=Train_data[Train_data.columns[9:10]]

boxplot(EstimatedSalary_df)

data=trace

layout=  go.Layout(

                        title = 'Estimated Salary plot box ')

                        



                                

fig = go.Figure(data=data, layout=layout)

pio.show(fig)

# One-Hot encoding our categorical attributes

category_list = ['Geography', 'Gender']

Train_data = pd.get_dummies(Train_data, columns = category_list, prefix = category_list)

Train_data.head()
X = Train_data.drop('Exited', axis=1)

y = Train_data.Exited

labels = X.columns

forest = RandomForestClassifier (n_estimators = 5000, random_state = 0, n_jobs = -1)

forest.fit(X, y)

importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]

# Vizualizing the graph

plt.title('Importance of the features')

plt.bar(range(X.shape[1]), importances[indices], color = "blue", align = "center")

plt.xticks(range(X.shape[1]), labels, rotation=45)

plt.show()
#Splitting my dataset into 70% training and 30% testing



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

#First Ransom Forest

classifier = RandomForestClassifier(n_estimators=5000, random_state=0)  

classifier.fit(X_train, y_train)  

predictions = classifier.predict(X_test)
#then we are going to evaluate how the algorithm we just trained perform by looking at different metrics as it will be printed below

print(classification_report(y_test,predictions ))  

print(accuracy_score(y_test, predictions ))
#Second: Logistic regression

# Initialization of the Logistic Regression

Logistic_model = LogisticRegression(penalty = 'l2', dual = False, tol = 0.0002, C = 1.0, fit_intercept = True,

                            intercept_scaling = 1, class_weight = None, 

                            random_state = None, solver = 'liblinear', max_iter = 100,

                            multi_class = 'ovr', verbose = 2)

# Fitting the model with training data 

Logistic_model.fit(X_train, y_train)

predict_logistic = Logistic_model.predict(X_test)

#then we are going to evaluate how the algorithm we just trained perform by looking at different metrics as it will be printed below

print(classification_report(y_test,predict_logistic ))  

print(accuracy_score(y_test, predict_logistic ))
#Third: K-Nearest Neighbor (KNN)



Knn_model = KNeighborsClassifier(n_neighbors = 5, weights = 'uniform', algorithm = 'ball_tree', leaf_size = 30, p = 2,

                             metric = 'minkowski', metric_params = None)

# Fitting the model with training data 

Knn_model.fit(X_train, y_train)

predict_KNN = Knn_model.predict(X_test)
#then we are going to evaluate how the algorithm we just trained perform by looking at different metrics as it will be printed below

print(classification_report(y_test,predict_KNN ))  

print(accuracy_score(y_test, predict_KNN ))