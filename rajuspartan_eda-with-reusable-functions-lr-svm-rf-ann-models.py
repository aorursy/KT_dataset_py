# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import plotly.offline as pyo

import plotly.graph_objs as go

import numpy as np

from plotly import tools

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

pyo.init_notebook_mode()

import warnings

from numpy import percentile

warnings.filterwarnings("ignore")



df=pd.read_csv('../input/heart.csv')

### This function helps in identifying number of NA values present in each column and generates a bar chart plotting overall NA values

#### Pass your dataset as parameter to this function

def plotNAvalues(data):

    df=data.isna().sum().reset_index()

    df.columns=['Variable','Number of NA values']

    trace=go.Bar(

        x=df['Variable'],

        y=df['Number of NA values'],

        marker=dict(color='#FFD700')

        )

    print(df)

    data=[trace]

    layout=go.Layout(

        title='Bar charts',

        xaxis=dict(title= 'Columns'),

        yaxis= dict(title= 'Number of NA values')

        )

    fig = go.Figure(data=data, layout=layout)

    pyo.iplot(fig)



plotNAvalues(df)
###This function helps in plotting distribution of each variable

### Parameters to be passed: 1) Dataset 2) Variable 3) Number of bins you want in the distribution

def Histogram(data,cat1,bins):

    data = [go.Histogram(

        x=data[cat1],

        nbinsx=bins

    )]



    layout = go.Layout(

    title="Histogram" + '   ' + cat1,

         xaxis = dict(title = cat1 + ' ' + "distribution"), # x-axis label

            yaxis = dict(title = "Number of records in respective bins" ) # y-axis label

    )

    fig = go.Figure(data=data, layout=layout)

    pyo.iplot(fig)



   

Histogram(df,'age',5)

Histogram(df,'trestbps',10)

Histogram(df,'chol',10)

Histogram(df,'target',2)

Histogram(df,'fbs',2)



###Similarly you can try distribution plots for other variables also

###This function helps in identifying whether any outliers are present in each continous varaible

### Parameters to be passed 1) dataset 2) continous variable1 3)continous variable2 etc..

def Boxplots(data,cont1,cont2,cont3):

    data = [

    go.Box(

        y=data[cont1],

        name=cont1

        ),

    go.Box(

        y=data[cont2],

        name=cont2

        ),

    go.Box(

        y=data[cont3],

        name=cont3

        )

    ]

    layout = go.Layout(

        title = 'Box plots' + '   ' + cont1 + ' &  ' + cont2 + '  & ' + cont3

    )

    fig = go.Figure(data=data, layout=layout)

    pyo.iplot(fig)



Boxplots(df,'trestbps','chol','thalach')
###This function helps in treating the outliers. It assigns "UPPER FENCE VALUE" to outliers in the top & 

###"LOWER FENCE VALUE"  to outliers in the bottom

####Parameters to be passed: 1) Dataset 2) continous variable

from numpy import percentile

def treatoutliers(data,cont1):

    quartiles = percentile(data[cont1], [25, 50, 75])

    data_max=data[cont1].max()

    data_min=data[cont1].min()

    IQR=(quartiles[2]-quartiles[0])

    for index,row in df.iterrows():

        if (data[cont1][index] > (quartiles[2] + 1.5*IQR)):

            data[cont1][index]= quartiles[2] + 1.5*IQR

        elif(data[cont1][index] < (quartiles[0] - 1.5*IQR)):

            data[cont1][index]= quartiles[0] - 1.5*IQR

        

treatoutliers(df,'trestbps')

treatoutliers(df,'chol')

treatoutliers(df,'thalach')
#### This function helps in plotting stacked bar chart of various continous variables w.r.t a categorical varaible

### Parameters to be passed: 1) Dataset 2)categorical variable 3) continous variable1 4) continous variable2 5) continous variable13



def Bar_Stack_charts(data,cat1,cont1,cont2,cont3):

    data=data.groupby('target').mean().reset_index()

    trace1 = go.Bar(

    x=data[cat1],  # NOC stands for National Olympic Committee

    y=data[cont1],

    name = cont1,

    marker=dict(color='#FFD700') # set the marker color to gold

    )

    trace2 = go.Bar(

    x=data[cat1],  # NOC stands for National Olympic Committee

    y=data[cont2],

    name = cont2,

    marker=dict(color='#9EA0A1') # set the marker color to gold

    )

    trace3 = go.Bar(

    x=data[cat1],  # NOC stands for National Olympic Committee

    y=data[cont3],

    name = cont3,

    marker=dict(color='#CD7F32') # set the marker color to gold

    )

    data = [trace1,trace2,trace3]

    layout = go.Layout(

        title='Bar charts',

        xaxis = dict(title = cat1), # x-axis label

        yaxis = dict(title = cont1 + '  ' + cont2 + '  ' + cont3), # y-axis label

        barmode='stack'

    )

    fig = go.Figure(data=data, layout=layout)

    pyo.iplot(fig)



Bar_Stack_charts(df,'target','trestbps','chol','thalach')
def scatterplot(data,cont1,cont2):

    data = [go.Scatter(

        x = data[cont1],

        y = data[cont2],

        mode = 'markers ',

        marker = dict(      # change the marker style

            size = 10,

            color = 'rgb(51,204,153)',

            symbol = 'pentagon',

            line = dict(

                width = 2,

            )

        )

        )]

    layout = go.Layout(

        title = 'Scatterplot ' + '' + cont1 + '     vs    ' + cont2, # Graph title

        xaxis = dict(title = cont1), # x-axis label

        yaxis = dict(title = cont2), # y-axis label

        hovermode ='closest' # handles multiple points landing on the same vertical

    )

    fig = go.Figure(data=data, layout=layout)

    pyo.iplot(fig)

    

scatterplot(df,'trestbps','chol')

scatterplot(df,'thalach','chol')

scatterplot(df,'thalach','trestbps')

scatterplot(df,'age','trestbps')

###Function helps in plotting bubble graphs  which shows distribution of continous varaibles for each category of Target variable

### Parameters 1)dataset 2) continous varaibles1 3)continous varaibles2 4) continous varaibles3 5) Categorical varaible

def Bubblecharts(data,cont1,cont2,cont3,cat1):

    

    data = [go.Scatter(

            x=df[cont1],

            y=df[cont2],

            #text=df['text2'],  # use the new column for the hover text

            mode='markers',

            marker=dict(size=0.1*df[cont3],color= df[cat1])

        )]

    layout = go.Layout(

        title='Bubble chart' + '  '  + cont1 + ' vs  ' + cont2 + '   Size of the bubbles: ' + cont3+ '    Color :' + cat1,

        xaxis = dict(title = cont1), # x-axis label

        yaxis = dict(title = cont2), # y-axis label

        hovermode='closest'

    )

    fig = go.Figure(data=data, layout=layout)

    pyo.iplot(fig)



Bubblecharts(df,'trestbps','thalach','chol','target')

Bubblecharts(df,'thalach','trestbps','chol','target')

Bubblecharts(df,'chol','trestbps','thalach','target')

Bubblecharts(df,'chol','oldpeak','thalach','target')

#Bubblecharts(df,'thalach','trestbps','chol','exang')





def Heatmaps(data,cat1,cat2,cont1,cont2,cont3):

    trace1 = go.Heatmap(

    x=data[cat1],

    y=data[cat2],

    z=data[cont1],

    colorscale='Jet' # add max/min color values to make each plot consistent

    )

    trace2 = go.Heatmap(

    x=data[cat1],

    y=data[cat2],

    z=data[cont2],

    colorscale='Jet' # add max/min color values to make each plot consistent

    )

    trace3 = go.Heatmap(

    x=data[cat1],

    y=data[cat2],

    z=data[cont3],

    colorscale='Jet' # add max/min color values to make each plot consistent

    )

    fig = tools.make_subplots(rows=1, cols=3,

    subplot_titles=('Sex vs Target : thalach','Sex vs Target : chol', 'Sex vs Target : trestbps'),

    shared_yaxes = True,  # this makes the hours appear only on the left

    )

    fig.append_trace(trace1, 1, 1)

    fig.append_trace(trace2, 1, 2)

    fig.append_trace(trace3, 1, 3)

    fig['layout'].update(      # access the layout directly!

    title='Distributions of value for Sex vs Target combination'

    )

    pyo.iplot(fig)

    

Heatmaps(df,'sex','target','thalach','chol','trestbps')

Heatmaps(df,'fbs','target','thalach','chol','trestbps')

Heatmaps(df,'cp','target','thalach','chol','trestbps')

def Barchart_group(data,cat1,cat2):

    trace1 = go.Bar(

        x=df[cat1].values,

        y=df[df[cat2]==1].groupby(cat1).count()[cat2].values,

        name='Patients with heart disease'

    )

    trace2 = go.Bar(

        x=df[cat1].values,

        y=df[df[cat2]==0].groupby(cat1).count()[cat2].values,

        name='Patients without heart disease'

    )



    data = [trace1, trace2]

    layout = go.Layout(

        barmode='group',

        xaxis=dict(title=cat1 + '  ' + cat2),

        yaxis=dict(title="Number of records")

    )

    

    fig = go.Figure(data=data, layout=layout)

    pyo.iplot(fig)

    

Barchart_group(df,'cp','target')
####Data preparation

from sklearn.model_selection import train_test_split

import pandas as pd

df['agecat']=['Young' if (x <=50) else ('Middleaged' if (x>50 and x<61) else 'Elder') for x in df['age']]

del df['age']



from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X = LabelEncoder()

df['agecat'] = labelencoder_X.fit_transform(df['agecat'])

df=pd.get_dummies(df,columns=['agecat'])

del df['agecat_2']

df['cp_cat']=[1 if x==0 else 0 for x in df['cp']]

del df['cp']

df['ca_cat']=[1 if x==0 else 0 for x in df['ca']]

del df['ca']

df['thal_cat']=[1 if x==2 else 0 for x in df['thal']]

del df['thal']

df['thalachcat']=[1 if x>140 else 0 for x in df['thalach']]

del df['thalach']

df['slope_cat']=[1 if x==2 else 0 for x in df['slope']]

del df['slope']

#df['trestbps']=[1 if x>130 else 0 for x in df['trestbps']]

#del df['trestbps']

y = df['target'].values

df1=df

del df1['target']

X = df1.values

####Splitting the data to train and test sets



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)



                                #####LOGISTIC REGRESSION###########

# Fitting Logistic Regression to the Training set

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)

classifier.fit(X_train, y_train)



# Predicting the Test set results

y_pred = classifier.predict(X_test)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

Accuracy=(cm[0][0] + cm[1][1])/(cm[0][0] + cm[1][1] + cm[0][1] + cm[1][0])

print("Accuracy of Logistic regression model : {}".format(Accuracy*100))

                                  ###KMM###

# Fitting K-NN to the Training set

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors = 3, metric = 'minkowski', p = 2)

classifier.fit(X_train, y_train)



# Predicting the Test set results

y_pred = classifier.predict(X_test)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

Accuracy=(cm[0][0] + cm[1][1])/(cm[0][0] + cm[1][1] + cm[0][1] + cm[1][0])

print("Accuracy of KNN model : {}".format(Accuracy*100))
                                    #####RANDOM FOREST######

# Fitting Random Forest Classification to the Training set

from sklearn.ensemble import RandomForestClassifier

def RF(N):

    classifier = RandomForestClassifier(n_estimators = N, criterion = 'entropy', max_features='auto',random_state = 0)

    classifier.fit(X_train, y_train)



    # Predicting the Test set results

    y_pred = classifier.predict(X_test)



    # Making the Confusion Matrix

    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_test, y_pred)

    return (cm[0][0] + cm[1][1])/(cm[0][0] + cm[1][1] + cm[0][1] + cm[1][0]) 



dict1={}  

for i in range(1,20):

    dict1[i]=RF(i)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

Accuracy= RF(max(dict1, key=lambda k: dict1[k]))

print("Accuracy of Random forest model : {}".format(Accuracy*100)  )



                            ################SVM############################

# Fitting SVM to the Training set

from sklearn.svm import SVC

classifier = SVC(kernel = 'linear', random_state = 0)

classifier.fit(X_train, y_train)



# Predicting the Test set results

y_pred = classifier.predict(X_test)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

(cm[0][0] + cm[1][1])/(cm[0][0] + cm[1][1] + cm[0][1] + cm[1][0])

Accuracy=(cm[0][0] + cm[1][1])/(cm[0][0] + cm[1][1] + cm[0][1] + cm[1][0])

print("Accuracy of SVM model : {}".format(Accuracy*100))
                ###################Kernel-SVM####################



# Fitting Kernel SVM to the Training set

from sklearn.svm import SVC

classifier = SVC(kernel = 'rbf', random_state = 0)

classifier.fit(X_train, y_train)



# Predicting the Test set results

y_pred = classifier.predict(X_test)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

Accuracy=(cm[0][0] + cm[1][1])/(cm[0][0] + cm[1][1] + cm[0][1] + cm[1][0])

print("Accuracy of kernel SVM model : {}".format(Accuracy*100))
import keras

import keras

from keras.models import Sequential

from keras.layers import Dense



# Initialising the ANN

classifier = Sequential()



# Adding the input layer and the first hidden layer

classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 14))





# Adding the output layer

classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))



# Compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



# Fitting the ANN to the Training set

classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)



# Part 3 - Making the predictions and evaluating the model



# Predicting the Test set results

y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

Accuracy=(cm[0][0] + cm[1][1])/(cm[0][0] + cm[1][1] + cm[0][1] + cm[1][0])

print("Accuracy of kernel ANN model : {}".format(Accuracy*100))