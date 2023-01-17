

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
# Plotly Libraris
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings("ignore")
passenger_list=pd.read_csv("/kaggle/input/passenger-list-for-the-estonia-ferry-disaster/estonia-passenger-list.csv")
df=passenger_list.copy()
df.head()
df.info()
df.shape
df.columns
df.describe().T
df.isnull().values.any()
df.isnull().sum()
df[df.duplicated() == True]
pclass=df['Category'].value_counts().to_frame().reset_index().rename(columns={'index':'Category','Category':'Count'})

fig = go.Figure(data=[go.Scatter(
    x=pclass['Category'], y=pclass['Count'],
    mode='markers',
    marker=dict(
        color=pclass['Count'],
        size=pclass['Count']*0.2,
        showscale=True
    ))])

fig.update_layout(title='The Type Of Passenger',xaxis_title="Class",yaxis_title="Count",title_x=0.5)
fig.show()
pclass=df['Category'].value_counts().to_frame().reset_index().rename(columns={'index':'Category','Category':'Count'})

colors=['cyan','darkcyan']
fig = go.Figure([go.Pie(labels=pclass['Category'], values=pclass['Count'])])
fig.update_traces(hoverinfo='label+percent', textinfo='percent+value', textfont_size=15,
                 marker=dict(colors=colors, line=dict(color='#000000', width=2)))
fig.update_layout(title="Passengers ",title_x=0.5)
fig.show()
df_name=df.Firstname.value_counts().to_frame().reset_index().rename(columns={'index':'Firstname','Firstname':'Count'}).sort_values('Count',ascending="False")
df_name=df_name[835:849]
df_name

fig = go.Figure(go.Bar(y=df_name['Firstname'], x=df_name['Count'], # Need to revert x and y axis
                      orientation="h")) # default orentation value is "v" - vertical ,we need to change it as orientation="h"
fig.update_layout(title_text='Top 15 Names',
                  xaxis_title="Count ",
                  yaxis_title="Names",
                  title_x=0.5)
fig.show()
df_name=df.Firstname.value_counts().to_frame().reset_index().rename(columns={'index':'Firstname','Firstname':'Count'})




fig = go.Figure([go.Pie(labels=df_name['Firstname'][0:15], values=df_name['Count'])])

fig.update_traces(hoverinfo='label+percent', textinfo='value+percent', textfont_size=12,insidetextorientation='radial')

fig.update_layout(title="Names",title_x=0.5)
fig.show()
df_Lastname=df.Lastname.value_counts().to_frame().reset_index().rename(columns={'index':'Lastname','Lastname':'Count'}).sort_values('Count',ascending="False")
df_Lastname=df_Lastname[759:774]



fig = go.Figure(go.Bar(y=df_Lastname['Lastname'], x=df_Lastname['Count'], # Need to revert x and y axis
                      orientation="h")) # default orentation value is "v" - vertical ,we need to change it as orientation="h"
fig.update_layout(title_text='Top 15 Last Names',
                  xaxis_title="Count ",
                  yaxis_title=" Last Names",
                  title_x=0.5)
fig.show()
df_Country=df['Country'].value_counts()[:12].to_frame().reset_index().rename(columns={'index':'Country','Country':'Count'})

fig = go.Figure(go.Bar(
    x=df_Country['Country'],y=df_Country['Count'],
    marker={'color': df_Country['Count'], 
    'colorscale': 'Viridis'},  
    text=df_Country['Count'],
    textposition = "outside",
))
fig.update_layout(title_text='Top Country',xaxis_title="Country",yaxis_title="Number Of Passengers",title_x=0.5)
fig.show()
#  Bubble Plot with Color gradient

df['age_category']=np.where((df['Age']<19),"below 19",
                                 np.where((df['Age']>18)&(df['Age']<=30),"19-30",
                                    np.where((df['Age']>30)&(df['Age']<=50),"31-50",
                                                np.where(df['Age']>50,"Above 50","NULL"))))

age=df['age_category'].value_counts().to_frame().reset_index().rename(columns={'index':'age_category','age_category':'Count'})


fig = go.Figure(data=[go.Scatter(
    x=age['age_category'], y=age['Count'],
    mode='markers',
    marker=dict(
        color=age['Count'],
        size=age['Count']*0.2,
        showscale=True
    ))])

fig.update_layout(title='Different Age People In Passengers',xaxis_title="Age Category",yaxis_title="Number Of People",title_x=0.5)
fig.show()
#  Basic Box Plot

df_age=df['Age']

fig = go.Figure(go.Box(y=df_age,name=" Age")) # to get Horizonal plot change axis :  x=df_age
fig.update_layout(title="Distribution of Age")
fig.show()
df_age=df['Age']

fig = go.Figure(data=[go.Histogram(x=df_age,  # To get Horizontal plot ,change axis - y=campus_computer
                                  marker_color="darkcyan",
                       xbins=dict(
                      start=0, #start range of bin
                      end=100,  #end range of bin
                      size=10   #size of bin
                      ))])
fig.update_layout(title="Distribution Of Age",xaxis_title="Age",yaxis_title="Counts",title_x=0.5)
fig.show()
df_target=df['Survived'].value_counts().to_frame().reset_index().rename(columns={'index':'Survived','Survived':'Count'})
# 0 = No, 1 = Yes

fig = go.Figure(go.Bar(
    x=df_target['Survived'],y=df_target['Count'],
    marker={'color': df_target['Count'], 
    'colorscale': 'Viridis'},  
    text=df_target['Count'],
    textposition = "outside",
))
fig.update_layout(title_text='Survived',xaxis_title="Survived Class",yaxis_title="Count",title_x=0.5)
fig.show()
df_sex_target=df.groupby(by =['Sex','Survived'])['Age'].count().to_frame().reset_index().rename(columns={'Sex':'Sex','Survived':'Survived','Age':'Count'})
df_sex_target['Survived']=df_sex_target['Survived'].astype('category')

fig = px.bar(df_sex_target, x="Sex", y="Count",color="Survived",barmode="group",
             
             )
fig.update_layout(title_text='Sex with Survived',title_x=0.5)
fig.show()
df_age_sex=df.groupby(by =['age_category','Sex'])['Age'].count().to_frame().reset_index().rename(columns={'age_category':'Age Category','Sex':'Sex','Age':'Count'})
df_age_sex['Sex']=df_age_sex['Sex'].astype('category')
df_age_sex

fig = px.bar(df_age_sex, x="Age Category", y="Count",
             color="Sex",barmode="group")
               
fig.update_layout(title_text='Sex With Age Class',title_x=0.5)
fig.show()
df_cp=df.groupby(by =['Survived','age_category','Sex'])['Age'].count().to_frame().reset_index().rename(columns={'Survived':'Survived','Sex':'Sex','age_category':'Age Category','Age':'Count'})
df_cp['Survived']=df_cp['Survived'].astype('category')
df_cp['Sex']=df_cp['Sex'].astype('category')

# Bar Chart

fig = px.bar(df_cp, x="Survived", y="Count",color="Age Category",barmode="group",
             facet_row="Sex"
             )
fig.update_layout(title_text='Age Category With Survived And Sex',title_x=0.5)
fig.show()
df_sex=df['Sex'].value_counts().to_frame().reset_index().rename(columns={'index':'Sex','Sex':'Count'})

fig = go.Figure([go.Pie(labels=df_sex['Sex'], values=df_sex['Count']
                        ,hole=0.3)])  # can change the size of hole 

fig.update_traces(hoverinfo='label+percent', textinfo='value+percent', textfont_size=15)
fig.update_layout(title="Sex Distribution ",title_x=0.5)
fig.show()
df_target=df.groupby(by =['Survived','age_category'])['Age'].count().to_frame().reset_index().rename(columns={'Survived':'Survived','age_category':'Age Category','Age':'Count'})
df_target['Survived']=df_target['Survived'].astype('category')
df_target

fig = px.bar(df_target, x="Survived", y="Count",color="Age Category",barmode="group",
             
             )
fig.update_layout(title_text='Age Category With Survived',title_x=0.5)
fig.show()
df_cat=df.groupby(by =['Survived','Category'])['Age'].count().to_frame().reset_index().rename(columns={'Survived':'Survived','Category':'Category','Age':'Count'})
df_cat['Survived']=df_cat['Survived'].astype('category')
df_cat

fig = px.bar(df_cat, x="Survived", y="Count",color="Category",barmode="group",
             
             )
fig.update_layout(title_text='Category With Survived',title_x=0.5)
fig.show()
df_sex=df.groupby(by =['Sex','Category'])['Age'].count().to_frame().reset_index().rename(columns={'Sex':'Sex','Category':'Category','Age':'Count'})
df_sex

fig = px.bar(df_sex, x="Sex", y="Count",color="Category",barmode="group",
             
             )
fig.update_layout(title_text='Category With Survived',title_x=0.5)
fig.show()
passenger_list=pd.read_csv("/kaggle/input/passenger-list-for-the-estonia-ferry-disaster/estonia-passenger-list.csv")
df=passenger_list.copy()
df.head()
df=df.drop(['PassengerId', 'Country','Firstname','Lastname'], axis=1)
df.head()
# Feature Generation
df=pd.get_dummies(df,drop_first=True)
df.head()
df.info()
y=df['Survived']
X=df.drop('Survived',axis=1)
X.head()
# Normalize
X = (X - np.min(X)) / (np.max(X) - np.min(X)).values
X.head()
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

X_train,X_test,y_train,y_test=train_test_split(X,y,
                                               test_size=0.2,
                                               random_state=42)
print('X_train',X_train.shape)
print('X_test',X_test.shape)
print('y_train',y_train.shape)
print('y_test',y_test.shape)
from sklearn.linear_model import LogisticRegression

loj = LogisticRegression(solver = "liblinear")
loj_model = loj.fit(X,y)
loj_model
loj_model.intercept_
loj_model.coef_
y_pred_loj = loj_model.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
confusion_matrix(y_test , y_pred_loj)
print("Training Accuracy :", loj_model.score(X_train, y_train))

print("Testing Accuracy :", loj_model.score(X_test, y_test))
print(classification_report(y_test, y_pred_loj))
cross_val_score(loj_model, X_test, y_test, cv = 10).mean()
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb_model = nb.fit(X_train, y_train)
nb_model
y_pred_nb = nb_model.predict(X_test)
accuracy_score(y_test, y_pred_nb)
# Cofusion Matrix
confusion_matrix(y_test , y_pred_nb)
cross_val_score(nb_model, X_test, y_test, cv = 10).mean()
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn_model = knn.fit(X_train, y_train)
knn_model
y_pred_knn = knn_model.predict(X_test)
# Cofusion Matrix
confusion_matrix(y_test , y_pred_knn)
print(classification_report(y_test, y_pred_knn))
knn_params = {"n_neighbors": np.arange(1,50)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, knn_params, cv=10)
knn_cv.fit(X_train, y_train)
print("Best Score_:" + str(knn_cv.best_score_))
print("Best Params: " + str(knn_cv.best_params_))
knn = KNeighborsClassifier(30)
knn_tuned = knn.fit(X_train, y_train)
y_pred_knn_tuned = knn_tuned.predict(X_test)
accuracy_score(y_test, y_pred_knn_tuned)
# Cofusion Matrix
confusion_matrix(y_test , y_pred_knn_tuned)
print(classification_report(y_test, y_pred_knn_tuned))
from sklearn.svm import SVC
svm_model = SVC(kernel = "linear").fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
accuracy_score(y_test, y_pred_svm)
# Cofusion Matrix
confusion_matrix(y_test , y_pred_svm)
from sklearn.neural_network import MLPClassifier
mlpc = MLPClassifier().fit(X_train, y_train)
y_pred_mlpc = mlpc.predict(X_test)
accuracy_score(y_test,y_pred_mlpc)
# Cofusion Matrix
confusion_matrix(y_test , y_pred_mlpc)
from sklearn.tree import DecisionTreeClassifier

cart = DecisionTreeClassifier()
cart_model = cart.fit(X_train, y_train)
y_pred_cart = cart_model.predict(X_test)
accuracy_score(y_test, y_pred_cart)
# Cofusion Matrix
confusion_matrix(y_test , y_pred_cart)
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier().fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
accuracy_score(y_test, y_pred_rf)
# Cofusion Matrix
confusion_matrix(y_test , y_pred_rf)
Importance = pd.DataFrame({"Importance": rf_model.feature_importances_*100},
                         index = X_train.columns)
Importance.sort_values(by = "Importance", 
                       axis = 0, 
                       ascending = True).plot(kind ="barh", color = "r")

plt.xlabel("Variable Significance Levels")
models = [
    knn_model,
    loj_model,
    svm_model,
    nb_model,
    mlpc,
    cart_model,
    rf_model
      
]

for model in models:
    names = model.__class__.__name__
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("-"*28)
    print(names + ":" )
    print("Accuracy: {:.4%}".format(accuracy))
result = []

results = pd.DataFrame(columns= ["Models","Accuracy"])

for model in models:
    names = model.__class__.__name__
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)    
    result = pd.DataFrame([[names, accuracy*100]], columns= ["Models","Accuracy"])
    results = results.append(result)
    
    
sns.barplot(x= 'Accuracy', y = 'Models', data=results, color="r")
plt.xlabel('Accuracy %')
plt.title('Accuracy Ratios of Models');    