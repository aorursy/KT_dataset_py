import warnings

warnings.simplefilter('ignore')



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os#Walking through directores



import plotly.graph_objects as go # Generate Graphs

from plotly.subplots import make_subplots #To Create Subplots



from sklearn import decomposition #pca

from sklearn.preprocessing import StandardScaler # Standardization ((X - X_mean)/X_std)



from sklearn.neighbors import KNeighborsClassifier #KNN Model

from sklearn.ensemble import RandomForestClassifier #RandomForest Model

from sklearn.linear_model import LogisticRegression #Logistic Model



from sklearn.model_selection import train_test_split # Splitting into train and test



from sklearn.model_selection import GridSearchCV# Hyperparameter Tuning

from sklearn.model_selection import cross_val_score#cross validation score



from sklearn.metrics import classification_report # text report showing the main classification metrics

from sklearn.metrics import confusion_matrix #to get confusion_matirx 



pd.set_option('display.max_columns', None)#Setting Max Columns Display to Max inorder to get glance of all features in dataframe
missing_values = ['?', '--', ' ', 'NA', 'N/A', '-'] #Sometimes Missing Values are't in form of NaN

df = pd.read_csv('../input/cardiovascular-disease-dataset/cardio_train.csv', delimiter = ';', na_values = missing_values)

print('There are Total {} datapoints in the dataset with {} Features listed as {}:'.format(df.shape[0], df.shape[1], df.columns.values))
df.head()
features_with_null = [features for feature in df.columns if df[feature].isnull().sum()>0]

if features_with_null:

    print('Features with Null Values {}'.format(features_with_null))

else:

    print('Dataset contains no Null Values')
df.info()
df.drop(columns=['id'], inplace=True)
duplicate_sum = df.duplicated().sum()

if duplicate_sum:

    print('Duplicates Rows in Dataset are : {}'.format(duplicate_sum))

else:

    print('Dataset contains no Duplicate Values')
duplicated = df[df.duplicated(keep=False)]

duplicated = duplicated.sort_values(by=['gender', 'height', 'weight'], ascending= False)

duplicated.head()
df.drop_duplicates(keep = 'first', inplace = True)

print('Total {} datapoints remaining with {} features'.format(df.shape[0], df.shape[1]))
Continuous_features = [feature for feature in df.columns if len(df[feature].unique())>25]

print('Continuous Values are : {}'.format(Continuous_features))
df[Continuous_features].head()
df[Continuous_features].describe()
fig = go.Figure()



fig.add_trace(go.Box(x=df['height'], name = 'Height', boxpoints='outliers',))

fig.add_trace(go.Box(x=df['weight'], name = 'Weight', boxpoints='outliers',))



fig.update_layout(title_text="Box Plot for Weight and Height with Outliers")

fig.show()


fig = make_subplots(rows=1, cols=2, subplot_titles=("Height Distribution", "Weight Distribution"))



trace0 = go.Histogram(x=df['height'], name = 'Height')

trace1 = go.Histogram(x=df['weight'], name = 'Weight')





fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 2)



fig.update_xaxes(title_text="Height", row=1, col=1)

fig.update_yaxes(title_text="Total Count", row=1, col=1)



fig.update_xaxes(title_text="Weight", row=1, col=2)

fig.update_yaxes(title_text="Total Count", row=1, col=2)



fig.update_layout(title_text="Histograph", height=700)





fig.show()
def outliers(df_out, drop = False):

    for each_feature in df_out.columns:

        feature_data = df_out[each_feature]

        Q1 = np.percentile(feature_data, 25.) # 25th percentile of the data of the given feature

        Q3 = np.percentile(feature_data, 75.) # 75th percentile of the data of the given feature

        IQR = Q3-Q1 #Interquartile Range

        outlier_step = IQR * 1.5 #That's we were talking about above

        outliers = feature_data[~((feature_data >= Q1 - outlier_step) & (feature_data <= Q3 + outlier_step))].index.tolist()  

        print('For the feature {}, No of Outliers is {}'.format(each_feature, len(outliers)))

outliers(df[['height', 'weight']])
outline_free_df = df.copy()

outline_free_df[['height', 'weight']] = np.log(outline_free_df[['height', 'weight']])

outliers(outline_free_df[['height', 'weight']])
outline_free_df = outline_free_df[(outline_free_df['weight'] > outline_free_df['weight'].quantile(0.005)) & (outline_free_df['weight'] < outline_free_df['weight'].quantile(0.995))]

outline_free_df = outline_free_df[(outline_free_df['height'] > outline_free_df['height'].quantile(0.005)) & (outline_free_df['height'] < outline_free_df['height'].quantile(0.995))]

outliers(outline_free_df[['height', 'weight']])
print('Handling outliners cost us {} datapoints'.format(len(df)-len(outline_free_df)))
outline_free_df = outline_free_df[outline_free_df['ap_lo']>=0]

outline_free_df = outline_free_df[outline_free_df['ap_hi']>=0]
print('There are total {} observations where ap_hi < ap_lo'.format(len(outline_free_df[outline_free_df['ap_hi'] < outline_free_df['ap_lo']])))
cleaned_data = outline_free_df[outline_free_df['ap_hi'] >= outline_free_df['ap_lo']].reset_index(drop=True)

print('Total observations preserved : {}'.format(len(cleaned_data)))
print('As per our assumptions we have total {} outliers'.format(len(cleaned_data[(cleaned_data["ap_hi"]>250) | (cleaned_data["ap_lo"]>200)])))
cleaned_data = cleaned_data[(cleaned_data["ap_hi"]<=250) & (cleaned_data["ap_lo"]<=200)]
print('Total {} datapoints remaining with {} features'.format(cleaned_data.shape[0], cleaned_data.shape[1]))
cleaned_data.head()
cleaned_data['age'] = cleaned_data['age'].div(365).apply(lambda x: int(x))
fig = go.Figure()

fig.add_trace(go.Histogram(x=cleaned_data['age'], name = 'Age'))

fig.show()
duplicate_sum = cleaned_data.duplicated().sum()

if duplicate_sum:

    print('Duplicates Rows in Dataset are : {}'.format(duplicate_sum))

else:

    print('Dataset contains no Duplicate Values')
duplicated = cleaned_data[cleaned_data.duplicated(keep=False)]

duplicated = duplicated.sort_values(by=['gender', 'height', 'weight'], ascending= False)

duplicated.head()
cleaned_data.drop_duplicates(keep = 'first', inplace = True)

print('Total {} datapoints remaining with {} features'.format(cleaned_data.shape[0], df.shape[1]))
fig = go.Figure(data=[go.Bar(x = cleaned_data[cleaned_data['cardio'] == 0]['age'].value_counts().index.to_list(), 

                             y =cleaned_data[cleaned_data['cardio'] == 0]['age'].value_counts().values, name = 'Non CVD'),

                      go.Bar(x = cleaned_data[cleaned_data['cardio'] == 1]['age'].value_counts().index.to_list(), 

                             y =cleaned_data[cleaned_data['cardio'] == 1]['age'].value_counts().values, name = 'CVD')]

               )



fig.update_layout(barmode='group', xaxis_tickangle=-45, title_text="Distribution of Age groups grouped by Target Value", 

                  yaxis=dict(

        title='Total Count',

        titlefont_size=16,

        tickfont_size=14,

    ),     xaxis=dict(

        title='Age',

        titlefont_size=16,

        tickfont_size=14,

    ))

fig.show()
fig = go.Figure([go.Pie(labels=cleaned_data[cleaned_data['cardio'] == 0]['age'].value_counts().index.to_list(),values=cleaned_data[cleaned_data['cardio'] == 0]['age'].value_counts().values)])

fig.update_layout(title_text="Distribution of Age group for Non CVD", template="plotly_white")

fig.data[0].marker.line.color = 'rgb(255, 255, 255)'

fig.data[0].marker.line.width = 2

fig.update_traces(textposition='inside')

fig.show()
fig = go.Figure([go.Pie(labels=cleaned_data[cleaned_data['cardio'] == 1]['age'].value_counts().index.to_list(),values=cleaned_data[cleaned_data['cardio'] == 1]['age'].value_counts().values)])

fig.update_layout(title_text="Distribution of Age group for CVD", template="plotly_white")

fig.data[0].marker.line.color = 'rgb(255, 255, 255)'

fig.data[0].marker.line.width = 2

fig.update_traces(textposition='inside')

fig.show()


fig = make_subplots(rows=2, cols=2, subplot_titles=("Height Distribution for CVD Population", "Height Distribution for non CVD Population", "Weight Distribution for CVD Population", "Weight Distribution for non CVD Population"))



trace0 = go.Histogram(x=np.exp(cleaned_data[cleaned_data['cardio'] == 0]['height']), name = 'Non CVD')

trace1 = go.Histogram(x=np.exp(cleaned_data[cleaned_data['cardio'] == 1]['height']), name = 'CVD')



trace2 = go.Histogram(x=np.exp(cleaned_data[cleaned_data['cardio'] == 0]['weight']), name = 'Non CVD')

trace3 = go.Histogram(x=np.exp(cleaned_data[cleaned_data['cardio'] == 1]['weight']), name = 'CVD')



fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 2)

fig.append_trace(trace2, 2, 1)

fig.append_trace(trace3, 2, 2)



fig.update_xaxes(title_text="Height", row=1, col=1)

fig.update_yaxes(title_text="Total Count", row=1, col=1)



fig.update_xaxes(title_text="Height", row=1, col=2)

fig.update_yaxes(title_text="Total Count", row=1, col=2)



fig.update_xaxes(title_text="Weight", row=2, col=1)

fig.update_yaxes(title_text="Total Count", row=2, col=1)



fig.update_xaxes(title_text="Weight", row=2, col=2)

fig.update_yaxes(title_text="Total Count", row=2, col=2)



fig.show()
fig = go.Figure(data=[go.Bar(x = cleaned_data[cleaned_data['cardio'] == 0]['ap_hi'].value_counts().index.to_list(), 

                             y =cleaned_data[cleaned_data['cardio'] == 0]['ap_hi'].value_counts().values, name = 'Non CVD'),

                      go.Bar(x = cleaned_data[cleaned_data['cardio'] == 1]['ap_hi'].value_counts().index.to_list(), 

                             y =cleaned_data[cleaned_data['cardio'] == 1]['ap_hi'].value_counts().values, name = 'CVD')]

               )



fig.update_layout(barmode='group', xaxis_tickangle=-45, title_text="Distribution of Systolic blood pressure Values grouped by Target Value", 

                  yaxis=dict(

        title='Total Count',

        titlefont_size=16,

        tickfont_size=14,

    ),     xaxis=dict(

        title='Systolic Blood Pressure Values',

        titlefont_size=16,

        tickfont_size=14,

    ))

fig.show()
fig = go.Figure([go.Pie(labels=cleaned_data[cleaned_data['cardio'] == 0]['ap_hi'].value_counts().index.to_list(),values=cleaned_data[cleaned_data['cardio'] == 0]['ap_hi'].value_counts().values)])

fig.update_layout(title_text="Distribution of Systolic blood pressure values for Non CVD", template="plotly_white")

fig.data[0].marker.line.color = 'rgb(255, 255, 255)'

fig.data[0].marker.line.width = 2

fig.update_traces(textposition='inside')

fig.show()
fig = go.Figure([go.Pie(labels=cleaned_data[cleaned_data['cardio'] == 1]['ap_hi'].value_counts().index.to_list(),values=cleaned_data[cleaned_data['cardio'] == 1]['ap_hi'].value_counts().values)])

fig.update_layout(title_text="Distribution of Systolic blood pressure values for CVD", template="plotly_white")

fig.data[0].marker.line.color = 'rgb(255, 255, 255)'

fig.data[0].marker.line.width = 2

fig.update_traces(textposition='inside')

fig.show()
fig = go.Figure(data=[go.Bar(x = cleaned_data[cleaned_data['cardio'] == 0]['ap_lo'].value_counts().index.to_list(), 

                             y =cleaned_data[cleaned_data['cardio'] == 0]['ap_lo'].value_counts().values, name = 'Non CVD'),

                      go.Bar(x = cleaned_data[cleaned_data['cardio'] == 1]['ap_lo'].value_counts().index.to_list(), 

                             y =cleaned_data[cleaned_data['cardio'] == 1]['ap_lo'].value_counts().values, name = 'CVD')]

               )



fig.update_layout(barmode='group', xaxis_tickangle=-45, title_text="Distribution of Diastolic blood pressure Values grouped by Target Value", 

        yaxis=dict(

        title='Total Count',

        titlefont_size=16,

        tickfont_size=14,

    ),     xaxis=dict(

        title='Diastolic Blood Pressure Values',

        titlefont_size=16,

        tickfont_size=14,

    ))

fig.show()
fig = go.Figure([go.Pie(labels=cleaned_data[cleaned_data['cardio'] == 0]['ap_lo'].value_counts().index.to_list(),values=cleaned_data[cleaned_data['cardio'] == 0]['ap_lo'].value_counts().values)])

fig.update_layout(title_text="Distribution of Daistolic blood pressure values for Non CVD", template="plotly_white")

fig.data[0].marker.line.color = 'rgb(255, 255, 255)'

fig.data[0].marker.line.width = 2

fig.update_traces(textposition='inside')

fig.show()
fig = go.Figure([go.Pie(labels=cleaned_data[cleaned_data['cardio'] == 1]['ap_lo'].value_counts().index.to_list(),values=cleaned_data[cleaned_data['cardio'] == 1]['ap_lo'].value_counts().values)])

fig.update_layout(title_text="Distribution of Daistolic blood pressure values for CVD", template="plotly_white")

fig.data[0].marker.line.color = 'rgb(255, 255, 255)'

fig.data[0].marker.line.width = 2

fig.update_traces(textposition='inside')

fig.show()
Categorial_features = [feature for feature in cleaned_data.columns if len(cleaned_data[feature].unique())<25]

print('Categorial Values are : {}'.format(Categorial_features))
for each_feature in Categorial_features:

    print('No of Categorial Values in Feature {} is {} as {}'.format(each_feature, len(cleaned_data[each_feature].unique()), cleaned_data[each_feature].unique()))
fig = go.Figure([go.Pie(labels=['Not Having CVD', 'Having CVD'],values=cleaned_data['cardio'].value_counts().values)])

fig.update_layout(title_text="Pie chart of Target Variable", template="plotly_white")

fig.data[0].marker.line.color = 'rgb(255, 255, 255)'

fig.data[0].marker.line.width = 2

fig.update_traces(hole=.4,)

fig.show()


fig = make_subplots(rows=2, cols=3,subplot_titles=("Alchoal Distribution", "Gender Distribution", "Choslesterol Distribution", "Glucose Distribution", "Smoking Distribution", "Fitness Distribution"), specs=[[{'type':'domain'}, {'type':'domain'}, {'type':'domain'}], [{'type':'domain'}, {'type':'domain'}, {'type':'domain'}]])



fig.add_trace(go.Pie(labels=['Non Alchoalic', 'Alchoalic'],values=cleaned_data['alco'].value_counts().values, name = 'Alchoal Status'), 1, 1)

fig.add_trace(go.Pie(labels=['Female', 'Male'],values=cleaned_data['gender'].value_counts().values, name = 'Gender Status'), 1, 2)



fig.add_trace(go.Pie(labels=['Normal', 'Above Normal', 'Well Above Normal'],values=cleaned_data['cholesterol'].value_counts().values, name = 'Cholesterol Level Status'), 1, 3)

fig.add_trace(go.Pie(labels=['Normal', 'Above Normal', 'Well Above Normal'],values=cleaned_data['gluc'].value_counts().values, name = 'Glucose Level Status'), 2, 1)



fig.add_trace(go.Pie(labels=['Non Smoker', 'Smoker'],values=cleaned_data['smoke'].value_counts().values, name = 'Smoking Status'), 2, 2)

fig.add_trace(go.Pie(labels=['Not Involved in Physical Activites', 'Involved in Physical Activites'],values=cleaned_data['active'].value_counts().values, name = 'Fitness Status'), 2, 3)



fig.update_traces(hole=.4,)

fig.update_layout(title_text="Distribution of Various Categorial Values")



fig.show()

target_value = cleaned_data['cardio']

cleaned_data_for_pca = cleaned_data.drop(['cardio'], axis=1)
scaled_data = StandardScaler().fit_transform(cleaned_data_for_pca)



pca = decomposition.PCA()

pca.n_components = 2

pca_data = pca.fit_transform(scaled_data)



pca_data = np.vstack((pca_data.T, target_value)).T

pca_df = pd.DataFrame(data = pca_data, columns = ('first', 'second', 'label'))
fig = go.Figure(data=go.Scattergl(

    x = pca_df['first'], 

    y = pca_df['second'],

    mode='markers',

    marker_color=pca_df['label']

))



fig.show()
def BMI(data):

    return np.exp(data['weight']) / (np.exp(data['height'])/100)**2 

 

cleaned_data['bmi'] = cleaned_data.apply(BMI, axis=1)
def pulse(data):

    return np.subtract(data['ap_hi'], data['ap_lo'])

 

cleaned_data['pulse'] = cleaned_data.apply(pulse, axis=1)
import seaborn as sns

import matplotlib.pyplot as plt



plt.rcParams['figure.figsize'] = (20, 15) 

sns.heatmap(cleaned_data.corr(), annot = True, linewidths=.5, cmap="YlGnBu")

plt.title('Corelation Between Features', fontsize = 30)

plt.show()
X = cleaned_data.drop(['cardio', 'bmi', 'weight', 'gluc', 'gender', 'smoke', 'alco', 'active'], axis =1)

Y = cleaned_data['cardio']
scaler = StandardScaler()

standard_X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(standard_X, Y, test_size=0.2, random_state=42, shuffle = True)
params = {'n_neighbors':list(range(0, 51)),

          'weights':['uniform', 'distance'],

          'p':[1,2]}
"""knn = KNeighborsClassifier()

knn_grid_cv = GridSearchCV(knn, param_grid=params, cv=10) 

knn_grid_cv.fit(X_train, y_train)

print("Best Hyper Parameters:\n",knn_grid_cv.best_params_)"""



print("Best Hyper Parameters: {'n_neighbors': 50, 'p': 1, 'weights': 'uniform'}")
knn = KNeighborsClassifier(n_neighbors=50, p=1, weights='uniform')

knn.fit(X_train, y_train) 
params = { 

    'n_estimators': [10, 50, 100, 150, 200, 300, 400, 500],

    'max_depth' : [10,20,30,40,50],

    'criterion' : ['entropy','gini']

}
'''rfc_gridcv = RandomForestClassifier(random_state=42)

rfc_gridcv = GridSearchCV(estimator=rfc_gridcv, param_grid=params, cv= 10, n_jobs = -1)

rfc_gridcv.fit(X_train, y_train)

print("Best Hyper Parameters:\n",rfc_gridcv.best_params_)'''



print("Best Hyper Parameters:{'criterion': 'entropy', 'max_depth': 10, 'n_estimators': 100}")

rfc = RandomForestClassifier(random_state=42, n_estimators=100, max_depth= 10, criterion = 'entropy')

rfc.fit(X_train, y_train)
params_for_l1 = { 

    'C' :  np.logspace(0, 4, 10),

    'solver' : ['liblinear', 'saga']

}



params_for_l2 = { 

    'C' :  np.logspace(0, 4, 10),

    'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']

}



params_for_elasticnet = { 

    'C' :  np.logspace(0, 4, 10),

    'l1_ratio' : np.arange (0.1, 1.0, 0.1),

    'solver' : ['saga']

}
'''logreg_with_l1_gridcv = LogisticRegression(penalty = 'l1')

logreg_with_l1_gridcv = GridSearchCV(estimator=logreg_with_l1_gridcv, param_grid=params_for_l1, cv= 10, n_jobs = -1)

logreg_with_l1_gridcv.fit(X_train, y_train)

print("Best Hyper Parameters:\n",logreg_with_l1_gridcv.best_params_)'''



print("Best Hyper Parameters:{'C': 166.81005372000593, 'solver': 'saga'}")
logreg_with_l1 = LogisticRegression(penalty = 'l1', C = 166.81005372000593, solver = 'saga')

logreg_with_l1.fit(X_train, y_train)
'''logreg_with_l2_gridcv = LogisticRegression(penalty = 'l2')

logreg_with_l2_gridcv = GridSearchCV(estimator=logreg_with_l2_gridcv, param_grid=params_for_l2, cv= 10, n_jobs = -1)

logreg_with_l2_gridcv.fit(X_train, y_train)

print("Best Hyper Parameters:\n",logreg_with_l2_gridcv.best_params_)'''



print("Best Hyper Parameters:{'C': 1.0, 'solver': 'liblinear'}")
logreg_with_l2 = LogisticRegression(penalty = 'l2', C = 1.0, solver = 'liblinear')

logreg_with_l2.fit(X_train, y_train)
'''logreg_with_elasticnet_gridcv = LogisticRegression(penalty = 'elasticnet')

logreg_with_elasticnet_gridcv = GridSearchCV(estimator=logreg_with_elasticnet_gridcv, param_grid=params_for_elasticnet, cv= 10, n_jobs = -1)

logreg_with_elasticnet_gridcv.fit(X_train, y_train)

print("Best Hyper Parameters:\n",logreg_with_elasticnet_gridcv.best_params_)'''





print("Best Hyper Parameters:{'C': 1291.5496650148827, 'l1_ratio': 0.6, 'solver': 'saga'}")
logreg_with_elasticnet = LogisticRegression(penalty = 'elasticnet', C = 1291.5496650148827, l1_ratio =  0.6, solver = 'saga')

logreg_with_elasticnet.fit(X_train, y_train)
scores = cross_val_score(knn, X_train, y_train, cv=10)

print('KNN Model gives an average accuracy of {0:.2f} % with minimun of {1:.2f} % and maximum of {2:.2f} % accuracy'.format(scores.mean() * 100, scores.min() * 100, scores.max() * 100))
Y_hat = knn.predict(X_test)

print(classification_report(y_test, Y_hat))
plt.rcParams['figure.figsize'] = (5, 5) 

sns.heatmap(confusion_matrix(y_test, Y_hat), annot = True, linewidths=.5, cmap="YlGnBu")

plt.title('Corelation Between Features')

plt.show()
print('True Positive Cases : {}'.format(confusion_matrix(y_test, Y_hat)[1][1]))

print('True Negative Cases : {}'.format(confusion_matrix(y_test, Y_hat)[0][0]))

print('False Positive Cases : {}'.format(confusion_matrix(y_test, Y_hat)[0][1]))

print('False Negative Cases : {}'.format(confusion_matrix(y_test, Y_hat)[1][0]))
scores = cross_val_score(rfc, X_train, y_train, cv=10)

print('Random Forest Model gives an average accuracy of {0:.2f} % with minimun of {1:.2f} % and maximum of {2:.2f} % accuracy'.format(scores.mean() * 100, scores.min() * 100, scores.max() * 100))
Y_hat = rfc.predict(X_test)

print(classification_report(y_test, Y_hat))
plt.rcParams['figure.figsize'] = (5, 5) 

sns.heatmap(confusion_matrix(y_test, Y_hat), annot = True, linewidths=.5, cmap="YlGnBu")

plt.title('Corelation Between Features')

plt.show()
print('True Positive Cases : {}'.format(confusion_matrix(y_test, Y_hat)[1][1]))

print('True Negative Cases : {}'.format(confusion_matrix(y_test, Y_hat)[0][0]))

print('False Positive Cases : {}'.format(confusion_matrix(y_test, Y_hat)[0][1]))

print('False Negative Cases : {}'.format(confusion_matrix(y_test, Y_hat)[1][0]))
scores = cross_val_score(logreg_with_l1, X_train, y_train, cv=10)

print('Logistic Model with L1 Penalty gives an average accuracy of {0:.2f} % with minimun of {1:.2f} % and maximum of {2:.2f} % accuracy'.format(scores.mean() * 100, scores.min() * 100, scores.max() * 100))
Y_hat = logreg_with_l1.predict(X_test)

print(classification_report(y_test, Y_hat))
plt.rcParams['figure.figsize'] = (5, 5) 

sns.heatmap(confusion_matrix(y_test, Y_hat), annot = True, linewidths=.5, cmap="YlGnBu")

plt.title('Corelation Between Features')

plt.show()
print('True Positive Cases : {}'.format(confusion_matrix(y_test, Y_hat)[1][1]))

print('True Negative Cases : {}'.format(confusion_matrix(y_test, Y_hat)[0][0]))

print('False Positive Cases : {}'.format(confusion_matrix(y_test, Y_hat)[0][1]))

print('False Negative Cases : {}'.format(confusion_matrix(y_test, Y_hat)[1][0]))
scores = cross_val_score(logreg_with_l2, X_train, y_train, cv=10)

print('Logistic Model with L2 Penalty gives an average accuracy of {0:.2f} % with minimun of {1:.2f} % and maximum of {2:.2f} % accuracy'.format(scores.mean() * 100, scores.min() * 100, scores.max() * 100))
Y_hat = logreg_with_l2.predict(X_test)

print(classification_report(y_test, Y_hat))
plt.rcParams['figure.figsize'] = (5, 5) 

sns.heatmap(confusion_matrix(y_test, Y_hat), annot = True, linewidths=.5, cmap="YlGnBu")

plt.title('Corelation Between Features')

plt.show()
print('True Positive Cases : {}'.format(confusion_matrix(y_test, Y_hat)[1][1]))

print('True Negative Cases : {}'.format(confusion_matrix(y_test, Y_hat)[0][0]))

print('False Positive Cases : {}'.format(confusion_matrix(y_test, Y_hat)[0][1]))

print('False Negative Cases : {}'.format(confusion_matrix(y_test, Y_hat)[1][0]))
scores = cross_val_score(logreg_with_elasticnet, X_train, y_train, cv=10)

print('Logistic Model with Elasticnet Penalty gives an average accuracy of {0:.2f} % with minimun of {1:.2f} % and maximum of {2:.2f} % accuracy'.format(scores.mean() * 100, scores.min() * 100, scores.max() * 100))
Y_hat = logreg_with_elasticnet.predict(X_test)

print(classification_report(y_test, Y_hat))
plt.rcParams['figure.figsize'] = (5, 5) 

sns.heatmap(confusion_matrix(y_test, Y_hat), annot = True, linewidths=.5, cmap="YlGnBu")

plt.title('Corelation Between Features')

plt.show()
print('True Positive Cases : {}'.format(confusion_matrix(y_test, Y_hat)[1][1]))

print('True Negative Cases : {}'.format(confusion_matrix(y_test, Y_hat)[0][0]))

print('False Positive Cases : {}'.format(confusion_matrix(y_test, Y_hat)[0][1]))

print('False Negative Cases : {}'.format(confusion_matrix(y_test, Y_hat)[1][0]))