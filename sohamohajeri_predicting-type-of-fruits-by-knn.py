import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import plotly.express as px

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import plotly.io as pio

pio.renderers.default='notebook'
df=pd.read_table('../input/fruits-with-colors-dataset/fruit_data_with_colors.txt')
df.head()
df.shape
df.info()
df['fruit_name'].unique()
df['fruit_subtype'].unique()
fig= px.sunburst(data_frame=df,path=['fruit_name','fruit_subtype'], color='mass',values='width', color_continuous_scale='algae')

fig.update_layout(

    title={

        'text': 'Classification of Types and Subtypes of Fruits',

        'y':0.95,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'})

fig.show()
fig=px.scatter_3d(df,x='width',y='height',z='mass',color='fruit_name',color_continuous_scale='teal')

fig.update_layout(

    title={

        'text': 'Width vs. Height vs. Mass for Different Type of Fruits',

        'y':0.92,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'})

fig.show()
ss=StandardScaler()
ss.fit(df[['mass', 'width', 'height']])
scaled=ss.transform(df[['mass', 'width', 'height']])
scaled_df=pd.DataFrame(data=scaled, columns=df.columns[3:6])
scaled_df.head()
scaled_df.shape
X=scaled_df

y=df['fruit_name']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
error_rate=[]

for n in range(1,40):

    knc=KNeighborsClassifier(n_neighbors=n)

    knc.fit(X_train,y_train)

    prediction=knc.predict(X_test)

    error_rate.append(np.mean(prediction!=y_test))

print(error_rate)
plt.figure(figsize=(9,6))

plt.plot(list(range(1,40)), error_rate, color='royalblue', marker='o',linewidth=2, markersize=12, markerfacecolor='deeppink', markeredgecolor='deeppink' )

plt.xlabel('Number of Neighbors', fontsize=12)

plt.ylabel('Error Rate', fontsize=12)

plt.title('Error Rate Versus Number of Neighbors by Elbow Method', fontsize=15)

plt.show()
knc=KNeighborsClassifier(n_neighbors=2)

knc.fit(X_train,y_train)
prediction_knn=knc.predict(X_test)
print(confusion_matrix(y_test,prediction_knn))

print('\n')

print(classification_report(y_test,prediction_knn))

print('\n')

print('Accuracy Score: ',round(accuracy_score(y_test,prediction_knn), ndigits=2))
scaled_df.head()
knc.predict([[0.529442, 1.598690, -0.291397]])
df['fruit_name'].iloc[0]
knc.predict([[-1.413709,-1.117409,-2.218131]])
df['fruit_name'].iloc[3]