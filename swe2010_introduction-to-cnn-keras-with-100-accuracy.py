import numpy
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import scikitplot as skplt
from keras.layers.convolutional import Convolution2D
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import plotly.figure_factory as ff 
import  plotly.offline as py
import plotly.graph_objs as go 

from plotly.offline import download_plotlyjs,init_notebook_mode, iplot, plot
from plotly import tools 
py.init_notebook_mode(connected = True)

import cufflinks as cf 
cf.go_offline()

df = pd.read_csv("../input/indian_liver_patient.csv")


df.info()
df.plot(kind='kde')


Data_types = df.dtypes.value_counts()
print(Data_types)

plt.figure(figsize = (14,4))
sns.barplot(x = Data_types.index, y = Data_types.values)
plt.title("Data Type Distribution")
df.columns[df.isnull().any()].tolist()



df["Albumin_and_Globulin_Ratio"].fillna("0.6", inplace = True) 


df.isnull().sum()



Liver = df[(df['Dataset'] == 1)]
NoLiver = df[(df['Dataset'] != 1)]

trace = go.Pie(labels = ['Liver Disease', 'No Liver Disease'], values = df['Dataset'].value_counts(), 
               textfont=dict(size=15), opacity = 0.8,
               marker=dict(colors=['lightskyblue','gold'], 
                           line=dict(color='#000000', width=1.5)))


layout = dict(title =  'Distribution of Number  patients')
           
fig = dict(data = [trace], layout=layout)
py.iplot(fig)
Male = df[(df['Gender'] == 1)]
Female = df[(df['Gender'] != 1)]

trace = go.Pie(labels = ['Male', 'Female'], values = df['Gender'].value_counts(), 
               textfont=dict(size=15), opacity = 0.8,
               marker=dict(colors=['lightskyblue','gold'], 
                           line=dict(color='#000000', width=1.5)))


layout = dict(title =  'Distribution of Gender patients')
           
fig = dict(data = [trace], layout=layout)
py.iplot(fig)
liver_corr = df.corr()
plt.figure(figsize=(20, 20))
sns.heatmap(liver_corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 15},
           cmap= 'YlGnBu')
plt.title('Correlation between features');

corr=df.select_dtypes(include=[np.number]).dropna().corr()
mask=np.zeros_like(corr)
mask[np.triu_indices_from(mask)]=True
sns.heatmap(corr,cmap=sns.diverging_palette(256,0,sep=80,n=7,as_cmap=True), annot=True, mask=mask)
sns.pairplot(df.select_dtypes(include=[np.number]), dropna=True)


#correlation
correlation = df.corr()
#tick labels
matrix_cols = correlation.columns.tolist()
#convert to array
corr_array  = np.array(correlation)
#Plotting
trace = go.Heatmap(z = corr_array,
                   x = matrix_cols,
                   y = matrix_cols,
                   xgap = 2,
                   ygap = 2,
                   colorscale='Viridis',
                   colorbar   = dict() ,
                  )
layout = go.Layout(dict(title = 'Correlation Matrix for variables',
                        autosize = False,
                        height  = 720,
                        width   = 800,
                        margin  = dict(r = 0 ,l = 210,
                                       t = 25,b = 210,
                                     ),
                        yaxis   = dict(tickfont = dict(size = 9)),
                        xaxis   = dict(tickfont = dict(size = 9)),
                       )
                  )
fig = go.Figure(data = [trace],layout = layout)
py.iplot(fig)

Liver = df[(df['Dataset'] == 1)]
NoLiver = df[(df['Dataset'] != 1)]


#------------COUNT-----------------------
trace = go.Bar(x = (len(Liver), len(NoLiver)), y = ['with liver disease', 'NO liver disease'], orientation = 'h', opacity = 0.8, marker=dict(
        color=[ 'gold', 'lightskyblue'],
        line=dict(color='#000000',width=1.5)))

layout = dict(title =  'Count of patients Gender variable')
                    
fig = dict(data = [trace], layout=layout)
py.iplot(fig)

#------------PERCENTAGE-------------------
trace = go.Pie(labels = ['NO liver disease', 'with liver disease'], values = df['Dataset'].value_counts(), 
               textfont=dict(size=15), opacity = 0.8,
               marker=dict(colors=['lightskyblue', 'gold'], 
                           line=dict(color='#000000', width=1.5)))


layout = dict(title =  'Distribution of diagnosis variable')
           
fig = dict(data = [trace], layout=layout)
py.iplot(fig)
def plot_feat1_feat2(feat1, feat2) :  
    trace0 = go.Scatter(
        x = Liver[feat1],
        y = Liver[feat2],
        name = 'Yes',
        mode = 'markers', 
        marker = dict(color = '#FFD700',
            line = dict(
                width = 1)))

    trace1 = go.Scatter(
        x = NoLiver[feat1],
        y = NoLiver[feat2],
        name = 'NO',
        mode = 'markers',
        marker = dict(color = '#7EC0EE',
            line = dict(
                width = 1)))

    layout = dict(title = feat1 +" "+"vs"+" "+ feat2,
                  yaxis = dict(title = feat2,zeroline = False),
                  xaxis = dict(title = feat1, zeroline = False)
                 )

    plots = [trace0, trace1]

    fig = dict(data = plots, layout=layout)
    py.iplot(fig)
plot_feat1_feat2('Total_Bilirubin','Direct_Bilirubin')
plot_feat1_feat2('Direct_Bilirubin','Total_Bilirubin')
plot_feat1_feat2('Alkaline_Phosphotase','Alamine_Aminotransferase')
plot_feat1_feat2('Albumin','Total_Protiens')
plot_feat1_feat2('Albumin','Age')









plot_feat1_feat2('Alkaline_Phosphotase','Alamine_Aminotransferase')
plot_feat1_feat2('Age','Total_Protiens')
plot_feat1_feat2('Albumin_and_Globulin_Ratio','Alkaline_Phosphotase')
plot_feat1_feat2('Albumin','Direct_Bilirubin')

df_sex = pd.get_dummies(df['Gender'])
df_new = pd.concat([df, df_sex], axis=1)
Droop_gender = df_new.drop(labels=['Gender' ],axis=1 )
Droop_gender.columns = ['Age', 'Total_Bilirubin', 'Direct_Bilirubin','Alkaline_Phosphotase','Alamine_Aminotransferase','Aspartate_Aminotransferase','Total_Protiens','Albumin','Albumin_and_Globulin_Ratio','Male','Fmale','Dataset']

X = Droop_gender.drop('Dataset',axis=1)
y = Droop_gender['Dataset']


# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
classifier = Sequential() # Initialising the ANN

classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# compile ANN
classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the data
hisroy =classifier.fit(X_train, y_train, batch_size = 20, epochs = 50)

plt.plot(hisroy.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.plot(hisroy.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.style.use("ggplot")
plt.figure()
N = 50
plt.plot(np.arange(0, N), hisroy.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), hisroy.history["acc"], label="train_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")

y_pred = classifier.predict(X_test)
y_pred = [ 1 if y>=0.5 else 0 for y in y_pred ]



print(classification_report(y_test, y_pred))



skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)



