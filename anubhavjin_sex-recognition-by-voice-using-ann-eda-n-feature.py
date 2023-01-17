# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df=pd.read_csv('/kaggle/input/voicegender/voice.csv')
df.head()
df.shape
df.describe()
df.info()
df['label'].value_counts()
df.columns
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
def calc_limits(feature):
    q1,q3=df[feature].quantile([0.25,0.75])
    iqr=q3-q1
    rang=1.5*iqr
    return(q1-rang,q3+rang)
def plot(feature):
    fig,axes=plt.subplots(1,2)
    sns.boxplot(data=df,x=feature,ax=axes[0])
    sns.distplot(a=df[feature],ax=axes[1],color='#ff4125')
    fig.set_size_inches(15,5)
    
    lower,upper = calc_limits(feature)
    l=[df[feature] for i in df[feature] if i>lower and i<upper] 
    print("Number of data points remaining if outliers removed : ",len(l))
plot('meanfreq')
plot('sd')
plot('median')
plot('Q25')
plot('IQR')
plot('skew')
plot('kurt')
temp=[]
for i in df['label']:
    if i=='male':
        temp.append(1)
    else:
        temp.append(0)
df['label']=temp
cor_mat=df[:].corr()
plt.figure(figsize=(20,20))
sns.heatmap(data=cor_mat,annot=True)
df.drop('centroid',axis=1,inplace=True)
g = sns.PairGrid(df[['meanfreq','sd','median','Q25','IQR','sp.ent','sfm','meanfun','label']], hue = "label")
g = g.map(plt.scatter).add_legend()
for col in df.columns:
    lower,upper=calc_limits(col)
    df = df[(df[col] >lower) & (df[col]<upper)]
temp_df=df.copy()

temp_df.drop(['skew','kurt','mindom','maxdom'],axis=1,inplace=True) # only one of maxdom and dfrange.
temp_df['meanfreq']=temp_df['meanfreq'].apply(lambda x:x*2)
temp_df['median']=temp_df['meanfreq']+temp_df['mode']
temp_df['median']=temp_df['median'].apply(lambda x:x/3)
temp_df['pear_skew']=temp_df['meanfreq']-temp_df['mode']
temp_df['pear_skew']=temp_df['pear_skew']/temp_df['sd']
temp_df.head(10)
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
scaled_df=scaler.fit_transform(temp_df.drop('label',axis=1))
X=scaled_df
Y=df['label'].as_matrix()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import History
from keras.utils import plot_model
from keras.optimizers import SGD

classifier=Sequential()
history = History()

#number of input variables =20
#first layer 
#input_dim is only for the first layer
classifier.add(Dense(output_dim=11,init='uniform',activation='relu',input_dim=16))
#first Hidden layer
classifier.add(Dense(output_dim=11,init='uniform',activation='relu'))
#Second Hidden
classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
#output layer
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
#Running the artificial neural network
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#fitting
classifier.fit(X_train,y_train,batch_size=10,epochs=50,validation_split=0.1,callbacks=[history],shuffle=2)
import sklearn.metrics as metrics
y_pred=classifier.predict(X_test)
y_pred = np.round(y_pred)

print('Accuracy we are able to achieve with our ANN is',metrics.accuracy_score(y_pred,y_test)*100,'%')

plt.plot(history.history['loss'], color = 'red',label='Variaton Loss over the epochs',)
plt.plot(history.history['accuracy'],color='cyan',label='Variation in Profit over the epochs')

plt.xlabel('Epochs')
plt.title('Loss/Accuracy VS Epoch')
plt.ylabel('Loss/Accuracy')
plt.legend(loc='best')
plt.show()
