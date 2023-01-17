# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')

df.head()
!pip install pywaffle
import pandas as pd

import numpy as np

from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import f1_score

from pywaffle import Waffle
sns.set_style('whitegrid')
def print_waffle(column):



    equilibre=df[column].value_counts()

    data = equilibre

    fig = plt.figure(

        FigureClass=Waffle, 

        title={'label': column},

        rows=8, 

        values=data, 

        colors=("#232066", "#983D3D"),

        # legend={'loc': 'upper left', 'bbox_to_anchor': (1, 1)},

        icons='child', icon_size=10, 

        icon_legend=True,

        labels=["{0}({1})".format(k,v) for k, v in data.items()],

        legend={'loc':'lower left','bbox_to_anchor':(0,-0.4),'ncol':len(data),'framealpha':0}

    )

    fig.gca().set_facecolor('#EEEEEE')

    fig.set_facecolor('#EEEEEE')

    plt.show()
print_waffle('DEATH_EVENT')      

print_waffle('diabetes')

print_waffle('smoking')
def double_histo_binary(df,column):



    toprint_1=df['DEATH_EVENT'].loc[df[column]==0].value_counts()

    toprint_2=df['DEATH_EVENT'].loc[df[column]==1].value_counts()

    df_tempo=pd.DataFrame({'Not_'+column:toprint_1,column:toprint_2})

    ax=plt.subplot(211)

    df_tempo.plot.bar(title=column+' vs non-'+column +' in absolute value',colormap='Accent',ax=ax)

    ax.set(xlabel='Death_event',ylabel='Count')

    total_Notsmoker=df_tempo['Not_'+column].sum()

    total_smoker=df_tempo[column].sum()

    df_tempo1=pd.DataFrame({'Not_'+column:(df_tempo['Not_'+column].values*100)/total_Notsmoker,column:(df_tempo[column].values*100)/total_smoker})

    ax=plt.subplot(212)

    df_tempo1.plot.bar(title=column+' vs non-'+column +' in %',colormap='Accent',ax=ax)

    ax.set(xlabel='Death_event',ylabel='%')

    

    plt.show()
double_histo_binary(df,'smoking')

double_histo_binary(df,'anaemia')

double_histo_binary(df,'diabetes')

double_histo_binary(df,'sex')
def plot_histogram(column):

    df_death=df.loc[df['DEATH_EVENT']==1]

    df_nodeath=df.loc[df['DEATH_EVENT']==0]

    sns.kdeplot(df_death[column],shade=True,color='red',label='Death_event=1')

    sns.kdeplot(df_nodeath[column],shade=True,color='blue',label='Death_event=0')

    plt.title(column+' distribution by class')

    plt.show()
plot_histogram('serum_creatinine')

plot_histogram('serum_sodium')

plot_histogram('ejection_fraction')

plot_histogram('platelets')

plot_histogram('high_blood_pressure')

plot_histogram('age')
sns.heatmap(df.corr(),linewidths=2,linecolor='black',cmap='viridis')

plt.show()
columns_goodplace=['age','anaemia','creatinine_phosphokinase','ejection_fraction','serum_creatinine','serum_sodium','time',

                   'smoking','diabetes','sex','platelets','high_blood_pressure']
y=df['DEATH_EVENT'].values

y=to_categorical(y)

df.drop(['DEATH_EVENT'],1,inplace=True)

df=df[columns_goodplace]

X=df.values
scaler=StandardScaler()

ss=scaler.fit(X)

X=scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=91)
from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import itertools

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')

        

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")

    plt.ylim(top=-0.5,bottom=1+0.5)

    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.show()



def evaluate_model(history,predicted_classes,y_test,model):

    

#    history.history['acc'][0]=0

#    history.history['val_acc'][0]=0

    print(history)

    fig1, ax_acc = plt.subplots()

    plt.plot(history.history['accuracy'])

    plt.plot(history.history['val_accuracy'])

    plt.xlabel('Epoch')

    plt.ylabel('Accuracy')

    plt.title('Model - Accuracy')

    plt.legend(['Training', 'Validation'], loc='lower right')

    plt.show()

#    

#    history.history['loss'][0]=0.5

#    history.history['val_loss'][0]=0.5

    fig2, ax_loss = plt.subplots()

    plt.xlabel('Epoch')

    plt.ylabel('loss')

    plt.title('Model- loss')

    plt.legend(['Training', 'Validation'], loc='upper right')

    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.show()

    target_names=['0','1']

    

    prediction=predicted_classes

    cnf_matrix = confusion_matrix(y_test, prediction)

    report=classification_report(y_test, prediction, target_names=target_names)

    print(cnf_matrix)

    print(report)

    np.set_printoptions(precision=2)

    plt.figure(figsize=(5, 5))

    plot_confusion_matrix(cnf_matrix, classes=['0', '1'],normalize=True,

                      title='Confusion matrix, with normalization')

    plt.show()

    plt.figure(figsize=(5, 5))

    plot_confusion_matrix(cnf_matrix, classes=['0', '1'],normalize=False,

                      title='Confusion matrix, with normalization')



    plt.show()
def create_model(Nbr_layer,nbr_per_layer,dropout,batch_size,class_0_weight,class_1_weight,X_train,y_train,X_test,y_test):

    model = Sequential()

    for i in range(Nbr_layer):

        model.add(Dense(nbr_per_layer, activation='relu'))

    model.add(Dropout(dropout))   

    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer="Adam",loss="binary_crossentropy",metrics=["accuracy"])

    

    callbacks = [EarlyStopping(monitor='val_loss', patience=5),

             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

    class_weight = {0: class_0_weight,

                    1: class_1_weight}

    

    history=model.fit(X_train,y_train,epochs=50,batch_size=batch_size,validation_data=(X_test,y_test),callbacks=callbacks,class_weight=class_weight)

    model.load_weights('best_model.h5')

    

    

    pred=model.predict(X_test)

    predicted_classes=np.argmax(pred,axis=1)

    y_true=np.argmax(y_test,axis=1)

    # evaluate_model(history,predicted_classes,y_true,model)

    score=f1_score(y_true,predicted_classes)

    return(model,score,history)

    
layer=[1,2,3]

number_per_layer=[10,20,30,40,60,80]

dropout=[0.1,0.2,0.3,0.4]

batch_size=[8,16,32]

class_0_weight=[1]

class_1_weight=[1]



score=0

best_model=0

best_param=[0,0,0,0]

best_history=0



score_lessdata=0

best_model_lessdata=0

best_param_lessdata=[0,0,0,0]

best_history_lessdata=0
for element in layer:

    for element1 in number_per_layer:

        for element2 in dropout:

            for element3 in batch_size:

                    for element4 in class_0_weight:

                        for element5 in class_1_weight:

                            print('+++++++++++++++++++++New_train+++++++++++++++++++')

                            print([element,element1,element2,element3,element4,element5])

                            model,score1,history=create_model(element,element1,element2,element3,element4,element5,X_train,y_train,X_test,y_test)

                            if score1>score:

                                score=score1

                                best_model=model

                                best_param=[element,element1,element2,element3,element4,element5]

                                best_history=history

                            model_lessdata,score_lessdata1,history_lessdata=create_model(element,element1,element2,element3,element4,element5,X_train[:,:7],y_train,X_test[:,:7],y_test)

                            if score_lessdata1>score_lessdata:

                                score_lessdata=score_lessdata1

                                best_model_lessdata=model_lessdata

                                best_param_lessdata=[element,element1,element2,element3,element4,element5]

                                best_history_lessdata=history_lessdata

                                                                        

pred=best_model.predict(X_test)

predicted_classes=np.argmax(pred,axis=1)

y_true=np.argmax(y_test,axis=1)

from sklearn.metrics import accuracy_score



accuracy_score(y_true,predicted_classes)

from sklearn.metrics import f1_score

f1_score(y_true,predicted_classes)
evaluate_model(best_history,predicted_classes,y_true,model)
pred1=best_model_lessdata.predict(X_test[:,:7])

predicted_classes1=np.argmax(pred1,axis=1)

y_true=np.argmax(y_test,axis=1)

evaluate_model(best_history_lessdata,predicted_classes1,y_true,best_model_lessdata)
accuracy_score(y_true,predicted_classes1)
f1_score(y_true,predicted_classes1)
!pip install ethik
import ethik
explainer = ethik.ClassificationExplainer()
X_test1=pd.DataFrame(X_test,columns=columns_goodplace)
explainer.plot_influence_ranking(

    X_test=X_test1,

    y_pred=predicted_classes,

    n_features=5,

)
explainer.plot_influence(

    X_test=X_test1["ejection_fraction"],

    y_pred=predicted_classes,

)
explainer.plot_influence(

    X_test=X_test1["age"],

    y_pred=predicted_classes,

)
bob=X_test1.iloc[1]

Mary=X_test1.iloc[4]
explainer.plot_influence_comparison(

    X_test=X_test1,

    y_pred=predicted_classes,

    reference=bob,

    compared=Mary,

)

y_pred=pd.DataFrame(pred,columns=[0,1])

explainer.plot_influence(

    X_test=X_test1["anaemia"],

    y_pred=y_pred[[0, 1]]

)
y_pred=pd.DataFrame(pred,columns=[0,1])

explainer.plot_influence(

    X_test=X_test1["high_blood_pressure"],

    y_pred=y_pred[[0, 1]]

)