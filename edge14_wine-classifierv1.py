



import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.ensemble import RandomForestClassifier



from sklearn.ensemble import GradientBoostingClassifier





from sklearn.model_selection import train_test_split



from sklearn.preprocessing import MinMaxScaler



from sklearn.metrics import classification_report,confusion_matrix,accuracy_score



from imblearn.combine import SMOTETomek

from sklearn.preprocessing import LabelEncoder



from sklearn.pipeline import Pipeline





#loading data

df=pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')



#data analysis

print(df.head())

print(df.columns)

print(df.info())

print(df['alcohol'].describe())

print(df.isnull().sum())







sns.countplot(df['quality'],hue=df['quality'])



print(df.corr())



print(df['quality'].unique())





#bining target variable into 2 classes

bins=(2,6,8)

labels=['bad','good']

df['quality']=pd.cut(df['quality'],bins=bins,labels=labels) # bining













#DATA PREPARATION

X=df.drop('quality',axis=1)

Y=df['quality']



le = LabelEncoder() # convert target from GOOD/BAD to 1/0

Y = le.fit_transform(Y)



smk=SMOTETomek(random_state=42) #upsampling the data

X_up,Y_up=smk.fit_sample(X,Y)





df1=pd.DataFrame(Y_up)





x_train,x_test,y_train,y_test=train_test_split(X_up,Y_up,test_size=0.4)









#pipeline for 2 classification models

p1=Pipeline([('sacler1',MinMaxScaler()),

     ('model1',RandomForestClassifier(n_estimators=20))])





p2 = Pipeline([('sacler2', MinMaxScaler()),

       ('model2', GradientBoostingClassifier())])



pipeline=[p1,p2]



for p in pipeline:

    p.fit(x_train,y_train)





pip_dict={0:'RandomForest',1:'Gradient Boost Classifier'}

for i,model in enumerate(pipeline):

    print(pip_dict[i],' : ',model.score(x_test,y_test))

    predictions=model.predict(x_test)

    print(classification_report(y_test,predictions))
