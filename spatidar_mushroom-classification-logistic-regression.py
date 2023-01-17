# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

from sklearn import metrics



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

df1=pd.read_csv("../input/mushrooms.csv",na_values="?")

print(df1.isnull().sum(axis=0))

##########drop 'stalk-root' column as it contains 30 % missing value 

df=df1.drop('stalk-root',1)



######----- print data type of each column-------

for col in df:

    print (col,df[col].dtypes)

    

########--------- frequecy of each column --------



for col in df:

    print (pd.crosstab(index=df[col],columns=["class"]))



data1 = pd.crosstab(index=df["bruises"],columns=df["class"])

print(data1)

#####-------One Hot encoding for all categorical variable except dependent variable--------------

number=LabelEncoder()

df['class'] = number.fit_transform(df['class'])

df['class']= df['class'].astype(str)



d1=pd.get_dummies(df,columns=df.columns.values.tolist()[1:])



####------Divide data into X and Y

X=d1.iloc[:,1:113]

Y=d1.iloc[:,0]



model_LR= LogisticRegression()



X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=4)



model_LR.fit(X_train,y_train)

y_prob = model_LR.predict(X_test)

model_LR.score(X_test, y_prob)

confusion_matrix=metrics.confusion_matrix(y_test,y_prob)

print(confusion_matrix)


