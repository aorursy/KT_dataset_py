

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


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
from IPython.display import display
df=pd.read_csv('../input/project2.csv')
print(df)
print(df.iloc[0:2])

df.columns=['SFH','Popup','SSLfinal_state','Request_url','URL_of_anchor','web_traffic','url_length','age_of_domain','having_IP_address','Result']
print(df)
display(df.head(5))
df.describe()

df.describe()

a=len(df[df.Result==0])
b=len(df[df.Result==-1])
c=len(df[df.Result==1])
print(a,"times 0 repeated in Result")
print(b,"times -1 repeated in Result")
print(c,"times 1 repeated in Result")
sns.countplot(df['Result'])
#from the above graph it is concluded that most of the websites in 1353 websites are -1 i.e.phishy and few are suspicious which are not included in the categories of both legiative and phisphy
result=df['Result']
features=df.drop(['Result'],axis=1)
for i in features.columns:
    plt.title("%s"%i)
    plt.figure(figsize=(10,6))
    sns.countplot(df[i],hue=df['Result'])
import seaborn
print(df.corr())
seaborn.heatmap(df.corr(),annot=True)

#we observe that there is no corelation among them it is difficult to find them
#to improve this use logistic regression with training and testing data treated 20% as testing data

#
from sklearn.model_selection import train_test_split

# Split the 'features' and 'result' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, 
                                                    result, 
                                                    test_size = 0.2, 
                                                    random_state = 5)

# Show the results of the split
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))
#using logistic regression
#import Evaluation metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

#create logistic regression object
clf_lr = LogisticRegression(random_state=5)
#Train the model using training data 
clf_lr=clf_lr.fit(X_train,y_train)

#Test the model using testing data
predictions = clf_lr.predict(X_test)

print("f1 score is ",f1_score(y_test,predictions,average='weighted'))
print("matthews correlation coefficient is ",matthews_corrcoef(y_test,predictions))

#secondary metric,we should not consider accuracy score because the classes are imbalanced.
print("Accuracy score is ",accuracy_score(y_test,predictions))
