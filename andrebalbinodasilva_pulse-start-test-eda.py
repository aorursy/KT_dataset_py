# By André Balbino da Silva





import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,confusion_matrix

from sklearn.naive_bayes import GaussianNB

import seaborn as sns



from yellowbrick.classifier import ConfusionMatrix



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
base = pd.read_csv('/kaggle/input/predicting-a-pulsar-star/pulsar_stars.csv')



base.head()




pulse_star = base['target_class'].loc[base.iloc[:,-1] ==1].count()

no_pulse = base['target_class'].loc[base.iloc[:,-1] ==0].count()





fig, axs = plt.subplots(1, 1, figsize=(6, 3), sharey=True)

axs.bar(['pulse','no pulse'], [pulse_star,no_pulse])





fig.suptitle('Pulse Star Plotting')



x = base.iloc[:,0:8].values

y = base.iloc[:,8].values







for i in range(0,8):

    print(f'Correlation X {i} and Y')

    print(np.corrcoef(x[:,i],y))

    print('#'*50)

    



    

#Select Features     

x2 = x[:,2:4]
x_train,x_test,y_train,y_test = train_test_split(x2,y,random_state=0,test_size=0.3)







model = GaussianNB()



model.fit(x_train,y_train)





predicts = model.predict(x_test)
v = ConfusionMatrix(model)



v.fit(x_train,y_train)



print(v.score(x_test,y_test))

v.poof()
df_predicts = pd.DataFrame(columns=['Predict',"Real",'Check'])



df_predicts['Predict'] = predicts

df_predicts['Real'] = y_test

df_predicts['Check'] = predicts == y_test



hits = sum((df_predicts['Check'].values == True))

miss = sum((df_predicts['Check'].values == False))





error_rate = ((miss/hits)*100)



hit_rate = (hits/len(df_predicts))*100



print(f'Hits :{hits}')

print(f'Miss :{miss}')



print(f' Error Rate: {error_rate}')

print('#-'*50)

print(f' Hit Rate: {hit_rate}')









#Real data

sns.countplot(y_test)

#Predict Data



sns.countplot(predicts)
sns.scatterplot(x2[:,1],x2[:,0])



np.corrcoef(x2[:,1],x2[:,0])