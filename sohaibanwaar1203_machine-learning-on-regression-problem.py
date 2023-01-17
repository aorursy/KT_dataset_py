# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import numpy

import pandas

import seaborn as sns

from keras.models import Sequential

from keras.layers import Dense ,Dropout,BatchNormalization

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/Admission_Predict_Ver1.1.csv")

#changing names because previous names are little bit confusing

df=df.rename(index=str, columns={"GRE Score": "GRE", "TOEFL Score": "TOEFL", "Chance of Admit ": "Admission_Chance"})

#we donot need serial number so its good to drop it because its just a number

df=df.drop("Serial No.",axis=1)

df.head(10)

def modiffy(row):

    if row['Admission_Chance'] >0.7 :

        return 1

    else :

        return 0

df['Admit'] = df.apply(modiffy,axis=1)
df.describe()
admit=np.asarray(df["Admission_Chance"])

len(np.unique(admit))

#we have 60 different values in the coloum [chance to predict]
import pandas as pd

import matplotlib.pyplot as plt

corr = df.corr()

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)

fig.colorbar(cax)

ticks = np.arange(0,len(df.columns),1)

ax.set_xticks(ticks)

plt.xticks(rotation=90)

ax.set_yticks(ticks)

ax.set_xticklabels(df.columns)

ax.set_yticklabels(df.columns)

plt.show()
import seaborn as sns

corr = df.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)
fig = plt.figure(figsize = (20, 25))

j = 0

for i in df.columns:

    plt.subplot(6, 4, j+1)

    j += 1

    sns.distplot(df[i][df['Admission_Chance']<0.72], color='r', label = 'Not Got Admission')

    sns.distplot(df[i][df['Admission_Chance']>0.72], color='g', label = 'Got Admission')

    plt.legend(loc='best')

fig.suptitle('Admission Chance In University ')

fig.tight_layout()

fig.subplots_adjust(top=0.95)

plt.show()
for column in df:

    plt.figure()

    sns.boxplot(x=df[column])
for column_1st in df:

    for coloum_2nd in df:

        jet=plt.get_cmap('jet')

        plt.figure(figsize=(15,5))

        plt.scatter(df[column_1st], df[coloum_2nd], s=30, c=df['Admission_Chance'], vmin=0, vmax=1, cmap=jet)

        plt.xlabel(column_1st,fontsize=40)

        plt.ylabel(coloum_2nd,fontsize=40)

        plt.colorbar()

        plt.show()
X=np.asarray(df.drop("Admission_Chance",axis=1))

Y=np.asarray(df["Admission_Chance"])

X_train, X_test, y_train, y_test = train_test_split(

     X,Y, test_size=0.2, random_state=0)
sns.heatmap(df.isnull())
#correlations_data = admt.corr()['Chance'].sort_values(ascending=False)

cor=df.corr()['Admission_Chance']

# Print the correlations

print(cor)
df.hist(figsize=(10,8),bins=10,color='#ffd700',linewidth='1',edgecolor='k')

plt.tight_layout()

plt.show()
df[(df['Admission_Chance']>0.90)].mean().reset_index()
plt.figure(figsize=(15,10))

plt.subplot(2,2,1)

sns.violinplot("Research","GRE",hue="Admit", data=df,split=True)

plt.subplot(2,2,2)

sns.violinplot("Research","TOEFL",hue="Admit", data=df,split=True)

plt.subplot(2,2,3)

sns.violinplot("Research","CGPA",hue="Admit", data=df,split=True)

plt.subplot(2,2,4)

sns.violinplot("Research","University Rating",hue="Admit", data=df,split=True)

plt.ioff()

plt.show()
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X_train,y_train)

y_head_lr = lr.predict(X_test)



print("real value of y_test[1]: " + str(y_test[1]) + " -> the predict: " + str(lr.predict(X_test[[1],:])))

print("real value of y_test[2]: " + str(y_test[2]) + " -> the predict: " + str(lr.predict(X_test[[2],:])))



from sklearn.metrics import r2_score

print("r_square score: ", r2_score(y_test,y_head_lr))



y_head_lr_train = lr.predict(X_train)

print("r_square score (train dataset): ", r2_score(y_train,y_head_lr_train))
#Visualising the Acutal and predicted Result

plt.plot(y_test, color = 'green', label = 'Actual')

plt.plot(y_head_lr , color = 'blue', label = 'Predicted')

plt.grid(alpha = 0.3)

plt.xlabel('Number of Candidate')

plt.ylabel('Score')

plt.title('Actual vs Predicted')

plt.legend()

plt.show()
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(n_estimators = 100, random_state = 42)

rfr.fit(X_train,y_train)

y_head_rfr = rfr.predict(X_test)



print("real value of y_test[1]: " + str(y_test[1]) + " -> the predict: " + str(rfr.predict(X_test[[1],:])))

print("real value of y_test[2]: " + str(y_test[2]) + " -> the predict: " + str(rfr.predict(X_test[[2],:])))



from sklearn.metrics import r2_score

print("r_square score: ", r2_score(y_test,y_head_rfr))



y_head_rfr_train = rfr.predict(X_train)

print("r_square score (train dataset): ", r2_score(y_train,y_head_rfr_train))
#Visualising the Acutal and predicted Result

plt.plot(y_test, color = 'green', label = 'Actual')

plt.plot(y_head_lr , color = 'blue', label = 'Predicted')

plt.grid(alpha = 0.3)

plt.xlabel('Number of Candidate')

plt.ylabel('Score')

plt.title('Actual vs Predicted')

plt.legend()

plt.show()
from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor(random_state = 42)

dtr.fit(X_train,y_train)

y_head_dtr = dtr.predict(X_test)



print("real value of y_test[1]: " + str(y_test[1]) + " -> the predict: " + str(dtr.predict(X_test[[1],:])))

print("real value of y_test[2]: " + str(y_test[2]) + " -> the predict: " + str(dtr.predict(X_test[[2],:])))



from sklearn.metrics import r2_score

print("r_square score: ", r2_score(y_test,y_head_dtr))



y_head_dtr_train = dtr.predict(X_train)

print("r_square score (train dataset): ", r2_score(y_train,y_head_dtr_train))
#Visualising the Acutal and predicted Result

plt.plot(y_test, color = 'green', label = 'Actual')

plt.plot(y_head_lr , color = 'blue', label = 'Predicted')

plt.grid(alpha = 0.3)

plt.xlabel('Number of Candidate')

plt.ylabel('Score')

plt.title('Actual vs Predicted')

plt.legend()

plt.show()
#we canot get improtant features by ndarrays so we have to train it again on dataframe to check the importance

#features



X=df.drop("Admission_Chance",axis=1)

X=X.drop("Admit",axis=1)

y=df["Admission_Chance"]

classifier = RandomForestRegressor()

classifier.fit(X,y)

feature_names = X.columns

importance_frame = pd.DataFrame()

importance_frame['Features'] = X.columns

importance_frame['Importance'] = classifier.feature_importances_

importance_frame = importance_frame.sort_values(by=['Importance'], ascending=True)

plt.barh([1,2,3,4,5,6,7], importance_frame['Importance'], align='center', alpha=0.5)

plt.yticks([1,2,3,4,5,6,7], importance_frame['Features'])

plt.xlabel('Importance')

plt.title('Feature Importances')

plt.show()
y = np.array([r2_score(y_test,y_head_lr),r2_score(y_test,y_head_rfr),r2_score(y_test,y_head_dtr)])

x = ["LinearRegression","RandomForestReg.","DecisionTreeReg."]

plt.bar(x,y)

plt.title("Comparison of Regression Algorithms")

plt.xlabel("Regressor")

plt.ylabel("r2_score")

plt.show()
print("real value of y_test[5]: " + str(y_test[5]) + " -> the predict: " + str(lr.predict(X_test[[5],:])))

print("real value of y_test[5]: " + str(y_test[5]) + " -> the predict: " + str(rfr.predict(X_test[[5],:])))

print("real value of y_test[5]: " + str(y_test[5]) + " -> the predict: " + str(dtr.predict(X_test[[5],:])))



print()



print("real value of y_test[50]: " + str(y_test[50]) + " -> the predict: " + str(lr.predict(X_test[[50],:])))

print("real value of y_test[50]: " + str(y_test[50]) + " -> the predict: " + str(rfr.predict(X_test[[50],:])))

print("real value of y_test[50]: " + str(y_test[50]) + " -> the predict: " + str(dtr.predict(X_test[[50],:])))
red = plt.scatter(np.arange(0,80,5),y_head_lr[0:80:5],color = "red")

green = plt.scatter(np.arange(0,80,5),y_head_rfr[0:80:5],color = "green")

blue = plt.scatter(np.arange(0,80,5),y_head_dtr[0:80:5],color = "blue")

black = plt.scatter(np.arange(0,80,5),y_test[0:80:5],color = "black")

plt.title("Comparison of Regression Algorithms")

plt.xlabel("Index of Candidate")

plt.ylabel("Chance of Admit")

plt.legend((red,green,blue,black),('LR', 'RFR', 'DTR', 'REAL'))

plt.show()