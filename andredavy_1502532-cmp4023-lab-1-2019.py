# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data_path = '../input/tables_1968_2018.csv' # Path to data file

data = pd.read_csv(data_path) 

data.head(15)
data.columns
def create_label_encoder_dict(df):

    from sklearn.preprocessing import LabelEncoder

    

    label_encoder_dict = {}

    for column in df.columns:

        # Only create encoder for categorical data types

        if not np.issubdtype(df[column].dtype, np.number) and column != 'season' and column != 'name'  and column != 'team':

            label_encoder_dict[column]= LabelEncoder().fit(df[column])

    return label_encoder_dict
label_encoders = create_label_encoder_dict(data)

print("Encoded Values for each Label")

print("="*32)

for column in label_encoders:

    print("="*32)

    print('Encoder(%s) = %s' % (column, label_encoders[column].classes_ ))

    print(pd.DataFrame([range(0,len(label_encoders[column].classes_))], columns=label_encoders[column].classes_, index=['Encoded Values']  ).T)
# Apply each encoder to the data set to obtain transformed values

data2 = data.copy() # create copy of initial data set

for column in data2.columns:

    if column in label_encoders:

        data2[column] = label_encoders[column].transform(data2[column])



print("Transformed data set")

print("="*32)

data2
data.columns
X_data = data2[['w','points']]

Y_data = data2['pos']
df = pd.DataFrame([[5.1, 3.5], [4.9, 3.0], [7.0, 3.2], [6.4, 3.2], [5.9, 3.0]], columns=['w', 'pos'])
ax1 = df.plot.scatter(x='w', y='pos', c='Blue')
df2 = pd.DataFrame([[5.1, 3.5], [4.9, 3.0], [7.0, 3.2], [6.4, 3.2], [5.9, 3.0]], columns=['points', 'pos'])
ax2 = df2.plot.scatter(x='points', y='pos', c='Red')
from sklearn.model_selection import train_test_split #train test split

X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.30)
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(X_train,y_train)
reg.coef_
X_train.columns
print("Regression Coefficients")

pd.DataFrame(reg.coef_,index=X_train.columns,columns=["Coefficient"])
# Intercept

reg.intercept_
# Make predictions using the testing set

test_predicted = reg.predict(X_test)

test_predicted [0:5]
data3 = X_test.copy()

data3['predicted_position']=test_predicted

data3['pos']=y_test

data3.head(10)
data3 = X_test.copy()

data3['predicted_position']=test_predicted

data3['w']=y_test

data3.head(10)
from sklearn.metrics import mean_squared_error, r2_score
# The mean squared error

print("Mean squared error: %.2f" % mean_squared_error(y_test, test_predicted))
from sklearn.metrics import mean_absolute_error

y_true = [3, 4, 2, 7]

y_pred = [8, 0, 2, 4]

mae = mean_absolute_error(y_true, y_pred)



print('Mean Absolute Error: %f' % mae)
# Explained variance score: 1 is perfect prediction

# R squared

print('Variance score: %.2f' % r2_score(y_test, test_predicted))
# Returns the coefficient of determination R^2 of the prediction.

reg.score(X_test,y_test)
from sklearn.decomposition import PCA
pca = PCA(n_components=1)
pca.fit(data2[X_train.columns])
pca.components_
pca.n_features_
pca.n_components_
X_test
X_reduced = pca.transform(X_test)

X_reduced
plt.scatter(X_reduced, y_test,  color='blue')
plt.scatter(X_reduced, y_test,  color='blue')

plt.plot(X_reduced, test_predicted, color='red',linewidth=1)

plt.plot(X_reduced, test_predicted, color='green',linewidth=1)



#plt.xticks(())

#plt.yticks(())



plt.show()
plt.plot(y_test, test_predicted, 'ro-')
np.std(np.abs(y_test-test_predicted))
data4=pd.DataFrame({'actual':y_test,'pred':test_predicted})

data4.head()
data4.sort_values('actual').plot(kind='line',x='actual',y='pred')
plt.scatter(reg.predict(X_train), reg.predict(X_train)-y_train,c='b',s=40,alpha=0.5)

plt.scatter(reg.predict(X_test),reg.predict(X_test)-y_test,c='g',s=40)

plt.hlines(y=0,xmin=np.min(reg.predict(X_test)),xmax=np.max(reg.predict(X_test)),color='red',linewidth=3)

plt.title('Residual Plot using Training (blue) and test (green) data ')

plt.ylabel('Residuals')
data.corr()
import seaborn as sns
sns.pairplot(data)
rng = np.random.RandomState(1)

x = 10 * rng.rand(50)

y = 2 * x - 5 + rng.randn(50)

plt.scatter(x, y);
rng = np.random.RandomState(1)

x2 = 10 * rng.rand(50)

y = 2 * x - 5 + rng.randn(50)

plt.scatter(x2, y);