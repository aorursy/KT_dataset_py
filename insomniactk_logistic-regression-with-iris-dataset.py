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
%matplotlib inline
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

data = load_iris()
print(data.keys())
labels = data['target_names']
description = data['DESCR']
feature_names = data['feature_names']

print(f"This is a dataset with {len(data.data)} samples.\n{len(labels)} classes.\n{len(feature_names)} features for each sample.")
print(f"Feature names: {feature_names}")
print(f"Label Names:{labels}")
print(f"---------\nSample input feature vector: {data['data'][0]}\nLabel: {data['target'][0]}")
x, y = data['data'][:,2],data['data'][:,3] # petal length and petal width 
c = data['target']

fig, ax = plt.subplots(figsize=(8,6))
scatter = ax.scatter(x, y, c=c,label='x')

# produce a legend with the unique colors from the scatter
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="best", title="Classes",fontsize=16)
ax.add_artist(legend1)
plt.title("0-Setosa,1-Versioclor,2-Virginica",fontsize=16)
plt.xlabel('Petal Length(cm)',fontsize=18)
plt.ylabel('Petal Width(cm)',fontsize=18)
plt.show()
from mpl_toolkits import mplot3d
c = []
for i in data['target']:
    if i == 0:
        c.append('red')
    elif i==1:
        c.append('green')
    else:
        c.append('blue')
fig = plt.figure(figsize=(5,5))
ax = mplot3d.Axes3D(fig)

# Data for a three-dimensional line
sepal_width,petal_length,petal_width  = data['data'][:,0],data['data'][:,1],data['data'][:,2]
ax.scatter(sepal_width,petal_length,petal_width,c=c)

plt.show()
# for this simplistic example I am just using one feature: sepal length
x = data['data'][:,:2]
y = (data['target']==0).astype('int')
# rough plot to see what we're dealing with 
plt.figure(figsize=(7,7))
for i in range(len(x)):
    if y[i] == 1:
        c = 'red'
        marker = '^'
    else:
        c = 'green'
        marker = '.'
    plt.plot(x[i,0],x[i,1],c=c,marker=marker)
plt.xlabel('Sepal Length(cm)',fontsize=14)
plt.ylabel('Sepal Width(cm)',fontsize=14)
plt.title("Red Square - Iris Setosa | Green Dot - Not iris Setosa",fontsize=12)
plt.show()
from sklearn.linear_model import LogisticRegression
lr_binary = LogisticRegression() 
lr_binary.fit(x,y)
lr_binary.__dict__
# let us check the score now
lr_binary.score(x,y)
sepal_l = data['data'][:,0] #sepal length
single_feature_lr_binary = LogisticRegression()
single_feature_lr_binary.fit(sepal_l.reshape(-1,1),y)
y_probab = single_feature_lr_binary.predict_proba(np.linspace(4.5,8.0,1000).reshape(-1,1))
# h(x) = theta_0 + theta_1*X
theta_0,theta_1 = single_feature_lr_binary.intercept_,single_feature_lr_binary.coef_
print(theta_0.shape)
print(theta_1.shape)
single_feature_lr_binary.decision_function(sepal_l[0].reshape(-1,1))
def h(theta_0,theta_1,x):
    return theta_0 + theta_1*x
single_feature_lr_binary.decision_function([[4.5],[8]])
single_feature_lr_binary.score(sepal_l.reshape(-1,1),y)
# new regularization constant 
lr_single_feature_high_regularization = LogisticRegression(C=0.5) # C is inverse of Regulariation Strength
lr_single_feature_low_regularization = LogisticRegression(C=10) 
crazy_regularized = LogisticRegression(C=0.01)
lr_single_feature_high_regularization.fit(sepal_l.reshape(-1,1),y)
lr_single_feature_low_regularization.fit(sepal_l.reshape(-1,1),y)
crazy_regularized.fit(sepal_l.reshape(-1,1),y)
no_regularization = LogisticRegression(C=10**5)
y_prob_high = lr_single_feature_high_regularization.predict_proba(np.linspace(4.5,8.0,1000).reshape(-1,1))
y_prob_low = lr_single_feature_low_regularization.predict_proba(np.linspace(4.5,8.0,1000).reshape(-1,1))
plt.figure(figsize=(10,10))

plt.subplot(221)
plt.title("C=1.0")
plt.plot(sepal_l,y,'ro')
plt.plot(np.linspace(4.5,8.0,1000),y_probab[:,0],label="P(Not Iris Setosa) | X ")
plt.plot(np.linspace(4.5,8.0,1000),y_probab[:,1],label="P(Iris Setosa) | X ")
plt.legend(loc='best')
plt.xlabel('Sepal Length(in cm)')

# C = 10 , LOW REGULARIZATION 
plt.subplot(222)
plt.title("C = 10")
plt.plot(sepal_l,y,'ro')
plt.plot(np.linspace(4.5,8.0,1000),y_prob_low[:,0],label="P(Not Iris Setosa) | X ")
plt.plot(np.linspace(4.5,8.0,1000),y_prob_low[:,1],label="P(Iris Setosa) | X ",color='black')
plt.legend(loc='best')
plt.xlabel('Sepal Length(in cm)')


# C = 0.5 , HIGH REULARIZATION 
plt.subplot(223)
plt.title("C = 0.5")
plt.plot(sepal_l,y,'ro')
plt.plot(np.linspace(4.5,8.0,1000),y_prob_high[:,0],label="P(Not Iris Setosa) | X ")
plt.plot(np.linspace(4.5,8.0,1000),y_prob_high[:,1],label="P(Iris Setosa) | X ",color='black')
plt.legend(loc='best')
plt.xlabel('Sepal Length(in cm)')
plt.show()

plt.tight_layout()
# scores 
print(lr_single_feature_high_regularization.score(sepal_l.reshape(-1,1),y))
print(lr_single_feature_low_regularization.score(sepal_l.reshape(-1,1),y))
crazy_regularized.score(sepal_l.reshape(-1,1),y)
no_regularization.fit(sepal_l.reshape(-1,1),y)
no_regularization.score(sepal_l.reshape(-1,1),y)
X = data['data'] # using all 4 features 
y = data['target'] # labels
lr_multi_class = LogisticRegression(multi_class='ovr') # we set the `multi_class` attribute to 'ovr'
lr_multi_class.fit(X,y)
y_hat = lr_multi_class.predict(X) # make predictions
lr_multi_class.score(X,y)
plt.figure(figsize=(6,6))
plt.title("OvR | Green+ = Correct Prediction | Redx = Incorrect | Mean Accuracy: 95.3%",fontsize=14)
for i in range(len(X)):
    # correctly predicted
    if y[i] == y_hat[i]:
        plt.plot(i,X[i,0],'g+')
    else:
        plt.plot(i,X[i,0],'rx')
plt.show()

from sklearn.multiclass import OneVsOneClassifier
model = LogisticRegression()
ovo = OneVsOneClassifier(model)
# fit
ovo.fit(X,y)
# score
print("Score:",ovo.score(X,y))
# predictions
y_hat = ovo.predict(X)

plt.figure(figsize=(6,6))
plt.title("OvO | Green+ = Correct Prediction | Redx = Incorrect | Mean Accuracy: 95.3%",fontsize=14)
for i in range(len(X)):
    # correctly predicted
    if y[i] == y_hat[i]:
        plt.plot(i,X[i,0],'g+')
    else:
        plt.plot(i,X[i,0],'rx')
plt.show()

lr_muticlasss_multinomial = LogisticRegression(multi_class='multinomial',max_iter=125)
lr_muticlasss_multinomial.fit(X,y)
print(lr_muticlasss_multinomial.score(X,y))
y_hat = lr_muticlasss_multinomial.predict(X)
plt.figure(figsize=(6,6))
plt.title("OvO | Green+ = Correct Prediction | Redx = Incorrect | Mean Accuracy: 95.3%",fontsize=14)
for i in range(len(X)):
    # correctly predicted
    if y[i] == y_hat[i]:
        plt.plot(i,X[i,0],'g+')
    else:
        plt.plot(i,X[i,0],'rx')
plt.show()

from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)
lr_scaled_features = LogisticRegression(multi_class="ovr") # one vs rest
lr_scaled_features.fit(X_scaled,y)
# score
lr_scaled_features.score(X_scaled,y)
lr_ovo_scaled = LogisticRegression()
ovo_scaled = OneVsOneClassifier(lr_ovo_scaled)
ovo_scaled.fit(X_scaled,y)
ovo_scaled.score(X_scaled,y)
lr_scaled_multinomial = LogisticRegression(multi_class="multinomial")
lr_scaled_multinomial.fit(X_scaled,y)
lr_scaled_multinomial.score(X_scaled,y)