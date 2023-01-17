# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np

import pandas as pd

from sklearn import preprocessing

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import ElasticNet,ElasticNetCV

from sklearn.metrics import r2_score,mean_squared_error

from sklearn.metrics import r2_score

from sklearn.linear_model import LogisticRegression



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("../input/chipotle-locations/chipotle_stores.csv")
plt.figure(figsize=(25,15))

g = sns.scatterplot(x=df['longitude'], y= df['latitude'], data=df, hue='state')

g.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

plt.savefig('map.png', bbox_inches="tight")

plt.show()
states = df.state.value_counts()

states
df_new = df.state.value_counts().rename_axis('state1').reset_index(name='count')

new_df = pd.merge(df, df_new, left_on='state', right_on='state1', how='left').drop('state1', axis=1)



le = preprocessing.LabelEncoder()

new_df['state'] = le.fit_transform(new_df['state'])

new_df['location'] = le.fit_transform(new_df['location'])

new_df = new_df.drop(columns = ['address'])
relation = new_df.corr()



fig = plt.figure(figsize = (10,10)) # Determines the size of the figure that will be displayed



ticks=[-1, -0.5 , 0 , +0.5, +1]  # Shows the interval of the colorbar displayed beside the heatmap



sns.heatmap(relation, vmin = -1, vmax = 1, square = True,center=0, cmap='BrBG', annot=True,robust=True, cbar_kws= {'shrink' : 0.8 , "ticks" : ticks}, linewidths= 0.2)

# vmin is the minimum range and vmax is the maximum range till which the heatmap will be displayed.

# cbar_kws shrinks the colobar to the same size as the heatmap

# linewidth is used to seperate the rows and coulnms by the given value to make the heatmap more presentable



plt.title("Relationship between Inputs and Outputs using Heatmap", fontsize = 16) ## Sets the title of the heatmap

plt.savefig('Heatmap.png', bbox_inches="tight")

plt.show() ## Displaying the heatmap
X = new_df.drop(columns = ['count'])

y = new_df[['count']]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25)



RandomForestmodel = RandomForestRegressor()

RandomForestmodel.fit(X_train,y_train)

y_pred = RandomForestmodel.predict(X_test)



mse = mean_squared_error(y_test,y_pred)

rmse = np.sqrt(mse)

r2_score_model = r2_score(y_test, y_pred)

print("RMSE value of the model is:", rmse)

print("R2 Score of the model is:", r2_score_model)
logisticRegr = LogisticRegression()

logisticRegr.fit(X_train, y_train)

y_pred = logisticRegr.predict(X_test)



mse = mean_squared_error(y_test,y_pred)

rmse = np.sqrt(mse)

r2_score_model = r2_score(y_test, y_pred)

print("RMSE value of the model is:", rmse)

print("R2 Score of the model is:", r2_score_model)
alphas = [0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1]



for a in alphas:

    model = ElasticNet(alpha=a).fit(X_train,y_train)   

    pred_y = model.predict(X_test)

    mse = mean_squared_error(y_test, pred_y)   

    score = r2_score(y_test,pred_y)

    print("Alpha:{0:.4f}, R2:{1:.2f}, MSE:{2:.2f}, RMSE:{3:.2f}"

       .format(a, score, mse, np.sqrt(mse)))