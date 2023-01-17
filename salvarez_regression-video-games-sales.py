%matplotlib notebook

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
#load data

df = pd.read_csv("../input/videogamesales/vgsales.csv")
df.describe(include = 'all')
df.isna().sum()
#drop na rows

df.dropna(inplace=True)
#select 10 games published as threashold



publisher_counts = df.Publisher.value_counts().reset_index()



publisher_threashold = []



for x in range(1,20):

    indx = publisher_counts.Publisher < x



    publisher_threashold.append([x,sum(indx)])



publisher_df = pd.DataFrame(publisher_threashold,columns=['n_games','n_publisher'])





fig, axs = plt.subplots(1,2)



axs[0].plot(publisher_df['n_publisher'],publisher_df['n_games'])

axs[0].set_xlabel("n_publisher")

axs[0].set_ylabel("n_games")

axs[0].set_title("N° games / Publisher")



axs[1].hist(df.Publisher.value_counts(),bins=20)

axs[1].set_xlabel("n publisher")

axs[1].set_ylabel("n games")

axs[1].set_title("N° games / Publisher")



plt.show()



#we set the 10 games published as threas

df_platform_sales = df.groupby('Platform').agg(['sum','count'])['Global_Sales'].sort_values(by=['sum']).reset_index()



fig, axs = plt.subplots()

axs.bar(df_platform_sales.Platform,df_platform_sales['sum'])



plt.xticks(rotation=90)



plt.show()
df_platform_sales = df.groupby('Platform').agg(['sum','count'])['Global_Sales'].sort_values(by=['sum']).reset_index()



#column revenues_platform

df_platform_sales.loc[df_platform_sales["sum"]<=150,"revenues_platform"] = "low"

df_platform_sales.loc[(df_platform_sales["sum"]>150) & (df_platform_sales["sum"]<500),"revenues_platform"] = "medium"

df_platform_sales.loc[df_platform_sales["sum"]>500,"revenues_platform"] = "high"



#column mean_game_revenue

df_platform_sales["mean_game_revenue"] = df_platform_sales["sum"] / df_platform_sales["count"]



df = df.merge(df_platform_sales[["Platform","revenues_platform","mean_game_revenue"]],how='inner',left_on="Platform",right_on="Platform")
# column published games



publisher_df = df.groupby("Publisher").count()["Rank"].reset_index()

publisher_df["published_games"] = publisher_df["Rank"]



df = df.merge(publisher_df[["Publisher","published_games"]],how="inner",left_on="Publisher",right_on="Publisher")
#drop unnecesary columns

df.drop(["Rank","Name","Platform","Publisher","NA_Sales",'EU_Sales',"JP_Sales","Other_Sales"],axis=1,inplace=True)
#one hot encoding

genre_one_hot = pd.get_dummies(df["Genre"],prefix="Genre")

revenues_platform_one_hot = pd.get_dummies(df["revenues_platform"],prefix="revenues_platform")



df = pd.concat([df,genre_one_hot,revenues_platform_one_hot],axis=1)



#drop columns

df.drop(["Genre","revenues_platform"],axis=1,inplace=True)
#the final look of the dataset

df.head()
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_val_score



results = []



X = df.drop("Global_Sales",axis=1)

y = df["Global_Sales"]



X = StandardScaler().fit_transform(X)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#simple multiple linear regresion

from sklearn.linear_model import LinearRegression



regressor = LinearRegression()



scores = cross_val_score(regressor,X,y,cv=15,scoring='neg_mean_squared_error')



results.append(['Linear Regression',scores.mean()])
#polinomial regresion

from sklearn.preprocessing import PolynomialFeatures



quadratic_featurizer = PolynomialFeatures(degree=2)

X_quadratic = quadratic_featurizer.fit_transform(X)



regressor_quadratic = LinearRegression()



scores = cross_val_score(regressor_quadratic,X_quadratic,y,cv=15,scoring='neg_mean_squared_error')



results.append(['Polinomial Regression (^2)',scores.mean()])
#stochastic gradient decent

from sklearn.linear_model import SGDRegressor





regressor = SGDRegressor(loss='squared_loss')

scores = cross_val_score(regressor,X,y,cv=15,scoring='neg_mean_squared_error')



results.append(['SGD',scores.mean()])
#neural network

from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasRegressor



def base_model():

    model = Sequential()

    model.add(Dense(18,input_dim=18,activation='relu'))

    model.add(Dense(1))

    model.compile(loss='mean_squared_error',optimizer='adam')

    return model



estimator = KerasRegressor(build_fn=base_model,epochs=50,batch_size=1,verbose=0)



scores = cross_val_score(estimator,X,y,cv=3)



results.append(['Neural Network',scores.mean()])
print(pd.DataFrame(results,columns=['Estimator','Neg Mean Squared Error']).sort_values(by=['Neg Mean Squared Error'],ascending=False))