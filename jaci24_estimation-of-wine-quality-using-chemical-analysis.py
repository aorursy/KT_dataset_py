import pandas as pd

import numpy as np



chemical1 = pd.read_csv('../input/wine-quality/winequalityN.csv')

print(chemical1.shape)

chemical1.info()
sensory1 = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv',engine='python', error_bad_lines=False)

print(sensory1.shape)

sensory1.info()
sensory2 = pd.read_csv('../input/wine-reviews/winemag-data-130k-v2.csv',engine='python', error_bad_lines=False)

print(sensory2.shape)

sensory2.info()
import matplotlib.pyplot as pit





total = len(chemical1.index)

white_wine = chemical1[chemical1['type']=='white']

print("Percentage of white wine: ",(len(white_wine.index)/total)*100,"%")

red_wine = chemical1[chemical1['type']=='red']

print("Percentage of red wine: ",(len(red_wine.index)/total)*100,"%")



pit.pie([len(white_wine.index),len(red_wine.index)], colors = ['#635e6b','#EF5350'], labels = ['white wine','red wine'],startangle=90)

pit.title('Types in the wine chemical datasets')

pit.show()
from prettytable import PrettyTable



labels = ['Quality 1','Quality 2','Quality 3','Quality 4','Quality 5','Quality 6','Quality 7','Quality 8','Quality 9','Quality 10']

t = PrettyTable(['Quality', 'white wine', 'red wine'])

values_white = []

values_red = []

for i in range(1,11):

  values_white.append(len(white_wine[white_wine['quality']==i]))

  values_red.append(len(red_wine[red_wine['quality']==i]))

  t.add_row([i,len(white_wine[white_wine['quality']==i]),len(red_wine[red_wine['quality']==i])])



print('Wines in each quality')

print(t)



colors_w = ['#fdf7ff','#f8f2ff','#f1ebfa','#e2ddec','#bfbac8','#a09ba9','#77727f','#635e6b','#433f4b','#221e29']

colors_r = ['#FFEBEE','#FFCDD2','#EF9A9A','#E57373','#EF5350','#F44336','#E53935','#D32F2F','#C62828','#B71C1C']



fig,(ax1,ax2)= pit.subplots(2,1)

fig.set_size_inches(10,9)

#ax1.pie(values_white, colors = colors_w, labels=labels, startangle=90)

ax1.bar(labels, values_white)

ax1.set_title('Number of white wines in each quality')

ax1.legend()



#ax2.pie(values_red, colors = colors_r, labels=labels, startangle=90)

ax2.bar(labels, values_red)

ax2.set_title('Number of white wines in each quality')

ax2.set_title('Number of red wines in each quality')

ax2.legend()



pit.show()

import seaborn as sns



feature_names = ['fixed acidity','volatile acidity', 'citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']



def find_MIN_MAX(df,q):

  print("quality: ", q)

  if(df.empty):

    print('no datasets')

    return

  else:

    t = PrettyTable(['Name', 'min', 'max'])

    min_list=[]

    max_list=[]

    for x in range(1,12):

      name = df.columns[x]

      mini = df[df.columns[x]].min()

      min_list.append(mini)

      maxi = df[df.columns[x]].max()

      max_list.append(maxi)

      t.add_row([name,mini,maxi])

    print(t)

    barWidth = 0.3

    pit.figure(figsize=(18, 5))

    r1 = np.arange(len(min_list))

    r2 = [x + barWidth for x in r1]

    pit.bar(r1, min_list, width = barWidth, color = 'blue', edgecolor = 'black',label='Min')

    pit.bar(r2, max_list, width = barWidth, color = 'cyan', edgecolor = 'black', label='Max')

    pit.xticks([r + barWidth for r in range(len(min_list))], feature_names)

    pit.ylabel('Values')

    pit.legend()

    pit.show()



    



print('Red Wine Data:')



for i in range(1,11):

   find_MIN_MAX(red_wine[red_wine['quality']==i],i)







print('White Wine Data:')



for i in range(1,11):

  find_MIN_MAX(white_wine[white_wine['quality']==i],i)

#for white wine

old_min = white_wine['quality'].min()

old_max = white_wine['quality'].max()



new_min = sensory1['points'].min()

new_max = sensory1['points'].max()

print(new_max)



white_wine['points'] = (((white_wine['quality'] - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min

white_wine.head()
#for red wine

old_min = red_wine['quality'].min()

old_max = red_wine['quality'].max()



red_wine['points'] = (((red_wine['quality'] - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min

red_wine.head()
uniq_variety1 = sensory1['variety'].unique()

uniq_vriety2 = sensory2['variety'].unique()

common_variety = list(set(uniq_variety1).intersection(uniq_vriety2))

print('Common and unique varieties in both csv are: ',common_variety)

print('Number of varieties :',len(common_variety))
white_varieties = ['Aligoté','Alvarinho', 'Auxerrois', 'Bacchus','Bual','Chardonnay','Chasselas','Chenin','Blanc','Colombard','Emerald','Riesling','Fumé','Blanc','Folle','Blanche','Furmint','Gewürztraminer','Grüner Veltliner','Hárslevelü','Jacquère','Kerner','Malvasia','Marsanne','Morio-Muscat','Müller-Thurgau','Muscadelle','Muscadet','Moscato','Palomino','Pedro Ximenez','Picolit','Pinot Blanc','Pinot Gris','Riesling','Rkatsiteli','Sacy','Savagnin','Sauvignon Blanc','Scheurebe','Sémillon','Sercial','Seyval Blan','Silvaner','Trebbiano','Verdelho','Verdicchio','Vidal','Viognier','Viura','Welschriesling']

red_varieties = ['Aglianico','Alicante','Baco','Noir','Barbera','Cabernet Franc','Cabernet Sauvignon','Carignan','Cinsault','de Chaunac','Dolcetto','Freisa','Gamay','Gamay Beaujolais','Grenache','Grignolino','Kadarka','Lambrusco','Malbec','Maréchal Foch','Merlot','Mourvèdre','Nebbiolo','Petite Sirah','Pinot Noir','Pinot','Meunier','Pinotae','primitivo','Ruby Cabernet','Sangiovese','Syrah','Tempranillo,''Touriga Naçional','Xynomavro','Zinfandel']

print('Number of white wine varieties considered: ', len(white_varieties))

print('Number of red wine varieties considered: ', len(red_varieties))
white = pd.concat([sensory1[sensory1['variety'].isin(white_varieties)] ,sensory2[sensory2['variety'].isin(white_varieties)]],sort=False)

white = white[np.isfinite(white['price'])]

print('shape of the white wine: ', white.shape)

white.head()


red = pd.concat([sensory1[sensory1['variety'].isin(red_varieties)],sensory2[sensory2['variety'].isin(red_varieties)]],sort=False)

red = red[np.isfinite(red['price'])]

print('shape of the red wine: ', red.shape)

red.head()
from sklearn.linear_model import LinearRegression,LogisticRegression,  BayesianRidge

from sklearn.model_selection import train_test_split

from sklearn import metrics



x = white[['points']]

y = white[['price']]



x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

model_price_white = LinearRegression()

model_price_white.fit(x_train,y_train)

y_pred = model_price_white.predict(x_test)

r =metrics.mean_squared_error(y_test,y_pred)

print("white wine::")

print("MEAN SQUARE ERROR: ", r)

print("ROOT MEAN SQUARE ERROR: ",np.sqrt(r))



pit.scatter(x,y)

pit.xlabel('quality points')

pit.ylabel('price')

pit.title('Price determined from Quality points for white wine')

pit.show()

x = red[['points']]

y = red[['price']]



x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

model_price_red = LinearRegression()

model_price_red.fit(x_train,y_train)

y_pred = model_price_red.predict(x_test)

r =metrics.mean_squared_error(y_test,y_pred)

print("red wine::")

print("MEAN SQUARE ERROR: ", r)

print("ROOT MEAN SQUARE ERROR: ",np.sqrt(r))



pit.scatter(x,y,color='r')

pit.xlabel('quality points')

pit.ylabel('price')

pit.title('Price determined from Quality points for red wine')

pit.show()
white_wine['price']=model_price_white.predict(np.array(white_wine['points']).reshape(-1,1))

red_wine['price']=model_price_red.predict(np.array(red_wine['points']).reshape(-1,1))
from sklearn.ensemble import RandomForestClassifier

import seaborn as sns



x_white = white_wine[feature_names]

y = white_wine['quality']

x_white.fillna(0, inplace=True)

x_train,x_test,y_train,y_test = train_test_split(x_white,y,test_size=0.3)

model = RandomForestClassifier(n_estimators=10)

model.fit(x_train,y_train)

y_pred = model.predict(x_test)
print("PRECISION RECALL: ", metrics.recall_score(y_test,y_pred,average='weighted'))



feature_imp = pd.Series(model.feature_importances_,index = feature_names).sort_values(ascending=False)

print(feature_imp)



sns.barplot(x=feature_imp,y=feature_imp.index)

pit.xlabel('Properties importance score')

pit.ylabel('Chemical Properties')

pit.title('Visualizing Important Features for white wine')

pit.legend()

pit.show()
x_red = red_wine[feature_names]

y = red_wine['quality']

x_red.fillna(0, inplace=True)



x_train,x_test,y_train,y_test = train_test_split(x_red,y,test_size=0.3)



model = RandomForestClassifier(n_estimators=10)

model.fit(x_train,y_train)

y_pred = model.predict(x_test)



print("PRECISION RECALL: ", metrics.recall_score(y_test,y_pred,average='weighted'))



feature_imp = pd.Series(model.feature_importances_,index = feature_names).sort_values(ascending=False)

print(feature_imp)



sns.barplot(x=feature_imp,y=feature_imp.index)

pit.xlabel('Properties importance score')

pit.ylabel('Chemical Properties')

pit.title('Visualizing Important Features for red wine')

pit.legend()

pit.show()
feature = ['alcohol','density','volatile acidity','free sulfur dioxide','total sulfur dioxide','sulphates']

x = white_wine[feature]

y = white_wine['quality']

x.fillna(0, inplace=True)



model_white = LogisticRegression() #RandomForestClassifier(n_estimators=10)

model_white.fit(x,y)



x = red_wine[feature]

y = red_wine['quality']

x.fillna(0, inplace=True)



model_red = LogisticRegression() #RandomForestClassifier(n_estimators=10)

model_red.fit(x,y)
white_wine['predicted quality'] = model_white.predict(x_white[feature])

red_wine['predicted quality'] = model_red.predict(x_red[feature])
white_wine['predicted points'] = (((white_wine['predicted quality'] - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min

white_wine['predicted price'] = model_price_white.predict(white_wine[['predicted points']])
red_wine['predicted points'] = (((red_wine['predicted quality'] - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min

red_wine['predicted price'] = model_price_red.predict(red_wine[['predicted points']])
total_price_white = sum(white_wine['price'])

total_price_red = sum(red_wine['price'])

total_price = total_price_white + total_price_red



print('Predicted price from sensory points')

print("Total price for white wine: ", "{0:.2f}".format(total_price_white))

print("Total price for red wine: ", "{0:.2f}".format(total_price_red))

print("Total price : ", "{0:.2f}".format(total_price))





predicted_total_price_white = sum(white_wine['predicted price'])

predicted_total_price_red = sum(red_wine['predicted price'])

predicted_total_price = predicted_total_price_white + predicted_total_price_red



print('Predicted price from chemical analysis')

print("Total predicted price for white wine: ", "{0:.2f}".format(predicted_total_price_white))

print("Total predicted price for red wine: ", "{0:.2f}".format(predicted_total_price_red))

print("Total predicted price : ", "{0:.2f}".format(predicted_total_price))



barWidth = 0.2

pit.figure(figsize=(8, 6))

r1 = np.arange(3)

r2 = [x + barWidth for x in r1]

pit.bar(r1, [total_price_white,total_price_red,total_price], width = barWidth, color = 'blue', edgecolor = 'black',label='Predicted price from sensory points')

pit.bar(r2, [predicted_total_price_white,predicted_total_price_red,predicted_total_price], width = barWidth, color = 'cyan', edgecolor = 'black', label='Predicted price from chemical analysis')

pit.xticks([r + barWidth for r in range(3)], ['white wine','red wine','total'])

pit.ylabel('Price')

pit.title('Comparing the price calculated from sensory points and estmated quality points from chemical analysis')

pit.legend()

pit.show()
print('Profit gained for white wine: ',((predicted_total_price_white-total_price_white)/total_price)*100,"%")

print('Loss gained for red wine: ',((total_price_red-predicted_total_price_red)/total_price)*100,"%")

print('Total Profit gained: ',((predicted_total_price-total_price)/total_price)*100,"%")