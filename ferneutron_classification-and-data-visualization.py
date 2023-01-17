import numpy as np
import pandas as pd
import seaborn as sb
import category_encoders as ce
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score

%matplotlib inline
australia = pd.read_csv('../input/weatherauscsv/weatherAUS.csv')
australia.head()
australia  = australia.drop(['Location','Date','Evaporation','Sunshine', 'Cloud9am','Cloud3pm',
                           'WindGustDir','WindGustSpeed','WindDir9am','WindDir3pm','WindSpeed9am',
                           'WindSpeed3pm'], axis=1)
Y =  australia.RainTomorrow
X = australia.drop(['RainTomorrow'], axis=1)
plot_sb = sb.countplot(Y, label='Total')
Rain, NotRain = Y.value_counts()
print('Rain: ',Rain)
print('Not Rain : ',NotRain)
X = X.replace({'No':0, 'Yes':1})
X = X.fillna(0)
Y = Y.replace({'No':0, 'Yes':1})
Y = Y.fillna(0)
X_scaled = (X - X.mean()) / (X.std())
X_scaled.head()
# Concatenate the target frame with just 20 columns from corpus_scaled
#X_plot = pd.concat([Y, X_scaled], axis=1) 
X_plot = pd.concat([Y, X_scaled.iloc[:,0:20]], axis=1) 

# Reshaping the frame
X_plot = pd.melt(X_plot, id_vars="RainTomorrow", var_name="Features", value_name='Values')
X_plot.head()
# Setting the plt object
plt.figure(figsize=(10,10))
# Setting the violinplot objetc with respecitve atributes
sb.violinplot(x="Features", y="Values", hue="RainTomorrow", data=X_plot, split=True, inner="quart")
# Rotation of x ticks
plt.xticks(rotation=90)
# Correlation is taken from Pearsonr value, 1 is totally correlated.
sb.jointplot(X_scaled.loc[:,'MinTemp'], 
              X_scaled.loc[:,'MaxTemp'], kind="regg", color="#ce1414")
f, ax = plt.subplots(figsize=(18, 18))
sb.heatmap(X_scaled.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.xticks(rotation=90)
x_train, x_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.25, random_state=28)
clf_rf = RandomForestClassifier(random_state=23)      
clr_rf = clf_rf.fit(x_train,y_train)
y_predict = clf_rf.predict(x_test)
accuracy = accuracy_score(y_test, y_predict )
print('Accuracy: ', accuracy)
conf_matrix = confusion_matrix(y_test, y_predict)
sb.heatmap(conf_matrix, annot=True, fmt="d")
clf_svm = SVC(kernel='linear', random_state=12)
clf_svm = clf_svm.fit(x_train, y_train)
y_predict = clf_svm.predict(x_test)
accuracy = accuracy_score(y_test, y_predict)
print('Accuracy: ', accuracy)
conf_matrix = confusion_matrix(y_test, y_predict)
sb.heatmap(conf_matrix, annot=True, fmt="d")
