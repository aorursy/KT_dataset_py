import pandas as pd 
df = pd.read_csv('../input/zoo.csv')
df.info()
print ('-------------------')
df.head()
df2 = pd.read_csv('../input/class.csv')
df2.info()
print ('----------------')
df2.head()
df3 = df.merge(df2,how='left',left_on='class_type',right_on='Class_Number')
df3.head()
g = df3.groupby(by='Class_Type')['animal_name'].count()
g / g.sum() * 100
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(df3['Class_Type'],label="Count",
             order = df3['Class_Type'].value_counts().index) #sort bars
plt.show()
feature_names = ['hair','feathers','eggs','milk','airborne','aquatic','predator','toothed',
                 'backbone','breathes','venomous','fins','legs','tail','domestic']

df3['ct'] = 1

for f in feature_names:
    g = sns.FacetGrid(df3, col="Class_Type",  row=f, hue="Class_Type")
    g.map(plt.hist, "ct")
    g.set(xticklabels=[])
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle(f)
gr = df3.groupby(by='Class_Type').mean()
columns = ['class_type','Class_Number','Number_Of_Animal_Species_In_Class','ct','legs'] #will handle legs separately since it's not binary
gr.drop(columns, inplace=True, axis=1)
plt.subplots(figsize=(10,4))
sns.heatmap(gr, cmap="YlGnBu")
sns.stripplot(x=df3["Class_Type"],y=df3['legs'])
#specify the inputs (x = predictors, y = class)
X = df[feature_names]
y = df['class_type'] #there are multiple classes in this column

#split the dataframe into train and test groups
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.2, test_size=.8)

#specify the model to train with
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier().fit(X_train, y_train)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning) #ignores warning that tells us dividing by zero equals zero

#let's see how well it worked
pred = clf.predict(X_test)
print('Accuracy of classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))
print()
print(confusion_matrix(y_test, pred))
print()
print(classification_report(y_test, pred))
df3[['Class_Type','class_type']].drop_duplicates().sort_values(by='class_type') #this is the order of the labels in the confusion matrix above
imp = pd.DataFrame(clf.feature_importances_)
ft = pd.DataFrame(feature_names)
ft_imp = pd.concat([ft,imp],axis=1,join_axes=[ft.index])
ft_imp.columns = ['Feature', 'Importance']
ft_imp.sort_values(by='Importance',ascending=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.1, test_size=.9) 

clf2 = DecisionTreeClassifier().fit(X_train, y_train)
pred = clf2.predict(X_test)
print('Accuracy of classifier on test set: {:.2f}'
     .format(clf2.score(X_test, y_test)))
print()
print(confusion_matrix(y_test, pred))
print()
print(classification_report(y_test, pred))
imp2 = pd.DataFrame(clf2.feature_importances_)
ft_imp2 = pd.concat([ft,imp2],axis=1,join_axes=[ft.index])
ft_imp2.columns = ['Feature', 'Importance']
ft_imp2.sort_values(by='Importance',ascending=False)
visible_feature_names = ['hair','feathers','toothed','fins','legs','tail']

X = df[visible_feature_names]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.2, test_size=.8)

clf3= DecisionTreeClassifier().fit(X_train, y_train)

pred = clf3.predict(X_test)
print('Accuracy of classifier on test set: {:.2f}'
     .format(clf3.score(X_test, y_test)))
print()
print(confusion_matrix(y_test, pred))
print()
print(classification_report(y_test, pred))
imp3= pd.DataFrame(clf3.feature_importances_)
ft = pd.DataFrame(visible_feature_names)
ft_imp3 = pd.concat([ft,imp3],axis=1,join_axes=[ft.index])
ft_imp3.columns = ['Feature', 'Importance']
ft_imp3.sort_values(by='Importance',ascending=False)
clf4= DecisionTreeClassifier(max_depth=2).fit(X_train, y_train)

pred = clf4.predict(X_test)
print('Accuracy of classifier on test set: {:.2f}'
     .format(clf4.score(X_test, y_test)))
print()
print(confusion_matrix(y_test, pred))
print()
print(classification_report(y_test, pred))
imp4= pd.DataFrame(clf4.feature_importances_)
ft_imp4 = pd.concat([ft,imp3],axis=1,join_axes=[ft.index])
ft_imp4.columns = ['Feature', 'Importance']
ft_imp4.sort_values(by='Importance',ascending=False)
columns = ['Model','Test %', 'Accuracy','Precision','Recall','F1','Train N']
df_ = pd.DataFrame(columns=columns)

df_.loc[len(df_)] = ["Model 1",20,.78,.80,.78,.77,81] #wrote the metrics down on paper and input into this dataframe
df_.loc[len(df_)] = ["Model 2",10,.68,.62,.68,.64,91]
df_.loc[len(df_)] = ["Model 3",20,.91,.93,.91,.91,81]
df_.loc[len(df_)] = ["Model 4",20,.57,.63,.57,.58,81]
ax=df_[['Accuracy','Precision','Recall','F1']].plot(kind='bar',cmap="YlGnBu", figsize=(10,6))
ax.set_xticklabels(df_.Model)