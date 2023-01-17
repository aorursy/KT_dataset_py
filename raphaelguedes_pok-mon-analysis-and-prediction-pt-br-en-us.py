import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df = pd.read_csv('../input/pokemon/pokemon_alopez247.csv')
df.info()
df.describe()
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
def impute_Type_2(cols):
    Type_2 = cols[0]
    
    if pd.isnull(Type_2):
        return "None"


    else:
        return Type_2
df['Type_2'] = df[['Type_2']].apply(impute_Type_2,axis=1)
def impute1(cols):
    Name = cols[0]
    return 1
df['Unic'] = df.apply(impute1,axis=1)
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
pt_df = df.pivot_table(values='Unic',index='Type_1',columns='Type_2',aggfunc=np.sum)
plt.figure(figsize=(12,8))
sns.heatmap(pt_df,cmap='coolwarm',annot=True)
plt.figure(figsize = (20,8))
sns.boxplot(data = df)
plt.figure(figsize=(12,8))
sns.axes_style("white")
sns.boxplot(x='Generation',y='Total',data=df,palette='rainbow')

plt.figure(figsize=(12,8))
sns.swarmplot(x='Generation',y='Total',data=df,palette='rainbow')
plt.figure(figsize=(12,8))
ax=sns.countplot(y='Body_Style',data=df)
for p in ax.patches:
    ax.annotate(int(p.get_width()),((p.get_x() + p.get_width()), p.get_y()), xytext=(15, -18),fontsize=9,color='#004d00',textcoords='offset points', horizontalalignment='right')

plt.figure(figsize=(12,8))
splot = sns.countplot(x='isLegendary',data=df, hue="hasGender")
for p in splot.patches:
    splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
plt.figure(figsize=(12,8))
sns.scatterplot(x='Weight_kg',y='Height_m',data=df,hue='Body_Style')
plt.figure(figsize=(12,8))
sns.scatterplot(x='Speed',y='Height_m',data=df,hue='Body_Style')
sns.set_style('white')
plt.figure(figsize=(12,8))
sns.scatterplot(x='Speed',y='Weight_kg',data=df,hue='Body_Style')
sns.set_style("darkgrid")
sns.lmplot(x='Weight_kg',y='Height_m',data=df,row='Body_Style',height=10)
sns.set_style('white')
sns.jointplot(x='Catch_Rate',y='Total',data=df,kind='scatter')
df.drop(['Number','Name','Type_1','Type_2','Color','Generation','Pr_Male','Egg_Group_1','Egg_Group_2','hasMegaEvolution','Body_Style','Unic'],axis=1,inplace=True)


df.info()
sns.pairplot(df,hue='isLegendary',palette='bwr',diag_kws={'bw': 0.2})

from sklearn.model_selection import train_test_split
X = df.drop('isLegendary',axis=1)
y = df['isLegendary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=600)
rfc.fit(X_train,y_train)
predictions = rfc.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))