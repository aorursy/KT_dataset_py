import pandas as pd
dataset=pd.read_csv("../input/haberman.csv",names=None)

X=dataset.iloc[1:,0:3].values
Y=dataset.iloc[1:,3].values
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
# create the RFE model and select 3 attributes
rfe = RFE(logreg, 2)
rfe = rfe.fit(X, Y)

# summarize the selection of the attributes
print(rfe.support_)
print(rfe.ranking_)
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
extree = ExtraTreesClassifier()
extree.fit(X, Y)
pos=[0,0.5,1]

# display the relative importance of each attribute
relval = extree.feature_importances_
l2=[]
for i in relval:
    l2.append(i)
print(l2)
l1=['age','year','positive_axillary_nodes']
plt.bar(l1,l2)


dataset.describe()
plt.scatter(dataset['Age'],dataset['positive_axillary_nodes'],color='green')
plt.xlabel('age')
plt.ylabel('Axial Nodes')
plt.scatter(dataset['Age'],dataset['year'],color='green')
plt.xlabel('age')
plt.ylabel('Year of operation')
plt.scatter(dataset['year'],dataset['positive_axillary_nodes'],color='green')
plt.xlabel('Year of operation')
plt.ylabel('Axial Nodes')
import seaborn as sns
plt.close()
sns.set_style('whitegrid')
sns.pairplot(dataset,hue='survival_status',size=4)
plt.show()
sns.boxplot(x='survival_status',y='positive_axillary_nodes',data=dataset)
sns.violinplot(x='survival_status',y='positive_axillary_nodes',data=dataset)
