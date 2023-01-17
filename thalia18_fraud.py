import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

fraud_file_path = '../input/PS_20174392719_1491204439457_log.csv'
fraud_data = pd.read_csv(fraud_file_path)
fraud_data=fraud_data.dropna()
fraud_data = fraud_data.drop_duplicates(fraud_data.columns, keep='last')
print(fraud_data.columns)
fraud_data.describe()
import seaborn as sns
from matplotlib.pyplot import plot
plt.title(r'Transactions Without Fraud')
print(fraud_data[fraud_data.isFraud==0].type.value_counts().head() / len(fraud_data))
(fraud_data[fraud_data.isFraud==0].type.value_counts().head() / len(fraud_data)).plot.bar(label='noFraud')
print(fraud_data[fraud_data.isFraud==1].type.value_counts().head() / len(fraud_data))
plt.title('Transactions With fraud')
(fraud_data[fraud_data.isFraud==1].type.value_counts().head() / len(fraud_data)).plot.bar(legend='isFraud')
plt.title(r'Step frecuency')
ax1=fraud_data[fraud_data.isFraud==1].step.value_counts().sort_index().plot.line(label='fraud')
ax1=fraud_data[fraud_data.isFraud==0].step.value_counts().sort_index().plot.line(ax=ax1)
ax1.legend(["Fraud", "No fraud"])
plt.title(r'Close up Fraud per unit of times')
ax1=fraud_data[fraud_data.isFraud==1].step.value_counts().sort_index().plot.line()
ax1.legend([ "Fraud"])
plt.title(r'Fraud Flagged or no per unit of time')
ax1=fraud_data[(fraud_data.isFraud==1)&(fraud_data.isFlaggedFraud==1)].step.value_counts().sort_index().plot.line(label='fraud')
ax1=fraud_data[(fraud_data.isFraud==1)&(fraud_data.isFlaggedFraud==0)].step.value_counts().sort_index().plot.line(label='fraud')
ax1.legend(["Fraud Flagged as Fraud","Fraud Not Flagged as Fraud"])
ax1=fraud_data[fraud_data.isFraud==1].plot.scatter(x='oldbalanceOrg', y='newbalanceOrig',c='blue',title='Relation old balance new balance',label='Fraud')
fraud_data[fraud_data.isFraud==0].plot.scatter(x='oldbalanceOrg', y='newbalanceOrig',c='orange',label='No Fraud',ax=ax1)
ax1.set_xlabel("Old balance Origin")
ax1.set_ylabel("New Balance Origin")
ax1=fraud_data[fraud_data.isFraud==0].plot.scatter(x='oldbalanceDest', y='newbalanceDest',c='orange',label='No Fraud')
fraud_data[fraud_data.isFraud==1].plot.scatter(x='oldbalanceDest', y='newbalanceDest',c='blue',title='Relation old balance new balance',label='Fraud',ax=ax1)
ax1.set_xlabel("Old balance Destination")
ax1.set_ylabel("New Balance Destination")
import seaborn as sns
g=sns.boxplot(x='type',y='amount',data=fraud_data,palette='rainbow',hue='isFraud')
g.set_yscale('log')
fraud_data.type.unique()
fraud_data=pd.get_dummies(data=fraud_data, columns=['type'])
features = ['amount','oldbalanceOrg', 'newbalanceOrig',
            'type_CASH_IN','type_CASH_OUT','type_DEBIT','type_PAYMENT','type_TRANSFER',
        'oldbalanceDest', 'newbalanceDest', 'isFraud']
data2 = fraud_data[features]
data2.describe()
data2['amount']=np.log1p(data2['amount'])
data2['oldbalanceOrg']=np.log1p(data2['oldbalanceOrg'])
data2['newbalanceOrig']=np.log1p(data2['newbalanceOrig'])
data2['oldbalanceDest']=np.log1p(data2['oldbalanceDest'])
data2['newbalanceDest']=np.log1p(data2['newbalanceDest'])
percent=[1,5,10,20,50,100]
df = pd.DataFrame({})
for value in percent:
    dataselect= data2.sample(frac=value/100)
    corr = dataselect.corr()
    df[str(value/100)]=corr['isFraud']
   # print(corr['isFraud'])#.sort_values(ascending=False))
df

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

finaldata=data2.sample(frac=.2)
colormap = plt.cm.magma
plt.figure(figsize=(11,11))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(data2.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
y = finaldata['isFraud']
X = finaldata.drop(['isFraud'], axis=1).values 
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

from sklearn import tree
from sklearn.metrics import f1_score, average_precision_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier

for i in range(1,11):
    decision_tree = DecisionTreeClassifier(max_depth = i)
    decision_tree.fit(train_X, train_y)
    print(i,"Accuracy:", decision_tree.score(val_X, val_y))
from IPython.display import Image as PImage
from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, average_precision_score, precision_score, recall_score

decision_tree = DecisionTreeClassifier(max_depth = 9)
decision_tree.fit(train_X, train_y)
y_pred_tree = decision_tree .predict(val_X)

DecisionTree=[decision_tree.score(val_X, val_y),precision_score(val_y, y_pred_tree),recall_score(val_y, y_pred_tree),f1_score(val_y, y_pred_tree)]
results=pd.DataFrame(DecisionTree,columns=['DecisionTree'],index=['Accuracy:', 'Precision:', 'Recall:','F1:'])


# Export our trained model as a .dot file
with open("treeisfraud.dot", 'w') as f:
     #f = Source(
         f=tree.export_graphviz(decision_tree,
                              out_file=f,
                              max_depth = 9,
                              impurity = True,
                              feature_names = ['amount','oldbalanceOrg', 'newbalanceOrig',
            'type_CASH_IN','type_CASH_OUT','type_DEBIT','type_PAYMENT','type_TRANSFER',
        'oldbalanceDest', 'newbalanceDest'],
                              class_names = ['No Fraud', 'Fraud'],
                              rounded = True,
                              filled= True )#)
        
#Convert .dot to .png to allow display in web notebook
check_call(['dot','-Tpng','treeisfraud.dot','-o','treeisfraud.png'])

# Annotating chart with PIL
img = Image.open("treeisfraud.png")
draw = ImageDraw.Draw(img)
font = ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf', 26)
draw.text((10, 0), # Drawing offset (position)
          '"Title <= Is Fraud', # Text to draw
          (0,0,255), # RGB desired color
          font=font) # ImageFont object with desired font
img.save('tree_isfraud.png')
PImage("tree_isfraud.png")

from sklearn.neighbors import KNeighborsClassifier

neiclassifier = KNeighborsClassifier(n_neighbors=5)  
neiclassifier.fit(train_X, train_y)
y_pred_nei = neiclassifier.predict(val_X)

KNeighbors=[neiclassifier.score(val_X, val_y),precision_score(val_y, y_pred_nei),recall_score(val_y, y_pred_nei),f1_score(val_y,y_pred_nei)]


results['KNeighbors']=KNeighbors
results
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(random_state=1)
forest.fit(train_X, train_y)
y_pred_for = forest.predict(val_X)

RandomForest=[forest.score(val_X, val_y),precision_score(val_y, y_pred_for),recall_score(val_y, y_pred_for),f1_score(val_y,y_pred_for)]

results['RandomForest']=RandomForest
results
from sklearn.linear_model import LogisticRegression
import warnings
#warnings.filterwarnings("ignore", category=FutureWarning)

logreg = LogisticRegression(solver='lbfgs')
logreg.fit(train_X, train_y)
y_pred_log =logreg.predict(val_X)

LogisticRegression=[logreg.score(val_X, val_y),precision_score(val_y, y_pred_log),recall_score(val_y, y_pred_log),f1_score(val_y,y_pred_log)]

results['LogisticRegression']=LogisticRegression
results
import matplotlib.pyplot as plt
import random

preds=pd.DataFrame()
preds['validation']=val_y
preds['tree']=y_pred_tree
preds['neighbors']=y_pred_nei
preds['forest']=y_pred_for
preds['logistic']=y_pred_log

preds.describe()
pr=preds.sample(n=100)
ind = np.linspace(0,200,100)# len(val_y),len(val_y))
plt.xlabel('Data index')
plt.ylabel('Fraud or No?') 

plt.plot(ind, pr['validation'],'X', markersize=6,label='Test data' )
plt.plot(ind,pr['tree'], '.', markersize=5,label='Decision Tree')
plt.plot(ind,pr['neighbors'], '.', markersize=5,label='K Neighbors')
plt.plot(ind,pr['forest'], '.', markersize=5,label='Random Forest')
plt.plot(ind,pr['logistic'], '.', markersize=5,label='Logistic Regression')
plt.legend(loc='right')
#plt.xlim(-2,1012)
plt.ylim(-.1,1.12)
plt.show()
