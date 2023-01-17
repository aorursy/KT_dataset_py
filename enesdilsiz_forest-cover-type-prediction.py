import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")
data = pd.read_csv('../input/forest-cover-type-dataset/covtype.csv')
data
print('Data Dimensions:','Rows(Records):', data.shape[0],'Columns(Features):', data.shape[1])
data.isnull().sum()
data.describe()
skewness = data.skew()
skewness
skew = pd.DataFrame(skewness, index=None, columns=['Skewness'])
plt.figure(figsize=(20,10))
sns.barplot(x=skew.index, y='Skewness', data=skew)
plt.xticks(rotation=90)
plt.title('Skewness of Features',fontsize=15)
class_dist = data.groupby('Cover_Type').size()
labels= '(Type1)Spruce/Fir', '(Type2)Lodgepole Pine', '(Type3)Ponderosa Pine', '(Type4)Cottonwood/Willow', '(Type5)Aspen', '(Type6)Douglas-fir', '(Type7)Krummholz'
fig1, ax1 = plt.subplots()
fig1.set_size_inches(15,10)
ax1.pie(class_dist, labels=labels, autopct='%1.1f%%')
ax1.axis('equal')
plt.title('Percentages of Cover Types',fontsize=15)
plt.show()
features = data.loc[:,'Elevation':'Horizontal_Distance_To_Fire_Points']
soiltype = data.loc[:,'Soil_Type1':'Soil_Type40']
wilderness = data.loc[:,'Wilderness_Area1':'Wilderness_Area4']
plt.figure(figsize=(30, 50))
for i,col in enumerate(features.columns.values):
    plt.subplot(5,2,i+1)
    sns.distplot(features[col])
    plt.title(col, fontsize=20)
    
plt.show()
plt.figure(figsize=(15,8))
plt.scatter(data["Elevation"], data["Horizontal_Distance_To_Hydrology"],c=data["Cover_Type"],cmap="rainbow")
plt.title('Elevation - Horizontal Distance to Water and Cover Types')
plt.xlabel('Elevation')
plt.ylabel('Horizontal Distance to Water')
plt.show()
plt.figure(figsize=(15,8))
plt.scatter(data["Elevation"], data["Horizontal_Distance_To_Roadways"],c=data["Cover_Type"],cmap="rainbow")
plt.title('Elevation - Horizontal Distance to Roadways and Cover Types')
plt.xlabel('Elevation')
plt.ylabel('Horizontal Distance to Roadways')
plt.show()
#data.shape = (581012, 55)
feat = features.copy()
feat['Cover_Type'] = data['Cover_Type']
feat
feat=feat.sort_values(by=['Cover_Type'])
cov1cnt = 0
covcnt = []
for k in range(1,8):
    for i in range(0,581012):
        if feat['Cover_Type'][i] == k:
            cov1cnt = cov1cnt +1
    covcnt.append(cov1cnt)
covcnt
cov1 = feat.iloc[0:211840]
cov2 = feat.iloc[211840:495141]
cov3 = feat.iloc[495141:530895]
cov4 = feat.iloc[530895:533642]
cov5 = feat.iloc[533642:543135]
cov6 = feat.iloc[543135:560502]
cov7 = feat.iloc[560502:581012]
avg1 = cov1.sum().div(211840)
avg2 = cov2.sum().div(495141-211840)
avg3 = cov3.sum().div(530895-495141)
avg4 = cov4.sum().div(533642-530895)
avg5 = cov5.sum().div(543135-533642)
avg6 = cov6.sum().div(560502-543135)
avg7 = cov7.sum().div(581012-560502)
from math import pi

# Set data
df = pd.DataFrame({
'group': ['Type1','Type2','Type3','Type4','Type5','Type6','Type7'],
'Elevation': [avg1[0],avg2[0],avg3[0],avg4[0],avg5[0],avg6[0],avg7[0]],
'Aspect': [avg1[1]*20,avg2[1]*20,avg3[1]*20,avg4[1]*20,avg5[1]*20,avg6[1]*20,avg7[1]*20],
'Slope': [avg1[2]*150,avg2[2]*150,avg3[2]*150,avg4[2]*150,avg5[2]*150,avg6[2]*150,avg7[2]*150],
'Horizontal_Distance_To_Hydrology': [avg1[3]*10,avg2[3]*10,avg3[3]*10,avg4[3]*10,avg5[3]*10,avg6[3]*10,avg7[3]*10],
'Vertical_Distance_To_Hydrology': [avg1[4]*50,avg2[4]*50,avg3[4]*50,avg4[4]*50,avg5[4]*50,avg6[4]*50,avg7[4]*50],
'Horizontal_Distance_To_Roadways': [avg1[5],avg2[5],avg3[5],avg4[5],avg5[5],avg6[5],avg7[5]],
'Hillshade_9am': [avg1[6]*10,avg2[6]*10,avg3[6]*10,avg4[6]*10,avg5[6]*10,avg6[6]*10,avg7[6]*10],
'Hillshade_Noon': [avg1[7]*10,avg2[7]*10,avg3[7]*10,avg4[7]*10,avg5[7]*10,avg6[7]*10,avg7[7]*10],
'Hillshade_3pm': [avg1[8]*20,avg2[8]*20,avg3[8]*20,avg4[8]*20,avg5[8]*20,avg6[8]*20,avg7[8]*20],
'Horizontal_Distance_To_Fire_Points': [avg1[9],avg2[9],avg3[9],avg4[9],avg5[9],avg6[9],avg7[9]]
})



plt.figure(figsize=(20,15)) 

# number of variable
categories=list(df)[1:]
N = len(categories)
 
# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
 
# Initialise the spider plot
ax = plt.subplot(111, polar=True)
 
# If you want the first axis to be on top:
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)
 
# Draw one axe per variable + add labels labels yet
plt.xticks(angles[:-1], categories)
 
# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([1000,2000,3000], ["1000","2000","3000"], color="black", size=5)
plt.ylim(0,3700)

plt.title('Radar Chart of Features',fontsize=15)

for i in range(0,7):
    values=df.loc[i].drop('group').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label="Type %d" %i)
    ax.fill(angles, values, 'b', alpha=0.2)
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.figure(figsize=(30, 50))
for i,col in enumerate(features.columns.values):
    plt.subplot(5,2,i+1)
    sns.boxplot(x=data['Cover_Type'], y=col, data=data)
    plt.title(col, fontsize=20)
    
plt.show()
def rev_code(row):
    for c in soiltype.columns:
        if row[c]==1:
            return c  
cover_soil_type = pd.DataFrame()
cover_soil_type['Cover_Type'] = data['Cover_Type']
cover_soil_type['Soil_Type']=soiltype.apply(rev_code, axis=1)
cover_soil_type
plt.figure(figsize=(20,10))
sns.countplot(x='Soil_Type', hue='Cover_Type',data=cover_soil_type, palette="rainbow")
plt.xticks(rotation=90)
plt.title('Count Plot of Soil Types',fontsize=15)
def rev_code(row):
    for c in wilderness.columns:
        if row[c]==1:
            return c  

cover_wild_type = pd.DataFrame()
cover_wild_type['Cover_Type'] = data['Cover_Type']
cover_wild_type['Wilderness_Area']=wilderness.apply(rev_code, axis=1)
cover_wild_type
plt.figure(figsize=(20,10))
sns.countplot(x='Wilderness_Area', hue='Cover_Type',data=cover_wild_type, palette="rainbow")
plt.xticks(rotation=90)
plt.title('Count Plot of Wildernes Areas',fontsize=15)
mask = np.zeros_like(features.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

plt.figure(figsize=(13,8))
plt.title('Correlation Table', fontsize=20)
sns.heatmap(features.corr(),mask=mask,cmap='Blues',linecolor='white',annot=True)
x = data.loc[:,'Elevation':'Soil_Type40']
y = data['Cover_Type']
remove = ['Hillshade_3pm','Soil_Type7','Soil_Type8','Soil_Type14','Soil_Type15',
     'Soil_Type21','Soil_Type25','Soil_Type28','Soil_Type36','Soil_Type37']
x.drop(remove, axis=1, inplace=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)
AC = [] # Accuracy comparisons of the algorithms
logreg = LogisticRegression(solver='liblinear', multi_class='ovr')
logreg.fit(x_train, y_train)
logreg_pred = logreg.predict(x_test)
logreg_accuracy = accuracy_score(logreg_pred , y_test)
AC.append(logreg_accuracy)
dectree = DecisionTreeClassifier()
dectree.fit(x_train, y_train)
dectree_pred = dectree.predict(x_test)
dectree_accuracy = accuracy_score(dectree_pred , y_test)
AC.append(dectree_accuracy)
plt.figure(figsize=(20,20))
plt.barh(x_train.columns.values, dectree.feature_importances_)
plt.title('Feature Importance for Decision Tree Algorithm',fontsize=20)
plt.ylabel('Feature Name')
plt.xlabel('Gini Value')
plt.show()
from sklearn import tree
plt.figure(figsize=(15,15))
tree.plot_tree(dectree, max_depth=5, fontsize=8)
plt.show()
randfor = RandomForestClassifier()
randfor.fit(x_train, y_train)
randfor_pred = randfor.predict(x_test)
randfor_accuracy = accuracy_score(randfor_pred , y_test)
AC.append(randfor_accuracy)
plt.figure(figsize=(20,20))
plt.barh(x_train.columns.values, randfor.feature_importances_)
plt.title('Feature Importance for Random Forest Algorithm',fontsize=20)
plt.ylabel('Feature Name')
plt.xlabel('Gini Value')
plt.show()
neighbors = np.arange(1,7)
train_accuracy =np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i,k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    train_accuracy[i] = knn.score(x_train, y_train)
    test_accuracy[i] = knn.score(x_test, y_test)
plt.figure(figsize=(15,10))
plt.title('k-NN Varying number of neighbors', fontsize=15)
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
knn_accuracy=knn.score(x_test,y_test)
print('KNN Accuracy:',knn_accuracy)
AC.append(knn_accuracy)
predicted = knn.predict(x_test)
predicted
pre = pd.DataFrame(data=predicted, columns=['Pre'])
pre
y_test
Y_test = pd.DataFrame(data=y_test)
yy_test = Y_test.reset_index()
yy_test.drop(columns=['index'])
hold = pd.DataFrame(columns=['Pre'])
a = []
counter = 0
for i in range(0,174304):
    if pre['Pre'][i] != yy_test['Cover_Type'][i]:
        a.append(pre['Pre'][i])
        counter = counter + 1
    else:
        a.append(0)
        counter = counter + 1

hold['Pre'] = a
A = range(0,174304)
plt.figure(figsize=(15,10))
sns.countplot(x='Pre', data=hold)
plt.title('The accurate predictions and others', fontsize=20)
plt.show()
accurate = 0
not_accurate = 0
for i in A:
    if hold['Pre'][i] == 0:
        accurate = accurate + 1
    else:
        not_accurate = not_accurate + 1
labels = 'Accurate', 'Not Accurate'
sizes = [accurate, not_accurate]

fig1, ax1 = plt.subplots()
fig1.set_size_inches(8,8)
ax1.pie(sizes, labels=labels, autopct='%1.1f%%')
ax1.axis('equal')
plt.title('Accuracy Rate', fontsize=20)
plt.show()
accuracydata = pd.DataFrame(data=None,columns=['Accuracy'], index=['Logistic Regression','Decision Tree Classifier','Random Forest Classifier','KNN Algorithm'])
accuracydata['Accuracy'][0] = AC[0]
accuracydata['Accuracy'][1] = AC[1]
accuracydata['Accuracy'][2] = AC[2]
accuracydata['Accuracy'][3] = AC[3]
accuracydata
sns.set(style="whitegrid")

f, ax = plt.subplots(figsize=(10, 3))
accuracydata = accuracydata.sort_values("Accuracy", ascending=False)

sns.set_color_codes("deep")
sns.barplot(x="Accuracy", y=accuracydata.index, data=accuracydata,label="Accuracy", color="g")

ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(xlim=(0, 1), ylabel="",xlabel="Accuracy Rate")

plt.title('Comparison of Accuracy Rates of the Algorithms', fontsize=20)
def predictor(el,asp,sc,hh,vh,hr,h9,hN,hf,wa,st):
    datap = {'Elevation':0, 'Aspect':0, 'Slope':0, 'Horizontal_Distance_To_Hydrology':0,
       'Vertical_Distance_To_Hydrology':0, 'Horizontal_Distance_To_Roadways':0,
       'Hillshade_9am':0, 'Hillshade_Noon':0, 'Horizontal_Distance_To_Fire_Points':0, 'Wilderness_Area1':0,
       'Wilderness_Area2':0, 'Wilderness_Area3':0, 'Wilderness_Area4':0,
       'Soil_Type1':0, 'Soil_Type2':0, 'Soil_Type3':0, 'Soil_Type4':0, 'Soil_Type5':0,
       'Soil_Type6':0, 'Soil_Type9':0, 'Soil_Type10':0,
       'Soil_Type11':0, 'Soil_Type12':0, 'Soil_Type13':0, 
       'Soil_Type16':0, 'Soil_Type17':0, 'Soil_Type18':0,
       'Soil_Type19':0, 'Soil_Type20':0, 'Soil_Type22':0,
       'Soil_Type23':0, 'Soil_Type24':0, 'Soil_Type26':0,
       'Soil_Type27':0, 'Soil_Type29':0, 'Soil_Type30':0,
       'Soil_Type31':0, 'Soil_Type32':0, 'Soil_Type33':0, 
       'Soil_Type34':0, 'Soil_Type35':0, 'Soil_Type38':0,
       'Soil_Type39':0, 'Soil_Type40':0}
    datapre = pd.DataFrame(data=datap,index=[0])
    datapre['Elevation'] = el
    datapre['Aspect'] = asp
    datapre['Slope'] = sc
    datapre['Horizontal_Distance_To_Hydrology'] = hh
    datapre['Vertical_Distance_To_Hydrology'] = vh
    datapre['Horizontal_Distance_To_Roadways'] = hr
    datapre['Hillshade_9am'] = h9
    datapre['Hillshade_Noon'] = hN
    datapre['Horizontal_Distance_To_Fire_Points'] = hf
    datapre['Wilderness_Area' + str(wa)] =1
    datapre['Soil_Type' + str(st)] =1
    predicted = knn.predict(datapre)
    print('The predicted cover type is: ' + str(predicted))
predictor(2884,33,12,350,58,872,218,214,1544,3,35)
predictedknn = knn.predict(x)
preknn = pd.DataFrame(data=predictedknn, columns=['Pre'])
preknn.to_csv('submission.csv', index=False)