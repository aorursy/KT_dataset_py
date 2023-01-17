from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

df = pd.read_csv("../input/adult.csv")
df.head()
print('df shape:', df.shape)
print('df size:', df.size)
print('df.ndim:', df.ndim, '\n')
print('df.index:', df.index)
print('df.columns:', df.columns)
df.info()
df.describe()
# Total number of records
records_number = len(df)
#Number of records where individual's income is more than $50,000
greater_50k = len(df.query('income == ">50K"'))
#Number of records where individual's income is at most $50,000
atmost_50k = len(df.query('income == "<=50K"'))
#Percentage of individuals's income exceeds $50,000
greater_50k_percent = (float(greater_50k) / records_number * 100)

print("Total number of records: {}".format(records_number))
print("individuals's income exceeds $50,000: {}".format(greater_50k))
print("individuals's income is at most $50,000: {}".format(atmost_50k))
print("Percentage of individuals's income exceeds $50,000: {:.2f}%".format(greater_50k_percent))
df.groupby(["workclass"]).size().plot(kind="bar",fontsize=14)
df.groupby(["income","workclass"]).size().unstack("income").plot(kind="bar",fontsize=14)
df.groupby(["education"]).size().plot(kind="bar",fontsize=14)
df.groupby(["income","education"]).size().unstack("income").plot(kind="bar",fontsize=14)
df.groupby(["marital.status"]).size().plot(kind="bar",fontsize=14)
df.groupby(["income","marital.status"]).size().unstack("income").plot(kind="bar",fontsize=14)
df.groupby(["occupation"]).size().plot(kind="bar",fontsize=14,x=df.groupby(["occupation"]).size(),y=df[["occupation"]])
df.groupby(["income","occupation"]).size().unstack("income").plot(kind="bar",fontsize=12)
df.groupby(["relationship"]).size().plot(kind="bar",fontsize=14)
df.groupby(["income","relationship"]).size().unstack("income").plot(kind="bar",fontsize=12)
df.groupby(["race"]).size().plot(kind="bar",fontsize=14)
df.groupby(["income","race"]).size().unstack("income").plot(kind="bar",fontsize=12)
df.groupby(["native.country"]).size().plot(kind="bar",fontsize=11)
df.groupby(["income","native.country"]).size().unstack("income").plot(kind="bar")
df.groupby(["sex"]).size().plot(kind="bar",fontsize=14)
df.groupby(["income","sex"]).size().unstack("income").plot(kind="bar")
df.pivot_table(df, index=['income'], aggfunc=np.mean)
import seaborn as sns
hmap = df.corr()
plt.subplots(figsize=(12, 9))
sns.heatmap(hmap, vmax=.8,annot=True,cmap="BrBG", square=True)
df=df.drop(columns='fnlwgt')#drops column:fnlwgt
def questionmark_number(x):
    return sum(x=='?')
df.apply(questionmark_number)
print('1.workclass:',set(df['workclass']) )
df['workclass'] = df['workclass'].map({'?':-1, 'Without-pay':0,'Never-worked':1, 'Local-gov':2,'State-gov':3, 'Federal-gov':3,
                                      'Private':4, 'Self-emp-not-inc':5, 'Self-emp-inc':6})
education_set = set(df['education']) 
print('2.education:',education_set)
df['education'] = df['education'].map({'Preschool':0,'1st-4th':1,'5th-6th':2, '7th-8th':3,
                                      '9th':4, '10th':5, '11th':6, '12th':7, 'Prof-school':8, 
                                      'HS-grad':9, 'Some-college':10, 'Assoc-voc':11, 'Assoc-acdm':12,
                                       'Bachelors':13, 'Masters':14, 'Doctorate':15})
maritalstatus_set = set(df['marital.status']) 
print('3.marital.status:',maritalstatus_set)
df['marital.status'] = df['marital.status'].map({'Never-married':0,'Widowed':1,'Divorced':2, 'Separated':3,
                                      'Married-spouse-absent':4, 'Married-civ-spouse':5, 'Married-AF-spouse':6})
print('4.occupation:',set(df['occupation']))
df['occupation'] = df['occupation'].map({'?':-1,'Priv-house-serv':0,'Protective-serv':1,'Handlers-cleaners':2, 'Machine-op-inspct':3,
                                      'Adm-clerical':4, 'Farming-fishing':5, 'Transport-moving':6, 'Craft-repair':7, 'Other-service':8,
                                       'Tech-support':9, 'Sales':10, 'Exec-managerial':11, 'Prof-specialty':12, 'Armed-Forces':13 })
print('5.relationship:',set(df['relationship']))
df['relationship'] = df['relationship'].map({'Unmarried':0,'Other-relative':1, 'Not-in-family':2,
                                      'Wife':3, 'Husband':4,'Own-child':5})
print('6.race:',set(df['race']) )
df['race'] = df['race'].map({'Black':0,'Asian-Pac-Islander':1,'Amer-Indian-Eskimo':2, 'Other':3,
                                      'White':4})
print('7.sex:',set(df['sex']))
df['sex'] = df['sex'].map({'Male':0,'Female':1})
print('8.native-country:',set(df['native.country']))
df['native.country'] = df['native.country'].map({'?':-1,'Puerto-Rico':0,'Haiti':1,'Cuba':2, 'Iran':3,
                                      'Honduras':4, 'Jamaica':5, 'Vietnam':6, 'Mexico':7, 'Dominican-Republic':8,
                                       'Laos':9, 'Ecuador':10, 'El-Salvador':11, 'Cambodia':12, 'Columbia':13,
                                         'Guatemala':14, 'South':15, 'India':16, 'Nicaragua':17, 'Yugoslavia':18, 
                                         'Philippines':19, 'Thailand':20, 'Trinadad&Tobago':21, 'Peru':22, 'Poland':23, 
                                         'China':24, 'Hungary':25, 'Greece':26, 'Taiwan':27, 'Italy':28, 'Portugal':29, 
                                         'France':30, 'Hong':31, 'England':32, 'Scotland':33, 'Ireland':34, 
                                         'Holand-Netherlands':35, 'Canada':36, 'Germany':37, 'Japan':38, 
                                         'Outlying-US(Guam-USVI-etc)':39, 'United-States':40
                                        })
print('9.income:',set(df['income']) )
df['income'] = df['income'].map({'<=50K':0,'>50K':1})
x = df.iloc[ : ,:-1].values.astype(int)
x
y = df[['income']]
hmap = df.corr()
plt.subplots(figsize=(12, 9))
sns.heatmap(hmap, vmax=.8,annot=True,cmap="BrBG", square=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
y_train.groupby(["income"]).size()
def transet_atmost_50k(x):
    return sum(x==0)
def transet_greater_than_50k(x):
    return sum(x==1)
transet_greater_than_50k_n=float(y_train.apply(transet_greater_than_50k)/y_train.size)
("Tranining set's Percentage of individuals making more than $50,000: ", "{:.2f}%".format(transet_greater_than_50k_n*100))
classifier1 = DecisionTreeClassifier()
classifier1.fit(x_train, y_train)
y_predict1_test=classifier1.predict(x_test)
y_predict1_train=classifier1.predict(x_train)
classifier1
classifier2 = DecisionTreeClassifier(max_leaf_nodes=8)
classifier2.fit(x_train, y_train)
y_predict2_test=classifier2.predict(x_test)
y_predict2_train=classifier2.predict(x_train)

classifier3 = DecisionTreeClassifier(max_leaf_nodes=16)
classifier3.fit(x_train, y_train)
y_predict3_test=classifier3.predict(x_test)
y_predict3_train=classifier3.predict(x_train)

classifier4 = DecisionTreeClassifier(max_leaf_nodes=32)
classifier4.fit(x_train, y_train)
y_predict4_test=classifier4.predict(x_test)
y_predict4_train=classifier4.predict(x_train)

classifier5 = DecisionTreeClassifier(max_leaf_nodes=64)
classifier5.fit(x_train, y_train)
y_predict5_test=classifier5.predict(x_test)
y_predict5_train=classifier5.predict(x_train)

classifier6 = DecisionTreeClassifier(max_leaf_nodes=128)
classifier6.fit(x_train, y_train)
y_predict6_test=classifier6.predict(x_test)
y_predict6_train=classifier6.predict(x_train)
classifier7 = DecisionTreeClassifier(min_impurity_decrease=0.001)
classifier7.fit(x_train, y_train)
y_predict7_test=classifier7.predict(x_test)
y_predict7_train=classifier7.predict(x_train)

classifier8 = DecisionTreeClassifier(min_impurity_decrease=0.01)
classifier8.fit(x_train, y_train)
y_predict8_test=classifier8.predict(x_test)
y_predict8_train=classifier8.predict(x_train)

classifier9 = DecisionTreeClassifier(min_impurity_decrease=0.02)
classifier9.fit(x_train, y_train)
y_predict9_test=classifier9.predict(x_test)
y_predict9_train=classifier9.predict(x_train)

classifier10 = DecisionTreeClassifier(min_impurity_decrease=0.03)
classifier10.fit(x_train, y_train)
y_predict10_test=classifier10.predict(x_test)
y_predict10_train=classifier10.predict(x_train)

classifier11 = DecisionTreeClassifier(min_impurity_decrease=0.04)
classifier11.fit(x_train, y_train)
y_predict11_test=classifier11.predict(x_test)
y_predict11_train=classifier11.predict(x_train)
classifier12 = DecisionTreeClassifier(min_samples_leaf=40, min_samples_split=80)
classifier12.fit(x_train, y_train)
y_predict12_test=classifier12.predict(x_test)
y_predict12_train=classifier12.predict(x_train)

classifier13 = DecisionTreeClassifier(min_samples_leaf=80, min_samples_split=160)
classifier13.fit(x_train, y_train)
y_predict13_test=classifier13.predict(x_test)
y_predict13_train=classifier13.predict(x_train)

classifier14 = DecisionTreeClassifier(min_samples_leaf=160, min_samples_split=320)
classifier14.fit(x_train, y_train)
y_predict14_test=classifier14.predict(x_test)
y_predict14_train=classifier14.predict(x_train)

classifier15 = DecisionTreeClassifier(min_samples_leaf=320, min_samples_split=640)
classifier15.fit(x_train, y_train)
y_predict15_test=classifier15.predict(x_test)
y_predict15_train=classifier15.predict(x_train)

classifier16 = DecisionTreeClassifier(min_samples_leaf=640, min_samples_split=1280)
classifier16.fit(x_train, y_train)
y_predict16_test=classifier16.predict(x_test)
y_predict16_train=classifier16.predict(x_train)
classifier17 = DecisionTreeClassifier(max_leaf_nodes=16,min_impurity_decrease=0.01)
classifier17.fit(x_train, y_train)
y_predict17_test=classifier17.predict(x_test)
y_predict17_train=classifier17.predict(x_train)

classifier18 = DecisionTreeClassifier(max_leaf_nodes=16,min_impurity_decrease=0.02)
classifier18.fit(x_train, y_train)
y_predict18_test=classifier18.predict(x_test)
y_predict18_train=classifier18.predict(x_train)

classifier19 = DecisionTreeClassifier(max_leaf_nodes=16,min_impurity_decrease=0.001)
classifier19.fit(x_train, y_train)
y_predict19_test=classifier19.predict(x_test)
y_predict19_train=classifier19.predict(x_train)
classifier20 = DecisionTreeClassifier(max_leaf_nodes=16,min_samples_leaf=40, min_samples_split=80)
classifier20.fit(x_train, y_train)
y_predict20_test=classifier20.predict(x_test)
y_predict20_train=classifier20.predict(x_train)

classifier21 = DecisionTreeClassifier(max_leaf_nodes=16,min_samples_leaf=80, min_samples_split=160)
classifier21.fit(x_train, y_train)
y_predict21_test=classifier21.predict(x_test)
y_predict21_train=classifier21.predict(x_train)

classifier22 = DecisionTreeClassifier(max_leaf_nodes=16,min_samples_leaf=160, min_samples_split=320)
classifier22.fit(x_train, y_train)
y_predict22_test=classifier22.predict(x_test)
y_predict22_train=classifier22.predict(x_train)
print("Classifier1(default) Accuracy:", '%f'%classifier1.score(x_test, y_test))
print("Classifier2(max_leaf_nodes=8) Accuracy:", '%f'%classifier2.score(x_test, y_test))
print("Classifier3(max_leaf_nodes=16) Accuracy:", '%f'%classifier3.score(x_test, y_test))
print("Classifier4(max_leaf_nodes=32) Accuracy:", '%f'%classifier4.score(x_test, y_test))
print("Classifier5(max_leaf_nodes=64) Accuracy:", '%f'%classifier5.score(x_test, y_test))
print("Classifier6(max_leaf_nodes=128) Accuracy:", '%f'%classifier6.score(x_test, y_test))
print("Classifier7(min_impurity_decrease=0.001) Accuracy:", '%f'%classifier7.score(x_test, y_test))
print("Classifier8(min_impurity_decrease=0.01) Accuracy:", '%f'%classifier8.score(x_test, y_test))
print("Classifier9(min_impurity_decrease=0.02) Accuracy:", '%f'%classifier9.score(x_test, y_test))
print("Classifier10(min_impurity_decrease=0.03) Accuracy:", '%f'%classifier10.score(x_test, y_test))
print("Classifier11(min_impurity_decrease=0.04) Accuracy:", '%f'%classifier11.score(x_test, y_test))
print("Classifier12(min_samples_leaf=40, min_samples_split=80) Accuracy:", '%f'%classifier12.score(x_test, y_test))
print("Classifier13(min_samples_leaf=80, min_samples_split=160) Accuracy:", '%f'%classifier13.score(x_test, y_test))
print("Classifier14(min_samples_leaf=160, min_samples_split=320) Accuracy:", '%f'%classifier14.score(x_test, y_test))
print("Classifier15(min_samples_leaf=320, min_samples_split=640) Accuracy:", '%f'%classifier15.score(x_test, y_test))
print("Classifier16(min_samples_leaf=640, min_samples_split=1280) Accuracy:", '%f'%classifier16.score(x_test, y_test))
print("Classifier17(max_leaf_nodes=16,min_impurity_decrease=0.001) Accuracy:", '%f'%classifier17.score(x_test, y_test))
print("Classifier18(max_leaf_nodes=16,min_impurity_decrease=0.01) Accuracy:", '%f'%classifier18.score(x_test, y_test))
print("Classifier19(max_leaf_nodes=16,min_impurity_decrease=0.02) Accuracy:", '%f'%classifier19.score(x_test, y_test))
print("Classifier20(max_leaf_nodes=16,min_samples_leaf=40, min_samples_split=80) Accuracy:", '%f'%classifier20.score(x_test, y_test))
print("Classifier21(max_leaf_nodes=16,min_samples_leaf=80, min_samples_split=160) Accuracy:", '%f'%classifier21.score(x_test, y_test))
print("Classifier22(max_leaf_nodes=16,min_samples_leaf=160, min_samples_split=320) Accuracy:", '%f'%classifier22.score(x_test, y_test))
print(" Classifier 5 confusion matrix:",'\n',confusion_matrix(y_test, y_predict5_test))  
### Model evaluation index
print("Classifier 5 model evaluation index：\n", classification_report(y_predict5_test, y_test, target_names=["at_most_50K","greater_than_50K"]))
## Step 3: Visualizing the decision tree model and analyzing the model result
from sklearn.tree import export_graphviz
import pydotplus
dot_data1 = export_graphviz(classifier3
                           ,out_file=None
                           ,feature_names=['age','workclass','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain', 'capital-loss','hours-per-week','native-country']
                           ,class_names=['at_most_50K','greater_than_50K']
                           ,filled=True
                           ,rounded=True
                           ,special_characters=True)
graph1 = pydotplus.graph_from_dot_data(dot_data1)
from IPython.display import Image
Image(graph1.create_png())