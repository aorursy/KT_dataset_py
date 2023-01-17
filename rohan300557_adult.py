import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
missing_val = '?'
df = pd.read_csv('../input/adult-census-income/adult.csv',na_values = missing_val)
df.head()
df.tail()
df.shape
df.info()
df.drop('fnlwgt',axis=1,inplace=True)
df.head()
df.isnull().any(axis=0)
df.isnull().sum()
df.loc[df['workclass'].isnull()==True,['workclass','occupation','native.country']]
le = round((1-(len(df.dropna())/len(df)))*100,2)
print('The Percentage of Rows that were removed while handling missing values : {0} %'.format(le))
df.dropna(inplace = True)
male = (df[df.sex == "Male"].sex.value_counts().values[0]/df.shape[0])*100
female = (df[df.sex == "Female"].sex.value_counts().values[0]/df.shape[0])*100
print('Percentage of Male is : %d %%' % round(male))
print('Percentage of Female is : %d %%' % round(female))
df.describe(include='number')
df.describe(include=["O"])
### Workclass
plt.figure(figsize=(10,5))
sb.countplot(df['workclass'],palette="Set3")
plt.show()
### Education 
plt.figure(figsize=(20,5))
sb.countplot(df['education'])
plt.show()
### marital.status
plt.figure(figsize=(15,5))
sb.countplot(df['marital.status'],palette="Set1")
plt.show()
### Occupation
plt.figure(figsize=(25,8))
sb.countplot(df['occupation'])
plt.show()
### Relationship
plt.figure(figsize=(10,5))
sb.countplot(df['relationship'], facecolor=(0, 0, 0, 0), linewidth=5,edgecolor=sb.color_palette("dark", 6))
plt.show()
### Race
plt.figure(figsize=(8,5))
sb.countplot(df['race'])
plt.show()
### Sex
sb.countplot(df['sex'],palette="gnuplot_r")
plt.show()
### Native Country
plt.ylabel('counts')
plt.xlabel('native.country')
sb.barplot(df['native.country'].value_counts().index[:5] , df['native.country'].value_counts().values[:5])
plt.show()
df['native.country'].value_counts().head()
### workclass 
plt.figure(figsize=(10,5))
plt.pie(df["workclass"].value_counts(),autopct='%.2f%%', shadow=True,labels=df.workclass.unique())
plt.show()
### Sex
plt.figure(figsize=(10,5))
plt.pie(df["sex"].value_counts(),autopct='%.2f%%', shadow=True,labels=df.sex.unique())
plt.show()
### Race
plt.figure(figsize=(10,5))
plt.pie(df["race"].value_counts(),autopct='%.2f%%', shadow=True,labels=df.race.unique())
plt.show()
### income
plt.figure(figsize=(10,5))
plt.pie(df["income"].value_counts(),autopct='%.2f%%', shadow=True,labels=df.income.unique())
plt.show()
### Age
plt.figure(figsize=(18,5))
plt.ylabel('Frequency')
plt.xlabel('Age')
sb.distplot(df['age'], rug=True)
plt.show()
### Educational.num : Number of Education Year
plt.figure(figsize=(10,5))
plt.ylabel('Frequency')
plt.xlabel('Education.num')
sb.distplot(df['education.num'], rug=True)
plt.show()
### Capital Gain
plt.figure(figsize=(15,5))
plt.subplot(121)
plt.ylabel('Frequency')
plt.xlabel('capital.gain')
sb.distplot(df['capital.gain'], kde_kws={'bw':0.1})
plt.subplot(122)
plt.ylabel('Frequency')
plt.xlabel('capital.loss')
sb.distplot(df['capital.loss'],kde_kws={'bw':0.1})
plt.show()
### Hours per week 
plt.figure(figsize=(15,5))
plt.ylabel('Frequency')
plt.xlabel('hours.per.week')
sb.distplot(df['hours.per.week'])
plt.show()
### Age
sb.boxplot(df['age'])
plt.show()
### Educational num
plt.figure(figsize=(10,5))
sb.boxplot(df['education.num'])
plt.show()
### Capital Gain
plt.figure(figsize=(15, 4))
plt.subplot(121)
sb.boxplot(df['capital.gain'])
plt.title('Capital Gain')
plt.subplot(122)
sb.boxplot(df['capital.loss'])
plt.title('Capital Loss')
plt.show()
### Hours Per Week
plt.figure(figsize=(10,5))
sb.boxplot(df['hours.per.week'])
plt.show()

### Age
plt.figure(figsize=(18,5))
plt.ylabel('Frequency')
plt.xlabel('Age')
sb.distplot(df['age'],hist = False,kde=True)
plt.show()
### Educational-num : Number of Education Year
plt.figure(figsize=(10,5))
plt.ylabel('Frequency')
plt.xlabel('Education.num')
sb.distplot(df['education.num'], hist=False, rug=True)
plt.show()
### Between Capital Gain and Capital Loss
plt.figure(figsize=(12,5))
sb.scatterplot(x = df['capital.gain'], y = df['capital.loss'])
plt.show()
sb.jointplot(x='age', y='hours.per.week',kind= 'hex', data=df, color='r')
plt.show()
sb.jointplot(x='capital.gain', y='capital.loss', data=df, kind='hex', color='g')
plt.show()
df.groupby(['income']).mean()
df.groupby(['income']).median()
### Age (Relationship with income)
fig = plt.figure(figsize=(10,5)) 
sb.boxplot(x="income", y="age", data=df)
plt.show()
### educational-num (Relationship with income)
fig = plt.figure(figsize=(10,5)) 
sb.boxplot(x="income", y="education.num", data=df)
plt.show()
### Captial Gain (Relationship with income)
fig = plt.figure(figsize=(10,5)) 
sb.boxplot(x="income", y="capital.gain", data=df)
plt.show()
### Captial Loss (Relationship with income)
fig = plt.figure(figsize=(10,5)) 
sb.boxplot(x="income", y="capital.loss", data=df)
plt.show()
### Hours per Week (Relationship with income)
fig = plt.figure(figsize=(10,5)) 
sb.boxplot(x="income", y="hours.per.week", data=df)
plt.show()
plt.figure(figsize=(15,5))
sb.boxplot(x="income", y="age",hue="sex",data=df)
plt.show()
plt.figure(figsize=(15,7))
sb.boxplot(x="income", y="hours.per.week",hue="sex",data=df)
plt.show()
plt.subplots(figsize=(16, 8))  
sb.boxplot(x='relationship', y='hours.per.week', hue='income', data=df)  
plt.show() 
plt.figure(figsize=(10,5))
sb.barplot(x="sex", y="age", hue="income", data=df)
plt.show()
plt.figure(figsize=(10,5))
sb.barplot(x="sex", y="hours.per.week", hue="income", data=df)
plt.show()
plt.figure(figsize=(15,5))
sb.barplot(x="race", y="hours.per.week", hue="income", data=df)
plt.show()
plt.figure(figsize=(15,5))
sb.barplot(x="relationship", y="hours.per.week", hue="income", data=df)
plt.show()
plt.figure(figsize=(25,7)) 
sb.pairplot(df,hue='income',diag_kws={'bw':'1.0'},palette="husl",markers=["o", "s"])
plt.show()
t1=df.corr()
plt.figure(figsize=(12,9))
ax = sb.heatmap(t1,linewidths=.5,annot=True)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.show()
cat = df.select_dtypes(include=['object'])
cat
encoder = OneHotEncoder(drop='first', sparse=False)
cat_df = pd.DataFrame(encoder.fit_transform(cat), 
                      columns=encoder.get_feature_names(cat.columns))
cat_df
num = df.select_dtypes(include=['int64'])
num
scaler = StandardScaler()
num_df = pd.DataFrame(scaler.fit_transform(num), 
                      columns = num.columns)
num_df.head()
new_df = pd.concat([num_df, cat_df], axis=1)
new_df
# X = new_df.drop('income_>50K',axis=1)
Y = new_df.pop('income_>50K')
X = new_df
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=.30,random_state=25)
lr = LogisticRegression( max_iter=250)
lr.fit(X_train,y_train)
y_predict = lr.predict(X_test)
y_predict
# Accuracy classification score.
accuracy_score(y_test, y_predict)
confusion_matrix(y_test, y_predict)
print(classification_report(y_test,y_predict))
from sklearn.feature_selection import RFE
lr_temp = LogisticRegression( max_iter=250)
## No. of features taking = 30
rfe = RFE(lr_temp, 30)
rfe = rfe.fit(X_train, y_train)
rfe
temp_df = pd.DataFrame({'Columns' : X_train.columns, 'Included' : rfe.support_, 'Ranking' : rfe.ranking_})
temp_df
imp_col = X_train.columns[rfe.support_]
imp_col
X_train_new = X_train[imp_col]
X_train_new
lrn = LogisticRegression(max_iter=250)
lrn.fit(X_train_new, y_train)
y_predict_new = lrn.predict(X_train_new)
y_predict_new
accuracy_score(y_test, y_predict)
pca = PCA(whiten=True)
pca.fit(X)
np.cumsum(pca.explained_variance_ratio_)
plt.figure(figsize = (12, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cummulative variance')
plt.show()
## Number of components = 42
pca = PCA(n_components=42,whiten=True)
X_PCA = pca.fit_transform(X)
X_PCA
x_train, x_test, y_train, y_test = train_test_split(X_PCA[:], Y, test_size=.30, random_state=25)
logist = LogisticRegression(max_iter = 250)
logist.fit(x_train,y_train)
##Predicting the test data
y_predict_logistic = logist.predict(x_test)
print('Accuracy score : {}'.format(accuracy_score(y_test,y_predict_logistic)))
print('\n')
print('Confusion matrix :')
print(confusion_matrix(y_test,y_predict_logistic))
print('\n')
print('Classification Report :')
print(classification_report(y_test,y_predict_logistic))
svm = svm.SVC()
svm.fit(x_train,y_train)
##Predicting the test data
y_predict_svm = svm.predict(x_test)
print('Accuracy score : {}'.format(accuracy_score(y_test,y_predict_svm)))
print('\n')
print('Confusion matrix :')
print(confusion_matrix(y_test,y_predict_svm))
print('\n')
print('Classification Report :')
print(classification_report(y_test,y_predict_svm))
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
##Predicting the test data
y_predict_knn = knn.predict(x_test)
print('Accuracy score : {}'.format(accuracy_score(y_test,y_predict_knn)))
print('\n')
print('Confusion matrix :')
print(confusion_matrix(y_test,y_predict_knn))
print('\n')
print('Classification Report :')
print(classification_report(y_test,y_predict_knn))
dtc = DecisionTreeClassifier()
dtc.fit(x_train,y_train)
##Predicting the test data
y_predict_dtc = dtc.predict(x_test)
print('Accuracy score : {}'.format(accuracy_score(y_test,y_predict_dtc)))
print('\n')
print('Confusion matrix :')
print(confusion_matrix(y_test,y_predict_dtc))
print('\n')
print('Classification Report :')
print(classification_report(y_test,y_predict_dtc))
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(x_train,y_train)
## Prediction the test data
y_predict_rfc = rfc.predict(x_test)
print('Accuracy score : {}'.format(accuracy_score(y_test,y_predict_rfc)))
print('\n')
print('Confusion matrix :')
print(confusion_matrix(y_test,y_predict_rfc))
print('\n')
print('Classification Report :')
print(classification_report(y_test,y_predict_rfc))
from sklearn import svm
def lets_try(train,labels):
    results={}
    
    def test_model(model):
        cv = KFold(n_splits=5,shuffle=True,random_state=15)
        predicted = cross_val_score(model, train, Y, cv=cv)
        scores=[predicted.mean()]
        return scores
    
    model = LogisticRegression()
    results["LogisticRegression"]=test_model(model)
    
    model = svm.SVC()
    results["SVM"]=test_model(model)
    
    model = KNeighborsClassifier(n_neighbors=3)
    results['KNeighborsClassifier']=test_model(model)
    
    model = DecisionTreeClassifier()
    results["DecisionTreeClassifier"]=test_model(model)
    
    model = RandomForestClassifier()
    results["RandomForestClassifier"]=test_model(model)
    
    results = pd.DataFrame.from_dict(results,orient='index')
    results.columns=["Accuracy Score"]
    
    results.plot(kind="bar",title="Model Scores")
    axes = plt.gca()
    axes.set_ylim([0.5,1])
    plt.show()
    
    return results

lets_try(X_PCA[:],Y)